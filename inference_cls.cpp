// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "platform.h"
#include "net.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

int parse_images_dir(const std::string& base_path, std::vector<std::string>& file_path)
{
    file_path.clear();

    const cv::String base_path_str(base_path);
    std::vector<cv::String> image_list;

    cv::glob(base_path_str, image_list, true);

    for (size_t i = 0; i < image_list.size(); i++)
    {
        const cv::String& image_path = image_list[i];
        file_path.push_back(image_path);
    }

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());
    int pred_idx;
    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        if(0==i)
        {
            pred_idx = index;
        }
        // fprintf(stderr, "%d = %f\n", index, score);
    }

    return pred_idx;
}

static int detect_net(const std::vector<std::string>& image_list, std::vector<float>& cls_scores, 
const std::string ncnn_param_file_path, const std::string ncnn_bin_file_path, const std::string out_layer)
{
    ncnn::Net net;
    size_t size = image_list.size();
    printf("Number of images: %lu\n", size);
    
#if NCNN_VULKAN
    net.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    net.load_param(&ncnn_param_file_path[0]);
    net.load_model(&ncnn_bin_file_path[0]);

    const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
    const float std_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
    int correct_count = 0;
    int label = -1;
    std::string folder_name = "dummy";
    for (size_t i = 0; i < image_list.size(); i++)
    {
        
        std::string img_name = image_list[i];

        std::istringstream f(img_name);
        std::string s;
        while(std::getline(f, s, '/'))
        {
            if((s.substr(0, 2) == "n0" || s.substr(0, 2) == "n1") && s.size() == 9 && folder_name != s)
            {
                label++;
                folder_name = s;
            }
        }

        if ((i + 1) % 1000 == 0)
        {
            fprintf(stderr, "          %d/%d, acc:%f\n", static_cast<int>(i + 1), static_cast<int>(size), static_cast<float>(correct_count)/static_cast<float>(i));
        }

#if OpenCV_VERSION_MAJOR > 2
        cv::Mat bgr = cv::imread(img_name, cv::IMREAD_COLOR);
#else
        cv::Mat bgr = cv::imread(img_name, CV_LOAD_IMAGE_COLOR);
#endif
        if (bgr.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", img_name.c_str());
            return -1;
        }

        ncnn::Mat resized = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, 256, 256);
        ncnn::Mat in;
        ncnn::copy_cut_border(resized, in, 16, 16, 16, 16);
        in.substract_mean_normalize(mean_vals, std_vals);

        ncnn::Extractor ex = net.create_extractor();
        ex.set_num_threads(2);

        ex.input("0", in);

        ncnn::Mat out;
        ex.extract(&out_layer[0], out);

        cls_scores.resize(out.w);
        for (int j=0; j<out.w; j++)
        {
            cls_scores[j] = out[j];
        }
        int pred_idx = print_topk(cls_scores, 3);
        // printf("label: %d, pred: %d\n", label, pred_idx);
        // printf("=======================================\n");
        if(pred_idx == label)
        {
            correct_count++;
        }
    }
    printf("Acc: %f\n", static_cast<float>(correct_count)/static_cast<float>(size));
    return 0;
}

int main(int argc, char** argv)
{
    const char* key_map =
        "{help h usage ? |   | print this message }"
        "{param p        |   | path to ncnn.param file }"
        "{bin b          |   | path to ncnn.bin file }"
        "{images i       |   | path to calibration images folder }"
        "{out_layer o       |   | name of the final layer (innerproduct or softmax) }"
    ;

    cv::CommandLineParser parser(argc, argv, key_map);
    const std::string image_folder_path = parser.get<cv::String>("images");
    const std::string ncnn_param_file_path = parser.get<cv::String>("param");
    const std::string ncnn_bin_file_path = parser.get<cv::String>("bin");
    const std::string out_layer = parser.get<cv::String>("out_layer");

    // check the input param
    if (image_folder_path.empty() || ncnn_param_file_path.empty() || ncnn_bin_file_path.empty())
    {
        fprintf(stderr, "One or more path may be empty, please check and try again.\n");
        return 0;
    }

    // parse the image file.
    std::vector<std::string> image_file_path_list;
    parse_images_dir(image_folder_path, image_file_path_list);

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif // NCNN_VULKAN

    std::vector<float> cls_scores;
    detect_net(image_file_path_list, cls_scores, ncnn_param_file_path, ncnn_bin_file_path, out_layer);

#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

    return 0;
}
