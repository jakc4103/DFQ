import torch
from modeling.detection.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from modeling.detection.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from modeling.detection.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from modeling.detection.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from dataset.detection.voc_dataset import VOCDataset
from dataset.detection.open_images import OpenImagesDataset
from utils.detection import box_utils, measurements
from utils.detection.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys
import time
from modeling.detection.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

from utils.relation import create_relation
from dfq import cross_layer_equalization, bias_absorption, bias_correction, clip_weight
from utils.layer_transform import switch_layers, replace_op, restore_op, set_quant_minmax, merge_batchnorm, quantize_targ_layer
from PyTransformer.transformers.torchTransformer import TorchTransformer
from utils.quantize import QuantConv2d, QuantLinear, QuantNConv2d, QuantNLinear, QuantMeasure, QConv2d, QLinear, set_layer_bits
from ZeroQ.distill_data import getDistilData
from improve_dfq import update_scale, transform_quant_layer, set_scale, update_quant_range, set_update_stat, bias_correction_distill

parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="mb2-ssd-lite",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str, default='modeling/detection/mb2-ssd-lite-mp-0_686.pth')

parser.add_argument("--dataset_type", default="voc07", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, default='/home/jakc4103/WDesktop/dataset/VOCdevkit/VOC2007/', help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, default='modeling/detection/voc-model-labels.txt', help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
parser.add_argument("--quantize", action='store_true')
parser.add_argument("--equalize", action='store_true')
parser.add_argument("--correction", action='store_true')
parser.add_argument("--absorption", action='store_true')
parser.add_argument("--distill_range", action='store_true')
parser.add_argument("--log", action='store_true')
parser.add_argument("--relu", action='store_true')
parser.add_argument("--clip_weight", action='store_true')
parser.add_argument("--trainable", action='store_true')
parser.add_argument("--equal_range", type=float, default=1e8)
parser.add_argument("--bits_weight", type=int, default=8)
parser.add_argument("--bits_activation", type=int, default=8)
parser.add_argument("--bits_bias", type=int, default=16)
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


if __name__ == '__main__':
    assert args.relu or args.relu == args.equalize, 'must replace relu6 to relu while equalization'
    assert args.equalize or args.absorption == args.equalize, 'must use absorption with equalize'
    
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    # timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    if args.dataset_type == "voc07":
        dataset = VOCDataset(args.dataset, is_test=True)
    elif args.dataset_type == "voc12":
        dataset = VOCDataset('/home/jakc4103/WDesktop/dataset/VOCdevkit/VOC2012/', is_test=False)
    elif args.dataset_type == 'open_images':
        dataset = OpenImagesDataset(args.dataset, dataset_type="test")

    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True, quantize=args.quantize)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)  

    # timer.start("Load Model")
    net.load(args.trained_model)
    # net = net.to(DEVICE)

    data = torch.ones((4, 3, 300, 300))

    if args.distill_range:
        import copy
        # define FP32 model 
        model_original = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True, quantize=False)
        model_original.load(args.trained_model)
        model_original.eval()
        transformer = TorchTransformer()
        transformer._build_graph(model_original, data, [QuantMeasure])
        graph = transformer.log.getGraph()
        bottoms = transformer.log.getBottoms()

        data_distill = getDistilData(model_original, 'imagenet', 64, bn_merged=False,\
            num_batch=8, gpu=True, value_range=[-1., 1.], size=[300, 300], max_value=1., early_break_factor=0.04)

    ## network surgery here
    transformer = TorchTransformer()
    module_dict = {}
    if args.quantize:
        if args.distill_range:
            module_dict[1] = [(torch.nn.Conv2d, QConv2d), (torch.nn.Linear, QLinear)]
        elif args.trainable:
            module_dict[1] = [(torch.nn.Conv2d, QuantConv2d), (torch.nn.Linear, QuantLinear)]
        else:
            module_dict[1] = [(torch.nn.Conv2d, QuantNConv2d), (torch.nn.Linear, QuantNLinear)]
    
    if args.relu:
        module_dict[0] = [(torch.nn.ReLU6, torch.nn.ReLU)]

    # transformer.summary(net, data)
    # transformer.visualize(net, data, 'graph_ssd', graph_size=120)
    
    net, transformer = switch_layers(net, transformer, data, module_dict, ignore_layer=[QuantMeasure], quant_op=args.quantize)

    graph = transformer.log.getGraph()
    bottoms = transformer.log.getBottoms()
    output_shape = transformer.log.getOutShapes()
    if args.quantize:
        if args.distill_range:
            targ_layer = [QConv2d, QLinear]
        elif args.trainable:
            targ_layer = [QuantConv2d, QuantLinear]
        else:
            targ_layer = [QuantNConv2d, QuantNLinear]
    else:
        targ_layer = [torch.nn.Conv2d, torch.nn.Linear]

    if args.quantize:
        set_layer_bits(graph, args.bits_weight, args.bits_activation, args.bits_bias, targ_layer)

    net = merge_batchnorm(net, graph, bottoms, targ_layer)

    #create relations
    if args.equalize or args.distill_range:
        res = create_relation(graph, bottoms, targ_layer, delete_single=not args.distill_range)
        if args.equalize:
            cross_layer_equalization(graph, res, targ_layer, visualize_state=False, converge_thres=2e-7, s_range=(1/args.equal_range, args.equal_range))

        # if args.distill:
        #     set_scale(res, graph, bottoms, targ_layer)

    if args.absorption:
        bias_absorption(graph, res, bottoms, 3)
    
    if args.clip_weight:
        clip_weight(graph, range_clip=[-15, 15], targ_type=targ_layer)

    if args.correction:
        bias_correction(graph, bottoms, targ_layer)

    if args.quantize:
        if not args.trainable and not args.distill_range:
            graph = quantize_targ_layer(graph, args.bits_weight, args.bits_bias, targ_layer)

        if args.distill_range:
            set_update_stat(net, [QuantMeasure], True)
            net = update_quant_range(net.cuda(), data_distill, graph, bottoms, is_detection=True)
            set_update_stat(net, [QuantMeasure], False)
        else:
            set_quant_minmax(graph, bottoms, is_detection=True)

        torch.cuda.empty_cache()
    
    net = net.to(DEVICE)

    # print(f'It took {timer.end("Load Model")} seconds to load the model.')
    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net,nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.quantize:
        replace_op()

    results = []
    print("Start Inference")
    for i in range(len(dataset)):
        # print("process image", i)
        # timer.start("Load Image")
        image = dataset.get_image(i)
        # print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        # timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        # print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))

    if args.quantize:
        restore_op()

    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            args.iou_threshold,
            args.use_2007_metric
        )
        
        aps.append(ap)
        print(f"{class_name}: {ap}")
    print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
    if args.log:
        with open("ssd_result.txt", 'a+') as ww:
            ww.write("{}, quant: {}, relu: {}, equalize: {}, absorption: {}, correction: {}, clip: {}, distill_range: {}\n".format(
                args.dataset_type, args.quantize, args.relu, args.equalize, args.absorption, args.correction, args.clip_weight, args.distill_range
            ))
            ww.write("mAP: {}\n\n".format(sum(aps)/len(aps)))
