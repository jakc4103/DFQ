import torch
import torch.onnx
import argparse
import os
import sys
from torchvision import transforms, datasets

from PyTransformer.transformers.torchTransformer import TorchTransformer
from utils.relation import create_relation
from dfq import cross_layer_equalization, bias_absorption, bias_correction, _quantize_error, clip_weight
from utils.quantize import QuantNConv2d, QuantNLinear, QuantMeasure, QConv2d, QLinear, set_layer_bits
from utils.layer_transform import switch_layers, replace_op, restore_op, set_quant_minmax, merge_batchnorm, quantize_targ_layer#, LayerTransform
from modeling.classification.MobileNetV2 import mobilenet_v2
from modeling.segmentation.deeplab import DeepLab
from ZeroQ.distill_data import getDistilData
from improve_dfq import update_scale, transform_quant_layer, set_scale, update_quant_range, set_update_stat

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action='store_true')
    parser.add_argument("--equalize", action='store_true')
    parser.add_argument("--distill_range", action='store_true')
    parser.add_argument("--correction", action='store_true')
    parser.add_argument("--relu", action='store_true')
    parser.add_argument("--clip_weight", action='store_true')
    parser.add_argument("--resnet", action='store_true')
    parser.add_argument("--bits_weight", type=int, default=8)
    parser.add_argument("--bits_activation", type=int, default=8)
    parser.add_argument("--bits_bias", type=int, default=32)
    parser.add_argument("--dis_batch_size", type=int, default=64)
    parser.add_argument("--dis_num_batch", type=int, default=8)
    parser.add_argument("--ncnn_build", type=str, default='/home/jakc4103/Documents/ncnn/build')
    parser.add_argument("--image_path", type=str, default='/home/jakc4103/workspace/DFQ/cali_images/')
    return parser.parse_args()

def main():
    args = get_argument()
    # An instance of your model
    if args.resnet:
        import torchvision.models as models
        model = models.resnet18(pretrained=True)
    else:
        model = mobilenet_v2('modeling/classification/mobilenetv2_1.0-f2a8633.pth.tar')
    model.eval()

    data = torch.ones((4, 3, 224, 224))#.cuda()

    if args.distill_range:
        import copy
        # define FP32 model 
        model_original = copy.deepcopy(model)
        model_original.eval()
        transformer = TorchTransformer()
        transformer._build_graph(model_original, data, [QuantMeasure])
        graph = transformer.log.getGraph()
        bottoms = transformer.log.getBottoms()
    
        data_distill = getDistilData(model_original, 'imagenet', args.dis_batch_size, bn_merged=False,\
            num_batch=args.dis_num_batch, gpu=True, value_range=[-2.11790393, 2.64], size=[224, 224], early_break_factor=1.2 if args.resnet else 0.5)

    transformer = TorchTransformer()
    module_dict = {}

    if args.distill_range:
        module_dict[1] = [(torch.nn.Conv2d, QConv2d), (torch.nn.Linear, QLinear)]
    else:
        module_dict[1] = [(torch.nn.Conv2d, QuantNConv2d), (torch.nn.Linear, QuantNLinear)]

    if args.relu or args.equalize:
        module_dict[0] = [(torch.nn.ReLU6, torch.nn.ReLU)]

    # transformer.summary(model, data)
    # transformer.visualize(model, data, 'graph_cls', graph_size=120)

    model, transformer = switch_layers(model, transformer, data, module_dict, ignore_layer=[QuantMeasure], quant_op=True)

    graph = transformer.log.getGraph()
    bottoms = transformer.log.getBottoms()
    if args.distill_range:
        targ_layer = [QConv2d, QLinear]
    else:
        targ_layer = [QuantNConv2d, QuantNLinear]

    set_layer_bits(graph, args.bits_weight, args.bits_activation, args.bits_bias, targ_layer)

    model = merge_batchnorm(model, graph, bottoms, targ_layer)

    #create relations
    if args.equalize or args.distill_range:
        res = create_relation(graph, bottoms, targ_layer, delete_single=False)
        if args.equalize:
            cross_layer_equalization(graph, res, targ_layer, visualize_state=False, converge_thres=2e-7)
    
    if args.clip_weight:
        clip_weight(graph, range_clip=[-15, 15], targ_type=targ_layer)

    if args.correction:
        bias_correction(graph, bottoms, targ_layer, bits_weight=args.bits_weight)

    if args.distill_range:
        set_update_stat(model, [QuantMeasure], True)
        model = update_quant_range(model.cuda(), data_distill, graph, bottoms)
        set_update_stat(model, [QuantMeasure], False)
    else:
        set_quant_minmax(graph, bottoms)

    torch.cuda.empty_cache()

    # restore custom conv layer to torch.nn.conv2d
    module_dict = {}
    if args.distill_range:
        module_dict[1] = [(QConv2d, torch.nn.Conv2d), (QLinear, torch.nn.Linear)]
    else:
        module_dict[1] = [(QuantNConv2d, torch.nn.Conv2d), (QuantNLinear, torch.nn.Linear)]

    model, transformer = switch_layers(model, transformer, data, module_dict, ignore_layer=[QuantMeasure], quant_op=False)

    # An example input you would normally provide to your model's forward() method
    x = torch.rand(1, 3, 224, 224)

    # Export the model    
    torch_out = torch.onnx._export(model, x, "model.onnx", export_params=True)

    os.system("python3 -m onnxsim model.onnx model-sim.onnx")
    
    cur_path = os.path.abspath(os.getcwd())
    os.system("cp model-sim.onnx {}".format(os.path.join(args.ncnn_build, 'tools/onnx', 'model-sim.onnx')))
    os.chdir(os.path.join(args.ncnn_build, 'tools/onnx'))
    os.system("./onnx2ncnn model-sim.onnx model.param model.bin")
    lines = [line.strip() for line in open("model.param", "r")]
    with open("model.param", 'w') as ww:
        for idx, line in enumerate(lines):
            if idx == 3 and 'input' in line:
                line += ' 0=224 1=224 2=3'
            ww.write(line+'\n')

    os.system("mv model.param {}".format(os.path.join(args.ncnn_build, 'tools/quantize', 'model.param')))
    os.system("mv model.bin {}".format(os.path.join(args.ncnn_build, 'tools/quantize', 'model.bin')))
    os.system("rm model-sim.onnx")
    os.chdir(os.path.join(args.ncnn_build, 'tools/quantize'))
    os.system("./ncnn2table --param=model.param --bin=model.bin\
             --images={} --output=model_int8.table\
             --mean={},{},{} --norm={},{},{} --size=224,224 --thread=2".format(
                 args.image_path, 0.485*255, 0.456*255, 0.406*255, 1/(0.229*255), 1/(0.224*255), 1/(0.225*255)))
    
    lines = [line.strip() for line in open("model.table", 'r')]

    if args.quantize:
        os.system("./ncnn2int8 model.param model.bin model_int8.param model_int8.bin model_int8.table")
        lines = [line.strip() for line in open("model_int8.param", "r")]
        with open("model_int8.param", 'w') as ww:
            for idx, line in enumerate(lines):
                if idx == 3 and 'input' in line:
                    line += ' 0=224 1=224 2=3'
                ww.write(line+'\n')

        os.system("cp model_int8.param {}".format(os.path.join(cur_path, 'model_int8.param')))
        os.system("cp model_int8.bin {}".format(os.path.join(cur_path, 'model_int8.bin')))
        os.system("cp model_int8.table {}".format(os.path.join(cur_path, 'model_int8.table')))
    else:
        os.system("cp model.param {}".format(os.path.join(cur_path, 'model.param')))
        os.system("cp model.bin {}".format(os.path.join(cur_path, 'model.bin')))

    os.chdir(cur_path)

if __name__ == '__main__':
    main()