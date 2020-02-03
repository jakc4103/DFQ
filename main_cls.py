import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from modeling.classification.MobileNetV2 import mobilenet_v2
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from utils.relation import create_relation
from dfq import cross_layer_equalization, bias_absorption, bias_correction, _quantize_error, clip_weight
from utils.layer_transform import switch_layers, replace_op, restore_op, set_quant_minmax, merge_batchnorm, quantize_targ_layer#, LayerTransform
from PyTransformer.transformers.torchTransformer import TorchTransformer

from utils.quantize import QuantConv2d, QuantLinear, QuantNConv2d, QuantNLinear, QuantMeasure, QConv2d, QLinear, set_layer_bits
from ZeroQ.distill_data import getDistilData
from improve_dfq import update_scale, transform_quant_layer, set_scale, update_quant_range, set_update_stat, bias_correction_distill

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action='store_true')
    parser.add_argument("--equalize", action='store_true')
    parser.add_argument("--distill", action='store_true')
    parser.add_argument("--correction", action='store_true')
    parser.add_argument("--absorption", action='store_true')
    parser.add_argument("--relu", action='store_true')
    parser.add_argument("--clip_weight", action='store_true')
    parser.add_argument("--trainable", action='store_true')
    parser.add_argument("--true_data", action='store_true')
    parser.add_argument("--bits_weight", type=int, default=8)
    parser.add_argument("--bits_activation", type=int, default=8)
    parser.add_argument("--bits_bias", type=int, default=16)
    parser.add_argument("--dis_batch_size", type=int, default=64)
    parser.add_argument("--dis_num_batch", type=int, default=8)
    return parser.parse_args()


def inference_all(model):
    print("Start inference")
    imagenet_dataset = datasets.ImageFolder('/home/jakc4103/WDesktop/dataset/ILSVRC/Data/CLS-LOC/val', transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ]))

    dataloader = DataLoader(imagenet_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for ii, sample in enumerate(dataloader):
            image, label = sample[0].cuda(), sample[1].numpy()
            logits = model(image)

            pred = torch.max(logits, 1)[1].cpu().numpy()
            
            num_correct += np.sum(pred == label)
            num_total += image.shape[0]
            # print(num_correct, num_total, num_correct/num_total)
    acc = num_correct / num_total
    return acc


def main():
    args = get_argument()
    assert args.relu or args.relu == args.equalize, 'must replace relu6 to relu while equalization'
    assert args.equalize or args.absorption == args.equalize, 'must use absorption with equalize'

    data = torch.ones((4, 3, 224, 224))#.cuda()

    model = mobilenet_v2('modeling/classification/mobilenetv2_1.0-f2a8633.pth.tar')
    model.eval()
    
    if args.distill:
        import copy
        # define FP32 model 
        model_original = copy.deepcopy(model)
        model_original.eval()
        transformer = TorchTransformer()
        transformer._build_graph(model_original, data, [QuantMeasure])
        graph = transformer.log.getGraph()
        bottoms = transformer.log.getBottoms()
    
        if not args.true_data:
            data_distill = getDistilData(model_original, 'imagenet', args.dis_batch_size, bn_merged=False, num_batch=args.dis_num_batch, gpu=True)
        else:
            imagenet_dataset = datasets.ImageFolder('/home/jakc4103/windows/Toshiba/workspace/dataset/ILSVRC/Data/CLS-LOC/train', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ]))
            data_distill = []
            dataloader = DataLoader(imagenet_dataset, batch_size=args.dis_batch_size, shuffle=True, num_workers=4, pin_memory=True)
            for idx, sample in enumerate(dataloader):
                if idx >= args.dis_num_batch:
                    break
                image = sample[0]
                data_distill.append(image)
            del dataloader, imagenet_dataset

        model_original = model_original.cpu()
        model_original = merge_batchnorm(model_original, graph, bottoms, [nn.Conv2d, nn.Linear])

    transformer = TorchTransformer()
    module_dict = {}
    if args.quantize:
        if args.distill:
            module_dict[1] = [(nn.Conv2d, QConv2d), (nn.Linear, QLinear)]
        elif args.trainable:
            module_dict[1] = [(nn.Conv2d, QuantConv2d), (nn.Linear, QuantLinear)]
        else:
            module_dict[1] = [(nn.Conv2d, QuantNConv2d), (nn.Linear, QuantNLinear)]
    
    if args.relu:
        module_dict[0] = [(torch.nn.ReLU6, torch.nn.ReLU)]

    # transformer.summary(model, data)
    # transformer.visualize(model, data, 'graph_cls', graph_size=120)

    model, transformer = switch_layers(model, transformer, data, module_dict, ignore_layer=[QuantMeasure], quant_op=args.quantize)

    graph = transformer.log.getGraph()
    bottoms = transformer.log.getBottoms()
    if args.quantize:
        if args.distill:
            targ_layer = [QConv2d, QLinear]
        elif args.trainable:
            targ_layer = [QuantConv2d, QuantLinear]
        else:
            targ_layer = [QuantNConv2d, QuantNLinear]
    else:
        targ_layer = [nn.Conv2d, nn.Linear]

    if args.quantize:
        set_layer_bits(graph, args.bits_weight, args.bits_activation, args.bits_bias, targ_layer)

    model = merge_batchnorm(model, graph, bottoms, targ_layer)

    #create relations
    if args.equalize or args.distill:
        res = create_relation(graph, bottoms, targ_layer)
        if args.equalize:
            cross_layer_equalization(graph, res, targ_layer, visualize_state=False, converge_thres=2e-7)

        if args.distill:
            set_scale(res, graph, bottoms, targ_layer)
    
    if args.absorption:
        bias_absorption(graph, res, bottoms, 3)
    
    if args.clip_weight:
        clip_weight(graph, range_clip=[-15, 15], targ_type=targ_layer)

    if args.correction:
        if args.distill:
            bias_correction_distill(model, model_original, data_distill, targ_layer, [nn.Conv2d, nn.Linear])
        else:
            bias_correction(graph, bottoms, targ_layer, bits_weight=args.bits_weight)

    if args.quantize:
        if not args.trainable and not args.distill:
            graph = quantize_targ_layer(graph, args.bits_weight, args.bits_bias, targ_layer)

        if args.distill:
            set_update_stat(model, [QuantMeasure], True)
            replace_op()
            model = update_quant_range(model.cuda(), data_distill)
            restore_op()
            set_update_stat(model, [QuantMeasure], False)
        else:
            set_quant_minmax(graph, bottoms)

    # if args.distill:
    #     model = update_scale(model, model_original, data_distill, graph, bottoms, res, targ_layer, num_epoch=1000)
    #     set_quant_minmax(graph, bottoms)

    model = model.cuda()
    model.eval()

    if args.quantize:
        replace_op()
    acc = inference_all(model)
    print("Acc: {}".format(acc))
    if args.quantize:
        restore_op()


if __name__ == '__main__':
    main()