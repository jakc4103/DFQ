import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from modeling.segmentation.deeplab import DeepLab
from torch.utils.data import DataLoader
from dataset.segmentation.pascal import VOCSegmentation
from utils.metrics import Evaluator

from utils.relation import create_relation
from dfq import cross_layer_equalization, bias_absorption, bias_correction, clip_weight
from utils.layer_transform import switch_layers, replace_op, restore_op, set_quant_minmax, merge_batchnorm, quantize_targ_layer#, LayerTransform
from PyTransformer.transformers.torchTransformer import TorchTransformer
from utils.quantize import QuantConv2d, QuantNConv2d, QuantMeasure, QConv2d, set_layer_bits
from ZeroQ.distill_data import getDistilData
from improve_dfq import update_scale, transform_quant_layer, set_scale, update_quant_range, set_update_stat, bias_correction_distill

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action='store_true')
    parser.add_argument("--equalize", action='store_true')
    parser.add_argument("--correction", action='store_true')
    parser.add_argument("--absorption", action='store_true')
    parser.add_argument("--distill", action='store_true')
    parser.add_argument("--log", action='store_true')
    parser.add_argument("--relu", action='store_true')
    parser.add_argument("--clip_weight", action='store_true')
    parser.add_argument("--dataset", type=str, default="voc12")
    parser.add_argument("--trainable", action='store_true')
    parser.add_argument("--bits_weight", type=int, default=8)
    parser.add_argument("--bits_activation", type=int, default=8)
    parser.add_argument("--bits_bias", type=int, default=16)
    return parser.parse_args()

def estimate_stats(model, state_dict, data, num_epoch=10, path_save='modeling/data_dependent_QuantConv2dAdd.pth'):
    import copy

    # model = DeepLab(sync_bn=False)
    model.eval()
    
    model = model.cuda()

    args = lambda: 0
    args.base_size = 513
    args.crop_size = 513
    voc_val = VOCSegmentation(args, split='train')
    dataloader = DataLoader(voc_val, batch_size=32, shuffle=True, num_workers=0)
    model.train()

    replace_op()
    ss = time.time()
    with torch.no_grad():
        for epoch in range(num_epoch):
            start = time.time()
            for sample in dataloader:
                image, _ = sample['image'].cuda(), sample['label'].cuda()

                _ = model(image)

            end = time.time()
            print("epoch {}: {} sec.".format(epoch, end-start))
    print('total time: {} sec'.format(time.time() - ss))
    restore_op()

    # load 'running_mean' and 'running_var' of batchnorm back from pre-trained parameters
    bn_dict = {}
    for key in state_dict:
        if 'running' in key:
            bn_dict[key] = state_dict[key]

    state = model.state_dict()
    state.update(bn_dict)
    model.load_state_dict(state)

    torch.save(model.state_dict(), path_save)

    return model


def inference_all(model, dataset='voc12', opt=None):
    print("Start inference")
    from utils.segmentation.utils import forward_all
    args = lambda: 0
    args.base_size = 513
    args.crop_size = 513
    if dataset == 'voc12':
        voc_val = VOCSegmentation(args, base_dir="/home/jakc4103/WDesktop/dataset/VOCdevkit/VOC2012/", split='val')
    elif dataset == 'voc07':
        voc_val = VOCSegmentation(args, base_dir="/home/jakc4103/WDesktop/dataset/VOCdevkit/VOC2007/", split='test')
    dataloader = DataLoader(voc_val, batch_size=32, shuffle=False, num_workers=2)

    forward_all(model, dataloader, visualize=False, opt=opt)


def main():
    args = get_argument()
    assert args.relu or args.relu == args.equalize, 'must replace relu6 to relu while equalization'
    assert args.equalize or args.absorption == args.equalize, 'must use absorption with equalize'
    data = torch.ones((4, 3, 513, 513))#.cuda()

    model = DeepLab(sync_bn=False)
    state_dict = torch.load('modeling/segmentation/deeplab-mobilenet.pth.tar')['state_dict']
    model.load_state_dict(state_dict)
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
    
        data_distill = getDistilData(model_original, 'imagenet', 32, bn_merged=False,\
            num_batch=8, gpu=True, value_range=[-2.11790393, 2.64], size=[513, 513], early_break_factor=0.2)

    transformer = TorchTransformer()

    module_dict = {}
    if args.quantize:
        if args.distill:
            module_dict[1] = [(nn.Conv2d, QConv2d)]
        elif args.trainable:
            module_dict[1] = [(nn.Conv2d, QuantConv2d)]
        else:
            module_dict[1] = [(nn.Conv2d, QuantNConv2d)]
    
    if args.relu:
        module_dict[0] = [(torch.nn.ReLU6, torch.nn.ReLU)]

    # transformer.summary(model, data)
    # transformer.visualize(model, data, 'graph_deeplab', graph_size=120)

    model, transformer = switch_layers(model, transformer, data, module_dict, ignore_layer=[QuantMeasure], quant_op=args.quantize)
    graph = transformer.log.getGraph()
    bottoms = transformer.log.getBottoms()

    if args.quantize:
        if args.distill:
            targ_layer = [QConv2d]
        elif args.trainable:
            targ_layer = [QuantConv2d]
        else:
            targ_layer = [QuantNConv2d]
    else:
        targ_layer = [nn.Conv2d]
    if args.quantize:
        set_layer_bits(graph, args.bits_weight, args.bits_activation, args.bits_bias, targ_layer)
    model = merge_batchnorm(model, graph, bottoms, targ_layer)

    #create relations
    if args.equalize or args.distill:
        res = create_relation(graph, bottoms, targ_layer)
        if args.equalize:
            cross_layer_equalization(graph, res, targ_layer, visualize_state=False)

        # if args.distill:
        #     set_scale(res, graph, bottoms, targ_layer)

    if args.absorption:
        bias_absorption(graph, res, bottoms, 3)
    
    if args.clip_weight:
        clip_weight(graph, range_clip=[-15, 15], targ_type=targ_layer)

    if args.correction:
        bias_correction(graph, bottoms, targ_layer)

    if args.quantize:
        if not args.trainable and not args.distill:
            graph = quantize_targ_layer(graph, 8, 16, targ_layer)
        
        if args.distill:
            set_update_stat(model, [QuantMeasure], True)
            model = update_quant_range(model.cuda(), data_distill, graph, bottoms)
            set_update_stat(model, [QuantMeasure], False)
        else:
            set_quant_minmax(graph, bottoms)

        torch.cuda.empty_cache()
    
    model = model.cuda()
    model.eval()

    if args.quantize:
        replace_op()
    inference_all(model, args.dataset, args if args.log else None)
    if args.quantize:
        restore_op()


if __name__ == '__main__':
    main()