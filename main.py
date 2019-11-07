import os
import inspect
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling.deeplab import DeepLab
from torch.utils.data import DataLoader
from dataset.pascal import VOCSegmentation
from utils.metrics import Evaluator

from utils.relation import create_relation
from dfq import cross_layer_equalization, bias_absorption
from utils.layer_transform import switch_layers, replace_op, restore_op, set_quant_minmax, merge_batchnorm#, LayerTransform
from PyTransformer.transformers.torchTransformer import TorchTransformer


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


def inference_all(model):
    from utils.utils import forward_all
    args = lambda: 0
    args.base_size = 513
    args.crop_size = 513
    voc_val = VOCSegmentation(args, split='val')
    dataloader = DataLoader(voc_val, batch_size=32, shuffle=False, num_workers=0)

    forward_all(model, dataloader, visualize=True)


def main():
    data = torch.ones((4, 3, 513, 513))#.cuda()
    # model = DeepLab(sync_bn=False)
    # state_dict = torch.load('modeling/deeplab-mobilenet.pth.tar')['state_dict']
    # model.load_state_dict(state_dict)
    # transformer = TorchTransformer()
    # switch_layers(model, transformer, data)
    # estimate_stats(model, state_dict, data)
    # return

    model = DeepLab(sync_bn=False)
    state_dict = torch.load('modeling/deeplab-mobilenet.pth.tar')['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    
    transformer = TorchTransformer()
    # layer_transform = LayerTransform()
    model = switch_layers(model, transformer, data)

    # use cpu to process
    transformer = TorchTransformer()
    model = model.cpu()
    data = torch.ones((4, 3, 513, 513))#.cuda()
    # transformer.summary(model, data)
    # transformer.visualize(model, data, 'deeplab_graph', graph_size=120)
    transformer._build_graph(model, data) # construt graph after all state_dict loaded

    graph = transformer.log.getGraph()
    bottoms = transformer.log.getBottoms()
    output_shape = transformer.log.getOutShapes()

    from PyTransformer.transformers.quantize import QuantConv2d
    
    model = merge_batchnorm(model, graph, bottoms, QuantConv2d)

    #create relations
    res = create_relation(graph, bottoms, QuantConv2d)
    cross_layer_equalization(graph, res, visualize_state=False)

    bias_absorption(graph, res)

    set_quant_minmax(graph, bottoms, output_shape)
    
    model = model.cuda()
    model.eval()

    # model = estimate_stats(model, model.state_dict(), data, path_save='modeling/equalized.pth')

    replace_op()
    inference_all(model)
    restore_op()


if __name__ == '__main__':
    main()