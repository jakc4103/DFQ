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
from dfq import cross_layer_equalization
from utils.layer_transform import merge_batchnorm, switch_layers, replace_op, restore_op
from PyTransformer.transformers.torchTransformer import TorchTransformer


def estimate_stats(data, num_epoch=10):
    import copy

    model = DeepLab(sync_bn=False)
    model.eval()
    state_dict = torch.load('modeling/deeplab-mobilenet.pth.tar')['state_dict']
    model.load_state_dict(state_dict)

    transformer = TorchTransformer()

    model = switch_layers(model, transformer, data)
    
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
    state_dict_old = torch.load('modeling/deeplab-mobilenet.pth.tar')['state_dict']
    bn_dict = {}
    for key in state_dict_old:
        if 'running' in key:
            bn_dict[key] = state_dict_old[key]

    state = model.state_dict()
    state.update(bn_dict)
    model.load_state_dict(state)

    # use cpu to process
    model = model.cpu()
    
    transformer._build_graph(model, data) # construt graph after all state_dict loaded

    graph = transformer.log.getGraph()
    bottoms = transformer.log.getBottoms()

    model = merge_batchnorm(model, graph, bottoms)
    torch.save(model.state_dict(), 'modeling/data_dependent_QuantConv2dAdd.pth')


def inference_all(model):
    from utils.utils import forward_all
    args = lambda: 0
    args.base_size = 513
    args.crop_size = 513
    voc_val = VOCSegmentation(args, split='val')
    dataloader = DataLoader(voc_val, batch_size=32, shuffle=False, num_workers=0)

    forward_all(model, dataloader, visualize=False)


def main():
    data = torch.ones((4, 3, 513, 513))#.cuda()

    # estimate_stats(data)
    # return

    model = DeepLab(sync_bn=False)
    model.eval()
    
    transformer = TorchTransformer()

    model = switch_layers(model, transformer, data)

    path_state_dict = 'modeling/data_dependent_QuantConv2dAdd.pth'
    if os.path.exists(path_state_dict):
        print("Load params from {}".format(path_state_dict))
        state = torch.load(path_state_dict)
 
        model.load_state_dict(state)

    # use cpu to process
    transformer = TorchTransformer()
    model = model.cpu()
    data = torch.ones((4, 3, 513, 513))#.cuda()
    # transformer.summary(model, data)
    # transformer.visualize(model, data, 'deeplab_graph', graph_size=120)
    transformer._build_graph(model, data) # construt graph after all state_dict loaded

    graph = transformer.log.getGraph()
    bottoms = transformer.log.getBottoms()

    # from PyTransformer.transformers.quantize import QuantMeasure
    # def print_module(model):
    #     for mm in model._modules:
    #         if len(model._modules[mm]._modules) > 0 and\
    #             not (len(model._modules[mm]._modules) == 1 and type(list(model._modules[mm]._modules.values())[0]) == QuantMeasure):
    #             print_module(model._modules[mm])
    #         else:
    #             print(id(getattr(model, mm)), getattr(model, mm))

    # print_module(model)

    from PyTransformer.transformers.quantize import QuantConv2d
    
    model = merge_batchnorm(model, graph, bottoms, QuantConv2d)

    #create relations
    res = create_relation(graph, bottoms, QuantConv2d)
    for ii in graph:
        print(ii, graph[ii])
    print('='*150)
    for rr in res:
        print(rr)
    print('='*150)
    return
    cross_layer_equalization(graph, res, visualize_state=False)
    
    model = model.cuda()
    model.eval()

    replace_op()
    inference_all(model)
    restore_op()


if __name__ == '__main__':
    main()