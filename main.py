import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling.deeplab import DeepLab
from torch.utils.data import DataLoader
from dataset.pascal import VOCSegmentation
from utils.metrics import Evaluator

from utils.relation import create_relation
from utils.quantize import QIdentity
from dfq import cross_layer_equalization
from PyTransformer.transformers.quantize import QConv2d, QuantMeasure
from PyTransformer.transformers.torchTransformer import TorchTransformer



def process_model(model):
    transformer = TorchTransformer()

    # transformer.register(nn.BatchNorm2d, QIdentity)
    # model = transformer.trans_layers(model, update=False)
    
    # replace layers
    transformer.register(nn.ReLU6, nn.ReLU)
    transformer.register(nn.Conv2d, QConv2d)
    model = transformer.trans_layers(model)

    # use cpu to process
    model = model.cpu()
    data = torch.ones((4, 3, 512, 512))#.cuda()
    # transformer.summary(model, data)
    # transformer.visualize(model, data, 'test.png', graph_size=120)
    transformer._build_graph(model, data)

    graph = transformer.log.getGraph()
    bottoms = transformer.log.getBottoms()
    
    with torch.no_grad():
        # merge bn params into QConv2d
        for layer_idx in graph:
            # print(bottoms[layer_idx])
            if bottoms[layer_idx] is None:
                continue
            for bot_idx in bottoms[layer_idx]:
                if type(graph[layer_idx]) == nn.BatchNorm2d and type(graph[bot_idx]) == QConv2d:
                    conv_weight = graph[bot_idx].weight
                    bn_weight = graph[layer_idx].weight
                    bn_var = graph[layer_idx].running_var
                    bn_eps = graph[layer_idx].eps

                    graph[bot_idx].weight.copy_(conv_weight.clone().mul(bn_weight.view(-1, 1, 1, 1) / torch.sqrt(bn_var.view(-1, 1, 1, 1) + bn_eps)))
                    
                    if graph[bot_idx].bias is None:
                        graph[bot_idx].bias = nn.Parameter(data=torch.zeros((graph[bot_idx].weight.size(0)), dtype=torch.float), requires_grad=False)

                    conv_bias = graph[bot_idx].bias
                    bn_bias = graph[layer_idx].bias
                    bn_mean = graph[layer_idx].running_mean

                    graph[bot_idx].bias.copy_(conv_bias.clone().mul(bn_weight.view(-1) / torch.sqrt(bn_var.view(-1) + bn_eps)).add(bn_bias.view(-1) -\
						 (bn_weight.view(-1) * bn_mean.view(-1)) / torch.sqrt(bn_var.view(-1) + bn_eps)))

                    graph[layer_idx].weight.mul_(0).add_(1)
                    graph[layer_idx].running_var.mul_(0).add_(1)
                    graph[layer_idx].bias.mul_(0)
                    graph[layer_idx].running_mean.mul_(0)
                    graph[layer_idx].eps = 0

                    break

    res = create_relation(graph, bottoms)

    cross_layer_equalization(graph, res, visualize_state=False)

    return model


def inference_all(model):
    from utils.utils import forward_all
    args = lambda: 0
    args.base_size = 512
    args.crop_size = 512
    voc_val = VOCSegmentation(args, split='val')
    dataloader = DataLoader(voc_val, batch_size=5, shuffle=False, num_workers=0)

    forward_all(model, dataloader, visualize=False)


def main():
    model = DeepLab(sync_bn=False)
    model.eval()
    model = model.cuda()

    # load trained parameters
    state_dict = torch.load('modeling/deeplab-mobilenet.pth.tar')['state_dict']
    model.load_state_dict(state_dict)
    model = process_model(model)
    model = model.cuda()
    inference_all(model)
    # return


if __name__ == '__main__':
    main()