import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling.deeplab import DeepLab
from torch.utils.data import DataLoader
from dataset.pascal import VOCSegmentation
from utils.metrics import Evaluator

from PyTransformer.transformers.quantize import QConv2d, QuantMeasure
from PyTransformer.transformers.torchTransformer import TorchTransformer

class QIdentity(nn.Module):
    def __init__(self):
        super(QIdentity, self).__init__()

    
    def forward(self, x):
        return x


def switch_layers(model):
    transformer = TorchTransformer()

    # transformer.register(nn.BatchNorm2d, QIdentity)
    # model = transformer.trans_layers(model, update=False)
    old_dict = model.state_dict()
    transformer.register(nn.ReLU6, nn.ReLU)
    transformer.register(nn.Conv2d, QConv2d)
    model = transformer.trans_layers(model)

    new_dict = model.state_dict()

    data = torch.ones((4, 3, 512, 512)).cuda()
    # transformer.summary(model, data)
    # transformer.visualize(model, data, 'test.png', graph_size=120)
    transformer._build_graph(model, data)
    print(transformer.log.getGraph())

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
    model = switch_layers(model)
    # inference_all(model)
    # return


if __name__ == '__main__':
    main()