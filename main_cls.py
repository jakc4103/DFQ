import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling.classification.MobileNetV2 import mobilenet_v2
from dataset.classification.imagenet_dataset import ImageNetDataset
from torch.utils.data import DataLoader

from utils.relation import create_relation
from dfq import cross_layer_equalization, bias_absorption, bias_correction
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
    args = lambda: 0
    args.base_size = 224
    args.crop_size = 224
    imagenet_dataset = ImageNetDataset(args, split='val')
    dataloader = DataLoader(imagenet_dataset, batch_size=256, shuffle=False, num_workers=0)

    num_correct = 0

    with torch.no_grad():
        for ii, sample in enumerate(dataloader):
            image, label = sample['image'].cuda(), sample['label'].numpy()

            logits = model(image)

            pred = torch.max(torch.softmax(logits, 1), 1)[1].cpu().numpy().astype(np.uint8)

            # print(torch.max(torch.softmax(logits, 1), 1)[0])
            # print(pred)
            # print(label)
            # print('='*150)

            num_correct += np.sum(pred == label)
            print(num_correct)

    print("Acc: {}".format(num_correct / len(imagenet_dataset)))


def main():
    data = torch.ones((4, 3, 224, 224))#.cuda()

    model = mobilenet_v2('modeling/classification/mobilenetv2_1.0-f2a8633.pth.tar')
    model.eval()
    inference_all(model.cuda())
    return
    
    transformer = TorchTransformer()
    # layer_transform = LayerTransform()
    model = switch_layers(model, transformer, data)

    # use cpu to process
    transformer = TorchTransformer()
    model = model.cpu()
    data = torch.ones((4, 3, 224, 224))#.cuda()
    # transformer.summary(model, data)
    # transformer.visualize(model, data, 'mobilev2_graph', graph_size=120)

    transformer._build_graph(model, data) # construt graph after all state_dict loaded

    graph = transformer.log.getGraph()
    bottoms = transformer.log.getBottoms()
    output_shape = transformer.log.getOutShapes()

    from PyTransformer.transformers.quantize import QuantConv2d
    
    model = merge_batchnorm(model, graph, bottoms, QuantConv2d)

    #create relations
    # res = create_relation(graph, bottoms, QuantConv2d)
    # cross_layer_equalization(graph, res, visualize_state=False)

    # bias_absorption(graph, res, bottoms, 3)
    # bias_correction(graph, bottoms)

    set_quant_minmax(graph, bottoms, output_shape)
    
    model = model.cuda()
    model.eval()

    # model = estimate_stats(model, model.state_dict(), data, path_save='modeling/equalized.pth')

    replace_op()
    inference_all(model)
    restore_op()


if __name__ == '__main__':
    main()