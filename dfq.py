import torch
import torch.nn as nn
import copy
from utils.utils import visualize_per_layer
from PyTransformer.transformers.quantize import QConv2d

def _layer_equalization(weight_first, weight_second, bias_first, s_range=(1e-8, 1e8), signed=False, eps=0):
    num_group = 1
    if weight_first.shape[0] != weight_second.shape[1]:
        # group convolution
        num_group = weight_first.shape[0] // weight_second.shape[1]
	
    group_channels_i = weight_first.shape[0] // num_group
    group_channels_o = weight_second.shape[0] // num_group

    # pdb.set_trace()
    for g in range(num_group):
        c_start_i = g * group_channels_i
        c_end_i = (g + 1) * group_channels_i
        weight_first_group = weight_first[c_start_i:c_end_i] # shape [k, c, h, w]

        c_start_o = g * group_channels_o
        c_end_o = (g + 1) * group_channels_o
        weight_second_group = weight_second[c_start_o:c_end_o]

        for ii in range(weight_second_group.shape[1]):
            if signed:
                range_1 = torch.max(torch.abs(weight_first_group[ii])) # signed
                range_2 = torch.max(torch.abs(weight_second_group[:, ii])) # signed

            else:
                range_1 = torch.max(weight_first_group[ii]) - torch.min(weight_first_group[ii]) # unsigned
                range_2 = torch.max(weight_second_group[:, ii]) - torch.min(weight_second_group[:, ii]) # unsigned
            
            # 1 / s = (1 / r1) * sqrt(r1 * r2)
            s = (1 / (range_1 + eps)) * torch.sqrt(range_1 * range_2 + eps)
            s = max(s_range[0], min(s_range[1], s))

            weight_first[c_start_i + ii].mul_(s)

            if bias_first is not None:
                bias_first[c_start_i + ii].mul_(s)

            weight_second[c_start_o:c_end_o, ii].mul_(1/s)

    return weight_first, weight_second, bias_first


def cross_layer_equalization(graph, relations, s_range=[1e-8, 1e8], converge_thres=1, signed=False, eps=0, visualize_state=False):
    with torch.no_grad():
        diff = 10
        while diff > converge_thres:
            state_prev = copy.deepcopy(graph)
            for rr in relations:
                layer_first, layer_second = rr.get_idxs()
                
                if visualize_state:
                    # print(torch.max(graph[layer_first].weight.detach()), torch.min(graph[layer_first].weight.detach()), 'before equal')
                    visualize_per_layer(graph[layer_first].weight.detach(), 'Before equalization')

                if graph[layer_first].bias is None: # add a fake bias term
                    graph[layer_first].bias = nn.Parameter(data=torch.zeros((graph[layer_first].weight.size(0)), dtype=torch.float), requires_grad=False)
                
                # layer eualization
                graph[layer_first].weight, graph[layer_second].weight, graph[layer_first].bias = \
                    _layer_equalization(graph[layer_first].weight,\
                                        graph[layer_second].weight,\
                                        graph[layer_first].bias, s_range=s_range, signed=signed, eps=eps)
                
                if visualize_state:
                    # print(torch.max(graph[layer_first].weight.detach()), torch.min(graph[layer_first].weight.detach()), 'after equal')
                    visualize_per_layer(graph[layer_first].weight.detach(), 'After equalization')

            diff_tmp = 0
            for layer_idx in graph:
                if type(graph[layer_idx]) == QConv2d:
                    diff_tmp += torch.mean(torch.abs(graph[layer_idx].weight - state_prev[layer_idx].weight))
            
            diff = min(diff, diff_tmp)
            # print('diff', diff)
    
    # return graph