import torch
import torch.nn as nn
import copy
from utils import visualize_per_layer
from PyTransformer.transformers.quantize import QConv2d

def _layer_equalization(weight_first, weight_second, bias_first, bn_weight=None, bn_bias=None, s_range=(1e-8, 1e8), signed=False, eps=0):
    num_group = 1
    if weight_first.shape[0] != weight_second.shape[1]:
        # group convolution
        num_group = weight_first.shape[0] // weight_second.shape[1]
	
    group_channels_i = weight_first.shape[0] // num_group
    group_channels_o = weight_second.shape[0] // num_group

    S = torch.zeros(weight_first.size(0))
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
            S[c_start_i + ii] = s

            weight_first[c_start_i + ii].mul_(s)
            
            if bn_weight is not None:
                bn_weight[c_start_i + ii].mul_(s)

            if bn_bias is not None:
                bn_bias[c_start_i + ii].mul_(s)

            if bias_first is not None:
                bias_first[c_start_i + ii].mul_(s)

            weight_second[c_start_o:c_end_o, ii].mul_(1/s)

    return weight_first, weight_second, bias_first, S


def cross_layer_equalization(graph, relations, s_range=[1e-8, 1e8], converge_thres=1, signed=False, eps=0, visualize_state=False):
    with torch.no_grad():
        diff = 10
        conv_type = type(graph[relations[0].get_idxs()[0]])
        while diff > converge_thres:
            state_prev = copy.deepcopy(graph)
            for rr in relations:
                layer_first, layer_second, bn_idx = rr.get_idxs()
                
                if visualize_state:
                    # print(torch.max(graph[layer_first].weight.detach()), torch.min(graph[layer_first].weight.detach()), 'before equal')
                    visualize_per_layer(graph[layer_first].weight.detach(), 'Before equalization')

                if graph[layer_first].bias is None: # add a fake bias term
                    graph[layer_first].bias = nn.Parameter(data=torch.zeros((graph[layer_first].weight.size(0)), dtype=torch.float), requires_grad=False)
                
                # layer eualization
                graph[layer_first].weight, graph[layer_second].weight, graph[layer_first].bias, S = \
                _layer_equalization(graph[layer_first].weight,\
                                        graph[layer_second].weight,\
                                        graph[layer_first].bias,\
                                        graph[bn_idx].fake_weight,\
                                        graph[bn_idx].fake_bias, s_range=s_range, signed=signed, eps=eps)
                # graph[layer_first].weight, graph[layer_second].weight, graph[layer_first].bias = \
                # _layer_equalization(graph[layer_first].weight,\
                #                         graph[layer_second].weight,\
                #                         graph[layer_first].bias, s_range=s_range, signed=signed, eps=eps)
                rr.set_scale_vec(S)
                if visualize_state:
                    # print(torch.max(graph[layer_first].weight.detach()), torch.min(graph[layer_first].weight.detach()), 'after equal')
                    visualize_per_layer(graph[layer_first].weight.detach(), 'After equalization')

            diff_tmp = 0
            for layer_idx in graph:
                if type(graph[layer_idx]) == conv_type:
                    diff_tmp += torch.mean(torch.abs(graph[layer_idx].weight - state_prev[layer_idx].weight))
            
            diff = min(diff, diff_tmp)
            print('diff', diff)
    
    # return graph

def bias_absorption(graph, relations, bottoms, N=3):
    def find_next_bn(layer_second, graph, idx):
        graph_list = list(graph.items())
        if idx is None:
            idx = 0

        while idx < len(graph_list) and graph_list[idx][0] != layer_second:
            idx += 1

        if idx >= len(graph_list) - 1:
            return None, 0, False

        if idx == len(graph_list) - 2 and type(graph_list[idx+1][1]) == torch.nn.BatchNorm2d:
            return graph_list[idx+1][0], idx + 1, False

        elif type(graph_list[idx+1][1]) == torch.nn.BatchNorm2d and type(graph_list[idx+2][1]) == torch.nn.ReLU:
            return graph_list[idx+1][0], idx + 1, True

        elif type(graph_list[idx+1][1]) == torch.nn.BatchNorm2d:
            return graph_list[idx+1][0], idx + 1, False

        return None, 0, False

    def is_relu_found(layer_second, layer_first, graph, bottoms):
        idx = layer_second
        while idx != layer_first:
            assert len(bottoms[idx]) == 1, 'graph in equalization relations should be 1-to-1 input-output'
            if type(graph[bottoms[idx][0]]) == torch.nn.ReLU:
                return True
            idx = bottoms[idx][0]
        return False

    idx = 0
    for rr in relations:
        layer_first, layer_second, bn_idx = rr.get_idxs()

        if not is_relu_found(layer_second, layer_first, graph, bottoms):
            continue

        bn_weight = getattr(graph[bn_idx], 'fake_weight').detach().clone()
        bn_bias = getattr(graph[bn_idx], 'fake_bias').detach().clone()
        
        weight = graph[layer_second].weight.detach().clone()
        size = weight.shape
        weight = weight.view(size[0], size[1], -1)

        num_group = graph[layer_first].weight.size(0) // graph[layer_second].weight.size(1)
        step_size_o = size[0] // num_group
        step_size_i = graph[layer_first].weight.size(0) // num_group

        c = (bn_bias - N * bn_weight)
        c.clamp_(0)

        S = rr.get_scale_vec()
        c[S<=1] = 0

        wc = torch.zeros(size[0])

        for g in range(num_group):
            wc[g*step_size_o:(g+1)*step_size_o] = torch.matmul(torch.sum(weight[g*step_size_o:(g+1)*step_size_o], -1), c[g*step_size_i:(g+1)*step_size_i])

        graph[layer_first].bias.add_(-c)
        graph[bn_idx].fake_bias.add_(-c)
        graph[layer_second].bias.add_(wc)
        bn_idx_next, idx, relu_attached = find_next_bn(layer_second, graph, idx)
        # if bn_idx_next is not None:
        #     #TODO should add bias than merge batch norm
        #     graph[bn_idx_next].fake_bias.add_(-wc)


def bias_correction(graph, bottoms):
    from PyTransformer.transformers.quantize import UniformQuantize, QuantConv2d
    from scipy.stats import norm

    standard_normal = lambda x: torch.exp(-(x * x) / 2) / 2.50662827463
    standard_cdf = lambda x: torch.from_numpy(norm.cdf(x)).float()
    expect_relu = None
    expect = None
    eps = None
    idx_bn_prev = None
    for idx_layer in graph:
        bot = bottoms[idx_layer]
        # if type(graph[idx_layer]) == QuantConv2d:
        #     # check if relu attached, then do bias correction
        #     idx_prev_conv = idx_layer

        #     bot_tmp = list(zip(bot[:], range(len(bot[:]))))
        #     bn_list = []
        #     while len(bot_tmp) > 0:
        #         idx_bot, bid = bot_tmp.pop(0)
        #         if idx_bot == 'Data':
        #             break
        #         if idx_bot not in bn_module:
        #             bot_tmp.extend(list(zip(bottoms[idx_bot], [bid]*len(bottoms[idx_bot]))))
        #         else:
        #             bn_list.append((bn_module[idx_bot], bid))

        #     # ignore all cat, add layers
        #     if len(bn_list) == 1:
        #         bn_weight = getattr(bn_list[0][0], 'fake_weight').detach().clone() # std
        #         bn_bias = getattr(bn_list[0][0], 'fake_bias').detach().clone() # mean

        #         weight = getattr(graph[idx_layer], 'weight').detach().clone()
        #         with torch.no_grad():
        #             weight_q = UniformQuantize().apply(weight, 8, float(weight.min()), float(weight.max()))

        #             eps = torch.mean((weight_q - weight).view(weight.size(0), -1), -1)

        #         expect = bn_weight * standard_normal(-bn_bias/bn_weight) + bn_bias * (1 - standard_cdf(-bn_bias/bn_weight))
        #         print(bn_weight.shape)
        #         print(weight.shape)
        #         print(eps, eps.shape)
        #         print(expect, expect.shape)
        #         bias_corrected_prev = -eps*expect

        #         graph[idx_layer].bias.add_(bias_corrected_prev)
        
        if type(graph[idx_layer]) == torch.nn.BatchNorm2d and len(bot) == 1 and type(graph[bot[0]]) == QuantConv2d:
            bn_weight = getattr(graph[idx_layer], 'fake_weight').detach().clone() # std
            bn_bias = getattr(graph[idx_layer], 'fake_bias').detach().clone() # mean

            weight = getattr(graph[bot[0]], 'weight').detach().clone()

            with torch.no_grad():
                weight_q = UniformQuantize().apply(weight, 8, float(weight.min()), float(weight.max())).detach()
                eps = torch.mean((weight_q - weight).view(weight.size(0), -1), -1)

                expect_relu = bn_weight * standard_normal(-bn_bias/bn_weight) + bn_bias * (1 - standard_cdf(-bn_bias/bn_weight))
                expect = bn_bias
            idx_bn_prev = idx_layer

        elif idx_bn_prev is not None and len(bot) == 1 and bot[0] == idx_bn_prev:
            with torch.no_grad():
                if type(graph[idx_layer]) == torch.nn.ReLU:
                    bias = eps * expect_relu
                else:
                    bias = eps * expect

                graph[bot[0]].bias.add_(-bias)
                graph[bottoms[bot[0]][0]].bias.add_(-bias)
            expect_relu = None
            expect = None
            eps = None
            idx_bn_prev = None

        else:
            expect_relu = None
            expect = None
            eps = None
            idx_bn_prev = None