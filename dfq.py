import torch
import torch.nn as nn
import copy
import numpy as np
from utils import visualize_per_layer
from utils.quantize import UniformQuantize

def _quantize_error(param, num_bits=8, reduction='sum'):
    """!
    reduction should be one of 'sum', 'mean', 'none', 'channel', default to 'sum'
    """
    param = param.detach().clone()
    with torch.no_grad():
        param_quant = UniformQuantize().apply(param, num_bits, float(param.min()), float(param.max()))
        eps = param_quant - param
        if reduction == 'sum':
            eps = torch.sum(torch.abs(eps))
        elif reduction == 'mean':
            eps = torch.mean(eps)
        elif reduction == 'channel':
            eps = torch.sum(torch.abs(torch.sum(eps.view(eps.size(0), -1), -1)))
        elif reduction == 'spatial':
            eps = torch.sum(torch.abs(torch.sum(eps.view(eps.size(0), eps.size(1), -1), -1)))

        return eps


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


def cross_layer_equalization(graph, relations, targ_type, s_range=[1e-8, 1e8], range_thres=0, converge_thres=2e-7, converge_count=20, signed=False, eps=0, visualize_state=False):
    print("Start cross layer equalization")
    with torch.no_grad():
        diff = 10
        count = 0
        while diff > converge_thres and count < converge_count:
            state_prev = copy.deepcopy(graph)
            for rr in relations:
                layer_first, layer_second, bn_idx = rr.get_idxs()
                
                if visualize_state:
                    visualize_per_layer(graph[layer_first].weight.detach(), 'Before equalization')

                if graph[layer_first].bias is None: # add a fake bias term
                    graph[layer_first].bias = nn.Parameter(data=torch.zeros((graph[layer_first].weight.size(0)), dtype=torch.float32), requires_grad=False)
                
                # layer eualization
                graph[layer_first].weight, graph[layer_second].weight, graph[layer_first].bias, S = \
                _layer_equalization(graph[layer_first].weight,\
                                        graph[layer_second].weight,\
                                        graph[layer_first].bias,\
                                        graph[bn_idx].fake_weight,\
                                        graph[bn_idx].fake_bias, s_range=s_range, signed=signed, eps=eps)
                rr.set_scale_vec(S)
                if visualize_state:
                    visualize_per_layer(graph[layer_first].weight.detach(), 'After equalization')

            diff_tmp = 0
            for layer_idx in graph:
                if type(graph[layer_idx]) in targ_type:
                    diff_tmp += float(torch.mean(torch.abs(graph[layer_idx].weight - state_prev[layer_idx].weight)))

            if abs(diff - diff_tmp) > 1e-9:
                count = 0
                diff = diff_tmp
                
            else:
                count += 1

            # print('diff', diff)
    
    # return graph

def bias_absorption(graph, relations, bottoms, N=3):
    print("Absorbing bias")
    def is_relu_found(layer_second, layer_first, graph, bottoms):
        idx = layer_second
        while idx != layer_first:
            assert len(bottoms[idx]) == 1, 'graph in equalization relations should be 1-to-1 input-output'
            if type(graph[bottoms[idx][0]]) == torch.nn.ReLU:
                return True
            idx = bottoms[idx][0]
        return False

    for rr in relations:
        layer_first, layer_second, bn_idx = rr.get_idxs()

        if not is_relu_found(layer_second, layer_first, graph, bottoms):
            continue

        bn_weight = getattr(graph[bn_idx], 'fake_weight').detach().clone()
        bn_bias = getattr(graph[bn_idx], 'fake_bias').detach().clone()
        
        weight = graph[layer_second].weight.detach().clone()
        size = weight.shape

        num_group = graph[layer_first].weight.size(0) // graph[layer_second].weight.size(1)
        step_size_o = size[0] // num_group
        step_size_i = graph[layer_first].weight.size(0) // num_group

        c = (bn_bias - N * bn_weight)
        c.clamp_(0)

        # S = rr.get_scale_vec()
        # c[S<=1] = 0

        weight = weight.view(size[0], size[1], -1)
        wc = torch.zeros(weight.size(0))
        for g in range(num_group):
            wc[g*step_size_o:(g+1)*step_size_o] = torch.matmul(torch.sum(weight[g*step_size_o:(g+1)*step_size_o], -1), c[g*step_size_i:(g+1)*step_size_i])

        graph[layer_first].bias.data.add_(-c)
        graph[bn_idx].fake_bias.data.add_(-c)
        graph[layer_second].bias.data.add_(wc)


def clip_weight(graph, range_clip=[-15, 15], targ_type=[nn.Conv2d, nn.Linear]):
    for idx in graph:
        if type(graph[idx]) in targ_type:
            graph[idx].weight.data.copy_(graph[idx].weight.data.clamp(range_clip[0], range_clip[1]))


def bias_correction(graph, bottoms, targ_type, bits_weight=8, bn_type=torch.nn.BatchNorm2d):
    """
    Perform bias correction.
    Expectation of input activations will be summed for elementwise addition, concate for torch.cat
    """
    from utils.layer_transform import find_prev_bn
    from scipy.stats import norm
    print("Start bias correction")
    # standard_normal = lambda x: torch.exp(-(x * x) / 2) / torch.sqrt(torch.tensor(2 * np.pi))
    standard_normal = lambda x: torch.from_numpy(norm(0, 1).pdf(x)).float()
    standard_cdf = lambda x: torch.from_numpy(norm.cdf(x)).float()
    calculate_mean = lambda weight, bias: weight * standard_normal(-bias/weight) + bias * (1 - standard_cdf(-bias/weight))
    # calculate_var = lambda weight, bias, mean: (1-standard_cdf(-bias/weight)) * (bias*bias + weight*weight + mean * mean - 2 * mean * bias) +\
    #                             weight * (bias - 2 * mean) * (standard_normal(-bias/weight)) + \
    #                             mean * mean * standard_cdf(-bias/weight)

    bn_module = {}
    bn_out_shape = {}
    relu_attached = {}
    bias_prev = None
    with torch.no_grad():
        for idx_layer in graph:
            bot = bottoms[idx_layer]

            if bot is None or bot[0] == 'Data':
                continue

            if type(graph[idx_layer]) == bn_type:
                bn_module[idx_layer] = graph[idx_layer]
                bn_out_shape[idx_layer] = graph[idx_layer]
                relu_attached[idx_layer] = False
                if bias_prev is not None:
                    graph[idx_layer].fake_bias.add_(bias_prev)
                    bias_prev = None
                continue
        
            if type(graph[idx_layer]) == torch.nn.ReLU:
                if bot[0] in bn_module:
                    relu_attached[bot[0]] = True

            if type(graph[idx_layer]) in targ_type: # 1 to many or 1 to 1
                bn_list, relu_attach_list, connect_type_list, _ = find_prev_bn(bn_module, relu_attached, graph, bottoms, bot[:])

                weight = getattr(graph[idx_layer], 'weight').detach().clone()
                # eps = _quantize_error(weight.cuda(), 8, reduction=None).cpu() ## different results on gpu or cpu, move to gpu
                eps = _quantize_error(weight, 8, reduction=None)
                eps = torch.sum(eps.view(weight.size(0), weight.size(1), -1), -1)

                bn_branch = {}
                for idx, tmp in enumerate(bn_list):
                    _, bid = tmp
                    if bid[0] in bn_branch:
                        bn_branch[bid[0]].append((tmp, relu_attach_list[idx], connect_type_list[idx]))
                    else:
                        bn_branch[bid[0]] = [(tmp, relu_attach_list[idx], connect_type_list[idx])]
                bn_res = {}
                for key in bn_branch:
                    tmp_list = sorted(bn_branch[key], key=lambda x: len(x[0][1]), reverse=True)
                    node_cur, use_relu, connect_type = tmp_list[0]
                    layer_cur, bid = node_cur
                    depth = len(bid)
                    tmp_list.pop(0)
                    bn_bias = layer_cur.fake_bias.detach().clone()
                    bn_weight = layer_cur.fake_weight.detach().clone()
                    
                    if use_relu:
                        expect = calculate_mean(bn_weight, bn_bias)
                        expect[expect < 0] = 0
                    else:
                        expect = bn_bias

                    while len(tmp_list) > 0:
                        idx_bound = 0
                        
                        while idx_bound < len(tmp_list) and len(tmp_list[idx_bound][0][1]) == depth:
                            idx_bound += 1

                        if idx_bound == 0 and len(tmp_list) > 0:
                            # cut depth, add node_cur back
                            depth = len(tmp_list[idx_bound][0][1])

                        else:
                            for idx in range(idx_bound):
                                node_tmp, use_relu_tmp, connect_type = tmp_list[idx]
                                bn_bias = node_tmp[0].fake_bias.detach().clone()
                                bn_weight = node_tmp[0].fake_weight.detach().clone()

                                if use_relu_tmp:
                                    expect_tmp = calculate_mean(bn_weight, bn_bias)
                                    expect_tmp[expect_tmp < 0] = 0
                                else:
                                    expect_tmp = bn_bias

                                if 'cat' == connect_type:
                                    expect = torch.cat([expect, expect_tmp], 0)
                                
                                else:
                                    expect += expect_tmp

                            tmp_list = tmp_list[idx_bound:]
                            # expect /= (idx_bound + 1)

                    bn_res[key] = (connect_type, expect)
                assert len(bn_res) == 1, "Error while calculating expectation for bias correction"
                if 'cat' == list(bn_res.values())[0][0]:
                    expect = torch.cat(list(zip(list(bn_res.values())[0]))[1], 0)

                # group operation
                num_group = expect.size(0) // eps.size(1)
                step_size_o = eps.size(0) // num_group
                step_size_i = expect.size(0) // num_group

                bias = torch.zeros(eps.size(0))
                for g in range(num_group):
                    bias[g*step_size_o:(g+1)*step_size_o] = torch.matmul(eps[g*step_size_o:(g+1)*step_size_o], expect[g*step_size_i:(g+1)*step_size_i])

                # bias = torch.matmul(eps, expect)
                graph[idx_layer].bias.add_(-bias)
                bias_prev = -bias
