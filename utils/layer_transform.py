import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from utils.quantize import QConv2d, QuantConv2d, QuantNConv2d, QLinear, QuantLinear, QuantNLinear, quantize, QuantMeasure
tensor_target = torch.Tensor
raw_tensor_magic_op = {}
tensor_target = torch.Tensor
tensor_magic_op_supported = ['__add__', 'add', '__iadd__']
raw_torch_op = {}
torch_op_supported = ['cat', 'mean']
raw_func_op = {}
func_op_sopprted = ['interpolate', 'softmax']

def ___add__(input, *args):
    # global name_tensor_op, idx_tensor_op_quantize
    global raw_tensor_magic_op, module_tensor_op
    _stack = inspect.stack()
    if 'forward' == _stack[1].function and '{}_{}_2'.format('add', _stack[1].lineno) == module_tensor_op.get_module_name():
        input = module_tensor_op(input)
        args = [module_tensor_op(args[0])]

        module_tensor_op.add_idx_name_tensor_op()

    x = raw_tensor_magic_op['__add__'](input, *args)

    del _stack

    return x


def ___iadd__(input, *args):
    # global name_tensor_op, idx_tensor_op_quantize
    global raw_tensor_magic_op, module_tensor_op
    _stack = inspect.stack()

    if 'forward' == _stack[1].function and '{}_{}_2'.format('iadd', _stack[1].lineno) == module_tensor_op.get_module_name():
        input = module_tensor_op(input)
        args = [module_tensor_op(args[0])]

        module_tensor_op.add_idx_name_tensor_op()

    x = raw_tensor_magic_op['__add__'](input, *args)

    del _stack

    return x


def _add(input,*args):
    return ___add__(input, *args)


def torch_cat(inputs, dim=0):
    global raw_torch_op, module_tensor_op
    _stack = inspect.stack()
    if 'forward' == _stack[1].function and 'torch_cat_{}_{}'.format(_stack[1].lineno, len(inputs)) == module_tensor_op.get_module_name():
        qinputs = []
        for t in inputs:
            qinputs.append(module_tensor_op(t))

        module_tensor_op.add_idx_name_tensor_op()
    else:
        qinputs = inputs

    x = raw_torch_op['cat'](tuple(qinputs), dim)

    del _stack

    return x


def torch_mean(input, dim=None, keepdim=False, out=None):
    global raw_torch_op, module_tensor_op
    _stack = inspect.stack()
    if 'forward' == _stack[1].function and 'torch_mean_{}_1'.format(_stack[1].lineno) == module_tensor_op.get_module_name():
        input = module_tensor_op(input)

        module_tensor_op.add_idx_name_tensor_op()

    if dim is None:
        x = raw_torch_op['mean'](input)
    else:
        x = raw_torch_op['mean'](input, dim)

    del _stack

    return x


def F_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    global raw_func_op, module_tensor_op
    _stack = inspect.stack()

    if 'forward' == _stack[1].function and 'F_interpolate_{}_1'.format(_stack[1].lineno) == module_tensor_op.get_module_name():
        input = module_tensor_op(input)

        module_tensor_op.add_idx_name_tensor_op()
    
    x = raw_func_op['interpolate'](input, size, scale_factor, mode, align_corners)
    
    del _stack

    return x

def F_softmax(input, dim=None, _stacklevel=3, dtype=None):
    global raw_func_op, module_tensor_op
    _stack = inspect.stack()
    if 'forward' == _stack[1].function and 'F_softmax_{}_1'.format(_stack[1].lineno) == module_tensor_op.get_module_name():
        input = module_tensor_op(input)
        module_tensor_op.add_idx_name_tensor_op()
    
    x = raw_func_op['softmax'](input, dim, _stacklevel, dtype)

    del _stack

    return x

def replace_op():
    global tensor_magic_op_supported, raw_tensor_magic_op, torch_op_supported, raw_torch_op, func_op_sopprted, raw_func_op

    for op_name in tensor_magic_op_supported:
        raw_op = getattr(torch.Tensor ,op_name)
        raw_tensor_magic_op[op_name] = raw_op
        setattr(tensor_target, op_name, globals()['_' + op_name])

    for op_name in torch_op_supported:
        raw_op = getattr(torch, op_name)
        raw_torch_op[op_name] = raw_op
        setattr(torch, op_name, globals()['torch_' + op_name])

    for op_name in func_op_sopprted:
        raw_op = getattr(F, op_name)
        raw_func_op[op_name] = raw_op
        setattr(F, op_name, globals()['F_' + op_name])

def restore_op():
    global tensor_magic_op_supported, raw_tensor_magic_op, torch_op_supported, raw_torch_op, func_op_sopprted, raw_func_op
    global tensor_target

    for op_name in tensor_magic_op_supported:
        setattr(tensor_target, op_name, raw_tensor_magic_op[op_name])

    for op_name in torch_op_supported:
        setattr(torch, op_name, raw_torch_op[op_name])

    for op_name in func_op_sopprted:
        setattr(F, op_name, raw_func_op[op_name])


def switch_layers(model, transformer, data, module_dict, ignore_layer=[], ignore_op=['pad'], quant_op=True):
    # replace layers
    for key in module_dict:
        for source, target in module_dict[key]:
            transformer.register(source, target)
        model = transformer.trans_layers(model, update=True if key == 1 else False)
    transformer._build_graph(model, data, ignore_layer) # construt graph after all state_dict loaded
    if not quant_op:
        return model, transformer

    global module_tensor_op
    tmp = transformer.log.getRecordTensorOP()
    for ig in ignore_op:
        idx = 0
        while idx < len(tmp):
            if ig in tmp[idx][1]:
                tmp.pop(idx)
                continue
            idx += 1

    tensor_op_quantize = []
    for layer_name, t in tmp:
        tok = t.split('_')
        qnum = int(tok[-1])
        while qnum > 0:
            tensor_op_quantize.append(QuantMeasure(num_bits=8, momentum=0.1))
            qnum -= 1

    name_tensor_op = tmp

    module_tensor_op = CustomTensorOP(tensor_op_quantize, name_tensor_op)
    model.add_module('custom_tensor_op', module_tensor_op)
    setattr(model, 'name_tensor_op', name_tensor_op)
    setattr(model, 'idx_name_tensor_op', 0)
    setattr(model, 'idx_tensor_op', 0)

    return model, transformer


class CustomTensorOP(nn.Module):
    """
    special module used for quantization of torch.xxx(), F.xxx() and torch.Tensor.__xxx__()
    """
    def __init__(self, tensor_op, name_tensor_op):
        super(CustomTensorOP, self).__init__()

        for idx, op in enumerate(tensor_op):
            self.add_module(str(idx), op)

        self.idx_tensor_op = 0
        self.len = len(tensor_op)
        self.name_tensor_op = name_tensor_op
        self.idx_name_tensor_op = 0
        self.num_op = len(name_tensor_op)

    def add_idx_tensor_op(self):
        self.idx_tensor_op = (self.idx_tensor_op + 1) % self.len
    
    def add_idx_name_tensor_op(self):
        self.idx_name_tensor_op = (self.idx_name_tensor_op + 1) % self.num_op
    
    def get_module_name(self):
        return self.name_tensor_op[self.idx_name_tensor_op][1]

    def get_graph_name(self):
        return self.name_tensor_op[self.idx_name_tensor_op][0]

    def get_module_next(self):
        mod = self._modules[str(self.idx_tensor_op)]
        self.add_idx_tensor_op()
        return mod

    def forward(self, x):
        x = self._modules[str(self.idx_tensor_op)](x)
        self.add_idx_tensor_op()

        return x


def merge_batchnorm(model, graph, bottoms, targ_type=[QConv2d]):
    """!
    This function will merge params and stats of BatchNorm into targ_type like QuantConv2d.
    Once the values is merged, the values of layer will be set to default (as an identity layer),
    and it creates buffer named 'fake_weight' adn 'fake_bias' for latter usage of set_quant_minmax
    """
    with torch.no_grad():
        # merge bn params into QConv2d
        for layer_idx in graph:
            # print(bottoms[layer_idx])
            if bottoms[layer_idx] is None:
                continue
            for bot_idx in bottoms[layer_idx]:
                if type(graph[layer_idx]) == nn.BatchNorm2d and type(graph[bot_idx]) in targ_type:
                    # TODO: suppport gpu version
                    conv_weight = graph[bot_idx].weight.detach()
                    bn_weight = graph[layer_idx].weight.detach()
                    bn_var = graph[layer_idx].running_var.detach()
                    bn_eps = graph[layer_idx].eps

                    graph[bot_idx].weight.copy_(conv_weight.mul(bn_weight.view(-1, 1, 1, 1) / torch.sqrt(bn_var.view(-1, 1, 1, 1) + bn_eps)))

                    if graph[bot_idx].bias is None: # add a bias term to conv or linear layer
                        graph[bot_idx].bias = nn.Parameter(data=torch.zeros((graph[bot_idx].weight.size(0)), dtype=torch.float32), requires_grad=False)

                    conv_bias = graph[bot_idx].bias.detach()
                    bn_bias = graph[layer_idx].bias.detach()
                    bn_mean = graph[layer_idx].running_mean.detach()

                    graph[bot_idx].bias.copy_(conv_bias.mul(bn_weight.view(-1) / torch.sqrt(bn_var.view(-1) + bn_eps)).add(bn_bias.view(-1) -\
						 (bn_weight.view(-1) * bn_mean.view(-1)) / torch.sqrt(bn_var.view(-1) + bn_eps)))

                    # store values for later usage. ex: set_quant_min_max and bias correction
                    graph[layer_idx].register_buffer('fake_weight', torch.abs(bn_weight.clone()))
                    graph[layer_idx].register_buffer('fake_bias', bn_bias.clone())

                    # set batch norm layer to the same to an identity layer
                    graph[layer_idx].weight.fill_(1)
                    graph[layer_idx].running_var.fill_(1)
                    graph[layer_idx].bias.fill_(0)
                    graph[layer_idx].running_mean.fill_(0)
                    graph[layer_idx].eps = 0

                    break

    return model


def quantize_targ_layer(graph, bit_weight=8, bits_bias=16, targ_type=None):
    print("Quantizing Layer parameters")
    assert targ_type != None, "targ_type cannot be None!"
    for layer_idx in graph:
        if type(graph[layer_idx]) in targ_type:
            with torch.no_grad():
                # quantization behave differently on cpu and gpu
                param = graph[layer_idx].weight.detach()#.cuda()
                tmp = quantize(param, bit_weight, float(param.min()), float(param.max()))

                graph[layer_idx].weight.data.copy_(tmp.data.cpu())
                if graph[layer_idx].bias is not None:
                    param = graph[layer_idx].bias.detach()#.cuda()
                    graph[layer_idx].bias.data.copy_(quantize(param, bits_bias, float(param.min()), float(param.max())).data.cpu())

    return graph


def find_prev_bn(bn_module, relu_attached, graph, bottoms, bot):
    """
    Find the batchnorm layers for calculation of expectation or min/max value of input activation.
    And find branching type(one, add, or cat).
    """
    bot_tmp = list(zip(bot, [str(x) for x in range(len(bot))]))
    type_tmp = {}
    targ_without_bn = {}
    for ii in range(len(bot)):
        type_tmp[str(ii)] = 'one'
    bn_list = []
    relu_attach_list = []
    connect_type_list = []
    cat_add_found = False
    while len(bot_tmp) > 0:
        idx_bot, bid = bot_tmp.pop(0)

        if type(graph[idx_bot]) == str:
            if 'add' in idx_bot:
                if idx_bot in relu_attached:
                    type_tmp[bid] = 'add_{}'.format(relu_attached[idx_bot])
                else:
                    type_tmp[bid] = 'add'
                cat_add_found = True
            elif 'cat' in idx_bot:
                type_tmp[bid] = 'cat'
                cat_add_found = True

        elif not cat_add_found and type(graph[idx_bot]) in [nn.Conv2d, QConv2d, QuantConv2d, QuantNConv2d, nn.Linear, QLinear, QuantLinear, QuantNLinear]:
            print("Warning: {} layer before first batch norm layer detected. The calculated value range might be off.".format(type(graph[idx_bot])))
            if bid[0] in targ_without_bn:
                assert False, "Multiple conv/linear layer without batch_norm is not supported."
            if type(graph[idx_bot]) in [nn.Conv2d, QConv2d, QuantConv2d, QuantNConv2d]:
                targ_without_bn[bid[0]] = ("conv", graph[idx_bot])
            else:
                targ_without_bn[bid[0]] = ("linear", graph[idx_bot])

        if idx_bot not in bn_module:
            bot_tmp.extend(list(zip(bottoms[idx_bot], [bid + bid[0]]*len(bottoms[idx_bot]))))
            type_tmp[bid + bid[0]] = type_tmp[bid]
        else:
            bn_list.append((bn_module[idx_bot], bid))
            relu_attach_list.append(relu_attached[idx_bot])
            connect_type_list.append(type_tmp[bid])

    return bn_list, relu_attach_list, connect_type_list, targ_without_bn


def set_quant_minmax(graph, bottoms, is_detection=False, bn_type=torch.nn.BatchNorm2d, N=6, verbose=True):
    """!
    This function set the running_min/running_max value of QuantMeasure using the statistics form previous BatchNorm layer.
    Since I handle the values layer by layer, there will be 3 + 1 cases in computing min/max:
        a. 1 to 1 mapping. ex: BatchNorm->ReLU->'QuantConv'
        b. 1 to many mapping. ex: BatchNorm->ReLU->
                                                    QuantAdd->'QuantConv'
                                  BatchNorm->ReLU->
        c. many to many. ex: BatchNorm->ReLU->
                                               QuantAdd->
                             BatchNorm->ReLU->     ;      'QuantAdd'
                                        BatchNorm->ReLU->

        d. layers without batch normalization. ex: BatchNorm->ReLU->QuantConv->'Concat' (in last layer of detection branch, SSD model)
    
    For elementwise addition, the input activations are assumed to be independent gaussian distributions.
    Thus I accumulate the mean and variance of distributions, then compute min/max at final step.

    For concatination, I will take the min/max values of input activations.

    For (d.) case, I use the min/max values of last available batch_norm layer to convolve with weight in conv layer.

    Otherwise, the min/max mean value of all input activations is used.
    """
    from collections import OrderedDict
    from scipy.stats import norm
    if verbose:
        print("SET QUANT MIN MAX")

    def get_quant_module(layer, bot):
        if type(layer) == str:
            global module_tensor_op, tensor_magic_op_supported, torch_op_supported, func_op_sopprted
            # print("layer name", layer, module_tensor_op.get_graph_name())
            if module_tensor_op.get_graph_name() == layer:
                quant_op = []
                for op_name in tensor_magic_op_supported + torch_op_supported + func_op_sopprted:
                    quant_op.append(op_name.replace('_', ''))

                for op_name in quant_op:
                    if op_name in layer:
                        module_list = []
                        num_op = int(module_tensor_op.get_module_name().split('_')[-1])
                        
                        while num_op > 0:
                            module_list.append(module_tensor_op.get_module_next())
                            num_op -= 1

                        module_tensor_op.add_idx_name_tensor_op()

                        return module_list
            
        elif hasattr(layer, 'quant'): # special keyword 'quant'
            return [getattr(layer, 'quant')]
        
        return None
    
    bn_module = {}
    relu_attached = {}
    bn_out_shape = {}
    eps = 1e-6
    get_max_value = lambda bias, weight, n: float(torch.max(bias + n * weight))
    get_min_value = lambda bias, weight, n: float(torch.min(bias - n * weight))
    standard_normal = lambda x: torch.from_numpy(norm(0, 1).pdf(x)).float()
    standard_cdf = lambda x: torch.from_numpy(norm.cdf(x)).float()
    calculate_mean = lambda weight, bias: weight * standard_normal(-bias/weight) + bias * (1 - standard_cdf(-bias/weight))
    calculate_var = lambda weight, bias, mean: (1-standard_cdf(-bias/weight)) * (bias*bias + weight*weight + mean * mean - 2 * mean * bias) +\
                                weight * (bias - 2 * mean) * (standard_normal(-bias/weight)) + \
                                mean * mean * standard_cdf(-bias/weight)
    calculate_mean_6 = lambda weight, bias: weight * (standard_normal(-bias/weight) - standard_normal((6-bias)/weight)) +\
                                            bias * (standard_cdf((6-bias)/weight) - standard_cdf(-bias/weight)) +\
                                            6 * (1 - standard_cdf((6-bias)/weight))
    calculate_var_6 = lambda weight, bias, mean: (standard_cdf((6-bias)/weight)-standard_cdf(-bias/weight)) * (bias*bias + weight*weight + mean * mean - 2 * mean * bias) +\
                                                weight * (-6) * standard_normal((6-bias)/weight) +\
                                                weight * (bias - 2 * mean) * (standard_normal(-bias/weight) - standard_normal((6-bias)/weight)) +\
                                                mean * mean * standard_cdf(-bias/weight) +\
                                                ((6 - mean) ** 2) * (1 - standard_cdf((6-bias)/weight))
    for idx_layer in graph:
        # print("process: {}, {}".format(idx_layer, type(graph[idx_layer])))
        bot = bottoms[idx_layer]

        if bot is None:
            continue

        if type(graph[idx_layer]) == bn_type:
            bn_module[idx_layer] = graph[idx_layer]
            bn_out_shape[idx_layer] = graph[idx_layer]
            relu_attached[idx_layer] = 'none'
            continue
        
        if type(graph[idx_layer]) == torch.nn.ReLU:
            # if bot[0] in bn_module:
            relu_attached[bot[0]] = 'relu'
        elif type(graph[idx_layer]) == torch.nn.ReLU6:
            relu_attached[bot[0]] = 'relu6'

        quant_module = get_quant_module(graph[idx_layer], bot)
        if len(bot) == 1 and bot[0] == 'Data':
            if is_detection:
                quant_module[0].running_max.fill_(1)
                quant_module[0].running_min.fill_(-1)
            else: # segmentation and classification
                quant_module[0].running_max.fill_(2.64) # use (1 - mean)/std as in data preprocess
                quant_module[0].running_min.fill_(-2.11790393) # use (0 - mean)/std as in data preprocess

        elif quant_module is not None: # set min/max w.r.t. previous layer (batch norm, add)
            bn_list, relu_attach_list, connect_type_list, targ_without_bn = find_prev_bn(bn_module, relu_attached, graph, bottoms, bot[:])
            if len(quant_module) == len(bn_list): # 1 to 1 mapping
                idx = 0
                while idx < len(bn_list):
                    bias = getattr(bn_list[idx][0], 'fake_bias').view(-1)
                    weight = getattr(bn_list[idx][0], 'fake_weight').view(-1)
                    
                    if bn_list[idx][1][0] in targ_without_bn:
                        # case (d.)
                        layer_type, obj_layer = targ_without_bn[bn_list[idx][1][0]]
                        layer_weight = getattr(obj_layer, 'weight').detach().data
                        layer_bias = getattr(obj_layer, 'bias').detach().data

                        if layer_type == 'conv':
                            layer_weight = layer_weight.view(layer_weight.size(0), layer_weight.size(1), -1).sum(-1)
                            layer_weight = layer_weight.view(layer_weight.size(0), layer_weight.size(1), 1, 1)
                            bias = torch.nn.functional.conv2d(bias.view(1, -1, 1, 1), layer_weight, layer_bias, 1, 0, 1, getattr(obj_layer, 'groups'))
                            weight = torch.nn.functional.conv2d(weight.view(1, -1, 1, 1), layer_weight, layer_bias, 1, 0, 1, getattr(obj_layer, 'groups'))
                        else:
                            bias = torch.nn.functional.linear(bias.view(1, -1), layer_weight, layer_bias)
                            weight = torch.nn.functional.linear(weight.view(1, -1), layer_weight, layer_bias)

                        value_max = get_max_value(bias, weight, N)
                        value_min = get_min_value(bias, weight, N)

                    else:
                        value_min = max(0., get_min_value(bias, weight, N)) if 'relu' in relu_attach_list[idx] else get_min_value(bias, weight, N)
                        value_max = min(6., get_max_value(bias, weight, N)) if 'relu6' in relu_attach_list[idx] else get_max_value(bias, weight, N)
                    # print("type 1, max {}, min {}".format(value_max, value_min))
                    quant_module[idx].running_max.fill_(value_max)
                    quant_module[idx].running_min.fill_(value_min)
                    idx += 1

            else:  # 1 to many or many to many
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
                    bias = layer_cur.fake_bias.detach().clone()
                    weight = layer_cur.fake_weight.detach().clone()
                    if 'add' in connect_type:
                        if 'relu' == use_relu:
                            mean = calculate_mean(weight, bias)
                            var = calculate_var(weight, bias, mean)
                        elif 'relu6' == use_relu:
                            mean = calculate_mean_6(weight, bias)
                            var = calculate_var_6(weight, bias, mean)
                        else:
                            mean = bias
                            var = weight * weight
                    else:
                        value_min = max(0., get_min_value(bias, weight, N)) if 'relu' in use_relu else get_min_value(bias, weight, N)
                        value_max = min(6., get_max_value(bias, weight, N)) if 'relu6' in use_relu else get_max_value(bias, weight, N)

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
                                bias = node_tmp[0].fake_bias.detach().clone()
                                weight = node_tmp[0].fake_weight.detach().clone()
                                if 'add' in connect_type:
                                    if 'relu' == use_relu_tmp:
                                        mean_tmp = calculate_mean(weight, bias)
                                        mean += mean_tmp
                                        var += calculate_var(weight, bias, mean_tmp)
                                    elif 'relu6' == use_relu_tmp:
                                        mean_tmp = calculate_mean_6(weight, bias)
                                        mean += mean_tmp
                                        var += calculate_var_6(weight, bias, mean_tmp)
                                    else:
                                        mean += bias
                                        var += weight * weight

                                    if 'relu6' in connect_type:
                                        mean_tmp = mean
                                        mean = calculate_mean_6(torch.sqrt(var + eps), mean)
                                        var = calculate_var_6(torch.sqrt(var + eps), mean_tmp, mean)
                                    elif 'relu' in connect_type:
                                        mean_tmp = mean
                                        mean = calculate_mean(torch.sqrt(var + eps), mean)
                                        var = calculate_var(torch.sqrt(var + eps), mean_tmp, mean)
                                else:
                                    if 'cat' == connect_type:
                                        value_min = min(value_min, max(0., get_min_value(bias, weight, N)) if 'relu' in use_relu_tmp else get_min_value(bias, weight, N))
                                        value_max = max(value_max, min(6., get_max_value(bias, weight, N)) if 'relu6' in use_relu_tmp else get_max_value(bias, weight, N))
                                    else:
                                        value_min += max(0., get_min_value(bias, weight, N)) if use_relu_tmp else get_min_value(bias, weight, N)
                                        value_max += get_max_value(bias, weight, N)
                                
                            tmp_list = tmp_list[idx_bound:]
                            if 'one' == connect_type:
                                value_min /= (idx_bound + 1)
                                value_max /= (idx_bound + 1)
                    if 'add' in connect_type:
                        bn_res[key] = (connect_type, mean, var)
                    else:
                        bn_res[key] = (connect_type, value_min, value_max)

                if len(quant_module) == 1 and len(quant_module) < len(bn_list): # 1 to many
                    assert len(list(bn_res.keys())) == 1, "Error occurs when setting min/max, should be 1 to many"
                    # connect_type, value_min, value_max = list(bn_res.values())[0]
                    if 'add' in list(bn_res.values())[0][0]:
                        _, mean, var = list(bn_res.values())[0]
                        value_min = get_min_value(mean, torch.sqrt(var + eps), N)
                        value_max = get_max_value(mean, torch.sqrt(var + eps), N)
                        # dont needed
                        # if 'relu' in list(bn_res.values())[0][0] and value_min < 0:
                        #     value_min = 0
                        # if 'relu6' in list(bn_res.values())[0][0] and value_max > 6:
                        #     value_max = 6
                    else:
                        _, value_min, value_max = list(bn_res.values())[0]

                    # print("type 2, max {}, min {}".format(value_max, value_min))
                    quant_module[0].running_max.fill_(value_max)
                    quant_module[0].running_min.fill_(value_min)

                elif len(quant_module) < len(bn_list): # many to many
                    idx = 0
                    assert len(bn_res) == len(quant_module), 'LENGTH NOT EQUAL {} vs {}'.format(len(bn_res), len(quant_module))
                    while idx < len(bn_res):
                        if 'add' in bn_res[str(idx)][0]:
                            _, mean, var = bn_res[str(idx)]
                            value_min = get_min_value(mean, torch.sqrt(var + eps), N)
                            value_max = get_max_value(mean, torch.sqrt(var + eps), N)
                            # if 'relu' in bn_res[str(idx)][0] and value_min < 0:
                            #     value_min = 0
                            # if 'relu6' in bn_res[str(idx)][0] and value_max > 6:
                            #     value_max = 6
                        else:
                            _, value_min, value_max = bn_res[str(idx)]

                        # print("type 3, max {}, min {}".format(value_max, value_min))
                        quant_module[idx].running_max.fill_(value_max)
                        quant_module[idx].running_min.fill_(value_min)
                        idx += 1
                else:
                    assert False, "Unknown error occured while setting min/max"