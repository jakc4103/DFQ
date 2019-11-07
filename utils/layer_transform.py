import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from PyTransformer.transformers.quantize import QConv2d, ReLUQuant, QuantConv2d, quantize, QuantMeasure
tensor_target = torch.Tensor
raw_tensor_magic_op = {}
tensor_target = torch.Tensor
tensor_magic_op_supported = ['__add__', 'add', '__iadd__']
raw_torch_op = {}
torch_op_supported = ['cat']
raw_func_op = {}
func_op_sopprted = ['interpolate']


# class LayerTransform():
#     def __init__(self):
#         self.raw_tensor_magic_op = {}
#         self.tensor_magic_op_supported = ['__add__', 'add', '__iadd__']
#         self.raw_torch_op = {}
#         self.torch_op_supported = ['cat']
#         self.raw_func_op = {}
#         self.func_op_sopprted = ['interpolate']
#         self.name_tensor_op = None
#         # self.module_tensor_op = None
#         self.idx_tensor_op_quantize = 0
#         self.model = None


#     def ___add__(self, input, *args):
#         # global name_tensor_op, idx_tensor_op_quantize
#         _stack = inspect.stack()
#         # name_tensor_op = getattr(_stack[8][0].f_locals['self'], 'name_tensor_op')
#         # idx_tensor_op_quantize = getattr(_stack[8][0].f_locals['self'], 'idx_tensor_op')
#         print('{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno))
#         if '{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno) == self.name_tensor_op[self.idx_tensor_op_quantize]:
#             # input = _stack[8][0].f_locals['self']._modules['custom_tensor_op'](input, self.idx_tensor_op_quantize*2)
#             # args = [_stack[8][0].f_locals['self']._modules['custom_tensor_op'](args[0], self.idx_tensor_op_quantize*2 + 1)]
#             input = self.model._modules['custom_tensor_op'](input, self.idx_tensor_op_quantize*2)
#             print("__add__", len(args), args)
#             print(type(self), type(input), type(args))
#             print(input.shape)
#             args = [self.model._modules['custom_tensor_op'](args[0], self.idx_tensor_op_quantize*2 + 1)]
#             self.idx_tensor_op_quantize += 1
#             self.idx_tensor_op_quantize %= len(self.name_tensor_op)

#         x = self.raw_tensor_magic_op['__add__'](input, *args)

#         del _stack

#         return x


#     def ___iadd__(self, input, *args):
#         # global name_tensor_op, idx_tensor_op_quantize
#         _stack = inspect.stack()
#         # name_tensor_op = getattr(_stack[8][0].f_locals['self'], 'name_tensor_op')
#         # idx_tensor_op_quantize = getattr(_stack[8][0].f_locals['self'], 'idx_tensor_op')
#         if '{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno) == self.name_tensor_op[self.idx_tensor_op_quantize]:
#             input = _stack[8][0].f_locals['self']._modules['custom_tensor_op'](input, self.idx_tensor_op_quantize*2)
#             args = [_stack[8][0].f_locals['self']._modules['custom_tensor_op'](args[0], self.idx_tensor_op_quantize*2 + 1)]
#             self.idx_tensor_op_quantize += 1
#             self.idx_tensor_op_quantize %= len(self.name_tensor_op)

#         x = self.raw_tensor_magic_op['__add__'](input, *args)

#         del _stack

#         return x

#     def _add(self, input,*args):
#         return self.___add__(input, *args)


def ___add__(input, *args):
    # global name_tensor_op, idx_tensor_op_quantize
    global raw_tensor_magic_op, module_tensor_op
    _stack = inspect.stack()
    if '{}_{}_2'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno) == module_tensor_op.get_name_next():
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

    if '{}_{}_2'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno) == module_tensor_op.get_name_next():
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
    if 'torch_cat_{}_{}'.format(_stack[1].lineno, len(inputs)) == module_tensor_op.get_name_next():
        qinputs = []
        for t in inputs:
            qinputs.append(module_tensor_op(t))

        module_tensor_op.add_idx_name_tensor_op()
    
    x = raw_torch_op['cat'](tuple(qinputs), dim)

    del _stack

    return x

def F_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    global raw_func_op, module_tensor_op
    _stack = inspect.stack()

    if 'F_interpolate_{}_1'.format(_stack[1].lineno) == module_tensor_op.get_name_next():
        input = module_tensor_op(input)

        module_tensor_op.add_idx_name_tensor_op()
    
    x = raw_func_op['interpolate'](input, size, scale_factor, mode, align_corners)
    
    del _stack

    return x


def replace_op():
    global tensor_magic_op_supported, raw_tensor_magic_op, torch_op_supported, raw_torch_op, func_op_sopprted, raw_func_op
    # global tensor_target
    # tensor_target = torch.Tensor
    # for op_name in layer_transform.tensor_magic_op_supported:
    for op_name in tensor_magic_op_supported:
        raw_op = getattr(torch.Tensor ,op_name)
        # layer_transform.raw_tensor_magic_op[op_name] = raw_op
        raw_tensor_magic_op[op_name] = raw_op
        # setattr(tensor_target, op_name, getattr(layer_transform, '_' + op_name))
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
    # tensor_target = torch.Tensor
    # for op_name in layer_transform.tensor_magic_op_supported:
    for op_name in tensor_magic_op_supported:
        # setattr(tensor_target, op_name, layer_transform.raw_tensor_magic_op[op_name])
        setattr(tensor_target, op_name, raw_tensor_magic_op[op_name])

    for op_name in torch_op_supported:
        setattr(torch, op_name, raw_torch_op[op_name])

    for op_name in func_op_sopprted:
        setattr(F, op_name, raw_func_op[op_name])


def switch_layers(model, transformer, data, ignore=['pad']):
    # replace layers
    transformer.register(nn.ReLU6, nn.ReLU)
    # transformer.register(nn.ReLU, ReLUQuant)
    model = transformer.trans_layers(model, update=False)
    # transformer.register(nn.Conv2d, QConv2d)
    transformer.register(nn.Conv2d, QuantConv2d)
    model = transformer.trans_layers(model)

    transformer._build_graph(model, data) # construt graph after all state_dict loaded

    global module_tensor_op
    tmp = transformer.log.getRecordTensorOP()
    for ig in ignore:
        idx = 0
        while idx < len(tmp):
            if ig in tmp[idx]:
                tmp.pop(idx)
                continue
            idx += 1

    tensor_op_quantize = []
    for t in tmp:
        tok = t.split('_')
        qnum = int(tok[-1])
        while qnum > 0:
            tensor_op_quantize.append(QuantMeasure(num_bits=8, momentum=0.1))
            qnum -= 1

    name_tensor_op = tmp
    # setattr(layer_transform, 'name_tensor_op', name_tensor_op)

    module_tensor_op = CustomTensorOP(tensor_op_quantize, name_tensor_op)
    model.add_module('custom_tensor_op', module_tensor_op)
    setattr(model, 'name_tensor_op', name_tensor_op)
    setattr(model, 'idx_name_tensor_op', 0)
    setattr(model, 'idx_tensor_op', 0)

    # setattr(layer_transform, 'model', model)

    return model


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

    # def __len__(self):
    #     return self.len

    def add_idx_tensor_op(self):
        self.idx_tensor_op = (self.idx_tensor_op + 1) % self.len
    
    def add_idx_name_tensor_op(self):
        self.idx_name_tensor_op = (self.idx_name_tensor_op + 1) % self.num_op
    
    def get_name_next(self):
        return self.name_tensor_op[self.idx_name_tensor_op]

    def get_module_next(self):
        mod = self._modules[str(self.idx_tensor_op)]
        self.add_idx_tensor_op()
        return mod

    def forward(self, x):
        x = self._modules[str(self.idx_tensor_op)](x)
        self.add_idx_tensor_op()

        return x


def merge_batchnorm(model, graph, bottoms, conv_type=QConv2d):
    with torch.no_grad():
        # merge bn params into QConv2d
        for layer_idx in graph:
            # print(bottoms[layer_idx])
            if bottoms[layer_idx] is None:
                continue
            for bot_idx in bottoms[layer_idx]:
                if type(graph[layer_idx]) == nn.BatchNorm2d and type(graph[bot_idx]) == conv_type:
                    # TODO: suppport gpu version
                    conv_weight = graph[bot_idx].weight.detach()
                    bn_weight = graph[layer_idx].weight.detach()
                    bn_var = graph[layer_idx].running_var.detach()
                    bn_eps = graph[layer_idx].eps

                    graph[bot_idx].weight.copy_(conv_weight.mul(bn_weight.view(-1, 1, 1, 1) / torch.sqrt(bn_var.view(-1, 1, 1, 1) + bn_eps)))

                    if graph[bot_idx].bias is None:
                        graph[bot_idx].bias = nn.Parameter(data=torch.zeros((graph[bot_idx].weight.size(0)), dtype=torch.float), requires_grad=False)

                    conv_bias = graph[bot_idx].bias.detach()
                    bn_bias = graph[layer_idx].bias.detach()
                    bn_mean = graph[layer_idx].running_mean.detach()

                    graph[bot_idx].bias.copy_(conv_bias.mul(bn_weight.view(-1) / torch.sqrt(bn_var.view(-1) + bn_eps)).add(bn_bias.view(-1) -\
						 (bn_weight.view(-1) * bn_mean.view(-1)) / torch.sqrt(bn_var.view(-1) + bn_eps)))

                    graph[layer_idx].register_buffer('fake_weight', bn_weight.clone())
                    graph[layer_idx].register_buffer('fake_bias', bn_bias.clone())

                    graph[layer_idx].weight.fill_(1)
                    graph[layer_idx].running_var.fill_(1)
                    graph[layer_idx].bias.fill_(0)
                    graph[layer_idx].running_mean.fill_(0)
                    graph[layer_idx].eps = 0

                    
                    # print(graph[layer_idx].running_var)

                    break

    return model


def set_quant_minmax(graph, bottoms, output_shape, bn_type=torch.nn.BatchNorm2d, N=6):
    from collections import OrderedDict
    print("SET QUANT MIN MAX")

    def get_quant_module(layer):
        if type(layer) == str:
            global module_tensor_op, tensor_magic_op_supported, torch_op_supported, func_op_sopprted
            quant_op = []
            for op_name in tensor_magic_op_supported + torch_op_supported + func_op_sopprted:
                quant_op.append(op_name.replace('_', ''))

            for op_name in quant_op:
                if op_name in layer:
                    module_list = []
                    num_op = int(module_tensor_op.get_name_next().split('_')[-1])
                    
                    while num_op > 0:
                        module_list.append(module_tensor_op.get_module_next())
                        num_op -= 1

                    module_tensor_op.add_idx_name_tensor_op()

                    return module_list
            
        elif hasattr(layer, 'quant'):
            return [getattr(layer, 'quant')]
        
        return None
    
    bn_module = {}
    relu_attached = {}
    bn_out_shape = {}
    get_max_value = lambda bias, weight, n: float(torch.max(bias + n * weight))
    get_min_value = lambda bias, weight, n: float(torch.min(bias - n * weight))
    for idx_layer in graph:
        # print("process: {}".format(idx_layer))
        bot = bottoms[idx_layer]

        if bot is None:
            continue

        if type(graph[idx_layer]) == bn_type:
            bn_module[idx_layer] = graph[idx_layer]
            bn_out_shape[idx_layer] = graph[idx_layer]
            relu_attached[idx_layer] = False
            continue
        
        if type(graph[idx_layer]) == torch.nn.ReLU:
            if bot[0] in bn_module:
                relu_attached[bot[0]] = True

        quant_module = get_quant_module(graph[idx_layer])
        if len(bot) == 1 and bot[0] == 'Data':
            quant_module[0].running_max.fill_(2.64) # use (1 - mean)/std as in data preprocess
            quant_module[0].running_min.fill_(-2.11790393) # use (0 - mean)/std as in data preprocess

        elif quant_module is not None: # set min/max w.r.t. previous layer (batch norm, add)
            bot_tmp = list(zip(bot[:], range(len(bot[:]))))
            bn_list = []
            relu_attach_list = []
            while len(bot_tmp) > 0:
                idx_bot, bid = bot_tmp.pop(0)
                if idx_bot not in bn_module:
                    bot_tmp.extend(list(zip(bottoms[idx_bot], [bid]*len(bottoms[idx_bot]))))
                else:
                    bn_list.append((bn_module[idx_bot], bid))
                    relu_attach_list.append(relu_attached[idx_bot])

            if len(quant_module) == len(bn_list): # 1 to 1 mapping
                idx = 0
                while idx < len(bn_list):
                    bias = getattr(bn_list[idx][0], 'bias') if not hasattr(bn_list[idx][0], 'fake_bias') else getattr(bn_list[idx][0], 'fake_bias')
                    weight = getattr(bn_list[idx][0], 'weight') if not hasattr(bn_list[idx][0], 'fake_weight') else getattr(bn_list[idx][0], 'fake_weight')
                    value_max = get_max_value(bias, weight, N)
                    value_min = get_min_value(bias, weight, N) if not relu_attach_list[idx] else 0.
                    # print("type 1, max {}, min {}".format(value_max, value_min))
                    quant_module[idx].running_max.fill_(value_max)
                    quant_module[idx].running_min.fill_(value_min)
                    idx += 1

            elif len(quant_module) == 1 and len(quant_module) < len(bn_list): # 1 to many
                idx = 0
                value_max = 0
                value_min = 0
                use_relu = True
                while idx < len(bn_list):
                    bias = getattr(bn_list[idx][0], 'bias') if not hasattr(bn_list[idx][0], 'fake_bias') else getattr(bn_list[idx][0], 'fake_bias')
                    weight = getattr(bn_list[idx][0], 'weight') if not hasattr(bn_list[idx][0], 'fake_weight') else getattr(bn_list[idx][0], 'fake_weight')
                    value_max += get_max_value(bias, weight, N)
                    value_min += get_min_value(bias, weight, N)
                    use_relu = use_relu and relu_attach_list[idx]
                    idx += 1
                # TODO: weighting w.r.t. tensor size
                value_max /= len(bn_list)
                value_min /= len(bn_list)
                # print("type 2, max {}, min {}".format(value_max, value_min if not use_relu else 0.))
                quant_module[0].running_max.fill_(value_max)
                quant_module[0].running_min.fill_(value_min if not use_relu else 0.)

            elif len(quant_module) < len(bn_list): # many to many
                idx = len(bn_list) - 1
                bn_res = []
                bias = 0
                weight = 0
                value_max = 0
                value_min = 0
                count = 0
                relu_attach_list = [x for x, _ in sorted(zip(relu_attach_list, bn_list), key=lambda x: x[1][1])]
                bn_list = sorted(bn_list, key=lambda x: x[1])
                use_relu = True
                while idx >= 0:
                    bn, bid = bn_list[idx]
                    bias = getattr(bn, 'bias') if not hasattr(bn, 'fake_bias') else getattr(bn, 'fake_bias')
                    weight = getattr(bn, 'weight') if not hasattr(bn, 'fake_weight') else getattr(bn, 'fake_weight')
                    value_max += get_max_value(bias, weight, N)
                    value_min += get_min_value(bias, weight, N)
                    use_relu = use_relu and relu_attach_list[idx]
                    idx -= 1
                    count += 1
                    if idx < 0 or bid != bn_list[idx][1]:
                        value_max /= count
                        value_min /= count
                        bn_res.append((value_max, value_min if not use_relu else 0.))
                        value_max = 0
                        value_min = 0
                        use_relu = True
                idx = 0

                assert len(bn_res) == len(quant_module), 'LENGTH NOT EQUAL {} vs {}'.format(len(bn_res), len(quant_module))
                while idx < len(bn_res):
                    value_max, value_min = bn_res[idx]
                    # print("type 3, max {}, min {}".format(value_max, value_min))
                    quant_module[idx].running_max.fill_(value_max)
                    quant_module[idx].running_min.fill_(value_min)
                    idx += 1

            else:
                print(len(quant_module), len(bn_list))
                assert False, "ERRORORRORORO"
