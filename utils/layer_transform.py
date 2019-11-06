import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from PyTransformer.transformers.quantize import QConv2d, ReLUQuant, QuantConv2d, quantize, QuantMeasure

class LayerTransform():
    def __init__(self):
        self.raw_tensor_magic_op = {}
        self.tensor_target = torch.Tensor
        self.tensor_magic_op_supported = ['__add__', 'add', '__iadd__']
        self.raw_torch_op = {}
        self.torch_op_supported = ['cat']
        self.raw_func_op = {}
        self.func_op_sopprted = ['adaptive_avg_pool2d', 'interpolate']
        self.name_tensor_op = None
        # self.module_tensor_op = None
        self.idx_tensor_op_quantize = 0
        self.model = None


    def ___add__(self, input, *args):
        # global name_tensor_op, idx_tensor_op_quantize
        _stack = inspect.stack()
        # name_tensor_op = getattr(_stack[8][0].f_locals['self'], 'name_tensor_op')
        # idx_tensor_op_quantize = getattr(_stack[8][0].f_locals['self'], 'idx_tensor_op')
        print('{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno))
        if '{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno) == self.name_tensor_op[self.idx_tensor_op_quantize]:
            # input = _stack[8][0].f_locals['self']._modules['custom_tensor_op'](input, self.idx_tensor_op_quantize*2)
            # args = [_stack[8][0].f_locals['self']._modules['custom_tensor_op'](args[0], self.idx_tensor_op_quantize*2 + 1)]
            input = self.model._modules['custom_tensor_op'](input, self.idx_tensor_op_quantize*2)
            print("__add__", len(args), args)
            print(type(self), type(input), type(args))
            print(input.shape)
            args = [self.model._modules['custom_tensor_op'](args[0], self.idx_tensor_op_quantize*2 + 1)]
            self.idx_tensor_op_quantize += 1
            self.idx_tensor_op_quantize %= len(self.name_tensor_op)

        x = self.raw_tensor_magic_op['__add__'](input, *args)

        del _stack

        return x


    def ___iadd__(self, input, *args):
        # global name_tensor_op, idx_tensor_op_quantize
        _stack = inspect.stack()
        # name_tensor_op = getattr(_stack[8][0].f_locals['self'], 'name_tensor_op')
        # idx_tensor_op_quantize = getattr(_stack[8][0].f_locals['self'], 'idx_tensor_op')
        if '{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno) == self.name_tensor_op[self.idx_tensor_op_quantize]:
            input = _stack[8][0].f_locals['self']._modules['custom_tensor_op'](input, self.idx_tensor_op_quantize*2)
            args = [_stack[8][0].f_locals['self']._modules['custom_tensor_op'](args[0], self.idx_tensor_op_quantize*2 + 1)]
            self.idx_tensor_op_quantize += 1
            self.idx_tensor_op_quantize %= len(self.name_tensor_op)

        x = self.raw_tensor_magic_op['__add__'](input, *args)

        del _stack

        return x


    def _add(self, input,*args):
        return self.___add__(input, *args)


    # def _cat(input, )

def replace_op(layer_transform):
    # global tensor_target, tensor_magic_op_supported

    for op_name in layer_transform.tensor_magic_op_supported:
        raw_op = getattr(layer_transform.tensor_target ,op_name)
        layer_transform.raw_tensor_magic_op[op_name] = raw_op
        print("replace op", getattr(layer_transform, '_' + op_name))
        setattr(layer_transform.tensor_target, op_name, getattr(layer_transform, '_' + op_name))


def restore_op(layer_transform):
    # global raw_tensor_magic_op, tensor_target, tensor_magic_op_supported

    for op_name in layer_transform.tensor_magic_op_supported:
        setattr(layer_transform.tensor_target, op_name, layer_transform.raw_tensor_magic_op[op_name])


def switch_layers(layer_transform, model, transformer, data):
    # replace layers
    transformer.register(nn.ReLU6, nn.ReLU)
    # transformer.register(nn.ReLU, ReLUQuant)
    model = transformer.trans_layers(model, update=False)
    # transformer.register(nn.Conv2d, QConv2d)
    transformer.register(nn.Conv2d, QuantConv2d)
    model = transformer.trans_layers(model)

    transformer._build_graph(model, data) # construt graph after all state_dict loaded

    # global name_tensor_op, module_tensor_op
    name_tensor_op = transformer.log.getRecordTensorOP()
    setattr(layer_transform, 'name_tensor_op', name_tensor_op)

    tensor_op_quantize = []
    for _ in range(len(name_tensor_op) * 2):
        tensor_op_quantize.append(QuantMeasure(num_bits=8, momentum=0.1))

    module_tensor_op = CustomTensorOP(tensor_op_quantize)
    model.add_module('custom_tensor_op', module_tensor_op)
    # setattr(model, 'name_tensor_op', self.name_tensor_op)
    # setattr(model, 'idx_tensor_op', 0)

    setattr(layer_transform, 'model', model)

    return model


class CustomTensorOP(nn.Module):
    def __init__(self, tensor_op):
        super(CustomTensorOP, self).__init__()

        for idx, op in enumerate(tensor_op):
            self.add_module(str(idx), op)

    
    def forward(self, x, idx):
        x = self._modules[str(idx)](x)

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

                    graph[layer_idx].weight.fill_(1)
                    graph[layer_idx].running_var.fill_(1)
                    graph[layer_idx].bias.fill_(0)
                    graph[layer_idx].running_mean.fill_(0)
                    graph[layer_idx].eps = 0

                    graph[layer_idx].register_buffer('fake_weight', bn_weight.clone())
                    graph[layer_idx].register_buffer('fake_bias', bn_bias.clone())
                    # print(graph[layer_idx].running_var)

                    break

    return model