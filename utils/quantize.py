"""
Module Source:
https://github.com/eladhoffer/quantized.pytorch
"""

import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class UniformQuantize(InplaceFunction):
    """!
    Perform uniform-quantization on input tensor

    Note that this function seems to behave differently on cpu or gpu.
    This is probably related to the underlaying implmentation difference of pytorch.
    Cpu seems to be more precise (with better accuracy for DFQ-models)
    """
    @staticmethod
    def forward(ctx, input, num_bits=8, min_value=None, max_value=None, inplace=False, symmetric=False, num_chunks=None):
        num_chunks = num_chunks = input.shape[0] if num_chunks is None else num_chunks
        if min_value is None or max_value is None:
            B = input.shape[0]
            y = input.view(B // num_chunks, -1)

        if min_value is None:
            min_value = y.min(-1)[0].mean(-1)  # C
            #min_value = float(input.view(input.size(0), -1).min(-1)[0].mean())

        if max_value is None:
            #max_value = float(input.view(input.size(0), -1).max(-1)[0].mean())
            max_value = y.max(-1)[0].mean(-1)  # C

        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input

        else:
            output = input.clone()

        if symmetric:
            qmin = -2. ** (num_bits - 1)
            qmax = 2 ** (num_bits - 1) - 1
            max_value = abs(max_value)
            min_value = abs(min_value)
            if max_value > min_value:
                scale = max_value / qmax
            else:
                max_value = min_value
                scale = max_value / (qmax + 1)

            min_value = 0.

        else:
            qmin = 0.
            qmax = 2. ** num_bits - 1.

            scale = (max_value - min_value) / (qmax - qmin)
            
        scale = max(scale, 1e-8)
        # test = np.array([scale])
        # print("test", test.dtype)

        output.add_(-min_value).div_(scale)

        output.clamp_(qmin, qmax).round_()  # quantize

        output.mul_(scale).add_(min_value)  # dequantize

        return output


    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None


def quantize(x, num_bits=8, min_value=None, max_value=None, inplace=False, symmetric=False, num_chunks=None):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, inplace, symmetric, num_chunks)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, momentum=0.1):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = momentum
        self.num_bits = num_bits


    def forward(self, input):
        if self.training:
            min_value = input.detach().view(input.size(0), -1).min(-1)[0].mean()
            max_value = input.detach().view(input.size(0), -1).max(-1)[0].mean()
            self.running_min.mul_(1 - self.momentum).add_(min_value * (self.momentum))
            self.running_max.mul_(1 - self.momentum).add_(max_value * (self.momentum))

        else:
            min_value = self.running_min
            max_value = self.running_max

        return quantize(input, self.num_bits, min_value=float(min_value), max_value=float(max_value), num_chunks=16)


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=None):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits


    def forward(self, input):
        qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                           min_value=float(self.weight.min()),
                           max_value=float(self.weight.max()))
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_weight)
        else:
            qbias = None
        
        output = F.conv2d(input, qweight, qbias, self.stride,
                            self.padding, self.dilation, self.groups)

        return output


class QuantConv2d(nn.Conv2d):
    """docstring for QuantConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=None, num_bits_bias=16, momentum=0.1):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_bias = num_bits_bias
        self.quant = QuantMeasure(num_bits=num_bits, momentum=momentum)


    def forward(self, input):
        input = self.quant(input)
        qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                           min_value=float(self.weight.min()),
                           max_value=float(self.weight.max()))
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_bias)
        else:
            qbias = None
        
        output = F.conv2d(input, qweight, qbias, self.stride,
                            self.padding, self.dilation, self.groups)

        return output

class QuantNConv2d(nn.Conv2d):
    """docstring for QuantNConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, momentum=0.1):
        super(QuantNConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.quant = QuantMeasure(num_bits=num_bits, momentum=momentum)


    def forward(self, input):
        input = self.quant(input)

        output = F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        return output

class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=None, num_bits_grad=None, biprecision=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits


    def forward(self, input):
        qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                           min_value=float(self.weight.min()),
                           max_value=float(self.weight.max()))
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_weight)
        else:
            qbias = None

        output = F.linear(input, qweight, qbias)

        return output

class QuantLinear(nn.Linear):
    """docstring for QuantLinear."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=None, num_bits_bias=16, momentum=0.1):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_bias = num_bits_bias
        self.quant = QuantMeasure(num_bits=num_bits, momentum=momentum)


    def forward(self, input):
        input = self.quant(input)
        qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                           min_value=float(self.weight.min()),
                           max_value=float(self.weight.max()))
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_bias)
        else:
            qbias = None

        output = F.linear(input, qweight, qbias)

        return output

class QuantNLinear(nn.Linear):
    """docstring for QuantLinear."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, momentum=0.1):
        super(QuantNLinear, self).__init__(in_features, out_features, bias)
        self.quant = QuantMeasure(num_bits=num_bits, momentum=momentum)


    def forward(self, input):
        input = self.quant(input)

        output = F.linear(input, self.weight, self.bias)

        return output

class ReLUQuant(nn.Module):
    def __init__(self, num_bits=8, momentum=0.1):
        super(ReLUQuant, self).__init__()
        self.quant = QuantMeasure(num_bits=num_bits, momentum=momentum)

    
    def forward(self, x):
        x = torch.clamp(x, min=0)
        x = self.quant(x)

        return x