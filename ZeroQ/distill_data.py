#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import os
import json
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from ZeroQ.utils import *
from improve_dfq import GradHook

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = (-1.0 * b.sum(-1)).mean()
        return b

def own_loss(A, B):
    """
	L-2 loss between A and B normalized by length.
	A and B should have the same shape
	"""
    return (A - B).norm()**2 / A.size(0)


class output_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer. 
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None

class InputHook(object):
    """
	Forward_hook used to get the input of the intermediate layer. 
	"""
    def __init__(self):
        super(InputHook, self).__init__()
        self.inputs = None

    def hook(self, module, input, output):
        self.inputs = input

    def clear(self):
        self.inputs = None

def getDistilData(teacher_model,
                  dataset,
                  batch_size,
                  num_batch=1,
                  bn_merged=False,
                  for_inception=False, gpu=True, value_range=[-10, 10], size=[224, 224], max_value=3., early_break_factor=1.):
    """
	Generate distilled data according to the BatchNorm statistics in the pretrained single-precision model.
	Currently only support a single GPU.

	teacher_model: pretrained single-precision model
	dataset: the name of the dataset
	batch_size: the batch size of generated distilled data
	num_batch: the number of batch of generated distilled data
	for_inception: whether the data is for Inception because inception has input size 299 rather than 224
	"""
    print("Start distilling data")
    # initialize distilled data with random noise according to the dataset
    dataloader = getRandomData(dataset=dataset,
                               batch_size=batch_size,
                               for_inception=for_inception, max_value=max_value)

    eps = 1e-6
    # initialize hooks and single-precision model
    hooks, hook_handles, bn_stats, refined_gaussian = [], [], [], []
    # grad_hook_handle = []
    # grad_hooks = []
    if gpu:
        teacher_model = teacher_model.cuda()
    teacher_model = teacher_model.eval()

    # get number of BatchNorm layers in the model
    layers = sum([
        1 if isinstance(layer, nn.BatchNorm2d) else 0
        for layer in teacher_model.modules()
    ])

    for n, m in teacher_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # register hooks on the convolutional layers to get the intermediate input to BatchNorm.
            # hook = output_hook()
            hook = InputHook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
            # register hooks to weight
            # grad_hook = GradHook(m.weight)
            # grad_hooks.append(grad_hook)
            # grad_hook_handle.append(m.weight.register_hook(grad_hook.mask_grad))
            # grad_hook_handle.append(m.register_backward_hook(grad_hook.hook_mask_grad_input))
        # elif isinstance(m, nn.Linear):
        #     grad_hook = GradHook(m.weight)
        #     grad_hooks.append(grad_hook)
            # grad_hook_handle.append(m.weight.register_hook(grad_hook.mask_grad))
            # grad_hook_handle.append(m.register_backward_hook(grad_hook.hook_mask_grad_input))

        if isinstance(m, nn.BatchNorm2d):
            # get the statistics in the BatchNorm layers
            if not bn_merged:
                if gpu:
                    bn_stats.append(
                        (m.running_mean.detach().clone().flatten().cuda(),
                        torch.sqrt(m.running_var +
                                    eps).detach().clone().flatten().cuda()))
                else:
                    bn_stats.append(
                        (m.running_mean.detach().clone().flatten(),
                        torch.sqrt(m.running_var +
                                    eps).detach().clone().flatten()))
            else:
                if gpu:
                    bn_stats.append((m.fake_bias.detach().clone().flatten().cuda(), m.fake_weight.detach().clone().flatten().cuda()))
                else:
                    bn_stats.append((m.fake_bias.detach().clone().flatten(), m.fake_weight.detach().clone().flatten()))

    assert len(hooks) == len(bn_stats)

    for i, gaussian_data in enumerate(dataloader):
        if i == num_batch:
            break
        # initialize the criterion, optimizer, and scheduler
        if gpu:
            gaussian_data = gaussian_data.cuda()
        gaussian_data.requires_grad = True
        # crit = EntropyLoss().cuda()
        optimizer = optim.Adam([gaussian_data], lr=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-7,
                                                         verbose=False,
                                                         patience=100)

        if gpu:
            input_mean = torch.zeros(1, 3).cuda()
            input_std = torch.ones(1, 3).cuda()
        else:
            input_mean = torch.zeros(1, 3)
            input_std = torch.ones(1, 3)

        for it in range(1000):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            output = teacher_model(gaussian_data.clamp(value_range[0], value_range[1]))
            # entropy_loss = crit(output)
            mean_loss = 0
            std_loss = 0

            # compute the loss according to the BatchNorm statistics and the statistics of intermediate output
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                tmp_output = hook.inputs[0]
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),
                                                      tmp_output.size(1), -1),
                                      dim=2)
                if tmp_output.view(tmp_output.size(0), tmp_output.size(1), -1).size(-1) != 1: # to prevent unbiased estimation results to NaN
                    tmp_std = torch.std(tmp_output.view(tmp_output.size(0), tmp_output.size(1), -1) + eps, dim=2) 
                else:
                    tmp_std = torch.std(tmp_output.view(tmp_output.size(1), -1) + eps, dim=1) 

                mean_loss += own_loss(bn_mean, tmp_mean)
                std_loss += own_loss(bn_std, tmp_std)
            tmp_mean = torch.mean(gaussian_data.view(gaussian_data.size(0), 3,
                                                     -1),
                                  dim=2)
            tmp_std = torch.std(gaussian_data.view(gaussian_data.size(0), 3, -1),
                          dim=2) 

            mean_loss += own_loss(tmp_mean, input_mean)
            std_loss += own_loss(tmp_std, input_std)
            total_loss = mean_loss + std_loss# + entropy_loss
            # print("mean: {}, std: {}, it: {}, num: {}, total_layers: {}".format(mean_loss, std_loss, it, i, layers+1))

            # update the distilled data
            total_loss.backward()

            optimizer.step()
            scheduler.step(total_loss.item())

            # early stop to prevent overfitting
            if total_loss <= (layers + 1) * early_break_factor:# and entropy_loss < 0.5:
                # print("Early break with loss: {}, (layer+1)*3: {}".format(total_loss, (layers+1)*3))
                break

        print("{} out of {} distilled.".format(i+1, num_batch))

        refined_gaussian.append(gaussian_data.detach().clone().clamp(value_range[0], value_range[1]))
        gaussian_data.cpu()

    for handle in hook_handles:
        handle.remove()
    # for handle in grad_hook_handle:
    #     handle.remove()
    return refined_gaussian
