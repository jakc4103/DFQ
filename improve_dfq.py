import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.quantize import QuantNConv2d, QuantNLinear, QuantConv2d, QuantLinear, QConv2d, QLinear
from PyTransformer.transformers.torchTransformer import TorchTransformer
from utils.layer_transform import set_quant_minmax, replace_op, restore_op
import numpy as np
import copy
from tensorboardX  import SummaryWriter
import time

class GradHook():
    def __init__(self, weight, scale=None, scale_prev=None, merge_scale=None, merge_scale_prev=None):
        self.weight = weight
        self.scale = scale
        self.scale_prev = scale_prev
        self.merge_scale = merge_scale
        self.merge_scale_prev = merge_scale_prev
        self.update_mask()
    
    def update_mask(self):
        weight = self.weight.detach()
        if self.scale_prev is not None:
            weight = self.merge_scale_prev(weight, self.scale_prev)
        
        if self.scale is not None:
            weight, _ = self.merge_scale(weight, None, self.scale)

        # print("Weight max: {}, min: {}, range: {}".format(weight.max(), weight.min(), weight.max()-weight.min()))

        std = torch.sqrt(torch.var(weight))
        mean = weight.mean()
        # mask = (weight > (mean + 2 * std)).float() + (weight < (mean - 2 * std)).float()
        mask = (weight < (mean + 2 * std)).long() + (weight > (mean - 2 * std)).long()
        weight[mask] = 0
        self.mask = torch.abs(weight) / torch.abs(weight).max()
        
    def get_weight_scaled(self):
        weight = self.weight
        if self.scale_prev is not None:
            # weight = self.merge_scale_prev(weight, torch.clamp(self.scale_prev, max=1))
            weight = self.merge_scale_prev(weight, self.scale_prev)
        
        if self.scale is not None:
            # weight, _ = self.merge_scale(weight, None, torch.clamp(self.scale, max=1))
            weight, _ = self.merge_scale(weight, None, self.scale)
        return weight

    def hook_mask_grad_tensor(self, grad):
        return grad
        # print(grad.shape, self.mask.shape, np.count_nonzero(np.array(grad.shape) != np.array(self.mask.shape)))

        if np.count_nonzero(np.array(grad.shape) != np.array(self.mask.shape)) != 0:
            # mask = self.mask.view(self.mask.size(0), -1).max(-1)[0].view(-1, 1, 1, 1)
            mask = self.mask.view(self.mask.size(0), -1).mean(-1).view(-1, 1, 1, 1)
        else:
            mask = self.mask

        return grad * mask

    def hook_mask_grad_input(self, m, grad_input, grad_output):
        return grad_input
        if type(m) == nn.Linear:
            # mask = self.mask.max(0)[0].view(1, -1)
            mask = self.mask.mean(0).view(1, -1)

            grad_input = list(grad_input)
            grad_input[1] *= mask
            grad_input = tuple(grad_input)
        elif type(m) == nn.Conv2d:
            mask = self.mask.view(self.mask.size(1), -1).mean(-1).view(1, -1, 1, 1)
            # mask = self.mask.view(self.mask.size(1), -1).max(-1)[0].view(1, -1, 1, 1)

            grad_input = list(grad_input)
            grad_input[0] *= mask
            grad_input = tuple(grad_input)
        else:
            raise NotImplementedError

        return grad_input

class ModuleHook(object):
    """
	Forward_hook used to get the output and module object of the intermediate layer. 
	"""
    def __init__(self):
        super(ModuleHook, self).__init__()
        self.input = None
        self.outputs = None
        self.module = None

    def hook(self, module, input, output):
        # if self.module is None:
        self.module = module
        self.inputs = input
        self.outputs = output

    def clear(self):
        self.module = None
        self.inputs = None
        self.outputs = None

def set_scale(res, graph, bottoms, targ_layer):
    # def _find_prev(graph, bottoms, layer_idx):
    #     bot = bottoms[layer_idx]
    #     last_bn = None
    #     prev_list = []
    #     while len(bot) == 1 and "Data" != bot[0]:
    #         if type(graph[bot[0]]) == nn.BatchNorm2d:
    #             last_bn = bot[0]

    #         if type(graph[bot[0]]) in targ_layer:
    #             prev_list.append((bot[0], last_bn))

    #         elif not(type(graph[bot[0]]) in [nn.BatchNorm2d, nn.ReLU] or
    #             (type(graph[bot[0]]) == str and ("F.pad" in bot[0] or "torch.mean" in bot[0]))):
    #             return None, None

    #         bot = bottoms[bot[0]]

    #     return None, None

    layer_first_list = []
    layer_second_list = []
    for rr in res:
        layer_first, layer_second, _ = rr.get_idxs()
        scale = rr.get_scale_vec()

        graph[layer_first].set_scale(scale=torch.ones(graph[layer_first].weight.shape[0]))
        graph[layer_second].set_scale(scale_prev=graph[layer_first].scale)
        layer_first_list.append(layer_first)
        layer_second_list.append(layer_second)

    # res_new = {}
    # for idx in graph:
    #     if type(graph[idx]) in targ_layer:
    #         if idx not in layer_first_list:
    #             res_new[idx] = []

    #         elif idx not in layer_second_list: 
    #             pass


def transform_quant_layer(model, graph, res, trainable=False):
    for rr in res:
        layer_first, layer_second, _ = rr.get_idxs()
        graph[layer_first].merge_scale_to_weight()
        graph[layer_second].merge_scale_to_weight()
        if hasattr(graph[layer_first], 'scale'):
            delattr(graph[layer_first], 'scale')

        if hasattr(graph[layer_first], 'scale_prev'):
            delattr(graph[layer_first], 'scale_prev')
        
        if hasattr(graph[layer_second], 'scale'):
            delattr(graph[layer_second], 'scale')
        
        if hasattr(graph[layer_second], 'scale_prev'):
            delattr(graph[layer_second], 'scale_prev')
        
    transformer = TorchTransformer()
    if trainable:
        transformer.register(QConv2d, QuantConv2d)
        transformer.register(QLinear, QuantLinear)
        
    else:
        transformer.register(QConv2d, QuantNConv2d)
        transformer.register(QLinear, QuantNLinear)

    model = transformer.trans_layers(model, update=True)

    return model

def kl_categorical(p_logit, q_logit, dim=-1):
    """
    https://blog.csdn.net/guotong1988/article/details/90262901
    """
    p = F.softmax(p_logit, dim=dim)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=dim)
                                  - F.log_softmax(q_logit, dim=dim)), dim)
    return torch.mean(_kl)

def norm2(weight, idx, writer, step):
    # print("Weight max: {}, min: {}, range: {}".format(round(float(weight.max()), 3), round(float(weight.min()), 3), round(float(weight.max()-weight.min()), 3)))
    writer.add_scalar("{}/max".format(idx), weight.max().data, step)
    writer.add_scalar("{}/min".format(idx), weight.min().data, step)
    writer.add_scalar("{}/range".format(idx), (weight.max()-weight.min()).data, step)
    mean = weight.mean()
    std = torch.sqrt(torch.var(weight)+1e-8)
    mask = ((weight < (mean - 2 * std)).float() + (weight > (mean + 2 * std)).float()) * (torch.abs(weight) > 2).float()
    return torch.sqrt((torch.abs(weight*mask) ** 2).sum()+1e-8) * (weight.max() - weight.min())

def update_scale(qmodel, model, data_distill, graph, bottoms, res, targ_layer, num_epoch=1000):
    """
    this function use data_distill to find optimized scale for DFQ
    """
    print("Start updating scale")
    writer = SummaryWriter("./tensorboard/exp_{}/".format(round(time.time())))
    qmodel = qmodel.eval().cuda()
    model = model.eval().cuda()
    for idx in range(len(data_distill)):
        data_distill[idx].requires_grad = False

    graph_original = copy.deepcopy(graph)

    optimizer = torch.optim.Adam([p for n, p in qmodel.named_parameters() if 'scale' in n], lr=0.001)
    terminate = False

    # hook params
    hooks = []
    hook_handle = []
    for name, module in qmodel.named_modules():
        if type(module) in targ_layer and hasattr(module, 'scale'):
            # print("Add hook to scale of {} module".format(type(module)))
            grad_hook = GradHook(module.weight, module.scale if hasattr(module, 'scale') else None,
                            module.scale_prev if hasattr(module, 'scale_prev') else None,
                            module.merge_scale if hasattr(module, 'scale') else None,
                            module.merge_scale_prev if hasattr(module, 'scale_prev') else None)
            hooks.append(grad_hook)
            # hook_handle.append(module.weight.register_hook(grad_hook.hook_mask_grad_tensor))
            hook_handle.append(module.scale.register_hook(grad_hook.hook_mask_grad_tensor))
    try:
        """
        TODO: check if graph and model contains same module parameters!!!
        """
        for epoch in range(num_epoch):
            for it in range(len(data_distill)):
                data = data_distill[it].cuda()
                with torch.no_grad():
                    logit = model(data)
                replace_op()
                qlogit = qmodel(data)
                restore_op()
                klloss = kl_categorical(qlogit, logit) #+ kl_categorical(logit, qlogit)
                normloss = 0
                for idx, hook in enumerate(hooks):
                    normloss += norm2(hook.get_weight_scaled(), idx, writer, epoch*len(data_distill)+it+1)
                loss =  klloss
                writer.add_scalar("loss", loss.data, epoch*len(data_distill)+it+1)
                writer.add_scalar("norm", normloss.data, epoch*len(data_distill)+it+1)
                writer.add_scalar("kldiv", klloss.data, epoch*len(data_distill)+it+1)
                print("loss: {}, klloss: {}, norm: {}, iter: {}, epoch: {}".format(loss.data, klloss.data, normloss.data, it+1, epoch+1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for rr in res:
                    layer_first, _, bn_idx = rr.get_idxs()
                    # scale = torch.clamp(graph[layer_first].scale.detach().data.view(-1), max=1)
                    scale = graph[layer_first].scale.detach().data.view(-1)
                    graph[bn_idx].fake_weight.copy_(graph_original[bn_idx].fake_weight * scale)
                    graph[bn_idx].fake_bias.copy_(graph_original[bn_idx].fake_bias * scale)

                set_quant_minmax(graph, bottoms, verbose=False)

                # for hook in hooks:
                #     hook.update_mask()
                #     print("iter: {}, epoch: {}, mean: {}".format(it, epoch, hook.weight.mean()))
                # print("="*150)

                if loss.data < 0.02:
                    terminate = True
                    break

            if terminate:
                break

    except KeyboardInterrupt:
        for rr in res:
            layer_first, _, bn_idx = rr.get_idxs()
            scale = graph[layer_first].scale.detach().data.view(-1)
            graph[bn_idx].fake_weight.copy_(graph_original[bn_idx].fake_weight * scale)
            graph[bn_idx].fake_bias.copy_(graph_original[bn_idx].fake_bias * scale)

    for handle in hook_handle:
            handle.remove()

    return qmodel

def update_quant_range(model, data, graph, bottoms, is_detection=False):
    with torch.no_grad():
        replace_op()
        for batch_data in data:
            batch_data = batch_data.cuda()
            _ = model(batch_data)
        restore_op()
        for idx in graph:
            if bottoms[idx] is None:
                continue
            if bottoms[idx][0] == "Data":
                if not is_detection:
                    graph[idx].quant.running_max.fill_(2.64)
                    graph[idx].quant.running_min.fill_(-2.11790393)
                else:
                    graph[idx].quant.running_max.fill_(1)
                    graph[idx].quant.running_min.fill_(-1)
    return model

def set_update_stat(model, targ_type, update_stat):
    """!
    this function turns on/off the update_stat flag in modules in targ_type
    """
    for module_name in model._modules:			
        # has children
        if len(model._modules[module_name]._modules) > 0 and type(getattr(model, module_name)) not in targ_type:
            set_update_stat(model._modules[module_name], targ_type, update_stat)
        else:
            if type(getattr(model, module_name)) in targ_type:
                getattr(model, module_name).set_update_stat(update_stat)

def bias_correction_distill(qmodel, model_original, data, targ_type, targ_type_original):
    """!
    do bias correction based on distilled data
    """
    qmodel = qmodel.cuda().eval()
    model_original = model_original.cuda().eval()
    hooks = []
    hooks_original = []
    hook_handles = []

    for name, module in qmodel.named_modules():
        if type(module) in targ_type:
            hook = ModuleHook()
            hooks.append(hook)
            hook_handles.append(module.register_forward_hook(hook.hook))

    for name, module in model_original.named_modules():
        if type(module) in targ_type_original:
            hook = ModuleHook()
            hooks_original.append(hook)
            hook_handles.append(module.register_forward_hook(hook.hook))

    error_list = {}
    assert len(hooks) == len(hooks_original), "len of hooks in 2 models must be the same"
    with torch.no_grad():
        for b, batch_data in enumerate(data):

            for hook in hooks:
                hook.clear()
                
            for hook in hooks_original:
                hook.clear()
            batch_data = batch_data.cuda()
            replace_op()
            out = qmodel(batch_data)
            restore_op()

            out = model_original(batch_data)
            for idx in range(len(hooks)):
                # print("Hook {}, error mean: {}, error sum: {}".format(idx, (hooks_original[idx].outputs.mean(0) - hooks[idx].outputs.mean(0)).cpu().mean(), (hooks_original[idx].outputs.mean(0) - hooks[idx].outputs.mean(0)).cpu().sum()))
                if b == 0:
                    error_list[idx] = [hooks[idx].outputs.mean(0).cpu(), hooks_original[idx].outputs.mean(0).cpu()]
                else:
                    error_list[idx][0] += (hooks[idx].outputs.mean(0)).cpu()
                    error_list[idx][1] += (hooks_original[idx].outputs.mean(0)).cpu()

                # error_list[idx].append((hooks[idx].outputs - hooks_original[idx].outputs).cpu())

        for idx, hook in enumerate(hooks):
            module = hook.module
            error = (error_list[idx][0] - error_list[idx][1]) / len(data)
            # print("Hook: {}, error_sum: {}, error_mean: {}".format(idx, error.sum(), error.mean()))
            # for idx_error in range(1, len(error_list[idx])):
                # error += error_list[idx][idx_error]
            error = error.view(error.size(0), -1).sum(-1)
            if not hasattr(module, "bias") or getattr(module, "bias") is None:
                module.bias = torch.nn.Parameter(torch.zeros(error.size(0)), requires_grad=False)
            module.bias.add_(-error.cuda())

    for handle in hook_handles:
        handle.remove()