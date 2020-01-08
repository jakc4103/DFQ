import torch
import torch.nn.functional as F
from utils.quantize import QuantNConv2d, QuantNLinear, QuantConv2d, QuantLinear, QConv2d, QLinear
from PyTransformer.transformers.torchTransformer import TorchTransformer

def set_scale(res, graph):
    for rr in res:
        layer_first, layer_second, _ = rr.get_idxs()
        scale = rr.get_scale_vec()

        graph[layer_first].set_scale(scale=scale)
        graph[layer_second].set_scale(scale_prev=graph[layer_first].scale)

def transform_quant_layer(model, graph, res, trainable=False):
    for rr in res:
        layer_first, layer_second, _ = rr.get_idxs()
        graph[layer_first].merge_scale_to_weight()
        graph[layer_second].merge_scale_to_weight()
        
    transformer = TorchTransformer()
    if trainable:
        transformer.register(QConv2d, QuantConv2d)
        transformer.register(QLinear, QuantLinear)
        
    else:
        transformer.register(QConv2d, QuantNConv2d)
        transformer.register(QLinear, QuantNLinear)

    model = transformer.trans_layers(model, True)

    return model

def kl_categorical(p_logit, q_logit, dim=-1):
    """
    https://blog.csdn.net/guotong1988/article/details/90262901
    """
    p = F.softmax(p_logit, dim=dim)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=dim)
                                  - F.log_softmax(q_logit, dim=dim)), 1)
    return torch.mean(_kl)

def update_scale(qmodel, model, data_distill, num_iteration=1000):
    qmodel = qmodel.eval().cuda()
    model = model.eval().cuda()
    data_distill.requires_grad = False
    data_distill = data_distill.cuda()

    optimizer = torch.optim.Adam([p for n, p in qmodel.named_parameters() if 'scale' in n], lr=0.1)

    for it in range(num_iteration):
        with torch.no_grad():
            logit = model(data_distill)

        qlogit = qmodel(data_distill)

        loss = kl_categorical(logit, qlogit)
        print("loss: {}, iter: {}".format(loss.data, it))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return qmodel