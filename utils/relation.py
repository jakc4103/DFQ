from collections import OrderedDict
from torch.nn import BatchNorm2d, ReLU, Dropout
from PyTransformer.transformers.quantize import QConv2d, QuantMeasure

class Relation():
    def __init__(self, layer_idx_1, layer_idx_2, bn_idx_1):
        self.layer_first = layer_idx_1
        self.layer_second = layer_idx_2
        self.bn_idx = bn_idx_1
        self.S = None


    def __repr__(self):
        return '({}, {})'.format(self.layer_first, self.layer_second)


    def get_idxs(self):
        return self.layer_first, self.layer_second, self.bn_idx

    def set_scale_vec(self, S):
        if self.S is None:
            self.S = S
        else:
            self.S *= S

    def get_scale_vec(self):
        return self.S


def create_relation(graph, bottoms, targ_type=[QConv2d]):
    name_pre = None
    top_pre = None
    bn_pre = None

    relation_dict = OrderedDict()

    for count, layer_idx in enumerate(graph):
        # print(type(graph[layer_idx]), layer_idx, top_pre, bottoms[layer_idx])
        if bottoms[layer_idx]:
            for bot in bottoms[layer_idx]:
                if bot in relation_dict:
                    relation_dict.pop(bot)
        if type(graph[layer_idx]) in targ_type:
            if top_pre in bottoms[layer_idx]:
                rel_tmp = Relation(name_pre, layer_idx, bn_pre)
                relation_dict[top_pre] = rel_tmp

            name_pre = layer_idx
            top_pre = layer_idx

        elif (type(graph[layer_idx]) in [BatchNorm2d, QuantMeasure, ReLU] or (type(graph[layer_idx]) == str and ('F.pad' in layer_idx or 'torch.mean' in layer_idx)))\
                and top_pre in bottoms[layer_idx]:
            if type(graph[layer_idx]) == BatchNorm2d:
                bn_pre = layer_idx
            top_pre = layer_idx

        else:
            name_pre = None
            bn_pre = None

    # only take the relations with more than 3 targ_layers, ex: Conv2d->Conv2d->Conv2d,, ignore Conv2d->Conv2d (in detection task)
    tmp = list(relation_dict.values())
    res_group = []
    for rr in tmp:
        group_idx = -1
        for idx, group in enumerate(res_group):
            for rr_prev in group:
                if rr.get_idxs()[0] == rr_prev.get_idxs()[1]:
                    group_idx = idx
                    break
        if group_idx != -1:
            res_group[group_idx].append(rr)
        else:
            res_group.append([rr])
    res = []
    for group in res_group:
        if len(group) > 1:
            res.extend(group)

    # print(len(res), len(list(relation_dict.values())))

    return res#list(relation_dict.values())
