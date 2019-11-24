from collections import OrderedDict
from torch.nn import BatchNorm2d, ReLU, Dropout
from PyTransformer.transformers.quantize import QConv2d, ReLUQuant, QuantMeasure

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


def create_relation(graph, bottoms, conv_type=QConv2d):
    name_pre = None
    top_pre = None
    bn_pre = None

    relation_dict = OrderedDict()

    for layer_idx in graph:
        # print(type(graph[layer_idx]), layer_idx, top_pre, bottoms[layer_idx])
        if bottoms[layer_idx]:
            for bot in bottoms[layer_idx]:
                if bot in relation_dict:
                    relation_dict.pop(bot)
        if type(graph[layer_idx]) == conv_type:
            if top_pre in bottoms[layer_idx]:
                rel_tmp = Relation(name_pre, layer_idx, bn_pre)
                relation_dict[top_pre] = rel_tmp

            name_pre = layer_idx
            top_pre = layer_idx

        elif (type(graph[layer_idx]) in [BatchNorm2d, ReLUQuant, QuantMeasure, ReLU] or (type(graph[layer_idx]) == str and 'F.pad' in layer_idx))\
                and top_pre in bottoms[layer_idx]:
            if type(graph[layer_idx]) == BatchNorm2d:
                bn_pre = layer_idx
            top_pre = layer_idx

        else:
            name_pre = None
            bn_pre = None

    return list(relation_dict.values())
