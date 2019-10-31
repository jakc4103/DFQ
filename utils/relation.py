from collections import OrderedDict
from torch.nn import BatchNorm2d
from PyTransformer.transformers.quantize import QConv2d, ReLUQuant, QuantMeasure
from utils.quantize import QIdentity

class Relation():
    def __init__(self, layer_idx_1, layer_idx_2):
        self.layer_first = layer_idx_1
        self.layer_second = layer_idx_2


    def __repr__(self):
        return '({}, {})'.format(self.layer_first, self.layer_second)


    def get_idxs(self):
        return self.layer_first, self.layer_second


def create_relation(graph, bottoms):
    name_pre = None
    top_pre = None

    relation_dict = OrderedDict()

    for layer_idx in graph:
        # print(type(graph[layer_idx]), layer_idx, top_pre, bottoms[layer_idx])
        if type(graph[layer_idx]) == QConv2d:
            if top_pre in bottoms[layer_idx]:
                rel_tmp = Relation(name_pre, layer_idx)
                relation_dict[top_pre] = rel_tmp

            name_pre = layer_idx
            top_pre = layer_idx

        elif (type(graph[layer_idx]) in [BatchNorm2d, ReLUQuant, QuantMeasure] or (type(graph[layer_idx]) == str and 'F.pad' in layer_idx))\
                and top_pre in bottoms[layer_idx]:
            top_pre = layer_idx

        else:
            name_pre = None

    return relation_dict.values()