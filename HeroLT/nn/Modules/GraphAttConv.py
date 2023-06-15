from ..Layers import GraphAttConvOneHead

import torch
from torch import nn

class GraphAttConv(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout):
        super(GraphAttConv, self).__init__()
        assert out_features % heads == 0
        out_perhead = out_features // heads

        self.graph_atts = nn.ModuleList([GraphAttConvOneHead(
               in_features, out_perhead, dropout=dropout) for _ in range(heads)])

        self.in_features = in_features
        self.out_perhead = out_perhead
        self.heads = heads

    def forward(self, input, adj):
        output = torch.cat([att(input, adj) for att in self.graph_atts], dim=1)
        # notice that original GAT use elu as activation func. 
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->[{}x{}])".format(
                    self.in_features, self.heads, self.out_perhead)