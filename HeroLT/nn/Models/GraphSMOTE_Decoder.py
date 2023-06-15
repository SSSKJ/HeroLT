
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import math

class Decoder(nn.Module):
    """
    Edge Reconstruction adopted in GraphSMOTE (https://arxiv.org/abs/2103.08826)
    """

    def __init__(self, nhid, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.de_weight = Parameter(torch.FloatTensor(nhid, nhid))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1,-2)))

        return adj_out