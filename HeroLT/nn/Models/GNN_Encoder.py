from ..Modules.GraphAttConv import GraphAttConv
from ..Layers.GraphConvolution import GraphConvolution

from torch import nn


class Encoder(nn.Module):
    def __init__(self, layer, nfeat, nhid, dropout, nhead=1, adj=None):
        super(Encoder, self).__init__()
        if layer == 'gcn':
            self.conv = GraphConvolution(nfeat, nhid)
            self.activation = nn.ReLU()
        elif layer == 'gat':
            self.conv = GraphAttConv(nfeat, nhid, nhead, dropout)
            self.activation = nn.ELU()
        
        self.dropout = nn.Dropout(p=dropout)
        self.adj = adj

    def forward(self, x, adj=None):
        if adj == None:
            adj = self.adj

        x = self.activation(self.conv(x, adj))
        output = self.dropout(x)

        return output