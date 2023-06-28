from ..Layers.GraphConvolution import GraphConvolution

from torch import nn

class Classifier(nn.Module):
    def __init__(self, layer, nhid, nclass, dropout, nhead=1, adj=None):
        super(Classifier, self).__init__()
        if layer == 'gcn':
            self.conv = GraphConvolution(nhid, nhid)
            self.activation = nn.ReLU()
        elif layer == 'gat':
            self.conv = GraphConvolution(nhid, nhid, nhead, dropout)
            self.activation = nn.ELU(True)
        
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)
        self.adj = adj

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj=None, logit=False):
        if adj == None:
            adj = self.adj
        x = self.activation(self.conv(x, adj))
        x = self.dropout(x)
        if logit:
            return x
        x = self.mlp(x)
        
        return x