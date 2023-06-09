import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch_sparse
from torch_scatter import scatter_max, scatter_add

class GraphAttConvOneHead(nn.Module):
    """
    Sparse version GAT layer, single head
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttConvOneHead, self).__init__()
        self.weight = Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = Parameter(torch.zeros(size=(1, 2*out_features)))
        # init 
        nn.init.xavier_normal_(self.weight.data, gain=nn.init.calculate_gain('relu')) # look at here
        nn.init.xavier_normal_(self.a.data, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
         
    def forward(self, input, adj):
        edge = adj._indices()
        h = torch.mm(input, self.weight)
        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t() # edge_h: 2*D x E
        # do softmax for each row, this need index of each row, and for each row do softmax over it
        alpha = self.leakyrelu(self.a.mm(edge_h).squeeze()) # E
        n = len(input)
        alpha = self.softmax(alpha, edge[0], n)
        output = torch_sparse.spmm(edge, self.dropout(alpha), n, n, h) # h_prime: N x out
        # output = torch_sparse.spmm(edge, self.dropout(alpha), n, n, self.dropout(h)) # h_prime: N x out
        return output

    def softmax(self, src, index, num_nodes=None):
        """
        sparse softmax
        """
        num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
        out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
        out = out.exp()
        out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
        return out