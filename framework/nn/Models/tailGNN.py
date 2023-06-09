from .TransGCN import TransGCN

import torch.nn as nn
import torch.nn.functional as F

class tailGNN(nn.Module):
    def __init__(self, params, ver=1):
        super(tailGNN, self).__init__()

        self.nhid = params['nhid']
        self.dropout = params['dropout']

        self.rel1 = TransGCN(params['nfeat'], self.nhid, g_sigma=params['g_sigma'], device=params['device'], ver=ver, ablation=params['ablation'])
        self.rel2 = TransGCN(self.nhid, params['nclass'], g_sigma=params['g_sigma'], device=params['device'], ver=ver, ablation=params['ablation'])

    def forward(self, x, adj, head):
        x1, out1 = self.rel1(x, adj, head)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2, out2 = self.rel2(x1, adj, head)

        return x2, F.log_softmax(x2, dim=1), [out1, out2]

    def embed(self, x, adj):
        x1, m1 = self.rel1(x, adj, False)
        x1 = F.elu(x1)
        x2, m2 = self.rel2(x1, adj, False)
        return F.log_softmax(x2, dim=1)