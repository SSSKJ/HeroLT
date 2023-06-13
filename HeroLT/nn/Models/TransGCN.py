from Layers import GraphConvolution
from Modules import Relation, Relationv2
from TailGNN_Generator import TailGNN_Generator

import torch
from torch import nn

class TransGCN(nn.Module):
    def __init__(self, nfeat, nhid, g_sigma, device, ver, ablation=0):
        super(TransGCN, self).__init__()

        self.device = device
        self.ablation = ablation

        if ver == 1:
            self.r = Relation(nfeat, ablation)
        else:
            self.r = Relationv2(nfeat, nhid, ablation)

        self.g = TailGNN_Generator(nfeat, g_sigma, ablation)
        self.gc = GraphConvolution(nfeat, nhid)

    def forward(self, x, adj, head):

        mean = F.normalize(adj, p=1, dim=1)
        neighbor = torch.mm(mean, x)

        output = self.r(x, neighbor)
        adj = adj + torch.eye(adj.size(0), device=self.device)

        if head or self.ablation == 2:
            norm = F.normalize(adj, p=1, dim=1)
            h_k = self.gc(x, norm)
        else:
            if self.ablation == 1:
                h_s = self.g(output)
            else:
                h_s = output

            h_k = self.gc(x, adj)
            h_s = torch.mm(h_s, self.gc.weight)
            h_k = h_k + h_s

            num_neighbor = torch.sum(adj, dim=1, keepdim=True)
            h_k = h_k / (num_neighbor + 1)

        return h_k, output