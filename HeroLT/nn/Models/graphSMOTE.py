from HeroLT.nn.Models import GNN_Encoder, GNN_Classifier, GraphSMOTE_Decoder
from HeroLT.nn.Samplers import recon_upsample
from HeroLT.utils import adj_mse_loss, normalize_adj, normalize_sym

import torch
from torch import nn
import torch.nn.functional as F

import copy


class graphSMOTE(nn.Module):
    def __init__(self, config, adj):
        super(graphSMOTE, self).__init__()
        self.config = config

        self.encoder = GNN_Encoder(layer=config['layer'], nfeat=config['nfeat'], nhid=config['nhid'], nhead=config['nhead'], dropout=config['dropout'], adj=adj)
        self.classifier = GNN_Classifier(layer=config['layer'], nhid=config['nhid'], nclass=config['nclass'], nhead=config['nhead'], dropout=config['dropout'], adj=adj)

        self.decoder = GraphSMOTE_Decoder(nhid=config['nhid'], dropout=config['dropout'])

    def forward(self, features, adj, labels, idx_train, pretrain=False):
        embed = self.encoder(features)

        ori_num = labels.shape[0]
        embed, labels_new, idx_train_new, adj_up = recon_upsample(embed, labels, idx_train, adj=adj.detach().to_dense(), portion=self.config['up_scale'], im_class_num=self.config['im_class_num'])
        generated_G = self.decoder(embed)  # generate edges

        loss_reconstruction = adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach().to_dense())  # Equation 6

        if pretrain:
            return loss_reconstruction

        adj_new = copy.deepcopy(generated_G.detach())
        threshold = 0.5
        adj_new[adj_new < threshold] = 0.0
        adj_new[adj_new >= threshold] = 1.0

        adj_new = torch.mul(adj_up, adj_new)  ###

        adj_new[:ori_num, :][:, :ori_num] = adj.detach().to_dense()
        adj_new = adj_new.detach() ##

        adj_new[adj_new != 0] = 1
        if self.config['adj_norm_1']:
            adj_new = normalize_adj(adj_new.to_sparse())
            
        elif self.config['adj_norm_2']:
            adj_new = normalize_sym(adj_new.to_sparse())

        output = self.classifier(embed, adj_new)
        loss_nodeclassification = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])  # Equation 11

        return loss_reconstruction, loss_nodeclassification