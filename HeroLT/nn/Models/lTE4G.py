from ..Models import GNN_Encoder, GNN_Classifier, GraphSMOTE_Decoder, MLP
from ...utils import adj_mse_loss

import torch
from torch import nn
import torch.nn.functional as F

class lTE4G(nn.Module):
    def __init__(self, config, adj):
        super(lTE4G, self).__init__()
        self.config = config
        self.expert_dict = {}

        self.encoder = GNN_Encoder(layer = config['layer'], nfeat = config['nfeat'], nhid = config['nhid'], nhead = config['nhead'], dropout = config['dropout'], adj = adj)
        if self.config['cls_og'] == 'GNN': # 'Cora', 'CiteSeer'
            self.classifier_og = GNN_Classifier(layer = config['layer'], nhid = config['nhid'], nclass = config['nclass'], nhead = config['nhead'], dropout = config['dropout'], adj = adj)
        elif self.config['cls_og'] == 'MLP': # 'cora_full'
            self.classifier_og = MLP(nhid = config['nhid'], nclass=config['nclass'])
        
        for sep in ['HH', 'H', 'TH', 'T']:
            num_class = config['sep_point'] if sep[0] == 'H' else config['nclass'] - config['sep_point']
            self.expert_dict[sep] = GNN_Classifier(layer = config['layer'], nhid = config['nhid'], nclass = num_class, nhead = config['nhead'], dropout = config['dropout'], adj=adj)
        
        if self.config['rec']:
            self.decoder = GraphSMOTE_Decoder(nhid=config['nhid'], dropout=config['dropout'])

    def forward(self, features, adj=None, labels=None, idx_train=None, classifier=None, embed=None, sep=None, teacher=None, pretrain=False, weight=None, is_og=False, is_expert=False, is_student=False):
        if embed == None:
            embed = self.encoder(features)

        if pretrain:
            generated_G = self.decoder(embed)
            loss_reconstruction = adj_mse_loss(generated_G, adj.detach().to_dense())
            return loss_reconstruction
            
        if is_og:
            output = self.classifier_og(embed)
            if self.config['class_weight']:
                ce_loss = -F.cross_entropy(output[idx_train], labels[idx_train], weight=weight)
                pt = torch.exp(-F.cross_entropy(output[idx_train], labels[idx_train]))
                loss_nodeclassfication = -((1 - pt) ** self.config['gamma']) * ce_loss
            else:
                ce_loss = -F.cross_entropy(output[idx_train], labels[idx_train])
                pt = torch.exp(-F.cross_entropy(output[idx_train], labels[idx_train]))
                loss_nodeclassfication = -((1 - pt) ** self.config['gamma']) * self.config['alpha'] * ce_loss
            
            if self.config['rec']:
                generated_G = self.decoder(embed)
                loss_reconstruction = adj_mse_loss(generated_G, adj.detach().to_dense())
                return loss_nodeclassfication, loss_reconstruction
            else:
                return loss_nodeclassfication
        
        if is_expert:
            pred = classifier(embed)

            if sep in ['T', 'TH', 'TT']:
                labels = labels - self.config['sep_point']

            loss_nodeclassfication = F.cross_entropy(pred[idx_train], labels[idx_train])

            return loss_nodeclassfication

        if is_student:
            # teacher
            teacher_head_degree = teacher[sep+'H']
            teacher_tail_degree = teacher[sep+'T']
            idx_train_head_degree = idx_train[sep+'H']
            idx_train_tail_degree = idx_train[sep+'T']
            idx_train_all = torch.cat((idx_train_head_degree, idx_train_tail_degree), 0)

            teacher_head_degree.eval()
            teacher_tail_degree.eval()
            
            out_head_teacher = teacher_head_degree(embed)[idx_train_head_degree]
            out_tail_teacher = teacher_tail_degree(embed)[idx_train_tail_degree]
                
            # student
            out_head_student = classifier(embed)[idx_train_head_degree]
            out_tail_student = classifier(embed)[idx_train_tail_degree]

            kd_head = F.kl_div(F.log_softmax(out_head_student / self.config['T'], dim=1), F.softmax(out_head_teacher / self.config['T'], dim=1), reduction='mean') * self.config['T'] * self.config['T']
            kd_tail = F.kl_div(F.log_softmax(out_tail_student / self.config['T'], dim=1), F.softmax(out_tail_teacher / self.config['T'], dim=1), reduction='mean') * self.config['T'] * self.config['T']
            
            if sep in ['T', 'TH', 'TT']:
                labels = labels - self.config['sep_point']

            ce_loss = F.cross_entropy(classifier(embed)[idx_train_all], labels[idx_train_all])

            return kd_head, kd_tail, ce_loss