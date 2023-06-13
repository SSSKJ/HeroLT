import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class CSCE(nn.Module):

    def __init__(self, para_dict=None):
        super(CSCE, self).__init__()
        self.num_class_list = para_dict["num_class_list"]
        self.device = para_dict["device"]

        cfg = para_dict["cfg"]
        scheduler = cfg.LOSS.CSCE.SCHEDULER
        self.step_epoch = cfg.LOSS.CSCE.DRW_EPOCH

        if scheduler == "drw":
            self.betas = [0, 0.999999]
        elif scheduler == "default":
            self.betas = [0.999999, 0.999999]
        self.weight = None

    def update_weight(self, beta):
        effective_num = 1.0 - np.power(beta, self.num_class_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def reset_epoch(self, epoch):
        idx = (epoch-1) // self.step_epoch
        beta = self.betas[idx]
        self.update_weight(beta)

    def forward(self, x, target, **kwargs):
        return F.cross_entropy(x, target, weight= self.weight)