import torch.nn as nn
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()

    def forward(self, output, target):
        output = output
        loss = F.cross_entropy(output, target)
        return loss