import torch
import torch.nn as nn
from .one_hot import one_hot

class FocalLoss(nn.Module):

    def __init__(self, weight=None, global_weight = 1., reduction='mean', gamma=2, eps=1e-7,
                 num_classes=4):
        super(FocalLoss, self).__init__()
        self.global_weight = global_weight
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
        self.num_classes = num_classes
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp

        if self.reduction == 'mean':
            return self.global_weight * loss.mean()
        else:
            return self.global_weight * loss.sum()