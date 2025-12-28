import torch
import torch.nn as nn
from .one_hot import one_hot

class CELoss(nn.Module):

    def __init__(self, weight=None, global_weight = 1., reduction='mean', num_classes=4):
        super(CELoss, self).__init__()
        self.global_weight = global_weight
        self.reduction = reduction
        self.num_classes = num_classes
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
            
    def forward(self, input, target):
        loss = self.ce(input, target)
        if self.reduction == 'mean':
            return self.global_weight * loss.mean()
        else:
            return self.global_weight * loss.sum()
    