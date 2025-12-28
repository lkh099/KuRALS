import torch
import torch.nn as nn
from .one_hot import one_hot

# Noisy Background Suppression Loss
class NBSLoss(nn.Module):

    def __init__(self, weight=None, global_weight = 1., reduction='mean', threshold=0.05, degree=2, 
                 num_classes=4, gamma=1, eps=1e-7):
        super(NBSLoss, self).__init__()
        self.global_weight = global_weight
        self.reduction = reduction
        self.threshold = threshold
        self.degree = degree
        self.num_classes = num_classes
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, input, target):
        """
        Args:
            Input: :math:`(N, C, H, W)` where C = number of classes.
            Target: :math:`(N, H, W)` where each value is :math:`0 ≤ targets[i] ≤ C-1`.
        """
        neg_logp = self.ce(input, target)
        p = torch.exp(-neg_logp)
        if self.degree == -1:
            recali = torch.zeros_like(p)
        # NBS1 loss
        elif self.degree == 1:
            recali = - p / (self.threshold + self.eps) + 1.0 - torch.log(torch.tensor(self.threshold + self.eps))
        # NBS2 loss
        elif self.degree == 2:
            recali = - p**2 / (2*self.threshold**2 + self.eps) + 0.5 - torch.log(torch.tensor(self.threshold + self.eps))
        else:
            raise ValueError(f'Arg degree only supports 1 and 2 yet, not {self.degree}')
        loss = torch.where((p < self.threshold) & (target == 0), recali, neg_logp)
        
        if self.reduction == 'mean':
            return self.global_weight * loss.mean()
        else:
            return self.global_weight * loss.sum()
        
    def thre_step(self, epoch, max_epoch):
        self.threshold = (epoch / (max_epoch - 1)) ** self.gamma

if __name__ == '__main__':
    pred = torch.ones(2, 2, 3, 3)
    target = torch.zeros((2, 3, 3), dtype=torch.long)
    target[:, 1, 1] = 1
    nbsloss = NBSLoss(threshold=0.05)
    loss = nbsloss(input=pred, target=target)
    print(loss)