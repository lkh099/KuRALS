import torch
import torch.nn as nn
from .one_hot import one_hot

# Noisy Background Suppression Loss with Focal Loss
class NBSFocalLoss(nn.Module):

    def __init__(self, weight=None, global_weight = 1., reduction='mean', threshold=0.05, degree=2, 
                 num_classes=4, gamma=2, eps=1e-7):
        super(NBSFocalLoss, self).__init__()
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
        focalloss = (1 - p) ** self.gamma * neg_logp
        if self.degree == -1:
            recali = torch.zeros_like(p)
        elif self.degree == 1:
            t = self.threshold
            g = self.gamma
            if self.threshold > 0:
                recali = (g*(1-t)**(g-1)*torch.log(torch.tensor(t)) - (1-t)**g/t)*p - \
                    (1-t)**g*torch.log(torch.tensor(t)) - t*g*(1-t)**(g-1)*torch.log(torch.tensor(t)) + \
                        (1-t)**g
            else:
                recali = focalloss
        elif self.degree == 2:
            t = self.threshold
            g = self.gamma
            if self.threshold > 0:
                recali = 1/2/t*(g*(1-t)**(g-1)*torch.log(torch.tensor(t)) - (1-t)**g/t)*p**2 - \
                    (1-t)**g*torch.log(torch.tensor(t)) - t/2*g*(1-t)**(g-1)*torch.log(torch.tensor(t)) + \
                        1/2*(1-t)**g
            else:
                recali = focalloss
        else:
            raise ValueError(f'Arg degree only supports 1 and 2 yet, not {self.degree}')
        loss = torch.where((p < self.threshold) & (target == self.down_index), recali, focalloss)
        
        if self.reduction == 'mean':
            return self.global_weight * loss.mean()
        else:
            return self.global_weight * loss.sum()

if __name__ == '__main__':
    pred = torch.ones(2, 2, 3, 3)
    target = torch.zeros((2, 3, 3), dtype=torch.long)
    target[:, 1, 1] = 1
    nbsfocal_loss = NBSFocalLoss(threshold=0.05)
    loss = nbsfocal_loss(input=pred, target=target)
    print(loss)