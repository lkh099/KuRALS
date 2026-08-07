"""CenterNet-style loss for the retired KuRALSNetNPU (see kurals/legacy/README.md):
a penalty-reduced focal loss on the per-class center heatmap plus a masked L1
loss on the sub-cell offset, gated to only the grid cells that hold a target.

The heatmap target is Gaussian-splatted (see dataloaders_detect.py), not a
hard single pixel: the (1-pos)^beta factor on the negative term is the
"penalty-reduced" part of penalty-reduced focal loss, discounting cells near
a splatted peak. num_pos is computed from `mask` (the hard single true-center
pixel), not by summing the heatmap, since summing a splatted heatmap would
count every splatted neighbor cell too and under-normalize the loss.

`beta` (default 4.0) and `gamma` (default 2.0, set to 3.0 in
kuralsnet_npu.json) are the standard CornerNet/CenterNet penalty-reduced
focal loss exponents. `heatmap_loss_type='softmax_ce'` is an alternative
positive-term loss (per-class spatial softmax cross-entropy instead of
per-cell independent focal loss) that was tried and reverted -- it produced
a wider int8 logit dynamic range as hypothesized, but was less stable to
train and did not improve on the focal-loss baseline's numbers; 'focal'
remains the default.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):

    def __init__(self, heatmap_weight=1.0, offset_weight=1.0, gamma=2.0, beta=4.0,
                 pos_weight=1.0, eps=1e-7, heatmap_loss_type='focal'):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.gamma = gamma
        self.beta = beta
        self.pos_weight = pos_weight
        self.eps = eps
        assert heatmap_loss_type in ('focal', 'softmax_ce')
        self.heatmap_loss_type = heatmap_loss_type

    def forward(self, pred, target):
        """
        pred: tensor (B, n_fg_classes + 2, Hs, Ws), raw logits/linear output
              of KuRALSNetNPU.
        target: dict with 'heatmap' (B, n_fg_classes, Hs, Ws),
                          'offset'  (B, 2, Hs, Ws),
                          'mask'    (B, 1, Hs, Ws)

        RETURNS
        -------
        loss, heatmap_loss, offset_loss: scalar tensors
        """
        n_fg_classes = target['heatmap'].shape[1]
        heatmap_logits = pred[:, :n_fg_classes]
        offset_pred = pred[:, n_fg_classes:]

        prob = torch.sigmoid(heatmap_logits)
        pos = target['heatmap']
        neg_weight = (1.0 - pos).pow(self.beta)
        neg_loss = -(prob.pow(self.gamma)) * torch.log((1.0 - prob).clamp(min=self.eps)) * neg_weight

        if self.heatmap_loss_type == 'focal':
            pos_loss = -((1.0 - prob).pow(self.gamma)) * torch.log(prob.clamp(min=self.eps)) * pos * self.pos_weight
        else:  # 'softmax_ce', see v28 section of this module's docstring
            b, c, h, w = heatmap_logits.shape
            log_prob_spatial = F.log_softmax(heatmap_logits.view(b, c, h * w), dim=-1).view(b, c, h, w)
            pos_loss = -pos * log_prob_spatial * self.pos_weight

        num_pos = target['mask'].sum().clamp(min=1.0)
        heatmap_loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        offset_mask = target['mask']  # (B, 1, Hs, Ws), broadcasts over the 2 offset channels
        offset_loss = (torch.abs(offset_pred - target['offset']) * offset_mask).sum()
        offset_loss = offset_loss / (offset_mask.sum() * 2.0).clamp(min=1.0)

        loss = self.heatmap_weight * heatmap_loss + self.offset_weight * offset_loss
        return loss, heatmap_loss.detach(), offset_loss.detach()
