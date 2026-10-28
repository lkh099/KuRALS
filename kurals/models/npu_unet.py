"""NPU-legal U-Net: a plain conv U-Net rebuilt from only the ops this NPU can run
(see docs/NPU_CONSTRAINTS.md), used as the resolution-agnostic reference model in
Exp03 (RD resolution ablation).

How it differs from kurals/models/unet.py (milesial U-Net, NOT deployable):
  * bilinear upsample -> nearest 2x fused onto the preceding conv (`upsample_after`)
  * nn.MaxPool2d as a free-standing op -> maxpool(2) fused onto a conv (`maxpool_after`)
  * a route/concat source conv is never also a fused-op conv (single-consumer rule)
  * concat is a 2-source route whose two operands are forced onto one quantization grid
  * every conv is QuantConvBNAct (BN folded, leaky, w_eff=2w+1 QAT path available)
  * no strided convs at all (downsampling is the fused maxpool)

The topology is described once by `layer_specs()` and consumed by both forward() and
kurals/npu_check.py, so the model and its legality check cannot drift apart.

Layout for depth=3 (two pools), base channels c:
  idx  name   conv            fusion
   0   stem   3x3  in->c
   1   e0a    3x3  c->c       route source (skip0)       <- CUTPOINT (first frame-reuse layer)
   2   e0b    3x3  c->c       maxpool_after
   3   e1a    3x3  c->2c      route source (skip1)
   4   e1b    3x3  2c->2c     maxpool_after
   5   e2a    3x3  2c->4c
   6   e2b    3x3  4c->4c
   7   u1     3x3  4c->2c     upsample_after, grid forced to e1a's
   8   d1a    3x3  4c->2c     <- concat(u1, e1a)
   9   u0     3x3  2c->c      upsample_after, grid forced to e0a's
  10   d0a    3x3  2c->c      <- concat(u0, e0a)
  11   d0b    3x3  c->c
  12   head   1x1  c->n_classes, linear
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from kurals.models.quant import QuantConvBNAct


class NPUUNet(nn.Module):
    ALIGN = 4  # two maxpool(2) stages

    def __init__(self, n_classes, n_frames, dataset_type='KuRALS_CW', base_ch=16, width_mult=1.0, act='leaky', fixed_input_scale=None):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.base_ch = max(8, int(round(base_ch * width_mult / 8.0)) * 8)
        self.act = act
        self.align = self.ALIGN
        self.specs = self.layer_specs(n_classes, n_frames, self.base_ch, act)
        self.convs = nn.ModuleDict({s['name']: QuantConvBNAct(s['in_ch'], s['out_ch'], kernel_size=s['kernel'],
                                                              stride=1, act=s['act'])
                                    for s in self.specs})
        self.convs['stem'].fixed_in_scale = fixed_input_scale  # 1.0 under the DSP log-code contract

    @staticmethod
    def layer_specs(n_classes, n_frames, c, act='leaky'):
        def L(name, k, i, o, a=act, **kw):
            d = dict(name=name, kernel=k, stride=1, in_ch=i, out_ch=o, act=a, depthwise=False)
            d.update(kw)
            return d
        return [
            L('stem', 3, n_frames, c),
            L('e0a', 3, c, c, route_target='d0a'),
            L('e0b', 3, c, c, maxpool_after=True),
            L('e1a', 3, c, 2 * c, route_target='d1a'),
            L('e1b', 3, 2 * c, 2 * c, maxpool_after=True),
            L('e2a', 3, 2 * c, 4 * c),
            L('e2b', 3, 4 * c, 4 * c),
            L('u1', 3, 4 * c, 2 * c, upsample_after=True, grid_of='e1a'),
            L('d1a', 3, 4 * c, 2 * c, route_source='u1+e1a'),
            L('u0', 3, 2 * c, c, upsample_after=True, grid_of='e0a'),
            L('d0a', 3, 2 * c, c, route_source='u0+e0a'),
            L('d0b', 3, c, c),
            L('head', 1, c, n_classes, a='linear'),
        ]

    @staticmethod
    def output_hw(h, w):
        return h, w

    def set_quant_enabled(self, enabled):
        for m in self.convs.values():
            m.quant_enabled = enabled

    def freeze_observers(self):
        for m in self.convs.values():
            m.freeze_observers()

    @staticmethod
    def _out_scale(layer):
        qmax = 255.0 if layer.act_name == 'relu' else 127.0
        return layer.out_observer.scale(qmax)

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        pad_h, pad_w = (-h) % self.align, (-w) % self.align
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        skips = {}
        pending = None  # (tensor from the previous upsample conv, skip name)
        for s in self.specs:
            conv = self.convs[s['name']]
            if 'route_source' in s:
                up_name, skip_name = s['route_source'].split('+')
                x = torch.cat((x, skips[skip_name]), dim=1)  # 2-source route
            if 'grid_of' in s:
                # concat has no rescale registers: force this conv's output onto its skip's grid
                x = conv(x, out_scale_override=self._out_scale(self.convs[s['grid_of']]))
            else:
                x = conv(x)
            if 'route_target' in s:
                skips[s['name']] = x
            if s.get('maxpool_after'):
                x = F.max_pool2d(x, 2, 2)
            if s.get('upsample_after'):
                x = F.interpolate(x, scale_factor=2, mode='nearest')
        if pad_h or pad_w:
            x = x[:, :, :h, :w]
        return x
