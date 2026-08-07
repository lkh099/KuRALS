"""KuRALS-Net-NPU-Seg: the same NPU-legal backbone as the retired
KuRALSNetNPU detection head (kurals/legacy/kuralsnet_npu.py), but with the
CenterNet heatmap+offset head replaced by a plain per-pixel segmentation
head -- reproducing baseline kuralsnet's task formulation (see kuralsnet.py)
without its NPU-illegal ops (dilated ASPP convs, ConvTranspose2d decoder,
Conv3d stem -- none of those compile on this hardware).

Backbone (stem through dec_a) is identical to KuRALSNetNPU: same blocks, same
channel widths, /2 intermediate resolution. Only head_out differs: instead of
(n_classes - 1) heatmap channels + 2 offset channels decoded via sigmoid +
CenterNet point-NMS, this outputs n_classes raw per-pixel logits (background
included, index 0) decoded via argmax off-NPU, matching kuralsnet.py's own
output convention so kurals/learners/model.py's Model/Tester and
test_kuralsnet_vs_cfar.py's argmax-based CFAR comparison work unmodified.

Stays at /2 resolution rather than adding a further learned /2->/1 decoder
hop: that variant was tried and reverted after repeated QAT-stability
failures without beating this /2 formulation's own numbers -- this is the
final architecture.

Result (binary any-foreground vs CFAR, KuRALS_CW test set): Prec=71.26%,
Pd=56.06%, FAR=0.00%. UAV-only: Prec=56.14%, Pd=55.56%, still below baseline
kuralsnet's 81.80%/94.83% -- the remaining gap is plausibly resolution-driven,
since this model's /2 output has no offset regression to correct sub-cell
position the way CenterNet's head did.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from kurals.models.quant import (
    QuantConvBNAct as ConvBNAct,
    QuantDepthwiseSeparableBlock as DepthwiseSeparableBlock,
    QuantResidualDWSeparableBlock as ResidualDWSeparableBlock,
)


class KuRALSNetNPUSeg(nn.Module):
    """Per-pixel segmentation variant of KuRALSNetNPU -- same NPU-legal
    backbone through dec_a, plain segmentation head instead of CenterNet's
    heatmap+offset head.

    PARAMETERS
    ----------
    n_classes: int
        Total classes including background at index 0 (4 for KuRALS_CW, 5
        for KuRALS_PD) -- matches kuralsnet.py's own convention, unlike
        KuRALSNetNPU which predicts n_classes - 1 foreground channels only.
    n_frames: int
        Number of consecutive frames stacked as input channels (no 3D conv
        on this NPU -- see module docstring).
    dataset_type: str
        Unused here (kept for signature compatibility with the other models
        in kurals/models/__init__.py and the branch in train.py).
    bottleneck_ch: int
        Channel width for down2/bott_c/bott_d1/bott_d2/upconv_b's input (the
        /8-resolution stage -- cheapest place to add capacity, since it has
        the smallest H*W of any stage). Default 64 reproduces the final
        architecture exactly. Widening this to 96 was tried (see
        kurals/config_files/kuralsnet_npu_seg_wide96.json) and did not beat
        the default -- kept as a tunable parameter for future capacity
        experiments, not because widening is expected to help.

    OUTPUT
    ------
    Tensor of shape (B, n_classes, H, W) -- native /1 resolution, matching
    the dataset's own rd_mask directly. head_out computes at /2 (STRIDE=2)
    and forward() does a bare nearest upsample of the logits to /1 (a learned
    /2->/1 decoder hop was tried and reverted -- see module docstring). Raw
    per-pixel class logits (softmax/argmax off-NPU, like CenterNet's sigmoid
    decode).
    output_hw(h, w) is still a plain identity -- forward() crops back to
    the caller's exact input (h, w) regardless of internal padding.
    """

    STRIDE = 2
    ALIGN = 8

    def __init__(self, n_classes, n_frames, dataset_type='KuRALS_CW', bottleneck_ch=64):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.stride = self.STRIDE
        self.align = self.ALIGN
        bc = bottleneck_ch

        # --- Encoder: identical to KuRALSNetNPU (kuralsnet_npu.py) ---
        self.stem = DepthwiseSeparableBlock(n_frames, 16, act='leaky', stride=2)  # /1 -> /2
        self.stageA = DepthwiseSeparableBlock(16, 32, act='leaky')                # /2 -> Skip (32ch)

        self.down1 = DepthwiseSeparableBlock(32, 64, act='leaky', stride=2)       # /2 -> /4
        self.down2 = DepthwiseSeparableBlock(64, bc, act='leaky', stride=2)       # /4 -> /8

        # --- Bottleneck: identical to KuRALSNetNPU when bottleneck_ch=64 ---
        self.bott_c = ResidualDWSeparableBlock(bc, act='leaky', dw_kernel_size=5)
        self.bott_d1 = ResidualDWSeparableBlock(bc, act='leaky', dw_kernel_size=5)
        self.bott_d2 = ResidualDWSeparableBlock(bc, act='leaky', dw_kernel_size=5)  # /8 -> P_deep

        # --- Decoder: identical to KuRALSNetNPU through dec_a (/2, 24ch) ---
        self.upconv_b = DepthwiseSeparableBlock(bc, 32, act='leaky')   # /8 -> /4 (fused upsample), no concat
        self.refine_b = DepthwiseSeparableBlock(32, 32, act='leaky')   # /4, plain refine

        self.upconv_a = DepthwiseSeparableBlock(32, 32, act='leaky')   # /4 -> /2 (fused upsample)
        self.dec_a = DepthwiseSeparableBlock(32 + 32, 24, act='leaky')  # concat w/ Skip -> /2

        # Segmentation head: n_classes raw logits (background included), plain
        # linear 1x1 conv -- decoded with argmax off-NPU, matching kuralsnet.py's
        # own convention so Model/Tester and test_kuralsnet_vs_cfar.py work
        # unmodified. Reads dec_a's /2 output directly; forward() does a bare
        # post-head_out nearest-upsample to /1 rather than a learned decoder
        # hop -- a real /2->/1 conv hop was tried and reverted after it failed
        # to beat this /2 formulation's own numbers (see module docstring).
        self.head_out = ConvBNAct(24, self.n_classes, kernel_size=1, stride=1, act='linear')

    @staticmethod
    def output_hw(h, w):
        """Final output (H, W) for a raw input (h, w) -- always identical to
        the input: forward() internally pads up to ALIGN and runs the /2
        backbone through dec_a and head_out, then a bare nearest upsample
        to /1, then crops back down to the caller's exact original (h, w),
        so no external caller (unlike KuRALSNetNPU's detection head, whose
        target encoder needs the true /STRIDE output shape) needs to
        account for padding or striding."""
        return h, w

    def set_quant_enabled(self, enabled):
        """Toggle the QAT path on every QuantConvBNAct in the model."""
        for m in self.modules():
            if isinstance(m, ConvBNAct):
                m.quant_enabled = enabled

    def freeze_observers(self):
        """Stop updating the EMA activation-range observers."""
        for m in self.modules():
            if isinstance(m, ConvBNAct):
                m.freeze_observers()

    def forward(self, x):
        # x: (B, n_frames, H, W)
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (-h) % self.align
        pad_w = (-w) % self.align
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # zero-pad right/bottom only

        def _out_scale(layer):
            """See KuRALSNetNPU.forward()'s _out_scale -- forces upconv_a's
            output onto stageA's quantization grid for the route/concat below
            (concat has no rescale registers on this hardware)."""
            qmax = 255.0 if layer.act_name == 'relu' else 127.0
            return layer.out_observer.scale(qmax)

        # --- Encoder ---
        x = self.stem(x)                  # /2, 16ch
        skip = self.stageA(x)             # /2, 32ch (also read by dec_a's concat)

        x = self.down1(skip)              # /4, 64ch
        x = self.down2(x)                 # /8, 64ch

        # --- Bottleneck ---
        x = self.bott_c(x)                # /8, 64ch (eltwise-add residual)
        x = self.bott_d1(x)               # /8, 64ch (eltwise-add residual)
        p_deep = self.bott_d2(x)          # /8, 64ch (eltwise-add residual)

        # --- Decoder ---
        u = self.upconv_b(p_deep)                              # /8, 32ch
        u = F.interpolate(u, scale_factor=2, mode='nearest')   # fused onto upconv_b.pw -- /4, 32ch
        u = self.refine_b(u)                                   # /4, 32ch, plain refine (no concat)

        u = self.upconv_a.dw(u)
        u = self.upconv_a.pw(u, out_scale_override=_out_scale(self.stageA.pw))  # /4, 32ch
        u = F.interpolate(u, scale_factor=2, mode='nearest')   # fused onto upconv_a.pw -- /2, 32ch
        u = torch.cat((u, skip), dim=1)                        # route/concat, 2-source -> /2, 64ch
        u = self.dec_a(u)                                      # /2, 24ch

        out = self.head_out(u)            # /2, n_classes, linear -- argmax decode off-NPU
        out = F.interpolate(out, scale_factor=2, mode='nearest')  # bare upsample of logits, /2 -> /1
        if pad_h or pad_w:
            out = out[:, :, :h, :w]  # crop back to the caller's original (unpadded) H, W
        return out
