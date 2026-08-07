"""KuRALS-Net-NPU: an anchor-free range-Doppler detector built only from ops
the target NPU can execute (see cfg_gen.py of the NPU toolchain):

  - conv2d, kernel size in {1, 3}, stride in {1, 2}          -> nn.Conv2d
  - depthwise conv2d, kernel size in {1, 3, 5}, stride in {1, 2} (per-layer
    flag, out_ch forced = in_ch) -- 5x5 depthwise confirmed by cfg_gen.py's
    3-bit filter_size register field (0-7 representable) plus a reference-
    compiled workload (efficientnetS_32x32.txt's MBConv blocks use
    "size=5 ... depthwise=1 ... pad=2", matching this file's own (k-1)//2
    padding exactly). Non-depthwise 5x5 has no such reference and is not used
    here -- its MAC cost scales with in_ch*out_ch instead of just out_ch.
                                                                -> nn.Conv2d(groups=in_ch)
  - 2x nearest-neighbor upsample (fused onto the prior conv) -> nn.Upsample(scale_factor=2, mode='nearest')
  - maxpool (fused onto the prior conv)                      -> nn.MaxPool2d(2, 2)
  - eltwise add (fused onto the prior conv, "shortcut")      -> tensor add
  - route/concat (2-source route)                            -> torch.cat
  - activation: leaky / relu / linear only (act_type is a 2-bit field) -> LeakyReLU / ReLU / Identity

Deliberately NOT used, because the NPU register generator has no path for them:
  - dilated convolution (no dilation field anywhere in the NPU cfg compiler)
  - bilinear interpolation / transposed conv (KuRALSNet's ASPP + ConvTranspose2d decoder)
  - 3D convolution (KuRALSNet's Double3DConvBlock) -- multi-frame input is instead
    channel-stacked into the first 2D conv, matching how the reference workloads
    (tiny-yolov2, yolov3, mobilenet-v1, resnet50) all consume a plain 2D tensor.
  - arbitrary padding -- kernel padding is always the implicit "same" pad the
    hardware applies (no pad register field), so only (k - 1) // 2 is used here.

BatchNorm2d is used at train time and is folded into the conv's per-output-channel
scale/bias (matching the NPU's "scale_bias_size = 2*out_ch + 2*out_ch" fields)
before deployment; it is not a separate runtime op.

Because the box annotations in this dataset are a fixed 3x3 window around a single
range-Doppler point (see get_box() in kurals/dataset_process/kuralscw_processing.py),
targets have no meaningful width/height to regress. The head is therefore CenterNet-style
and anchor-free: a per-class center heatmap (decoded with sigmoid off-NPU, exactly like
the excluded 'region'/'yolo' Darknet layers) plus a 2-channel sub-cell (x, y) offset
regression, both produced by a final linear (no-activation) 1x1 conv.

SIZING FOR A 128-MAC ARRAY @ 100MHz, 20Hz TARGET
-------------------------------------------------
Ideal (100% utilization) MAC budget per inference: 128 * 100e6 * 0.050s = 640M MACs.
KuRALS_CW's native resolution is 124x2048 (not a nicer power-of-two size), so an
earlier revision of this model that ran full-resolution 3x3 convs before its first
downsampling stage cost ~5.7 GMAC/inference -- ~9x over budget (~2.2Hz even at ideal
utilization). This version instead downsamples immediately (stride-2 stem conv, cheap
because in_ch = n_frames is tiny) and uses depthwise-separable convs everywhere except
the stem.

Iteratively tuned through 24 revisions (v1-v24), covering channel widths and
receptive field, output stride (v5: /8 -> /4, later /1), Gaussian-splat
target radius, temporal frame-stacking, dual-scale heads, quantization
warmup/freeze timing, loss hyperparameters (focal gamma, offset weight),
weight decay, learning-rate schedules, and gradient clipping. Best recorded
result (v24): Pd 35.0% at FAR 2.26% (threshold 0.05), or Pd 23.6% at FAR
0.16% (threshold 0.5) on KuRALS_CW -- still well below CFAR's 61-88% Pd and
baseline kuralsnet's precision/recall. Every further hyperparameter lever
tried on top of v24 (v25-v32) regressed or collapsed, which is what motivated
abandoning this sparse single-point-supervision approach for the dense
per-pixel segmentation head in kurals/models/kuralsnet_npu_seg.py -- see that
file's module docstring for the result.

QUANTIZATION
------------
Every conv layer here is a QuantConvBNAct (see kurals/models/quant.py), which trains
in plain float32 by default and switches to a QAT path -- simulating the real FPGA
int8 activation pipeline (scale-multiply, truncating shift, signed-int16 bias add,
truncating shift, then activation) -- once KuRALSNetNPU.set_quant_enabled(True) is
called. That's a deliberate two-phase recipe: float first so BN stats/weights settle,
then flip on quantization noise for the remainder of training so the network adapts
to it, rather than having it sprung on a frozen float model post-hoc (PTQ).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from kurals.models.quant import (
    QuantConvBNAct as ConvBNAct,
    QuantDepthwiseSeparableBlock as DepthwiseSeparableBlock,
    QuantResidualDWSeparableBlock as ResidualDWSeparableBlock,
    QuantMBConvBlock as MBConvBlock,
)


class KuRALSNetNPU(nn.Module):
    """Anchor-free range-Doppler detector, entirely NPU-legal ops, sized to
    run at 20Hz on a 128-MAC array @ 100MHz (see module docstring).

    PARAMETERS
    ----------
    n_classes: int
        Total classes including background at index 0 (same convention as
        the rest of this repo, e.g. 4 for KuRALS_CW, 5 for KuRALS_PD).
        The detection head predicts n_classes - 1 foreground heatmap channels.
    n_frames: int
        Number of consecutive frames stacked as input channels (no 3D conv
        on this NPU, so temporal fusion happens via channel concatenation
        at the stem instead of KuRALSNet's Double3DConvBlock).
    dataset_type: str
        Unused here (kept for signature compatibility with the other models
        in kurals/models/__init__.py and the branch in train.py).

    OUTPUT
    ------
    Tensor of shape (B, n_classes - 1 + 2, ceil_align(H) / STRIDE, ceil_align(W) / STRIDE),
    where ceil_align(n) rounds n up to the next multiple of ALIGN. Always compute the
    matching training-target resolution with KuRALSNetNPU.output_hw(h, w) rather
    than a plain h // stride -- see that method's docstring.
        channels [0 : n_classes-1]   -> per-class center-heatmap logits (apply
                                         sigmoid off-NPU, like the excluded
                                         'region'/'yolo' Darknet decode layer)
        channels [n_classes-1 : +2]  -> (x, y) sub-cell offset regression, linear
    """

    STRIDE = 2    # total downsampling from input to head output (v23, see docstring; v12-v20: 1).
    ALIGN = 8     # input H, W are zero-padded up to a multiple of this before the
                  # network runs, so every 2x downsample / 2x upsample pair lines up
                  # exactly. Must equal 2 ** (number of 2x reductions before the deepest
                  # feature map -- stem + down1 + down2 = 3, same depth as every prior
                  # revision, only how each step is *expressed* changed in v13/v14).

    def __init__(self, n_classes, n_frames, dataset_type='KuRALS_CW'):
        super().__init__()
        self.n_classes = n_classes
        self.n_fg_classes = n_classes - 1
        self.n_frames = n_frames
        self.stride = self.STRIDE
        self.align = self.ALIGN

        # --- Encoder: tiny-YOLOv3-style, ONE skip connection instead of v13's three
        # (see module docstring's v14 section for why). stem goes straight from raw
        # input to /2 (stride=2 on its depthwise layer) -- no skip stored at /1, since
        # nothing routes back to it this time. stageA (/2, unfused) is the network's
        # only skip: its raw output is safely readable later by dec_a's route/concat
        # (only *fused* ops, not plain reads, conflict with a tensor having multiple
        # consumers -- see module docstring's v13 section on cfg_gen.py's fusion rule).
        # down1/down2 continue to /8 as plain stride-2 convs with nothing stored --
        # no later consumer needs their pre-downsample values.
        self.stem = DepthwiseSeparableBlock(n_frames, 16, act='leaky', stride=2)  # /1 -> /2 (v17 reverted, see docstring)
        self.stageA = DepthwiseSeparableBlock(16, 32, act='leaky')                # /2 -> Skip (32ch)

        self.down1 = DepthwiseSeparableBlock(32, 64, act='leaky', stride=2)       # /2 -> /4
        self.down2 = DepthwiseSeparableBlock(64, 64, act='leaky', stride=2)       # /4 -> /8

        # --- Bottleneck: unchanged from v9-v13 -- three stacked residual blocks at the
        # network's coarsest, cheapest-to-widen resolution.
        self.bott_c = ResidualDWSeparableBlock(64, act='leaky', dw_kernel_size=5)
        self.bott_d1 = ResidualDWSeparableBlock(64, act='leaky', dw_kernel_size=5)
        self.bott_d2 = ResidualDWSeparableBlock(64, act='leaky', dw_kernel_size=5)  # /8 -> P_deep

        # --- Decoder: three upsample hops (/8->/4->/2->/1, same depth as v13), but only
        # ONE of them (upconv_a, at /4->/2) is followed by a route/concat -- the other
        # two are a fused upsample straight into a plain refine conv, no concat, no
        # scale-forcing. upconv_a's pw is fused with `upsample_after` (single consumer:
        # only dec_a's concat reads it), matching stageA's own scale (concat has no
        # rescale registers on this hardware -- see _out_scale() in forward()).
        self.upconv_b = DepthwiseSeparableBlock(64, 32, act='leaky')   # /8 -> /4 (fused upsample), no concat
        self.refine_b = DepthwiseSeparableBlock(32, 32, act='leaky')   # /4, plain refine (v15 residual variant reverted -- see docstring)

        self.upconv_a = DepthwiseSeparableBlock(32, 32, act='leaky')   # /4 -> /2 (fused upsample)
        # v23 tried widening this 24->32ch to spend v21's freed MAC headroom --
        # regressed (confidence ceiling ~0.16, see module docstring's v23 section).
        # Reverted to v21's exact 24ch.
        self.dec_a = DepthwiseSeparableBlock(32 + 32, 24, act='leaky')  # concat w/ Skip -> /2

        # v21/v23: upconv_f/refine_f (the final /2->/1 hop) removed -- see module
        # docstring's v21 section. head_out now reads dec_a's /2 output directly.

        # head_out is still a real NPU conv layer (its output is DMA'd to DRAM for host
        # readout, not skipped by hardware), so it goes through the same quantized
        # pipeline as everything else. Tried out_percentile=99.9 here (scoping the
        # percentile-calibration fix -- see quant.py's EMAObserver docstring -- to just
        # this terminal layer): training stayed stable, but the output was still
        # quantized to only ~22 distinct sigmoid values with the same gaps plain
        # abs-max's ~19 had, and Pd was worse than v14 at low thresholds -- no real
        # improvement, so reverted to None (plain abs-max, matching v14 exactly). See
        # kuralsnet_npu.json's comments field for the full writeup. v22 tried
        # in_edge_margin=10 here (scoping a *spatial* exclusion instead of a value
        # percentile) at STRIDE=1 -- real Pd gains but no plateau fix and a worse
        # false-alarm rate at 0.05 (see docstring); not carried into v23, which pursued
        # STRIDE=2's already-working plateau fix instead. v24 (this revision) stacks
        # the same in_edge_margin=10 onto v21/v23's STRIDE=2 topology instead --
        # calibration-only, doesn't change dec_a's or head_out's channel shape, so it
        # can't hit the confidence-ceiling failure v18/v23 both hit from perturbing
        # what feeds head_out's input tensor shape. See module docstring's v24 section.
        self.head_out = ConvBNAct(24, self.n_fg_classes + 2, kernel_size=1, stride=1, act='linear',
                                   in_edge_margin=10)  # v24's value; margin sweep (5/10/20, v25/v26) confirmed 10 is best, see docstring

    @staticmethod
    def output_hw(h, w, align=None, stride=None):
        """Head output (H, W) for a raw input (h, w), accounting for the
        alignment padding applied in forward(). Dataset target encoding must
        use this exact formula (not a plain h // stride) to stay consistent,
        since align (8) is coarser than stride (1): padding always rounds up
        to a multiple of 8 regardless of whether h is already a multiple of 1."""
        align = KuRALSNetNPU.ALIGN if align is None else align
        stride = KuRALSNetNPU.STRIDE if stride is None else stride
        pad_h = (-h) % align
        pad_w = (-w) % align
        return (h + pad_h) // stride, (w + pad_w) // stride

    def set_quant_enabled(self, enabled):
        """Toggle the QAT path on every QuantConvBNAct in the model. Call this
        after a float warm-up period (see kurals/learners/model_detect.py's
        quant_warmup_iters) rather than from the start of training."""
        for m in self.modules():
            if isinstance(m, ConvBNAct):
                m.quant_enabled = enabled

    def freeze_observers(self):
        """Stop updating the EMA activation-range observers (call once QAT
        training has stabilized), leaving only the weights to keep training."""
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
            """Read a QuantConvBNAct's calibrated output scale -- used to force the
            second operand of a route/concat onto the first's quantization grid, since
            concat has no rescale registers on this hardware (cfg_gen.py's concat_en
            path only sets memory-addressing fields) -- same constraint as the
            eltwise-add shortcut. Skip (stageA) is always primary (calibrates
            independently); upconv_a's output is forced to match it, mirroring the
            pattern every prior revision used for p_skip/neck_reduce."""
            qmax = 255.0 if layer.act_name == 'relu' else 127.0
            return layer.out_observer.scale(qmax)

        # --- Encoder: single skip (see module docstring's v14 section) ---
        x = self.stem(x)                  # /2, 16ch
        skip = self.stageA(x)             # /2, 32ch (unfused -- also read later by dec_a's concat)

        x = self.down1(skip)              # /4, 64ch
        x = self.down2(x)                 # /8, 64ch

        # --- Bottleneck ---
        x = self.bott_c(x)                # /8, 64ch (eltwise-add residual)
        x = self.bott_d1(x)               # /8, 64ch (eltwise-add residual)
        p_deep = self.bott_d2(x)          # /8, 64ch (eltwise-add residual)

        # --- Decoder: only the /4->/2 hop concats with the skip; the other two hops
        # are a fused upsample straight into a plain refine, no route, no scale-forcing.
        u = self.upconv_b(p_deep)                              # /8, 32ch
        u = F.interpolate(u, scale_factor=2, mode='nearest')   # fused onto upconv_b.pw -- /4, 32ch
        u = self.refine_b(u)                                   # /4, 32ch, plain refine (no concat)

        # upconv_a.pw takes out_scale_override directly (DepthwiseSeparableBlock.forward
        # doesn't pass through kwargs, so dw/pw are called explicitly here).
        u = self.upconv_a.dw(u)
        u = self.upconv_a.pw(u, out_scale_override=_out_scale(self.stageA.pw))  # /4, 32ch
        u = F.interpolate(u, scale_factor=2, mode='nearest')   # fused onto upconv_a.pw -- /2, 32ch
        u = torch.cat((u, skip), dim=1)                        # route/concat, 2-source -> /2, 64ch
        u = self.dec_a(u)                                      # /2, 24ch

        out = self.head_out(u)            # /2, n_fg_classes + 2, linear (v21/v23: final /2->/1 hop removed)
        return out
