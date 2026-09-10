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
    shallow_encoder: bool
        Legacy flag, kept for exact backward compatibility with existing configs/
        checkpoints -- equivalent to encoder_depth=1 (see below) when encoder_depth
        isn't given explicitly.
    encoder_depth: int or None
        How many of {down1, down2} actually stride (each halves resolution): 2 (default)
        = both, full /8 bottleneck; 1 = down1 only, /4 bottleneck (matches
        shallow_encoder=True); 0 = neither, /2 bottleneck -- same resolution as
        stageA's skip connection, for inputs where even /4 is too thin (e.g. the SoC
        8x64 buffer's 8-wide Doppler axis is already down to 2 at /4, 1 at /8). The
        decoder's two upsample hops mirror this count exactly, so dec_a/head_out
        always land back at /2 regardless of this setting.

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

    def __init__(self, n_classes, n_frames, dataset_type='KuRALS_CW', bottleneck_ch=64, shallow_encoder=False,
                 bottleneck_kernel_size=5, dropout_rate=0, encoder_depth=None, stem_stride=None):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        # stem_stride=1 keeps the whole network at native input resolution (head_out
        # predicts per-pixel directly, no final nearest-upsample of the logits). Only
        # worth it for inputs small enough that the /2 head's 2x2-blocky output is a
        # real accuracy cap rather than a cheap win -- e.g. the SoC 8x64 buffer, where
        # a target is a couple of cells wide and 2x2 blocks can't localize it at all.
        self.stem_stride = self.STRIDE if stem_stride is None else stem_stride
        assert self.stem_stride in (1, 2), 'stem_stride must be 1 or 2'
        self.stride = self.stem_stride
        self.align = self.ALIGN
        # encoder_depth generalizes the old shallow_encoder bool into a count of how
        # many of {down1, down2} actually stride (2 = both, the full /8 default;
        # 1 = down1 only, exactly shallow_encoder=True's old /4-bottleneck behavior;
        # 0 = neither -- stops at /2, the same resolution as `skip`/stageA's output,
        # for inputs where even /4 leaves too little spatial extent). If not given
        # explicitly, derived from shallow_encoder so every existing config/checkpoint
        # keeps its exact old meaning.
        self.encoder_depth = (1 if shallow_encoder else 2) if encoder_depth is None else encoder_depth
        assert self.encoder_depth in (0, 1, 2), 'encoder_depth must be 0, 1, or 2'
        self.shallow_encoder = shallow_encoder
        bc = bottleneck_ch
        # Whole-channel dropout on the bottleneck's output -- the most overfitting-prone
        # stage (smallest spatial extent, so fewest independent samples per BatchNorm
        # channel). Identity at eval (net.eval() disables Dropout automatically), so this
        # never appears in the exported/deployed NPU graph -- unlike BatchNorm, no folding
        # needed since it's simply absent at inference.
        self.bott_dropout = nn.Dropout2d(p=dropout_rate)

        # --- Encoder: identical to KuRALSNetNPU (kuralsnet_npu.py) ---
        self.stem = DepthwiseSeparableBlock(n_frames, 16, act='leaky', stride=self.stem_stride)  # /1 -> /2 (or /1 -> /1)
        self.stageA = DepthwiseSeparableBlock(16, 32, act='leaky')                # /2 -> Skip (32ch)

        # encoder_depth=0 also drops down1's stride (/2 -> /2 instead of /2 -> /4) -- for
        # inputs tiny enough that even /4 leaves too little spatial extent (e.g. the SoC
        # 8x64 buffer's 8-wide Doppler axis -> 2 at /4, 1 at /8). Matching decoder upsamples
        # are dropped symmetrically below.
        self.down1 = DepthwiseSeparableBlock(32, 64, act='leaky', stride=2 if self.encoder_depth >= 1 else 1)  # /2 -> /4
        # encoder_depth<=1 drops this stage's stride (/4 -> /4 instead of /4 -> /8) --
        # for inputs already tiny enough that /8 leaves near-zero bottleneck spatial extent
        # (e.g. an 8-wide axis -> 1 at /8). upconv_b's upsample is dropped to match (below).
        self.down2 = DepthwiseSeparableBlock(64, bc, act='leaky', stride=2 if self.encoder_depth >= 2 else 1)

        # --- Bottleneck: identical to KuRALSNetNPU when bottleneck_ch=64, bottleneck_kernel_size=5 ---
        # bottleneck_kernel_size default 5 was tuned for the /8-resolution bottleneck's spatial
        # extent; for tiny inputs where the bottleneck is only a few cells wide (e.g. shallow_encoder
        # or a small SoC RD buffer), a 5-wide depthwise kernel mostly convolves over padding -- pass 3.
        bk = bottleneck_kernel_size
        self.bott_c = ResidualDWSeparableBlock(bc, act='leaky', dw_kernel_size=bk)
        self.bott_d1 = ResidualDWSeparableBlock(bc, act='leaky', dw_kernel_size=bk)
        self.bott_d2 = ResidualDWSeparableBlock(bc, act='leaky', dw_kernel_size=bk)  # /8 (or /4 if shallow_encoder) -> P_deep

        # --- Decoder: identical to KuRALSNetNPU through dec_a (/2, 24ch) ---
        self.upconv_b = DepthwiseSeparableBlock(bc, 32, act='leaky')   # /8 -> /4 (fused upsample), no concat -- plain conv, no upsample, if shallow_encoder
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
        p_deep = self.bott_dropout(p_deep)  # train-only whole-channel dropout, identity at eval

        # --- Decoder --- (upsample count mirrors encoder_depth exactly, so dec_a always
        # lands back at /2 -- same resolution as `skip` -- regardless of bottleneck depth)
        u = self.upconv_b(p_deep)                              # bottleneck resolution, 32ch
        if self.encoder_depth >= 2:
            u = F.interpolate(u, scale_factor=2, mode='nearest')   # fused onto upconv_b.pw -- /8 -> /4
        u = self.refine_b(u)                                   # /4 (or bottleneck res if encoder_depth==0), plain refine (no concat)

        u = self.upconv_a.dw(u)
        u = self.upconv_a.pw(u, out_scale_override=_out_scale(self.stageA.pw))  # same resolution, 32ch
        if self.encoder_depth >= 1:
            u = F.interpolate(u, scale_factor=2, mode='nearest')   # fused onto upconv_a.pw -- /4 -> /2 (or /2 -> /2 no-op skipped when encoder_depth==0)
        u = torch.cat((u, skip), dim=1)                        # route/concat, 2-source -> /2, 64ch
        u = self.dec_a(u)                                      # /2, 24ch

        out = self.head_out(u)            # /stem_stride, n_classes, linear -- argmax decode off-NPU
        if self.stem_stride == 2:
            out = F.interpolate(out, scale_factor=2, mode='nearest')  # bare upsample of logits, /2 -> /1
        if pad_h or pad_w:
            out = out[:, :, :h, :w]  # crop back to the caller's original (unpadded) H, W
        return out
