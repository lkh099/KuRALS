"""Quantization-aware training (QAT) primitives that simulate the exact FPGA
activation pipeline (as specified for this NPU):

    acc = conv(x_int8, w_int8)                 # integer accumulator
    t1  = acc * scale[c]                       # scale[c]: signed int16, per out-channel
    t2  = t1 >> shift1                          # shift1 == "bias_shift" register field, in [5, 20], truncating (floor), no rounding
    t3  = t2 + bias[c]                          # bias[c]: signed int16, per out-channel
    t4  = t3 >> shift2                          # shift2 == "act_shift" register field, in [0, 7], truncating (floor), no rounding
    out = activation(t4)                        # leaky / relu / linear
    next_x = clamp_int8_or_uint8(out)            # signed unless activation == 'relu'

No zero-point / asymmetric offset exists in this register format (no such field
for either weights or activations), so quantization is symmetric throughout:
signed int8 weights, int8/uint8 activations, no bias correction term.

Both shift1 and shift2 are single values *per layer* (not per channel) -- the
hardware register only carries one bias_shift/act_shift per convolutional
layer, while scale/bias vary per output channel. All the per-channel weight
scale variation is therefore absorbed into scale[c] (with a shift shared
across channels); bias[c] shares the layer's single bias-domain scale.

Scale/shift/bias here are treated as non-differentiable calibration constants
(derived each forward pass from the current weights and from EMA-tracked
activation ranges), matching standard observer-based QAT. Only the rounding/
flooring/clamping steps use a straight-through estimator (STE) so gradients
still reach the underlying float weights.

Exactness note: this simulates the pipeline in float (not literal int32/int64
arithmetic), which is standard practice for QAT -- it needs to reproduce the
same *rounding convention* (truncating shift, no zero-point) so trained
weights become robust to it, not bit-exact hardware replication.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def ste_round(x):
    """Forward: round(x). Backward: identity (straight-through estimator)."""
    return x + (torch.round(x) - x).detach()


def ste_floor(x):
    """Forward: floor(x) (truncating, matches an arithmetic right shift on
    two's complement integers -- rounds toward -inf, not to nearest).
    Backward: identity (STE)."""
    return x + (torch.floor(x) - x).detach()


def ste_clamp(x, lo, hi):
    """Forward: clamp(x, lo, hi). Backward: identity (STE)."""
    return x + (torch.clamp(x, lo, hi) - x).detach()


class EMAObserver(nn.Module):
    """Tracks a running symmetric range (EMA of abs-max) for per-tensor
    activation quantization. Weights don't need this -- their full range is
    available fresh every forward pass, no streaming statistic required.

    Tried and reverted network-wide: a 99.9th-percentile version of this
    observer (see kuralsnet_npu.json's comments field for the full writeup),
    meant to fix a real problem -- raw abs-max lets one rare outlier
    activation set the scale for a whole tensor, and a trained checkpoint's
    head_out output was measured quantized to only ~19-24 distinct sigmoid
    values across a whole eval pass, with large gaps straddling the 0.3-0.7
    decision region. Applying percentile calibration to *every* observer in
    the network caused a full training collapse (every output cell 0.0
    across the entire test set) -- checkpoint inspection showed
    running_absmax growing monotonically with depth, from ~0.7 at the stem to
    33,421 at upconv_a, a ~10,000x blowup from where a BN-normalized
    leaky-relu network's activations should sit. Root cause: raw abs-max
    reacts to *any* single large activation and instantly widens the scale
    for the whole tensor -- a negative feedback loop that keeps magnitudes
    bounded, load-bearing at every layer whose output feeds a later one.
    Percentile deliberately ignores its top tail, so nothing pushed back once
    training (focal-BCE's asymptotic pressure to push background logits
    arbitrarily negative is a plausible driver here, given ~1 positive target
    per sample) grew a thin slice of activations -- and that slice's
    magnitude compounded through every later layer.

    `percentile` (default None, meaning plain abs-max) exists to apply that
    same fix in one place only: head_out, the network's terminal layer. Nothing
    reads head_out's output *as an activation* -- it's DMA'd straight to the
    host and sigmoid-decoded off-chip -- so a percentile-driven scale here
    can't feed a runaway loop into deeper layers the way a mid-network one
    did. Pass percentile=None everywhere it isn't explicitly wanted.

    `edge_margin` (default None, meaning no spatial exclusion) is a different
    fix for a related but distinct problem, added for the retired
    kuralsnet_npu.py's decoder head (see kurals/legacy/README.md):
    percentile calibration assumes the problem is a *thin rare-value* tail
    (which quantile-based rejection can ignore); a live investigation instead
    found head_out's actual input is elevated ~1.7-1.9x edge-vs-interior at
    the doppler/range boundary, present in 80%+ of the whole grid across
    frames -- a broad, spatially-located shift, not a rare outlier, which is
    exactly why the earlier out_percentile experiment (see above) barely
    helped. `edge_margin` excludes the first/last `edge_margin` columns
    (dim=-1, the doppler axis) from the abs-max computation ONLY -- those
    columns are still quantized and passed through using whatever scale
    results, just not allowed to influence what that scale *is*. Deliberately
    doppler-only, not range: a ground-truth check (2,586 targets across
    Train/Val/Test) found 0.04% of real targets fall in the doppler boundary
    (safe to exclude) versus 35.2% in the range boundary (excluding it would
    risk clipping genuine range-boundary target activations down to the same
    ceiling as the padding artifact, since the calibration would never see a
    real large activation from that region to widen the scale for). Scoped to
    head_out's in_observer only, same reasoning as `percentile` above."""

    # torch.quantile sorts its whole input and refuses more than this many
    # elements (a hard PyTorch limit) -- irrelevant for head_out's small
    # (n_fg_classes+2)-channel output, but subsampling first keeps this
    # correct if ever pointed at a bigger tensor.
    _QUANTILE_MAX_ELEMENTS = 2_000_000

    def __init__(self, momentum=0.9, percentile=None, edge_margin=None):
        super().__init__()
        self.momentum = momentum
        self.percentile = percentile
        self.edge_margin = edge_margin
        self.register_buffer('running_absmax', torch.tensor(0.0))
        self.register_buffer('initialized', torch.tensor(False))
        self.frozen = False

    @torch.no_grad()
    def observe(self, x):
        if self.frozen:
            return
        if self.edge_margin is not None:
            m = self.edge_margin
            x = x[..., m:-m]
        if self.percentile is None:
            cur = x.detach().abs().amax()
        else:
            flat = x.detach().abs().flatten().float()
            if flat.numel() > self._QUANTILE_MAX_ELEMENTS:
                idx = torch.randint(0, flat.numel(), (self._QUANTILE_MAX_ELEMENTS,), device=flat.device)
                flat = flat[idx]
            cur = torch.quantile(flat, self.percentile / 100.0)
        if not bool(self.initialized):
            self.running_absmax.copy_(cur)
            self.initialized.fill_(True)
        else:
            self.running_absmax.mul_(self.momentum).add_(cur * (1.0 - self.momentum))

    def scale(self, qmax):
        return self.running_absmax.clamp(min=1e-8) / qmax

    def freeze(self):
        self.frozen = True


def pick_shift_and_int16(multiplier, shift_lo, shift_hi):
    """Given a desired real-valued multiplier (scalar or per-channel tensor),
    find the largest shift in [shift_lo, shift_hi] (i.e. most precision) such
    that round(multiplier * 2**shift) still fits in signed int16 for every
    element, then return (shift, quantized_int16_values). Falls back to the
    smallest shift (clamping overflow) if even that doesn't fit -- matches a
    hardware saturating store, just with reduced precision rather than a crash.
    """
    with torch.no_grad():
        max_abs = multiplier.detach().abs().amax().item()
        chosen = shift_lo
        for shift in range(shift_hi, shift_lo - 1, -1):
            if max_abs * (2.0 ** shift) <= 32767.0:
                chosen = shift
                break
    int_vals = ste_clamp(ste_round(multiplier * (2.0 ** chosen)), -32768, 32767)
    return chosen, int_vals


_ACTIVATIONS = {
    'leaky': lambda x: F.leaky_relu(x, 0.125),
    'relu': lambda x: F.relu(x),
    'linear': lambda x: x,
}


class QuantConvBNAct(nn.Module):
    """conv2d + BN + activation, with an alternate quantized forward path that
    simulates the real FPGA pipeline documented in this module's docstring.

    Behaves exactly like the previous plain-float ConvBNAct when
    `quant_enabled` is False (the default) -- flip it on with
    KuRALSNetNPU.set_quant_enabled(True) once training has warmed up.
    """

    # Hardware register field constraints (cfg_gen.py's NPU spec). Named asserts
    # below are a deliberate tripwire: editing either tuple without also updating
    # its assert fails loudly at import time, instead of silently drifting.
    # NAMING, verified against the RTL (batch_norm_quant_act.sv), because it is the
    # reverse of what you would guess: the chain is var_shifter(bias_shift) -> +bias ->
    # var_shifter1(act_shift), so the shift applied BEFORE the bias add is the one the
    # register map calls "bias_shift" (= shift1 here), and the one applied AFTER it is
    # "act_shift" (= shift2). This file previously had the two labels swapped, which is
    # how kurals/manifest_to_cfg_input.py came to emit them the wrong way round -- every
    # layer's output saturated to -128 on real hardware before that was caught. The
    # reference input/tiny_yolov2 cfg corroborates the correct mapping: bias_shift=19
    # (in shift1's [5, 20] range), act_shift=4 (in shift2's [0, 7] range).
    SHIFT1_RANGE = (5, 20)   # "bias_shift" register field: scale-multiply shift, pre-bias-add
    SHIFT2_RANGE = (0, 7)    # "act_shift" register field: post-bias-add shift
    assert SHIFT1_RANGE == (5, 20), "bias_shift legal range is (5, 20) per NPU spec -- do not widen without re-verifying against cfg_gen.py"
    assert SHIFT2_RANGE == (0, 7), "act_shift legal range is (0, 7) per NPU spec -- do not widen without re-verifying against cfg_gen.py"

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, act='leaky', depthwise=False,
                 out_percentile=None, in_edge_margin=None):
        super().__init__()
        # reg[2]'s filter_size field is 3 bits (cfg_gen.py: out_ch<<18 | in_ch<<5 |
        # maxpool_stride<<4 | stride<<3 | filter_size), so 0-7 is representable in
        # hardware. 5x5 depthwise is a confirmed-legal, reference-compiled case (see
        # e.g. input/efficientnetS_32x32.txt's MBConv blocks: "size=5 ... depthwise=1
        # ... pad=2", matching this layer's own (k-1)//2 padding exactly). A 5x5
        # *non-depthwise* conv is representable in the same register field but has no
        # confirmed reference workload, and costs in_ch*out_ch (not just out_ch) more
        # MACs per tap, so it's excluded here pending separate verification.
        assert kernel_size in (1, 3) or (kernel_size == 5 and depthwise), \
            "NPU conv reg field only confirmed for 1x1 / 3x3 (any) and 5x5 (depthwise only)"
        assert stride in (1, 2), "NPU conv stride register is a single bit: 1 or 2 only"
        if depthwise:
            assert out_ch == in_ch, "NPU depthwise layers force out_ch == in_ch"
            groups = in_ch
        else:
            groups = 1
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.groups = groups
        self.act_name = act

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=self.padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = _ACTIVATIONS[act]

        self.in_observer = EMAObserver(edge_margin=in_edge_margin)
        self.out_observer = EMAObserver(percentile=out_percentile)
        self.quant_enabled = False

    def freeze_observers(self):
        self.in_observer.freeze()
        self.out_observer.freeze()

    def _fold_bn(self):
        """Standard BN-folding: conv(x)*gamma/std + (beta - mean*gamma/std).
        Matches the hardware's per-layer (scale, bias) fields exactly -- every
        NPU conv layer has this affine built in, whether or not it was called
        BatchNorm during training."""
        gamma = self.bn.weight
        beta = self.bn.bias
        mean = self.bn.running_mean
        std = torch.sqrt(self.bn.running_var + self.bn.eps)
        w_factor = gamma / std
        w_folded = self.conv.weight * w_factor.view(-1, 1, 1, 1)
        b_folded = beta - mean * w_factor
        return w_folded, b_folded

    @torch.no_grad()
    def export_int8(self, out_scale_override=None):
        """Finalize this layer's real hardware parameters from the current
        (should already be frozen -- see freeze_observers()) observer
        statistics. Call only in eval() mode after training.

        Sign note: this simulation always fake-quantizes activations to
        *signed* int8 ([-128, 127]), even where real hardware could use the
        extra bit available when the producing layer was ReLU (unsigned
        uint8 [0, 255]). Exporting unsigned_in=False for every layer keeps
        the export numerically consistent with what was actually trained;
        exploiting the unsigned input range would need input-sign-aware
        calibration this simulation doesn't implement -- a known, deliberate
        conservatism (costs a small amount of precision, not correctness).

        RETURNS
        -------
        dict of exported per-layer values: weight_int8 (out_ch, in_ch/groups,
        k, k), scale_int16 (out_ch,), bias_int16 (out_ch,), shift1, shift2
        (python ints), in_scale, out_scale (python floats, float32-precision --
        the dequantization scales; there is no zero-point/offset anywhere in
        this format, quantization is symmetric throughout, so "zero" is always
        0.0), unsigned_in, unsigned_out, depthwise, groups, kernel_size,
        stride, in_ch, out_ch.
        """
        w_folded, b_folded = self._fold_bn()
        w_absmax = w_folded.abs().amax(dim=(1, 2, 3)).clamp(min=1e-8)
        # Hardware weight datapath: the stored int8 byte w_stored is transformed to
        # w_eff = 2*w_stored + 1 before it's ever multiplied against an activation --
        # see forward()'s matching comment for the full derivation. Effective range is
        # therefore the odd integers in [-255, 255] (never zero), not [-127, 127], so
        # w_scale is calibrated against 255 (the achievable |w_eff| ceiling), not 127.
        w_scale = w_absmax / 255.0
        w_stored = torch.clamp(torch.round((w_folded / w_scale.view(-1, 1, 1, 1) - 1.0) / 2.0), -128, 127)
        w_int8 = w_stored.to(torch.int8)

        in_scale = self.in_observer.scale(127.0)
        unsigned_out = (self.act_name == 'relu')
        out_qmax = 255.0 if unsigned_out else 127.0
        out_scale = out_scale_override if out_scale_override is not None else self.out_observer.scale(out_qmax)

        shift2, bias_int = pick_shift_and_int16(b_folded / out_scale, *self.SHIFT2_RANGE)
        # s2 is the bias-domain scale: t2/t3 (after shift1, before shift2) are
        # quantized at s2 such that a further /2**shift2 lands exactly on out_scale,
        # i.e. s2 * 2**shift2 == out_scale => s2 = out_scale / 2**shift2 (NOT
        # out_scale * 2**shift2 -- that sign error corrupts multiplier1 below and
        # was caught by comparing this simulation against kurals/int8_inference.py's
        # independent pure-integer replay on real data).
        s2 = out_scale / (2.0 ** shift2)
        multiplier1 = (in_scale * w_scale) / s2
        shift1, scale_int = pick_shift_and_int16(multiplier1, *self.SHIFT1_RANGE)

        return {
            'weight_int8': w_int8.cpu().numpy(),
            'scale_int16': scale_int.round().to(torch.int16).cpu().numpy(),
            'bias_int16': bias_int.round().to(torch.int16).cpu().numpy(),
            'shift1': int(shift1),
            'shift2': int(shift2),
            'in_scale': float(in_scale),
            'out_scale': float(out_scale),
            'act_name': self.act_name,
            'unsigned_in': False,
            'unsigned_out': bool(unsigned_out),
            'depthwise': bool(self.groups > 1),
            'groups': int(self.groups),
            'kernel_size': int(self.conv.kernel_size[0]),
            'stride': int(self.stride),
            'in_ch': int(self.conv.in_channels),
            'out_ch': int(self.conv.out_channels),
        }

    def forward(self, x, out_scale_override=None):
        if not self.quant_enabled:
            return self.act(self.bn(self.conv(x)))

        unsigned_out = (self.act_name == 'relu')
        out_qmin, out_qmax = (0.0, 255.0) if unsigned_out else (-128.0, 127.0)

        # ---- input activation: fake-quantize to signed int8 ----
        # (the NPU's in_act_unsign is inherited from the *previous* layer's
        # output sign; since every QuantConvBNAct quantizes its own output to
        # the correct sign already, treating this layer's input range as
        # already representing that quantized tensor is consistent.)
        self.in_observer.observe(x)
        x_scale = self.in_observer.scale(127.0)
        x_q = ste_clamp(ste_round(x / x_scale), -128.0, 127.0)

        # ---- weights: fake-quantize to signed int8, per output channel. Hardware
        # datapath: the stored int8 byte w_stored is NOT what gets multiplied against
        # the input -- it's transformed to w_eff = 2*w_stored + 1 first (always odd,
        # never zero), so the effective weight range is [-255, 255], not [-127, 127].
        # w_scale is calibrated against that 255 ceiling; w_stored is solved for by
        # inverting the transform (w_folded/w_scale - 1)/2, then rounded/clamped to
        # the full signed-int8 range [-128, 127] (not [-127, 127] -- the old symmetric
        # weight range was specific to the old w_eff == w_stored convention). ----
        w_folded, b_folded = self._fold_bn()
        w_absmax = w_folded.detach().abs().amax(dim=(1, 2, 3)).clamp(min=1e-8)
        w_scale = w_absmax / 255.0
        w_stored = ste_clamp(ste_round((w_folded / w_scale.view(-1, 1, 1, 1) - 1.0) / 2.0), -128.0, 127.0)
        w_q = 2.0 * w_stored + 1.0

        # ---- integer accumulator (simulated in float, exact) ----
        acc = F.conv2d(x_q, w_q, stride=self.stride, padding=self.padding, groups=self.groups)

        # ---- differentiable reference: a standard (non-hardware-specific) fake-quant
        # conv, i.e. dequantize acc straight back into float units and add bias, no
        # truncating-shift split. This is what carries the gradient -- see note below. ----
        soft_out = self.act(acc * (x_scale * w_scale).view(1, -1, 1, 1) + b_folded.view(1, -1, 1, 1))

        # ---- calibrate the output scale from that reference (or take the caller's
        # override, used by the eltwise-add residual block so both operands share
        # one scale) ----
        self.out_observer.observe(soft_out)
        out_scale = out_scale_override if out_scale_override is not None else self.out_observer.scale(out_qmax)

        # ---- exact hardware pipeline: acc*scale[c] >> shift1, + bias[c], >> shift2.
        # Deliberately no_grad: differentiating *through* the two truncating shifts
        # directly (as an earlier version of this code did) means the backward pass
        # multiplies by 2**-shift1 * 2**-shift2 at every single layer -- with shift1
        # up to 25, that compounds across depth and underflows gradients to exactly
        # zero by the time they reach the stem. Standard fake-quant avoids this by
        # keeping the backward pass in the original float units (soft_out above)
        # rather than the truncated integer-domain ones; only the forward *value*
        # needs to be hardware-exact. ----
        with torch.no_grad():
            shift2, bias_int = pick_shift_and_int16(b_folded / out_scale, *self.SHIFT2_RANGE)
            # s2 = out_scale / 2**shift2, not out_scale * 2**shift2 -- see the matching
            # comment in export_int8() for the full derivation of why.
            s2 = out_scale / (2.0 ** shift2)
            multiplier1 = (x_scale * w_scale) / s2
            shift1, scale_int = pick_shift_and_int16(multiplier1, *self.SHIFT1_RANGE)

            t1 = acc * scale_int.view(1, -1, 1, 1)
            t2 = torch.floor(t1 / (2.0 ** shift1))
            t3 = t2 + bias_int.view(1, -1, 1, 1)
            t4 = torch.floor(t3 / (2.0 ** shift2))
            # activation is positively-homogeneous (leaky/relu/linear all satisfy
            # f(a*x) = a*f(x) for a > 0), so applying it in the integer domain
            # (as hardware does) or after dequantizing by out_scale gives the same
            # result. Dequantize by out_scale here, NOT s2 (t4 is at out_scale
            # resolution by construction -- s2 is only the intermediate bias-domain
            # scale used to derive scale_int/shift1 above).
            hard_out = self.act(t4).clamp(out_qmin, out_qmax) * out_scale

        # forward value = exact hardware result; backward gradient = soft_out's
        # well-behaved one (this is the same STE construction as ste_round/ste_floor,
        # just applied around the whole hardware pipeline instead of one op).
        return soft_out + (hard_out - soft_out).detach()


class QuantDepthwiseSeparableBlock(nn.Module):
    """MobileNet-style separable conv: depthwise 'convolutional' layer
    followed by a pointwise (1x1) 'convolutional' layer -- two distinct NPU
    layers, matching how mobilenet_v1_ssd.txt is compiled.

    dw_kernel_size: 3 (default, matches mobilenet_v1_ssd.txt) or 5 (matches
    efficientnetS_32x32.txt's MBConv blocks) -- both reference-compiled.

    stride: 1 (default) or 2 -- the depthwise layer is the one that changes
    resolution (the expensive in_ch*out_ch channel-mixing stays at stride 1,
    only the cheap per-channel depthwise pass does the actual downsampling).

    downsample: 'stride' (default) applies `stride` directly to the depthwise
    conv -- a real, unfused stride-2 conv, legal in the NPU's row-reuse
    domain. 'maxpool' keeps the depthwise conv at stride 1 and fuses a
    maxpool(k=stride) after it instead: the hardware does not support a
    strided depthwise conv in the frame-reuse domain at all, and every
    KuRALSNetNPUSeg downsampling stage below stageA lands there (its dec_a
    route/concat forces the row->frame transition to happen at/before
    stageA.pw -- see CLAUDE.md's "Row-reuse vs frame-reuse grouping"), so
    'maxpool' is what down1/down2 actually use whenever they stride."""

    def __init__(self, in_ch, out_ch, act='leaky', dw_kernel_size=3, stride=1, downsample='stride'):
        super().__init__()
        assert downsample in ('stride', 'maxpool')
        self.downsample = downsample
        self.pool_stride = stride
        dw_stride = stride if downsample == 'stride' else 1
        self.dw = QuantConvBNAct(in_ch, in_ch, kernel_size=dw_kernel_size, stride=dw_stride, act=act, depthwise=True)
        self.pw = QuantConvBNAct(in_ch, out_ch, kernel_size=1, stride=1, act=act, depthwise=False)

    def forward(self, x):
        x = self.dw(x)
        if self.downsample == 'maxpool' and self.pool_stride > 1:
            x = F.max_pool2d(x, kernel_size=self.pool_stride, stride=self.pool_stride)
        return self.pw(x)


class QuantResidualDWSeparableBlock(nn.Module):
    """Depthwise-separable block with the NPU's fused eltwise-add ("shortcut"),
    used where channel count stays constant so the shortcut is a plain identity.

    Hardware constraint this simulates: an eltwise-add's two operands must
    share one activation scale (the register format has no per-operand rescale
    for the add itself). So in the quantized path, the pointwise conv's output
    scale is forced to match the block's own input scale rather than being
    independently calibrated.
    """

    def __init__(self, ch, act='leaky', dw_kernel_size=3, eltwise_act=None):
        super().__init__()
        self.dw = QuantConvBNAct(ch, ch, kernel_size=dw_kernel_size, stride=1, act=act, depthwise=True)
        self.pw = QuantConvBNAct(ch, ch, kernel_size=1, stride=1, act='linear', depthwise=False)
        # eltwise_act='linear' is the only form this NPU can actually run: the fused add
        # has no verified activation stage, so the nonlinearity has to come from the next
        # conv layer instead (see export_int8.py's LEGAL_ELTWISE_ACTS, which refuses to
        # export anything else). Defaults to `act` to keep already-trained checkpoints
        # loading and evaluating unchanged -- those are not deployable as-is.
        eltwise_act = act if eltwise_act is None else eltwise_act
        self.act_elt = _ACTIVATIONS[eltwise_act]
        self.act_elt_name = eltwise_act

    def forward(self, x):
        if not (self.dw.quant_enabled and self.pw.quant_enabled):
            out = self.pw(self.dw(x))
            return self.act_elt(out + x)

        self.dw.in_observer.observe(x)
        x_scale = self.dw.in_observer.scale(127.0)

        dw_out = self.dw(x)
        pw_out = self.pw(dw_out, out_scale_override=x_scale)

        summed = pw_out + x  # both operands on the same quantization grid (x_scale)
        unsigned_out = (self.act_elt_name == 'relu')
        qmin, qmax = (0.0, 255.0 * x_scale) if unsigned_out else (-128.0 * x_scale, 127.0 * x_scale)
        # int8 + int8 can need 9 bits before the hardware re-clamps/saturates -- approximate that here.
        summed = ste_clamp(summed, -256.0 * x_scale, 255.0 * x_scale)
        out = self.act_elt(summed)
        return ste_clamp(out, qmin, qmax)


class QuantMBConvBlock(nn.Module):
    """EfficientNet-style inverted bottleneck: pointwise expand -> depthwise ->
    pointwise project, with an optional eltwise-add shortcut when shape-eligible.
    Matches efficientnetS_32x32.txt's compiled MBConv structure exactly (a real
    NPU-compilable reference workload -- see cfg_gen.py's INPUT_FILE options and
    kuralsnet_npu.py's module docstring): 1x1 conv (relu, expand) -> KxK depthwise
    conv (relu) -> 1x1 conv (linear, project) [-> shortcut]. No new NPU op
    categories -- this is QuantConvBNAct/QuantResidualDWSeparableBlock's own
    eltwise-add, just at a wider mid-block channel count than
    QuantDepthwiseSeparableBlock (which has no expand step at all) ever uses.

    `mid_ch` is explicit (not an expand_ratio float) so callers pick a PF=16-aligned
    filter count directly, matching every other channel width used in this codebase.
    `use_shortcut` defaults to shape-inference (in_ch == out_ch and stride == 1) but
    takes an explicit override so a caller can force it off even when shape-eligible
    -- an unchecked running_absmax blowup through an eltwise-add shortcut collapsed
    training once during this block's development, so the override exists to isolate
    the expand/project structure from that risk when needed.
    """

    def __init__(self, in_ch, out_ch, mid_ch, act='leaky', dw_kernel_size=3,
                 stride=1, use_shortcut=None):
        super().__init__()
        self.expand = QuantConvBNAct(in_ch, mid_ch, kernel_size=1, stride=1, act=act)
        self.dw = QuantConvBNAct(mid_ch, mid_ch, kernel_size=dw_kernel_size,
                                  stride=stride, act=act, depthwise=True)
        self.project = QuantConvBNAct(mid_ch, out_ch, kernel_size=1, stride=1, act='linear')
        self.act_elt = _ACTIVATIONS[act]
        self.act_elt_name = act
        self.use_shortcut = (in_ch == out_ch and stride == 1) if use_shortcut is None else use_shortcut
        if self.use_shortcut:
            assert in_ch == out_ch and stride == 1, \
                "eltwise-add shortcut requires matching shape (in_ch == out_ch, stride == 1)"

    def forward(self, x):
        if not self.use_shortcut:
            return self.project(self.dw(self.expand(x)))

        if not (self.expand.quant_enabled and self.dw.quant_enabled and self.project.quant_enabled):
            out = self.project(self.dw(self.expand(x)))
            return self.act_elt(out + x)

        # Same forced-scale eltwise-add as QuantResidualDWSeparableBlock -- the
        # project conv's output scale is forced to match this block's own input
        # scale, since the hardware add has no per-operand rescale.
        self.expand.in_observer.observe(x)
        x_scale = self.expand.in_observer.scale(127.0)

        branch_out = self.dw(self.expand(x))
        proj_out = self.project(branch_out, out_scale_override=x_scale)

        summed = proj_out + x
        unsigned_out = (self.act_elt_name == 'relu')
        qmin, qmax = (0.0, 255.0 * x_scale) if unsigned_out else (-128.0 * x_scale, 127.0 * x_scale)
        summed = ste_clamp(summed, -256.0 * x_scale, 255.0 * x_scale)
        out = self.act_elt(summed)
        return ste_clamp(out, qmin, qmax)
