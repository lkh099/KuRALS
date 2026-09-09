"""Exact-integer replay of KuRALSNetNPUSeg's real deployed pipeline, driven by an
export_int8.py manifest. Distinct from kurals.models.quant's training-time
simulation (which exists to be differentiable, via STE) -- this is
inference-only, no gradients, and uses the final calibrated/frozen weights
and scale/bias/shift values instead of live nn.Parameters.

Numeric note: conv accumulation is done in torch.float64 rather than literal
int32/int64 arithmetic, and is exact (not an approximation) for this model:
every value involved -- int8 weights/activations, int16 scale/bias -- fits
comfortably within float64's 52-bit exact-integer range even after the
largest per-output-element sum in this network (a few hundred terms of at
most 127*127 each, decades of headroom under 2**52). float64 conv2d gives the
literal correct integer result here, just computed via float hardware instead
of hand-rolled integer arithmetic.

Wire representation note: between layers, values are kept *dequantized*
(float, real units) rather than as raw shared-scale int8 bytes. This exactly
matches what the QAT training simulation did (every layer independently
calibrates its own input scale and requantizes), which is very slightly more
permissive than a hypothetical minimal-hardware interpretation with zero
rescale between two directly-chained conv layers with no fused pooling
between them (real hardware has no such rescale op, only the scale/shift/bias
pipeline within each conv's own accumulator) -- but it is the numerically
*correct* choice for reproducing what the exported weights were actually
trained to be robust to. At the two points where hardware genuinely has no
rescale capability at all -- the eltwise-add shortcut and the route/concat --
the exported scales were forced equal at export time (see export_int8.py), so
no rescale happens there either way, matching real hardware exactly.
"""
import numpy as np
import torch
import torch.nn.functional as F

_ACT_FNS = {
    'leaky': lambda x: F.leaky_relu(x, 0.125),
    'relu': lambda x: F.relu(x),
    'linear': lambda x: x,
}


def conv_layer(x_float, arrays, layer_meta, out_scale_override=None):
    """x_float: (B, C, H, W) float64 tensor, dequantized (real units).
    Returns the dequantized float64 output -- mirrors QuantConvBNAct.forward()'s
    hard (exact hardware) path exactly, minus the STE/gradient machinery."""
    name = layer_meta['name']
    w_stored = torch.from_numpy(arrays[f'{name}.weight_int8'].astype(np.float64))
    # Hardware weight datapath: the stored int8 byte is transformed to
    # 2*w_stored + 1 before it's multiplied against the input -- see
    # kurals/models/quant.py's QuantConvBNAct.forward()/export_int8() for the
    # full derivation. Never zero, effective range is the odd integers in
    # [-255, 255].
    w = 2.0 * w_stored + 1.0
    scale = torch.from_numpy(arrays[f'{name}.scale_int16'].astype(np.float64)).view(1, -1, 1, 1)
    bias = torch.from_numpy(arrays[f'{name}.bias_int16'].astype(np.float64)).view(1, -1, 1, 1)

    x_int = torch.clamp(torch.round(x_float / layer_meta['in_scale']), -128, 127)

    k = layer_meta['kernel_size']
    padding = (k - 1) // 2
    acc = F.conv2d(x_int, w, stride=layer_meta['stride'], padding=padding, groups=layer_meta['groups'])
    # Sanity check that float64 really did carry this exactly (see module docstring).
    assert torch.allclose(acc, torch.round(acc)), f'{name}: conv accumulator not integer-exact in float64'

    t1 = acc * scale
    t2 = torch.floor(t1 / (2.0 ** layer_meta['shift1']))
    t3 = t2 + bias
    t4 = torch.floor(t3 / (2.0 ** layer_meta['shift2']))

    out_int = _ACT_FNS[layer_meta['act_name']](t4)
    out_qmax = 255.0 if layer_meta['unsigned_out'] else 127.0
    out_qmin = 0.0 if layer_meta['unsigned_out'] else -128.0
    out_int = torch.clamp(out_int, out_qmin, out_qmax)

    out_scale = out_scale_override if out_scale_override is not None else layer_meta['out_scale']
    return out_int * out_scale


def residual_block(x, arrays, L, name):
    """Replay one eltwise-add residual block (dw -> pw -> add x -> act -> clamp),
    matching QuantResidualDWSeparableBlock's quantized forward exactly. Used for
    the bottleneck (bott_c/bott_d1/bott_d2, residual since v4/v9)."""
    dw_out = conv_layer(x, arrays, L[f'{name}.dw'])
    pw_meta = L[f'{name}.pw']
    pw_out = conv_layer(dw_out, arrays, pw_meta)  # already forced onto x's own scale at export time
    block_scale = pw_meta['out_scale']
    summed = pw_out + x
    summed = torch.clamp(summed, -256.0 * block_scale, 255.0 * block_scale)  # approximate int9 add-then-saturate
    eltwise_act = _ACT_FNS[pw_meta['eltwise_act']]
    block_out = eltwise_act(summed)
    unsigned = (pw_meta['eltwise_act'] == 'relu')
    qmin, qmax = (0.0, 255.0 * block_scale) if unsigned else (-128.0 * block_scale, 127.0 * block_scale)
    return torch.clamp(block_out, qmin, qmax)


def upconv_block(x, arrays, L, name):
    """Replay one decoder upconv step (dw -> pw -> fused 2x nearest upsample). pw's
    scale may or may not be forced to a skip's scale at export time, depending on
    whether this hop is followed by a concat (see export_int8.py's v14 topology --
    only upconv_a is; upconv_b/upconv_f are followed by a plain refine, no forcing)."""
    dw_out = conv_layer(x, arrays, L[f'{name}.dw'])
    pw_out = conv_layer(dw_out, arrays, L[f'{name}.pw'])
    return F.interpolate(pw_out, scale_factor=2, mode='nearest')


def refine_block(x, arrays, L, name):
    """Replay one plain (no-concat) refine block (dw -> pw). Used for refine_b/refine_f
    (v14) -- the decoder hops with no route/concat and therefore no scale-forcing."""
    dw_out = conv_layer(x, arrays, L[f'{name}.dw'])
    return conv_layer(dw_out, arrays, L[f'{name}.pw'])


def kuralsnet_npu_seg_int8_forward(x_int8, manifest, arrays):
    """
    PARAMETERS
    ----------
    x_int8: np.ndarray, shape (n_frames, H, W), dtype int8
        As produced by kurals.quantize_input.quantize_input().
    manifest, arrays: as written by export_int8.py.

    RETURNS
    -------
    logits: numpy float32 array, shape (n_classes, H, W) at native /1
    resolution -- dequantized raw per-pixel class logits (background included
    at index 0), decoded off-NPU via argmax (no sigmoid/offset split -- see
    kurals/models/kuralsnet_npu_seg.py's module docstring).
    """
    L = {l['name']: l for l in manifest['topology']}
    align = manifest['align']

    x = torch.from_numpy(x_int8.astype(np.float64))[None]  # (1, n_frames, H, W)
    x_float = x * L['stem.dw']['in_scale']  # dequantize the raw int8 input into wire units

    h, w = x_float.shape[-2], x_float.shape[-1]
    pad_h, pad_w = (-h) % align, (-w) % align
    if pad_h or pad_w:
        x_float = F.pad(x_float, (0, pad_w, 0, pad_h))

    # stem goes straight to /2 (no skip at /1). stageA is the network's only
    # skip -- a plain, unfused conv whose stored output is read again later
    # by dec_a's concat. down1/down2 continue to /8 as plain stride-2 convs
    # with nothing stored.
    x_float = conv_layer(x_float, arrays, L['stem.dw'])
    x_float = conv_layer(x_float, arrays, L['stem.pw'])

    skip = conv_layer(x_float, arrays, L['stageA.dw'])
    skip = conv_layer(skip, arrays, L['stageA.pw'])

    x_float = conv_layer(skip, arrays, L['down1.dw'])
    x_float = conv_layer(x_float, arrays, L['down1.pw'])
    x_float = conv_layer(x_float, arrays, L['down2.dw'])
    x_float = conv_layer(x_float, arrays, L['down2.pw'])

    # Bottleneck: three stacked eltwise-add residual blocks.
    block_input = x_float
    for stage_name in ('bott_c', 'bott_d1', 'bott_d2'):
        block_input = residual_block(block_input, arrays, L, stage_name)
    p_deep = block_input

    # Decoder: two upsample hops (/8->/4->/2), only the second one (upconv_a)
    # followed by a concat(skip) -> dec_a refine; upconv_b is followed by a
    # plain refine, no concat.
    u = upconv_block(p_deep, arrays, L, 'upconv_b')
    u = refine_block(u, arrays, L, 'refine_b')

    u = upconv_block(u, arrays, L, 'upconv_a')
    u = torch.cat((u, skip), dim=1)
    u = conv_layer(u, arrays, L['dec_a.dw'])
    u = conv_layer(u, arrays, L['dec_a.pw'])

    out = conv_layer(u, arrays, L['head_out'])  # /2, n_classes, linear logits
    # Bare (unfused, values-preserving) nearest upsample /2 -> /1, matching
    # KuRALSNetNPUSeg.forward()'s final step -- not a quantized layer, just a
    # duplication of already-dequantized values.
    out = F.interpolate(out, scale_factor=manifest['final_upsample'], mode='nearest')
    if pad_h or pad_w:
        out = out[:, :, :h, :w]

    return out[0].float().numpy()
