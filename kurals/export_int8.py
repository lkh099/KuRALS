"""Export a trained (QAT) KuRALSNetNPUSeg as int8 weights + the hardware's exact
per-layer dequantization parameters (signed int16 scale/bias, truncating
shift1/shift2), for the "sample -> quantize -> infer -> dequantize ->
benchmark" pipeline (see kurals/int8_inference.py) and for repacking into the
real NPU toolchain.

Two files are written:
  <out>.npz  -- one weight_int8/scale_int16/bias_int16 array per layer,
                keyed "<layer_name>.<field>"
  <out>.json -- everything else: shift1/shift2, in_scale/out_scale (float32,
                the dequantization scales -- there is no zero-point/offset
                anywhere in this format, quantization is symmetric
                throughout, so it's not exported because it's always 0), and
                a "topology" list describing execution order plus the fused
                ops (maxpool/upsample/route-concat/eltwise-add) between
                layers. json['input_scale'] / json['output_scale'] are the
                stem's and head_out's scales -- kurals/quantize_input.py needs
                exactly those two. json['input_assembly'] records how the
                stem's input should be physically assembled on real hardware:
                for n_frames=2, as a 2-source route/concat of two
                independently-addressed frame reads rather than one wide DMA
                read, since a circular input buffer can put the two frames on
                opposite sides of its wrap point -- purely a hardware-topology
                note; doesn't affect this repo's own int8 numeric simulation,
                since concat-then-conv == conv fed a pre-stacked input either
                way. json['final_upsample'] records the bare (unfused,
                values-preserving) nearest upsample forward() applies to
                head_out's raw logits (/2 -> /1) -- not a quantized layer,
                just a duplication of already-dequantized values, but
                int8_inference.py needs the factor to replay it.

Sign note (see QuantConvBNAct.export_int8's docstring): every layer is
exported with unsigned_in=False (signed int8 activations throughout), even
though real hardware could use the extra bit available downstream of a ReLU.
This keeps the export numerically consistent with what was actually
simulated during QAT -- it costs a bit of precision, not correctness.
"""
import argparse
import json

import numpy as np
import torch

from kurals.utils.checkpoint_io import load_segmentation_model

# Hardware constraint: the fused eltwise-add has no verified activation stage of its
# own. An add must be followed by a conv layer, which supplies the nonlinearity via
# its own act_type -- "add -> leaky" with nothing in between is illegal, even though
# cfg_gen.py's register format has an act_elt field that can encode it (that field is
# unexercised: tiny_yolov2, the only silicon-verified workload, has no shortcut at
# all). 'linear' means "write the sum through unmodified", which is legal.
LEGAL_ELTWISE_ACTS = ('linear',)


def _export_layer(name, layer, out_scale_override=None):
    d = layer.export_int8(out_scale_override=out_scale_override)
    return name, d


def export_kuralsnet_npu_seg(net):
    """Walk KuRALSNetNPUSeg's fixed topology in the exact order forward()
    executes it, exporting every QuantConvBNAct. Returns (arrays, manifest)."""
    assert not net.training, "call net.eval() before exporting"

    arrays = {}
    layers = []  # ordered list of per-layer metadata dicts (the "topology")

    # Tripwire for a real constraint: each of these fusion tags means "the op
    # is fused directly onto this conv's output pipeline, with no
    # intermediate write to memory" -- only valid when that raw output has
    # exactly one consumer (the fused op itself). route_source/route_target
    # means a *second* consumer needs the raw output again later, which is
    # incompatible with any of these on the same conv -- including
    # eltwise_add_with (residual): a conv cannot be both a residual/shortcut
    # AND a route/concat source on this NPU. If a conv's output genuinely
    # needs both, export the fused op as its own standalone topology entry
    # instead, rather than tagging it onto the conv.
    _SINGLE_CONSUMER_FUSIONS = ('eltwise_add_with', 'maxpool_after', 'upsample_after')

    def add(name, layer, out_scale_override=None, **fusion):
        used = [k for k in _SINGLE_CONSUMER_FUSIONS if k in fusion]
        has_route = ('route_source' in fusion) or ('route_target' in fusion)
        assert not (used and has_route), (
            f"{name}: {used} assumes a single consumer (fused, no intermediate memory "
            f"write) but route_source/route_target={fusion.get('route_source') or fusion.get('route_target')!r} "
            "means a second consumer needs the raw output too -- not both on one conv."
        )
        _, d = _export_layer(name, layer, out_scale_override)
        arrays[f'{name}.weight_int8'] = d['weight_int8']
        arrays[f'{name}.scale_int16'] = d['scale_int16']
        arrays[f'{name}.bias_int16'] = d['bias_int16']
        entry = {
            'name': name,
            'shift1': d['shift1'], 'shift2': d['shift2'],
            'in_scale': d['in_scale'], 'out_scale': d['out_scale'],
            'act_name': d['act_name'],
            'unsigned_in': d['unsigned_in'], 'unsigned_out': d['unsigned_out'],
            'depthwise': d['depthwise'], 'groups': d['groups'],
            'kernel_size': d['kernel_size'], 'stride': d['stride'],
            'in_ch': d['in_ch'], 'out_ch': d['out_ch'],
        }
        entry.update(fusion)
        layers.append(entry)
        return d['out_scale']

    # stem goes straight from raw input to /2 (a single stride-2 conv).
    # stageA (/2, unfused) is the network's only skip, read again by dec_a's
    # route/concat. down1/down2 continue to /8 as plain stride-2 convs,
    # storing nothing.
    #
    # Input assembly: stem.in_ch == n_frames raw consecutive frame(s),
    # channel-stacked. When n_frames==2, a single conv2d op has one input
    # base address and implicitly assumes its whole input span is contiguous
    # -- safe for one frame, not safe for "N frames back-to-back" once the
    # ring buffer they're read from wraps around between them. route/concat's
    # two independent, independently-addressed operands are exactly the
    # mechanism for combining two reads that aren't guaranteed contiguous, so
    # on real hardware stem's input should be assembled as a 2-source
    # route/concat of two separately-addressed frame reads (frame_t,
    # frame_t-1), not one wide DMA read -- recorded here for the real
    # toolchain's benefit, not exercised by the numeric simulation in
    # int8_inference.py (which is handed an already-assembled tensor, same
    # either way since concat-then-conv == conv with a pre-stacked input).
    input_route_source = 'input_frame_t+input_frame_t-1' if net.n_frames == 2 else None

    add('stem.dw', net.stem.dw, **({'route_source': input_route_source} if input_route_source else {}))
    add('stem.pw', net.stem.pw)
    add('stageA.dw', net.stageA.dw)
    stageA_out_scale = add('stageA.pw', net.stageA.pw, route_target='dec_a.dw')

    # maxpool_after mirrors net.down1/down2's own downsample mode (see
    # QuantDepthwiseSeparableBlock's docstring and KuRALSNetNPUSeg.__init__).
    add('down1.dw', net.down1.dw, maxpool_after=(net.down1.downsample == 'maxpool'))
    add('down1.pw', net.down1.pw)
    add('down2.dw', net.down2.dw, maxpool_after=(net.down2.downsample == 'maxpool'))
    add('down2.pw', net.down2.pw)

    # Bottleneck: eltwise-add residual blocks -- pw's output scale is forced
    # to match the block's own input scale (its dw's in_observer), matching
    # the hardware constraint that an eltwise-add's two operands share one
    # activation scale. The sum is recorded as eltwise metadata on this
    # layer's entry rather than in pw's own (linear) activation.
    for stage_name, stage in (('bott_c', net.bott_c), ('bott_d1', net.bott_d1), ('bott_d2', net.bott_d2)):
        assert stage.act_elt_name in LEGAL_ELTWISE_ACTS, (
            f"{stage_name}: eltwise-add feeding '{stage.act_elt_name}' directly is not legal on "
            f"this NPU -- an eltwise-add must be followed by a conv layer before any activation, "
            f"so only {LEGAL_ELTWISE_ACTS} may sit on the add itself (see this module's "
            f"LEGAL_ELTWISE_ACTS and CLAUDE.md's 'Eltwise-add activation' section). Rebuild the "
            f"block with eltwise_act='linear' and let the following conv apply the nonlinearity; "
            f"a checkpoint trained with the illegal form cannot be deployed as-is.")
        add(f'{stage_name}.dw', stage.dw)
        block_in_scale = float(stage.dw.in_observer.scale(127.0))
        add(f'{stage_name}.pw', stage.pw, out_scale_override=block_in_scale,
            eltwise_add_with=f'{stage_name}.block_input', eltwise_act=stage.act_elt_name)

    # Decoder: only upconv_a (the /4->/2 hop) is followed by a route/concat --
    # upconv_b is a fused upsample straight into a plain refine conv, no
    # route, no scale-forcing. upconv_a's upsample is fused (single consumer
    # -- only dec_a's route reads it), forced onto stageA's own scale since
    # concat has no rescale registers on this hardware (same constraint as
    # the eltwise-add above -- see the matching comment in
    # KuRALSNetNPUSeg.forward()). dec_a.dw's route_source names both concat
    # operands. There is no further learned decoder hop after dec_a -- see
    # head_out below.
    add('upconv_b.dw', net.upconv_b.dw)
    # Decoder upsample count mirrors net.encoder_depth (see KuRALSNetNPUSeg.forward()):
    # upconv_b's fused upsample only exists when both encoder hops actually strided.
    add('upconv_b.pw', net.upconv_b.pw, upsample_after=(net.encoder_depth >= 2))
    add('refine_b.dw', net.refine_b.dw)
    add('refine_b.pw', net.refine_b.pw)

    add('upconv_a.dw', net.upconv_a.dw)
    add('upconv_a.pw', net.upconv_a.pw, out_scale_override=stageA_out_scale, upsample_after=(net.encoder_depth >= 1))
    add('dec_a.dw', net.dec_a.dw, route_source='upconv_a.pw+stageA.pw')
    add('dec_a.pw', net.dec_a.pw)

    # Segmentation head: n_classes raw per-pixel logits (background included
    # at index 0), plain linear 1x1 conv, decoded off-NPU via argmax (no
    # sigmoid/offset split -- see KuRALSNetNPUSeg's module docstring).
    head_out_scale = add('head_out', net.head_out)

    manifest = {
        'model': 'kuralsnet_npu_seg',
        'stride': net.stride,
        'align': net.align,
        'n_classes': net.n_classes,
        'n_frames': net.n_frames,
        'input_assembly': 'route_concat_2source' if net.n_frames == 2 else 'single_dma_read',
        'input_scale': layers[0]['in_scale'],   # stem's input scale -- see quantize_input.py
        'output_scale': head_out_scale,         # head_out's output scale -- for dequantizing logits
        'output_unsigned': layers[-1]['unsigned_out'],
        # bare nearest upsample of head_out's logits back to input resolution; 1 (i.e. no
        # upsample at all) when stem_stride=1, since the head already predicts per-pixel.
        'final_upsample': net.stem_stride,
        'topology': layers,
    }
    return arrays, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config_files/kuralsnet_npu_seg.json')
    parser.add_argument('--dataset', default='KuRALS_CW')
    parser.add_argument('--model-path', dest='model_path', required=True)
    parser.add_argument('--out', default='kuralsnet_npu_seg_int8')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    with open(args.cfg, 'r') as fp:
        cfg = json.load(fp)
    cfg['dataset'] = args.dataset
    cfg['nb_classes'] = 4 if args.dataset == 'KuRALS_CW' else 5

    net, quant_enabled = load_segmentation_model(cfg, args.model_path, torch.device(args.device), quant_mode='on')
    if not quant_enabled:
        raise RuntimeError('quant_enabled came back False even with quant_mode="on" -- unexpected.')

    arrays, manifest = export_kuralsnet_npu_seg(net)

    np.savez(f'{args.out}.npz', **arrays)
    with open(f'{args.out}.json', 'w') as fp:
        json.dump(manifest, fp, indent=2)

    print(f'Exported {len(manifest["topology"])} layers to {args.out}.npz / {args.out}.json')
    print(f'input_scale={manifest["input_scale"]:.6g}  output_scale={manifest["output_scale"]:.6g}')


if __name__ == '__main__':
    main()
