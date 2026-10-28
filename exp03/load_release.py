"""Load a release package back into the (arrays, manifest) pair kurals/int8_inference.py expects.

make_release.py writes a package; until this script existed nothing read one back, so the exact-
integer replay could only be driven from a live torch model built from a .pt checkpoint. That made
the packages non-self-describing: a simulation server could compare its own output against
golden_outputs/, but could not step the Python reference implementation to find out *where* a
mismatch started. This closes that gap -- a package plus this file is enough.

  # replay every sample input and check it against the package's own golden outputs
  python exp03/load_release.py --package exp03/model_release_d32_v01

  # one sample, and dump the per-layer comparison
  python exp03/load_release.py --package exp03/model_release_v01 --sample 3 --layers

  # regenerate the cfg_gen.py input for a package, no checkpoint involved
  python exp03/load_release.py --package exp03/model_release_d32_v01 --emit-manifest /tmp/d32.json
  python kurals/manifest_to_cfg_input.py /tmp/d32.json --out kurals/input/kuralsnet_npu_seg_32x64.txt

In your own code:

  from exp03.load_release import load_release
  import kurals.int8_inference as q8
  arrays, manifest = load_release('exp03/model_release_d32_v01')
  logits = q8.kuralsnet_npu_seg_int8_forward(x_int8, manifest, arrays)   # (n_classes, H, W) float32

Reading a package needs only numpy. Replaying it additionally needs torch, because
int8_inference.py uses F.conv2d (in float64, which is exact for this network -- see its docstring).

Byte layout, from make_release.py: weights_int.bin is int8 and `weight_offset`/`weight_bytes` are
element counts into it; scales.bin and biases.bin are little-endian int16 and `scale_offset` /
`bias_offset` are BYTE offsets while `n_scale` / `n_bias` are element counts. Layers with no
parameters (pure fused ops) carry no offset keys and contribute no arrays.
"""
import argparse
import json
import os

import numpy as np

PARAM_KEYS = ('weight_offset', 'weight_bytes', 'weight_shape', 'scale_offset', 'n_scale', 'bias_offset', 'n_bias')


def load_release(pkg):
    """Return (arrays, manifest), the same pair export_int8.export_kuralsnet_npu_seg() returns."""
    arch = json.load(open(f'{pkg}/architecture.json'))
    quant = json.load(open(f'{pkg}/quantization.json'))

    w = np.fromfile(f'{pkg}/weights_int.bin', dtype=np.int8)
    s = np.fromfile(f'{pkg}/scales.bin', dtype='<i2')
    b = np.fromfile(f'{pkg}/biases.bin', dtype='<i2')

    arrays = {}
    for L in arch['layers']:
        if 'weight_offset' not in L:
            continue
        name = L['name']
        wo, n = L['weight_offset'], L['weight_bytes']
        arrays[f'{name}.weight_int8'] = w[wo:wo + n].reshape(L['weight_shape'])
        so = L['scale_offset'] // 2            # stored as a byte offset into an int16 buffer
        arrays[f'{name}.scale_int16'] = s[so:so + L['n_scale']]
        bo = L['bias_offset'] // 2
        arrays[f'{name}.bias_int16'] = b[bo:bo + L['n_bias']]
        if arrays[f'{name}.weight_int8'].size != n:
            raise ValueError(f'{name}: weights_int.bin is short -- wanted {n} bytes at {wo}')

    n_frames, H, W = arch['input_shape']
    manifest = dict(model=arch['model'], stride=arch['stride'], align=arch['align'],
                    n_classes=arch['n_classes'], n_frames=n_frames,
                    input_assembly=arch['input_assembly'], final_upsample=arch['final_upsample'],
                    input_scale=quant['input_scale'], output_scale=quant['output_scale'],
                    output_unsigned=quant['output_unsigned'],
                    # the offset keys are extra; int8_inference.py reads topology entries by key, so
                    # carrying them is harmless and keeps the package round-trippable.
                    topology=arch['layers'])
    return arrays, manifest


def emit_manifest(pkg, path):
    """Write an export_int8.py-format manifest JSON, so kurals/manifest_to_cfg_input.py can turn a
    package straight into a cfg_gen.py input file without a checkpoint. The per-layer byte offsets
    are dropped here, since export_int8()'s own manifest does not carry them."""
    _, manifest = load_release(pkg)
    manifest = dict(manifest, topology=[{k: v for k, v in L.items() if k not in PARAM_KEYS}
                                        for L in manifest['topology']])
    json.dump(manifest, open(path, 'w'), indent=1)
    print(f'{os.path.basename(pkg)} -> {path} ({len(manifest["topology"])} layers, '
          f'final_upsample={manifest["final_upsample"]})')
    return path


def grid(pkg):
    n_frames, H, W = json.load(open(f'{pkg}/architecture.json'))['input_shape']
    return n_frames, H, W


def verify(pkg, sample=None, show_layers=False):
    """Replay the packaged sample inputs and compare against the packaged golden outputs."""
    import kurals.int8_inference as q8

    arrays, manifest = load_release(pkg)
    ids = ([sample] if sample is not None else
           sorted(int(f[7:9]) for f in os.listdir(f'{pkg}/sample_inputs') if f.endswith('_int8.npy')))
    print(f'{os.path.basename(pkg)}: {len(manifest["topology"])} layers, '
          f'input {tuple(manifest["input_shape"]) if "input_shape" in manifest else grid(pkg)}, '
          f'{len(arrays) // 3} parameterised layers, final_upsample={manifest["final_upsample"]}')

    worst = 0.0
    for i in ids:
        x = np.load(f'{pkg}/sample_inputs/sample_{i:02d}_int8.npy')
        want = np.load(f'{pkg}/golden_outputs/sample_{i:02d}_logits_f32.npy')

        if show_layers:
            rec = {}
            orig = q8.conv_layer

            def traced(x_float, arrs, meta, out_scale_override=None):
                out = orig(x_float, arrs, meta, out_scale_override)
                rec[meta['name']] = out[0].float().numpy()
                return out

            q8.conv_layer = traced
            try:
                got = q8.kuralsnet_npu_seg_int8_forward(x, manifest, arrays)
            finally:
                q8.conv_layer = orig
            golden = np.load(f'{pkg}/golden_layer_outputs/sample_{i:02d}.npz')
            print(f'  sample {i:02d} per-layer:')
            for name in golden.files:
                if name in rec:
                    d = float(np.abs(rec[name].astype(np.float64) - golden[name].astype(np.float64)).max())
                    print(f'    {name:14s} {str(golden[name].shape):16s} max|diff| = {d:.3e}')
        else:
            got = q8.kuralsnet_npu_seg_int8_forward(x, manifest, arrays)

        d = float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max())
        am_got, am_want = got.argmax(0), want.argmax(0)
        agree = float((am_got == am_want).mean())
        worst = max(worst, d)
        flag = 'OK' if d == 0.0 else ('close' if d < 1e-5 else 'MISMATCH')
        print(f'  sample {i:02d}: logits max|diff| = {d:.3e}  argmax agreement = {agree:.6f}  {flag}')

    print(f'  worst logits max|diff| over {len(ids)} samples = {worst:.3e}')
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--package', required=True, help='a model_release_* directory')
    ap.add_argument('--sample', type=int, default=None, help='verify just this sample index')
    ap.add_argument('--layers', action='store_true', help='also compare every layer against golden_layer_outputs')
    ap.add_argument('--emit-manifest', metavar='PATH', default=None,
                    help='write an export_int8-format manifest JSON for kurals/manifest_to_cfg_input.py and exit')
    a = ap.parse_args()
    if a.emit_manifest:
        emit_manifest(a.package.rstrip('/'), a.emit_manifest)
        raise SystemExit(0)
    worst = verify(a.package.rstrip('/'), a.sample, a.layers)
    raise SystemExit(0 if worst < 1e-5 else 1)


if __name__ == '__main__':
    main()
