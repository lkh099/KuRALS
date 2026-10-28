"""Example input feature maps (IFMs) for any RD grid, in the formats the NPU and the
radar DSP actually exchange.

Contract (npu_riscv `docs/integration/radar_dsp_npu_input.md`, generalised from 8x64 to HxW):

    log8(P) = {leading_one_index(P), three_bits_below_leading_one(P)}   # 5.3-bit log2
    q       = clamp(log8(P) - LOG_SCALE, 0, 127)
    pixel   = {8'h00, 8'h00, 8'h00, q}                                  # 32'h000000qq

    CTRL[2]=1  int8, Doppler-major  - what the NPU reads:
               byte_address = 4 * (doppler * W + range) + channel       # only channel 0 carries data
    CTRL[2]=0  uint32, range-major  - raw I^2+Q^2, bring-up only:
               byte_address = 4 * (range * H + doppler)

A frame is 4*H*W bytes in both modes (2,048 B at 8x64). The model itself consumes the bare
int8 codes with input_scale 1.0, so three artefacts are written per example:

    <name>_int8.npy       (1, H, W) int8 in [0,127]   - model / int8_inference.py replay input
    <name>_npu_u32.bin    4*H*W bytes, Doppler-major  - CTRL[2]=1 frame the NPU DMAs
    <name>_raw_u32.bin    4*H*W bytes, range-major    - CTRL[2]=0 raw power frame (bring-up)

The raw and int8 frames of one example are consistent: the int8 codes are what the hardware
quantizer produces from that raw power frame at the given LOG_SCALE, so a sim server can feed
either mode and expect the same network input.

  python exp03/make_example_ifm.py                         # 8x64, 16x64, 32x64
  python exp03/make_example_ifm.py --grids 8x64,64x128 --log-scale 22
"""
import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def log8(P):
    """Hardware 5.3-bit log2: 8*floor(log2(P)) + the three bits below the leading one. 0 for P==0."""
    P = np.asarray(P, dtype=np.uint64)
    out = np.zeros(P.shape, dtype=np.int64)
    nz = P > 0
    if not nz.any():
        return out
    v = P[nz]
    e = np.floor(np.log2(v.astype(np.float64))).astype(np.int64)         # leading-one index
    e = np.minimum(e, 63)
    shift_down = np.maximum(e - 3, 0)
    m = np.where(e >= 3, (v >> shift_down.astype(np.uint64)) & np.uint64(7),
                 (v << np.maximum(3 - e, 0).astype(np.uint64)) & np.uint64(7))
    out[nz] = 8 * e + m.astype(np.int64)
    return out


def quantize(P, log_scale):
    """Raw uint32 power -> int8 NPU code, exactly as the DSP's quantizer does it."""
    return np.clip(log8(P) - log_scale, 0, 127).astype(np.int8)


def pack_npu_u32(code):
    """(H, W) int8 codes -> CTRL[2]=1 frame: 4 bytes per pixel, Doppler-major, channel 0 only."""
    H, W = code.shape
    buf = np.zeros((H, W, 4), dtype=np.uint8)
    buf[:, :, 0] = code.astype(np.uint8)          # channels 1..3 are the unused input-channel bytes
    return buf.reshape(-1).tobytes()


def pack_raw_u32(P):
    """(H, W) uint32 power -> CTRL[2]=0 frame: range-major, 4 bytes per cell, little-endian."""
    return np.ascontiguousarray(P.T.astype('<u4')).tobytes()


# ---------------------------------------------------------------- synthetic power fields
def synth(name, H, W, rng):
    """Raw uint32 power fields. Chosen to exercise the quantizer's whole 0..127 code range and
    the datapath's addressing, not to look like radar."""
    d = np.arange(H)[:, None]
    r = np.arange(W)[None, :]
    if name == 'zeros':                      # all-zero power -> all-zero codes
        return np.zeros((H, W), dtype=np.uint64)
    if name == 'noise_floor':                # flat mid-scale background
        return np.full((H, W), 1 << 20, dtype=np.uint64)
    if name == 'range_ramp':                 # monotone in range: checks the range axis / transpose
        return (np.uint64(1) << (4 + (24 * r // max(W - 1, 1))).astype(np.uint64)) * np.ones((H, 1), dtype=np.uint64)
    if name == 'doppler_ramp':               # monotone in Doppler: the other axis
        return (np.uint64(1) << (4 + (24 * d // max(H - 1, 1))).astype(np.uint64)) * np.ones((1, W), dtype=np.uint64)
    if name == 'corner_markers':             # distinct value in each corner: orientation check
        P = np.full((H, W), 1 << 12, dtype=np.uint64)
        for (i, j), e in (((0, 0), 20), ((0, W - 1), 24), ((H - 1, 0), 28), ((H - 1, W - 1), 31)):
            P[i, j] = np.uint64(1) << np.uint64(e)
        return P
    if name == 'targets':                    # sparse bright cells on a dim floor, like a real sparse scene
        P = np.full((H, W), 1 << 10, dtype=np.uint64)
        for k, (di, ri) in enumerate(((H // 4, W // 8), (H // 2, W // 2), (3 * H // 4, 7 * W // 8))):
            P[di % H, ri % W] = np.uint64(1) << np.uint64(26 + k)
            for dd, rr in ((0, 1), (0, -1), (1, 0), (-1, 0)):           # a little skirt around each peak
                P[(di + dd) % H, (ri + rr) % W] = np.uint64(1) << np.uint64(22 + k)
        return P
    if name == 'random':                     # full-range random powers
        return (np.uint64(1) << rng.integers(0, 32, size=(H, W)).astype(np.uint64))
    if name == 'saturate':                   # above the top of the code window -> everything clamps to 127
        return np.full((H, W), (1 << 63) - 1, dtype=np.uint64)
    raise ValueError(name)


SYNTH = ['zeros', 'noise_floor', 'range_ramp', 'doppler_ramp', 'corner_markers', 'targets', 'random', 'saturate']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grids', default='8x64,16x64,32x64', help='comma-separated HxW (Doppler x range)')
    ap.add_argument('--out', default=f'{ROOT}/example_ifm')
    ap.add_argument('--log-scale', type=int, default=0, help='LOG_SCALE register value (APB 0x0c)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-real', action='store_true', help='skip copying real frames out of the release packages')
    a = ap.parse_args()

    grids = [tuple(int(v) for v in g.lower().split('x')) for g in a.grids.split(',')]
    rng = np.random.default_rng(a.seed)
    manifest = dict(contract='npu_riscv docs/integration/radar_dsp_npu_input.md',
                    log_scale=a.log_scale, int8_range=[0, 127], input_scale=1.0,
                    doppler_order='natural FFT order (zero-Doppler at index 0)',
                    npu_frame='4 bytes per pixel, Doppler-major, byte = 4*(doppler*W + range) + channel; only channel 0 carries data',
                    raw_frame='4 bytes per cell, range-major, byte = 4*(range*H + doppler), little-endian uint32 power',
                    grids={})

    for H, W in grids:
        d = f'{a.out}/{H}x{W}'
        os.makedirs(d, exist_ok=True)
        entries = []
        for name in SYNTH:
            P = synth(name, H, W, rng)
            code = quantize(P, a.log_scale)
            np.save(f'{d}/{name}_int8.npy', code[None])                  # (1, H, W), as the model expects
            open(f'{d}/{name}_npu_u32.bin', 'wb').write(pack_npu_u32(code))
            open(f'{d}/{name}_raw_u32.bin', 'wb').write(pack_raw_u32(P))
            entries.append(dict(name=name, kind='synthetic', code_min=int(code.min()), code_max=int(code.max()),
                                nonzero_cells=int((code > 0).sum())))

        # real frames: reuse the int8 inputs already verified inside a release package for this grid
        if not a.no_real:
            for rel in sorted(glob.glob(f'{ROOT}/model_release*')):
                cfgs = glob.glob(f'{rel}/config.yaml')
                if not cfgs:
                    continue
                txt = open(cfgs[0]).read()
                # newer packages carry explicit bin counts; older ones only the rd_resolution name
                if not ((f'doppler_bins: {H}' in txt and f'range_bins: {W}' in txt)
                        or f'rd_resolution: R{W}-D{H}' in txt):
                    continue
                for i, f in enumerate(sorted(glob.glob(f'{rel}/sample_inputs/sample_*_int8.npy'))):
                    code = np.load(f).reshape(H, W).astype(np.int8)
                    base = f'real_{os.path.basename(rel)}_{i:02d}'
                    np.save(f'{d}/{base}_int8.npy', code[None])
                    open(f'{d}/{base}_npu_u32.bin', 'wb').write(pack_npu_u32(code))
                    entries.append(dict(name=base, kind='real (from the release package; no raw power available)',
                                        code_min=int(code.min()), code_max=int(code.max()),
                                        nonzero_cells=int((code > 0).sum()), source=os.path.relpath(f, ROOT)))
                break

        manifest['grids'][f'{H}x{W}'] = dict(doppler_bins=H, range_bins=W, cells=H * W,
                                             int8_bytes=H * W, frame_bytes=4 * H * W, examples=entries)
        print(f'{H}x{W}: {len(entries)} examples, frame = {4 * H * W} B ({H * W} cells) -> {d}')

    json.dump(manifest, open(f'{a.out}/manifest.json', 'w'), indent=1)
    open(f'{a.out}/README.md', 'w').write(f"""# Example IFMs

Generated by `exp03/make_example_ifm.py` (LOG_SCALE = {a.log_scale}). Contract:
`npu_riscv docs/integration/radar_dsp_npu_input.md`, generalised from 8x64 to any HxW.

Per example, three files:

| file | what | consumed by |
|---|---|---|
| `<name>_int8.npy` | `(1, H, W)` int8 codes in [0,127] | the model, `kurals/int8_inference.py` replay |
| `<name>_npu_u32.bin` | `4*H*W` B, Doppler-major, `4*(d*W + r) + ch`, channel 0 only | the NPU first-layer DMA (CTRL[2]=1) |
| `<name>_raw_u32.bin` | `4*H*W` B, range-major, `4*(r*H + d)`, LE uint32 power | DSP bring-up mode (CTRL[2]=0) |

The raw and int8 frames of one synthetic example are consistent: the codes are what the
hardware quantizer produces from that raw frame at this LOG_SCALE, so either mode can be fed
and the network sees the same input. Real frames come from a release package's verified
`sample_inputs/` and have no raw-power counterpart (the dataset stores post-FFT magnitude).

Synthetic patterns: {', '.join(SYNTH)}. `corner_markers` and the two ramps are orientation /
transpose checks; `saturate` clamps every cell to 127; `zeros` is the all-zero frame.

Shapes present: {', '.join(f'{h}x{w}' for h, w in grids)}. See `manifest.json` for per-example
code ranges and byte counts.
""")
    print('manifest:', f'{a.out}/manifest.json')


if __name__ == '__main__':
    main()
