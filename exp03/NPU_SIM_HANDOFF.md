# NPU simulation handoff — R64-D8 / D16 / D32

What to copy to the NPU sim server, what each file means, and how to regenerate it.
Everything here is resolution-agnostic: nothing assumes an 8x64 grid.

## Packages

| package | grid | width | MACs | peak NPU buffer | int8 replay test mIoU | source run |
|---|---|---|---|---|---|---|
| `model_release_v01` | R64-D8 | 0.5x | 1,569,280 | 16,384 B | 0.6002 | `Q_…wm0.5_ss1_R64-D8_qlr0.0001_s44` |
| `model_release_w1.0_v01` | R64-D8 | 1.0x | 4,952,576 | 32,768 B | 0.6152 | `Q_…wm1.0_ss1_R64-D8_qlr1e-05_s45` |
| `model_release_d16_v01` | R64-D16 | 0.5x | 3,138,560 | 32,768 B | 0.5939 | `Q_…wm0.5_ss1_R64-D16_qlr0.0001_s45` |
| `model_release_d32_v01` | R64-D32 | 0.5x (stem_stride 2) | 1,569,280 | 16,384 B | 0.3262 | `Q_…wm0.5_ss2_R64-D32_qlr0.0001_s44` |

All four pass `gen_cfg.py` allocation at 32 KiB with CUTPOINT 3 and 23 layers.
**D16 and the 1.0x design sit exactly at the 32,768 B bank limit, with no headroom.**

Each package contains `architecture.json` (per-layer shapes, shifts, scales, fusion flags),
`weights_int.bin` / `scales.bin` / `biases.bin`, `quantization.json`, `preprocessing.json`
(input contract), `class_map.json` (output contract), `sample_inputs/`, `golden_outputs/`,
`golden_layer_outputs/`, `metrics.json` and `config.yaml`. `npu_cfg.c` is generated on demand
from the manifest by `gen_cfg.py`; it is not checked in.

Rebuild any of them:

```bash
python exp03/make_release.py --run <QAT run tag> --out exp03/model_release_<name>
# a grid with no preset config in exp03/configs/:
python exp03/make_release.py --run <tag> --out <dir> \
    --range-bins 128 --doppler-bins 64 --dataset-root /path/to/KuRALS_CW
```

`make_release.py` takes the grid from the run tag (`R<range>-D<doppler>`) and the dataset root
from `exp03/configs/rd_r<W>_d<H>.yaml`; `--range-bins` / `--doppler-bins` / `--dataset-root`
override both. The release is verified by replaying the exported integers over the whole Test
split and comparing against the QAT simulation (`pixel_agreement_with_qat_sim` in
`metrics.json`; 1.000000 for D16/D32/1.0x, 0.999993 for D8).

## Example IFMs

`exp03/example_ifm/<H>x<W>/` — 16 examples per grid (8 synthetic + 8 real), three files each:

| file | what | consumed by |
|---|---|---|
| `<name>_int8.npy` | `(1, H, W)` int8 codes in [0,127] | the model / `kurals/int8_inference.py` |
| `<name>_npu_u32.bin` | `4*H*W` B, Doppler-major, `4*(d*W + r) + ch` | the NPU first-layer DMA (`CTRL[2]=1`) |
| `<name>_raw_u32.bin` | `4*H*W` B, range-major, `4*(r*H + d)`, LE uint32 | DSP bring-up mode (`CTRL[2]=0`) |

Frame size is `4*H*W` bytes in both modes: 2,048 B at 8x64, 4,096 B at 16x64, 8,192 B at 32x64.
The raw and int8 frames of a synthetic example are consistent — the codes are exactly what the
hardware quantizer produces from that raw frame at the given `LOG_SCALE` — so either mode can
be driven and the network sees the same input. Synthetic patterns: `zeros`, `noise_floor`,
`range_ramp`, `doppler_ramp`, `corner_markers`, `targets`, `random`, `saturate`; the ramps and
`corner_markers` are orientation/transpose checks, `saturate` clamps every cell to 127.

```bash
python exp03/make_example_ifm.py --log-scale 22                  # the three study grids
python exp03/make_example_ifm.py --grids 8x64,64x128 --log-scale 0
```

Contract and byte layout: `npu_riscv docs/integration/radar_dsp_npu_input.md`. The generator
implements that document's `log8()` and `q = clamp(log8(P) - LOG_SCALE, 0, 127)` directly, and
the packing is unit-checked against the stated byte addresses.

## Caveats to carry across

* **No NPU cycle or latency number exists for any of these models.** Do not infer one from the
  MAC count; measure it.
* **D32's package is from a stride-2 head.** Every stride-1 candidate at D32 exceeds the
  32,768 B bank limit, so its output head runs at /2 and the logits are nearest-upsampled —
  every 2x2 output block is identical. Its low accuracy is that buffer limit, not evidence
  about Doppler information content.
* **D16/D32 int8 QAT is being re-run** at lower LRs (`exp03/qat_redo_d16_d32.py`): at the
  pipeline's chosen 1e-4 the D16 final seeds 43 and 44 peaked at their *first* validation
  (epoch 2 of 100) and never improved. The D16 package above is built from seed 45, the one
  seed that trained through, but rebuild it if the redo produces a better validation score.
* **D32's trainer-recorded test mIoU (0.3482) does not match its replay (0.3262).** Replay and
  QAT simulation agree exactly with each other, so the discrepancy is in the trainer's
  evaluation path, not in quantization. The same class of gap exists for the 1.0x package
  (0.5947 recorded vs 0.6152 replayed) and is unexplained; quote the replay number, which is
  the one that was verified end to end.
* All RD inputs are block-max-pool emulations of the native KuRALS-CW map, **not** native SoC
  FFT output. See the SoC caveat in `CLAUDE.md`.
