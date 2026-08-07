# CLAUDE.md

Project context for whoever (or whichever Claude Code session) picks this up next. For how to actually run things, see [`docs/USAGE.md`](docs/USAGE.md). For install, see [`README.md`](README.md).

## What this is

A radar target-detection pipeline for the KuRALS range-Doppler dataset, with one branch specifically targeting deployment on a **128-MAC-array NPU @ 100MHz**, 20Hz target frame rate -- a hard **640M MAC/inference budget** (`128 * 100e6 * 0.050s`). `kurals/cfg_gen.py`'s op set (conv2d k∈{1,3} stride∈{1,2}, depthwise conv k∈{1,3,5}, fused 2x nearest upsample/maxpool/eltwise-add, 2-source route/concat, leaky/relu/linear activation only -- no dilation, no transposed conv, no 3D conv) is the hard constraint every NPU-targeted model in this repo is built from.

## Current best model

**`kuralsnet_npu_seg`** (`kurals/models/kuralsnet_npu_seg.py`, `bottleneck_ch=64`, config `kurals/config_files/kuralsnet_npu_seg.json`). NPU-legal backbone, `/2` internal resolution, plain per-pixel segmentation head (argmax-decoded, no threshold/NMS) trained with quantization-aware training (float32 warmup, then int8-simulated QAT, then frozen observers).

Best checkpoint: epoch 137, **val dice 0.5144 / test dice 0.4807**. Binary any-foreground vs CFAR: Prec=71.26%, Pd=56.06%, FAR=0.00%. UAV-only: Prec=56.14%, Pd=55.56% (baseline `kuralsnet`, a non-NPU-legal model with dilated ASPP convs / `ConvTranspose2d` / `Conv3d`, reaches 81.80%/94.83% -- the remaining gap is plausibly resolution-driven, since this model's `/2` output has no offset regression to correct sub-cell position).

This replaced an earlier CenterNet-style heatmap+offset detection head (`kurals/legacy/`, see `kurals/legacy/README.md`) that topped out at Pd 35.0%/FAR 2.26% after 24 tuning revisions -- its sparse single-point supervision and threshold-based decode were the likely bottleneck, not model capacity.

## What's been tried and ruled out (don't re-attempt without a new idea)

- **Bottleneck widening** (`bottleneck_ch=64 -> 96`, `kuralsnet_npu_seg_wide96.json`): clean negative, ~0.11-0.12 dice below the default despite 1.21x MAC headroom and stable QAT training. Third capacity-increase attempt in this codebase's history to fail (after two similar attempts on the retired CenterNet model).
- **Class-weighted (WCE) loss** (`kuralsnet_npu_seg_wce.json`): clean negative, val dice 0.2520 -- `NBSLoss`'s un-renormalized per-pixel mean collapses the gradient under this dataset's extreme class-weight skew (~800,000x background:vehicle).
- **`/1` native-resolution decoder hop** (extra learned upconv/refine stage past `dec_a`): tried twice, both attempts had severe QAT-stability collapses and never beat the `/2` checkpoint's dice, while leaving near-zero MAC headroom (638.12M/640M).

## Where things live

- **Checkpoints and per-run metrics are NOT in this repo** (`.gitignore` excludes `*.pt`) -- they live under the `logs` path from `kurals/config_files/config.ini` (set via `kurals/utils/set_paths.py`), one directory per run, with `val_doppler_model.pt`/`test_doppler_model.pt`/`latest_model.pt` + matching `*_results.json` (full per-class confusion matrices).
- **Tracker**: `kurals/utils/tracker.py` (SORT-style: `scipy.ndimage.label` centroids, constant-velocity Kalman filter, Hungarian assignment via `scipy.optimize.linear_sum_assignment`), demoed by `kurals/demo_tracker.py`.
- **int8 export/deployment pipeline**: `export_int8.py` -> `dummy_ifm.py`/`quantize_input.py` -> `benchmark_int8_pipeline.py`, all targeting `kuralsnet_npu_seg`'s topology (ported from the retired CenterNet model's equivalent pipeline -- see `docs/USAGE.md`).
