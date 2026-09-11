# KuRALS trained checkpoints

Committed to git despite the general `*.pt`/`*.npz` exclusion in `.gitignore` -- this
directory is explicitly whitelisted (`!checkpoints/**/*.pt`) so a fresh clone comes with
these checkpoints already in hand, no separate archive/transfer needed. See `CLAUDE.md`
at the repo root for the full writeup this package summarizes. Each directory below is a
trimmed `train.py` run directory (TensorBoard `boards/` logs dropped -- machine-specific,
not needed to use the checkpoint): `val_doppler_model.pt` / `test_doppler_model.pt` /
`latest_model.pt` are bare `state_dict`s, loadable with
`net.load_state_dict(torch.load(path), strict=True)`; `*_results.json` holds the full
per-class confusion matrix + dice/precision/recall for that checkpoint; `config.json` is
the exact training config that produced it.

To use one, just point scripts at the `.pt` file directly in place, e.g.:

```bash
python demo_tracker.py --model-path ../checkpoints/native_best_0.5144/results/val_doppler_model.pt \
    --dataset KuRALS_CW --split Test --sequence 两个无人机 --output tracker_demo.gif
```

(paths relative to `kurals/`, where these scripts are run from). No need to copy it into
`<logs>/KuRALS_CW/<model_name>/<run_name>/results/` first -- that's only where `train.py`
itself writes new checkpoints.

## What's here

- **`native_best_0.5144/`** -- `kuralsnet_npu_seg`, native 124x2048 resolution, the
  current best/final model overall. val dice 0.5144 / test dice 0.4807.
  Config: `kurals/config_files/kuralsnet_npu_seg.json`.

- **`kuralsnet_baseline/`** -- `kuralsnet`, the original non-NPU-legal model (dilated
  ASPP / `ConvTranspose2d` / `Conv3d` -- not deployable on the target NPU, reference only).
  Binary Prec/Pd 81.80%/94.83%. Config: `kurals/config_files/kuralsnet.json`.

- **`8x64_anchor_0.3757_bc64/`** -- `kuralsnet_npu_seg`, block-max-pooled 8x64 SoC-buffer
  emulation, `bottleneck_ch=64` (default), `bottleneck_kernel_size=3`, `dropout_rate=0.1`,
  finetuned from the native checkpoint above. **Was the best 8x64 result for a long time;
  superseded by `8x64_stemstride1_0.6401_bc64/` below.** val dice 0.3757 / test dice 0.3635 (reproduced from a fresh training run
  on 2026-08-22, matches the previously documented number to 4 decimal places).
  Config: `kurals/config_files/kuralsnet_npu_seg_8x64_kernel3_finetune_dropout.json`.
  **Read CLAUDE.md's SoC section before trusting this number as a deployment estimate --
  it's measured against an emulated dataset (block max-pooling), not real SoC hardware
  output; the real hardware computes its 8x64 grid via its own small FFT, which is not
  numerically equivalent.**

- **`8x64_width_sweep/`** -- `bottleneck_ch` in {32, 96, 128, 192} at the same 8x64
  resolution, same finetune+dropout recipe. None beat the bc=64 anchor above -- **this
  sweep has a confound**: bc=64 is the only width that exactly matches the native source
  checkpoint's own width, so it's the only one with full pretrained-weight transfer for
  the bottleneck layers. See CLAUDE.md's "Bottleneck width sweep" section before treating
  these as a clean capacity ablation. Folder name = `bc<width>_<val dice>`:
  - `bc32_0.3062/` -- `bottleneck_ch=32`. Config: `kuralsnet_npu_seg_8x64_bc32_kernel3_finetune_dropout.json`.
  - `bc96_0.3028/` -- `bottleneck_ch=96`. Config: `kuralsnet_npu_seg_8x64_bc96_kernel3_finetune_dropout.json`.
  - `bc128_0.3090/` -- `bottleneck_ch=128`. Config: `kuralsnet_npu_seg_8x64_bc128_kernel3_finetune_dropout.json`.
  - `bc192_0.3353/` -- `bottleneck_ch=192`. Config: `kuralsnet_npu_seg_8x64_bc192_kernel3_finetune_dropout.json`.

  (all configs under `kurals/config_files/`)

- **`8x64_oracle_ceiling_reference_NOT_deployable/`** -- ground-truth-based background
  flattening (`oracle_detect_and_damp_dataset.py`), trained *and evaluated* on that
  privileged distribution throughout. val dice 0.4821 / test dice 0.4379 -- proves real
  headroom exists for the 8x64 target, but **cannot be deployed or trusted as a
  performance estimate**: it requires ground truth at inference (impossible in the
  field), and a follow-up experiment (privileged train / realistic eval) showed
  continued training on this distribution actively *degrades* realistic performance
  (peaks at 0.3772 around epoch 100, collapses to 0.2315 by epoch 300) rather than just
  failing to transfer. Keep for reference only -- see CLAUDE.md's "Oracle ceiling" section.

- **`8x64_leaky0125_wq2x1_0.3468_bc64/`** -- same kernel3+finetune+dropout recipe as the
  anchor above, retrained under two new hardware-matching changes to
  `kurals/models/quant.py`: leaky ReLU slope `0.1 -> 0.125` (`2^-3`, a shift-friendly
  value), and the weight-multiply datapath `w_eff = 2*w_stored + 1` instead of
  `w_eff = w_stored` (stored int8 byte is never what's actually multiplied against the
  input -- see that file for the full derivation). val dice 0.3468 / test dice 0.3581
  (epoch 126) -- **below the anchor above, and not yet a settled result**: this specific
  run later collapsed to trivial all-background prediction (dice ~0.2499) by epoch 285,
  well after this checkpoint's own peak, in a run that was separately cut short by an
  unrelated session restart before reaching epoch 300. A follow-up run
  (`kuralsnet_npu_seg_8x64_kernel3_finetune_dropout_leaky0125_wq2x1_latefreeze.json`,
  `quant_freeze_iters` moved 60000->92000 to test whether freezing the quantization grid
  earlier -- leaving a long weight-only tail -- is what let the collapse happen) is what
  produced this exact checkpoint's replacement candidate; check git log / re-run status
  before treating this specific `.pt` as final. Config: `kuralsnet_npu_seg_8x64_kernel3_finetune_dropout.json`
  (same file as the anchor -- only the shared `quant.py` activation/weight-quant code
  changed, not this config). **Superseded by `8x64_stemstride1_0.6401_bc64/` below**, which
  uses the same hardware scheme and nearly doubles this number; kept only as the reference
  point for the ceiling analysis in CLAUDE.md.

- **`8x64_stemstride1_0.6401_bc64/`** -- best 8x64 dice, but **NOT DEPLOYABLE**: its three
  bottleneck blocks do `add -> leaky` with no conv between, which this NPU cannot run (see
  CLAUDE.md's "Eltwise-add activation"; `export_int8.py` refuses to export it). Kept as the
  research result and as the init for the legal variant below.
  val dice 0.6401 / test dice 0.6556 (epoch 275). Comparable only to other 8x64 entries
  here -- the native-resolution checkpoints are scored on a different dataset (see
  CLAUDE.md).
  Same `2w+1` / leaky-0.125 hardware scheme as the checkpoint above, but with
  `stem_stride=1` and `encoder_depth=1` (both `kurals/models/kuralsnet_npu_seg.py`
  constructor params): the network stays at native 8x64 through stem/stageA/dec_a/head_out
  and takes exactly one downsample hop to a 4x32 bottleneck, so `head_out` predicts
  per-pixel and there is no final logit upsample at all. Independently verified outside the
  training loop -- `test_kuralsnet_vs_cfar.py` on Test (binary fg/bg, QAT int8-simulated):
  Prec 87.70% / Pd 73.21% / FAR 0.06% / mDice 0.8713, and the pure-integer replay
  (`benchmark_int8_pipeline.py`, the actual deployed arithmetic) gives
  Prec 89.02% / Pd 70.27% / mDice 0.8696. Per-class val dice
  `[bg 0.9996, 0.8481, 0.6174, 0.0952]` -- the fourth class is ~18 pixels in val and is
  not meaningfully learnable at that sample count. Config:
  `kuralsnet_npu_seg_8x64_stemstride1_finetune_leaky0125_wq2x1.json`.

- **`8x64_eltwiselinear_0.6328_bc64/`** -- **the deployable checkpoint: same as the entry
  above but NPU-legal.** `eltwise_act='linear'` (see CLAUDE.md's "Eltwise-add activation"),
  finetuned directly from the 0.6401 checkpoint's own weights since the state_dict is
  unchanged. val dice 0.6328 / test dice 0.6491 at epoch 48. Pure-integer replay on Test --
  the actual deployed arithmetic -- gives Acc 0.9989 / Prec 0.9017 / Pd 0.7606 / FAR 0.0005
  / mDice 0.8906, i.e. **better than the illegal checkpoint's 0.8902 / 0.7027 / 0.8696**, so
  moving the nonlinearity after the conv cost nothing. This is the checkpoint
  `kurals/input/kuralsnet_npu_seg_8x64.txt` and its manifest were generated from. Config:
  `kuralsnet_npu_seg_8x64_stemstride1_eltwiselinear.json`.

  Two caveats on the contents. Only `test_doppler_model.pt` is here: that run's own
  `val_doppler_model.pt` was epoch 22, **before QAT engaged at epoch 44**, so it holds
  float32-era weights that cannot be exported -- it is deliberately not included. And this
  is not the best the configuration reached: the best post-QAT epoch was 0.6660 val at
  epoch 206, above the illegal run's 0.6401, but it was never written to disk because the
  pre-QAT float32 peak (0.6921) outranked it under the old selection rule. That rule is
  fixed in `kurals/learners/model.py` (pre-QAT epochs are now skipped), so a re-run of this
  same config should land nearer 0.666 -- worth doing before treating 0.6328 as this
  architecture's ceiling.

## Not included

The ~40 retired `kuralsnet_npu` (old CenterNet detection-head) run directories --
that architecture is fully superseded (see `kurals/legacy/README.md`), not part of any
current result.
