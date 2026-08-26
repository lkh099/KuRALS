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
  finetuned from the native checkpoint above. **Current best/deployable result for the
  8x64 target.** val dice 0.3757 / test dice 0.3635 (reproduced from a fresh training run
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
  these as a clean capacity ablation.

- **`8x64_oracle_ceiling_reference_NOT_deployable/`** -- ground-truth-based background
  flattening (`oracle_detect_and_damp_dataset.py`), trained *and evaluated* on that
  privileged distribution throughout. val dice 0.4821 / test dice 0.4379 -- proves real
  headroom exists for the 8x64 target, but **cannot be deployed or trusted as a
  performance estimate**: it requires ground truth at inference (impossible in the
  field), and a follow-up experiment (privileged train / realistic eval) showed
  continued training on this distribution actively *degrades* realistic performance
  (peaks at 0.3772 around epoch 100, collapses to 0.2315 by epoch 300) rather than just
  failing to transfer. Keep for reference only -- see CLAUDE.md's "Oracle ceiling" section.

## Not included

The ~40 retired `kuralsnet_npu` (old CenterNet detection-head) run directories --
that architecture is fully superseded (see `kurals/legacy/README.md`), not part of any
current result.
