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

## SoC 8x64 buffer emulation

A deployment target with a much smaller, fixed RD buffer (8 Doppler x 64 Range, vs.
this repo's native 124x2048) surfaced after the handoff above. **Critical caveat:
confirmed directly with the person who owns the SoC spec -- the real hardware computes
its 8x64 grid via its own small range/Doppler FFT directly from raw samples. It does
NOT downsample a larger FFT's output the way this repo's block max-pooling
(`kurals/dataset_process/resize_extracted_dataset.py` / `kuralscw_processing.py
--doppler-bins/--range-bins`) does.** Small-FFT and decimated-large-FFT outputs differ
in bin edges, windowing/leakage, and noise statistics -- they are not equivalent
signals. Raw IQ/ADC data isn't available in this repo (even the "raw complex-valued RD
data" in the README's dataset description is already post-FFT magnitude/power data, no
phase), so faithfully reproducing the real small-FFT signal isn't currently possible.
Block max-pooling the native-resolution RD map is a best-effort emulation given that
constraint, not validated hardware data. **Every dice/precision/recall number below is
a measurement against this emulated dataset, not a deployment accuracy estimate**,
until real hardware output or the SoC's exact FFT/windowing spec becomes available.

**To regenerate the 8x64 dataset from scratch** (needed on a fresh machine -- it isn't
committed, see "Where things live" below): run `kuralscw_processing.py` on the raw
`.mat` data (or, if you already have a native-resolution extracted dataset,
`resize_extracted_dataset.py` is cheaper and numerically identical):
```bash
python -m kurals.dataset_process.resize_extracted_dataset \
    --src /path/to/native/KuRALS_CW --dst /path/to/KuRALS_CW_8x64/KuRALS_CW \
    --doppler-bins 8 --range-bins 64
```
Then point `config.ini` at it (`kurals/utils/set_paths.py --cwr .../KuRALS_CW_8x64/KuRALS_CW ...`).

**Current best/deployable result**: `kuralsnet_npu_seg` with `bottleneck_kernel_size=3`
and `dropout_rate=0.1` (both new constructor params, see the model file), initialized
by finetuning from the native-resolution checkpoint above (partial-transfer for the
kernel-size-mismatched bottleneck layers via `kurals/expand_kernel3.py`, since
`bott_c/d1/d2`'s depthwise weights don't shape-match between kernel_size=5 and 3) --
val dice 0.3757 / test dice 0.3635
(`kurals/config_files/kuralsnet_npu_seg_8x64_kernel3_finetune_dropout.json`,
`norm_type: "tvt"` -- fixed global min/max stats, not per-batch). This has survived
every follow-up attempt below; nothing has beaten it yet.

Root cause diagnosed directly (not assumed): block max-pooling to a small buffer
preserves a target's own peak value correctly (~94% of sampled targets), but also
inflates background bins via order statistics (max of ~500 native pixels per output
cell), collapsing measured target-vs-background contrast from ~19.4x (native) to ~6.7x
(pooled). What's been tried (see the corresponding
`kurals/config_files/kuralsnet_npu_seg_8x64_*.json` files' own `comments` for full
detail): architecture changes (width/depth/kernel size) mostly didn't help alone;
finetuning from the native checkpoint was the single biggest lever (0.3070 -> 0.3684
val dice, same architecture); dropout is the only regularizer that helped when combined
with finetuning+kernel3 -- weight decay (at two calibrated strengths), data
augmentation, longer/stretched LR schedules, label smoothing, stacking multiple
regularizers together, exhaustive (vs random) flip coverage, and a CFAR-based local
contrast normalization all underperformed plain finetuning when tested. TTA
(hflip/vflip/both, averaged) helps the plain finetune checkpoint (0.3684 -> 0.3820 val)
but hurt the kernel3+finetune+dropout checkpoint (0.3757 -> 0.3135) -- not a universal
win, checkpoint-dependent.

### CFAR-based contrast preservation (ruled out at every alpha tried)

Direct attempt at fixing the diagnosed root cause: `kurals/dataset_process/cfar_detect_and_damp_dataset.py`
runs a real CA-CFAR detector (`kurals/models/cfar.py`'s `CFAR2D_Parallel`, ref_cells=1,
guard_cells=1) at native resolution, flattens every non-detected pixel to a fixed
background baseline (median native pixel value) *before* pooling, so max-pooling can no
longer inflate background bins (no variance left to exploit). Swept `alpha` (detection
strictness) at 4.0/5.0/7.0 -- val dice 0.3357/0.3749/0.3035 respectively. **None beat
plain max-pooling**, and the relationship with alpha is not monotonic (5.0 is a local
peak, not an endpoint) -- don't assume "higher alpha always better" and re-sweep blindly.

A precision/recall check directly against ground truth (`scipy.ndimage.label` per-target
connected components, sampled ~120 real targets) explains why: CFAR's pixel precision
only reaches 0.96%/3.4%/10.4%/25.3%/46.3%/65.0% at alpha=4/5/6/7/8/10, while per-target
recall drops 99.2%/92.5%/85.0%/80.0%/74.2%/55.8% over the same range -- CFAR never gets
close to the oracle's effective ~100% precision (see next section) at any usable recall.
This is a **detector-capability ceiling, not a threshold-tuning problem**: a per-pixel
local-average threshold test structurally cannot distinguish an isolated noise spike from
an isolated real target using only local intensity. A from-scratch Gaussian-peak-finding
detector would face the identical limitation (same information -- single-frame local
intensity -- different shape of decision boundary). The one lever that's information-wise
different and untested: multi-frame/temporal consistency (real targets persist
frame-to-frame, noise doesn't) -- `kurals/utils/tracker.py`'s SORT-style tracker already
exists in this repo but isn't used anywhere in the preprocessing pipeline.

### Oracle ceiling: real headroom exists, but doesn't transfer to deployment

`kurals/dataset_process/oracle_detect_and_damp_dataset.py` replaces CFAR's detector with
ground-truth dense annotation masks (channels 1-3 of `annotations/dense/*/range_doppler.npy`)
to decide what survives flattening -- not deployable (no ground truth at inference), but a
clean ceiling check for the entire "detect real peaks, flatten the rest" approach family
(CFAR, Gaussian, or anything else). Trained *and evaluated* on this privileged distribution
throughout: **val dice 0.4821 / test dice 0.4379** -- decisively beats plain pooling,
confirming real headroom exists and CFAR's failure is a detector-quality problem, not a
dead-end theory.

**But this number doesn't represent anything deployable**, and testing it properly showed
it's worse than a "close tie" -- it's an active overfitting trap. `kurals/dataset_process/build_mixed_train_dataset.py`
builds a dataset where *only* the Train split (2064 frames) gets the oracle-damped
transform (legitimate use of labels available during training) while Validation/Test stay
on the realistic plain-max-pooled distribution the model will actually see. Result
(`kuralsnet_npu_seg_8x64_oracle_train_only_kernel3_finetune_dropout.json`): val/test dice
peaks at **0.3772 (epoch 100)** -- a marginal, likely-noise-level edge over 0.3757 -- then
**collapses to 0.2315 by epoch 300** as training continues. The model learns the shortcut
"anything above the (near-zero-variance) oracle background = target," which is
catastrophic against real noisy backgrounds where almost every pixel is technically
nonzero. Best-val-checkpoint selection is the only reason the reported number isn't a
clear loss -- it happened to catch the peak before the pathology took hold. **Don't build
training data with a privileged/label-requiring transform and assume more training-on-it
is safe just because early epochs look fine** -- check the full trajectory, not just the
best epoch, especially when train and eval distributions differ.

### Bottleneck width sweep (confounded -- read before reusing bc≠64 checkpoints)

MAC headroom at 8x64 is enormous (`kurals/expand_bottleneck_width.py`'s docstring +
below), so unlike native resolution's `wide96` failure (tight 68%-used budget), a width
sweep here isn't budget-constrained. Swept `bottleneck_ch` in {32, 96, 128, 192} (vs. the
default 64), each initialized via `kurals/expand_bottleneck_width.py` (generalizes
`expand_kernel3.py`'s approach: copy every state_dict key whose shape matches the native
bc=64 source, random-init the rest). Val dice: 32=0.3062, 96=0.3028, 128=0.3090,
192=0.3353 -- **none beat the bc=64 anchor's 0.3757**.

**Confound, discovered after training, not before**: bc=64 is the *only* width that
exactly matches the native source checkpoint's own bottleneck width, so it's the only
one that gets full pretrained-weight transfer for `down2`/`bott_c`/`bott_d1`/`bott_d2`/`upconv_b`
(41 of 230 state_dict keys are random-initialized for every other width). Given
finetuning-from-native is independently documented as the single biggest lever for this
whole branch, this sweep mostly measured "full transfer vs. partial transfer," not width
alone -- it is **not** a clean capacity ablation. What partial evidence survives the
confound: within the four non-64 widths there's a rough *upward* trend with width
(32 worst -> 192 best), unlike `wide96`'s clean negative at native resolution *despite*
having full transfer there -- so capacity doesn't look actively harmful at 8x64, just
dominated by the transfer-loss effect. A clean re-run would need either width-matched
native sources (expensive) or all widths trained from scratch with no finetune init
(cheaper, but answers a different question -- raw capacity value, not best-recipe-per-width).

### MAC headroom (measured via `thop`, not the same accounting as the native model's `638.12M/640M` figure quoted above -- that one's provenance/methodology isn't preserved anywhere in this repo)

- Native (124x2048), default kernel5/no-dropout: **436.27M / 640M MACs (68.2%)** --
  203.73M headroom (31.8%), not enough for even a single extra TTA pass (2x -> 872M,
  over budget) or a 2-model ensemble.
- 8x64, kernel3+dropout=0.1: **0.83M / 640M MACs (0.13%)** -- 639.17M headroom (99.9%
  free). Input is ~496x smaller (512 vs 253,952 pixels), so conv cost scales down
  proportionally. TTA or even a large ensemble is trivially affordable here; MAC budget
  is a non-issue at this resolution. Affordability doesn't imply benefit, though -- TTA
  is already measured to hurt the exact checkpoint above (0.3757 -> 0.3135).

### Normalization methodology note

The established anchor config uses `norm_type: "tvt"` (fixed, precomputed global
min/max from `cwr_rd_stats_all.json`). Some configs from this branch's CFAR/oracle
experiments instead used `norm_type: "local"` (per-batch adaptive min/max) -- this may
have been a deliberate necessity for CFAR/oracle-damped data specifically (their
flattened background could fall outside `tvt`'s precomputed range and get badly
mis-scaled), but the reasoning wasn't preserved before a later session picked this back
up, so treat it as unconfirmed. The width sweep above deliberately used `tvt` throughout
to match the anchor exactly. **Check each config's own `norm_type` before comparing dice
numbers across configs in this branch** -- it's a real, previously-unflagged confound
axis, not just a stylistic difference.

## Where things live

- **Checkpoints and per-run metrics are NOT in this repo** (`.gitignore` excludes `*.pt`) -- they live under the `logs` path from `kurals/config_files/config.ini` (set via `kurals/utils/set_paths.py`), one directory per run, with `val_doppler_model.pt`/`test_doppler_model.pt`/`latest_model.pt` + matching `*_results.json` (full per-class confusion matrices). Kept checkpoints are pruned to: the native best (0.5144, `..._2026-08-05-21:17:46`), the `kuralsnet` baseline (only run that exists), and the current 8x64 best/anchor (0.3757/0.3635, kernel3+finetune+dropout) plus its width-sweep family (bc=32/96/128/192) for reference -- everything else this branch tried has its config's `comments` field as the record, but no surviving weights (see git log / this file's history if a specific old run's numbers are needed).
- **Datasets are also NOT in this repo** (`.gitignore` excludes `*.npy`) -- regenerate the 8x64 one with `resize_extracted_dataset.py` (see the SoC section above) from a native-resolution extracted dataset.
- **Dataset preprocessing**: `kurals/dataset_process/kuralscw_processing.py` (raw `.mat` -> native-resolution extracted dataset), `resize_extracted_dataset.py` (native -> any smaller RD buffer via block max-pooling), `cfar_detect_and_damp_dataset.py` / `oracle_detect_and_damp_dataset.py` (contrast-preservation attempts, both ruled out as deployable -- see SoC section), `cfar_normalize_dataset.py` (continuous SCR normalization, also underperformed), `build_mixed_train_dataset.py` (train-only privileged-transform dataset builder).
- **Checkpoint adaptation for finetuning across a shape mismatch**: `expand_kernel3.py` (kernel_size 5->3), `expand_stem_channels.py` (n_frames 1->N), `expand_bottleneck_width.py` (bottleneck_ch 64->N) -- all follow the same pattern: build a fresh target-shaped model, copy every state_dict key whose shape matches the source checkpoint, random-init the rest.
- **Tracker**: `kurals/utils/tracker.py` (SORT-style: `scipy.ndimage.label` centroids, constant-velocity Kalman filter, Hungarian assignment via `scipy.optimize.linear_sum_assignment`), demoed by `kurals/demo_tracker.py`. Visually validated against the native-resolution checkpoint (0.5144) on the two-drone Test sequence: correctly tracks both targets simultaneously with proper Hungarian association when both are visible, but shows real ID churn (many short-lived candidate tracks spawned and discarded before `confirm_hits` -- the underlying per-frame detector is fairly noisy) and drifts/drops tracks during stretches of missed detections rather than degrading gracefully (Kalman-predicted position visibly separates from the true target before the track is dropped) -- both consistent with the model's known ~56% UAV recall, not tracker logic bugs.
- **int8 export/deployment pipeline**: `export_int8.py` -> `dummy_ifm.py`/`quantize_input.py` -> `benchmark_int8_pipeline.py`, all targeting `kuralsnet_npu_seg`'s topology (ported from the retired CenterNet model's equivalent pipeline -- see `docs/USAGE.md`).
