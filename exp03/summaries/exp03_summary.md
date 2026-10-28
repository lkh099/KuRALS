# Exp03 summary (auto-generated; regenerate with `python exp03/report.py`)

## 1. Resolution-only reference (Experiment A)

Common model NPU-UNet 0.5x, ONE LR for all resolutions chosen on the tuning seed: LR = 0.001 (mean val mIoU over D8/D16/D32 per candidate: {'0.003': 0.5606115936558372, '0.002': 0.5403318543930835, '0.001': 0.5685606673050604, '0.0003': 0.4332806284965858}). Final statistics over seeds [43, 44, 45].

- R64-D8 (512 cells): test mIoU 0.5572 ± 0.0541, test Dice 0.6470 ± 0.0729, UAV IoU 0.6759 ± 0.0187 (n=3)
- R64-D16 (1024 cells): test mIoU 0.5299 ± 0.0508, test Dice 0.6275 ± 0.0598, UAV IoU 0.5947 ± 0.0940 (n=3)
- R64-D32 (2048 cells): test mIoU 0.5048 ± 0.0829, test Dice 0.5989 ± 0.1033, UAV IoU 0.5659 ± 0.0373 (n=3)

Supports statements about resolution itself (same architecture, optimiser, loss, schedule, augmentation, selection rule, evaluation code, batch).

## 2. Resolution-specific co-designed systems (Experiment B)

Maximise validation mIoU subject to peak NPU buffer <= 32 KB per bank (see docs/NPU_BUFFER_MODEL.md), tie within 0.01 mIoU -> fewer MACs. Identical search grid at every resolution (width x stem_stride x 2 LRs), chosen on the tuning seed; every candidate is in `codesign_search_log.csv`.

- D8: width 0.5, stem_stride 1, lr 0.000333333: test mIoU 0.5212 ± 0.0551, Dice 0.6119 ± 0.0823, UAV IoU 0.6658 ± 0.0178; 9587 params, 1.57 M MACs, peak NPU buffer 16384 B (n=3)
- D16: width 0.5, stem_stride 1, lr 0.001: test mIoU 0.4646 ± 0.0552, Dice 0.5502 ± 0.0599, UAV IoU 0.5132 ± 0.1647; 9587 params, 3.14 M MACs, peak NPU buffer 32768 B (n=3)
- D32: width 0.5, stem_stride 2, lr 0.000333333: test mIoU 0.2931 ± 0.0122, Dice 0.3273 ± 0.0204, UAV IoU 0.0701 ± 0.0286; 9587 params, 1.57 M MACs, peak NPU buffer 16384 B (n=3)
- Feasible candidates per resolution (of 12): {8: 10, 16: 8, 32: 4}. Fewer feasible options at higher resolution is a direct consequence of the buffer limit, not of a smaller search budget (the grid is identical; infeasible candidates are logged, not trained).

Supports statements about co-design; does NOT isolate resolution (architecture and training differ across rows).

## 3. R64-D8 model comparison (Experiment C)

- NPU-UNet 0.5x: test mIoU 0.5572 ± 0.0541, UAV IoU 0.6759 ± 0.0187, 30992 params, 3.30 M MACs, peak act 20480 B, NPU legal: not checked (baseline), latency: pending
- NPU-UNet 1.0x: test mIoU 0.5894 ± 0.0247, UAV IoU 0.6652 ± 0.0044, 123032 params, 13.08 M MACs, peak act 40960 B, NPU legal: not checked (baseline), latency: pending
- NPU-UNet 1.5x: test mIoU 0.5985 ± 0.0548, UAV IoU 0.6932 ± 0.0149, 276128 params, 29.36 M MACs, peak act 61440 B, NPU legal: not checked (baseline), latency: pending
- Proposed (D8 co-designed KuRALSNetNPUSeg): test mIoU 0.5212 ± 0.0551, UAV IoU 0.6658 ± 0.0178, 9587 params, 1.57 M MACs, peak act 49152 B, NPU legal: yes, latency: pending
- KuRALS-Net (algorithmic reference): test mIoU 0.5699 ± 0.0247, UAV IoU 0.7094 ± 0.0231, 1151460 params, 311.18 M MACs, peak act 131072 B, NPU legal: not checked (baseline), latency: N/A

Only the proposed model is required to be NPU-legal; baselines (U-Net, KuRALS-Net) are not legality-checked. Claims here are about accuracy-complexity operating points, not latency.

## 4. QAT (separate from FP32)

- D8: paired quantization delta (FP32 - QAT test mIoU) -0.0754 ± 0.0295 over 3 seeds; failed runs: []
- D16: paired quantization delta (FP32 - QAT test mIoU) -0.0430 ± 0.0274 over 3 seeds; failed runs: []
- D32: paired quantization delta (FP32 - QAT test mIoU) -0.0281 ± 0.0231 over 3 seeds; failed runs: []

QAT starts from the best-validation FP32 checkpoint of the same seed; QAT LR chosen on seed 42. Stability details: summaries/qat_stability.json.

## 5. NPU feasibility

Per-candidate buffer demand is in `codesign_search_log.csv` (peak_npu_buffer, feasible, rejection_reason). Definition of the 32 KB constraint: `docs/NPU_BUFFER_MODEL.md` (per-bank allocator watermark from the NPU8 cfg generator; weights stream from DRAM in the frame-reuse region; line buffers/accumulators excluded).

## 6. Pending: measured NPU latency

No NPU cycle/latency numbers exist yet. MACs, parameter count, peak NPU buffer and activation memory are implementation proxies only; this report makes no latency claim. When cycle counts arrive, add latency-vs-mIoU plots (do not replace the MAC plots) and rerun the Pareto analysis.

## Data preparation of R64-D8 / D16 / D32

- Source: KuRALS-CW native range-Doppler maps, 124 Doppler x 2048 range bins (Doppler axis = raw bins [2:126] of 128; the crop is identical for all three).
- Resolution reduction (`kurals/dataset_process/resize_extracted_dataset.py`, same code for all three): **block max-pooling** of the linear-magnitude map over contiguous, near-equal bin groups (`np.array_split`): range 2048 -> 64 = exactly 32 bins per cell; Doppler 124 -> {8,16,32} = groups of 15/16, 7/8, 3/4 bins (not an integer ratio, so group sizes differ by one).
- Labels: not resampled as images; every foreground cell of the native one-hot dense mask is remapped through the same bin lookup, background = complement (dilate_radius = 0, so no label dilation at any resolution).
- The inputs are produced **after** the native FFT by pooling, i.e. an emulation; the real SoC computes its own small FFT directly from raw samples (not reproducible here, no raw IQ). Accuracy differences between resolutions therefore include this emulation effect.
- Normalisation (same for all): DSP 5.3-bit log2 code, `q = clamp(log8(floor(v) >> 16) - 0, 0, 127)` (npu_riscv proxy bridge, LOG_SCALE = 0), fed with input scale 1.0; Doppler axis ifftshifted to natural FFT order for both input and labels.
- Augmentation: random horizontal/vertical flip (identical in every run).
- Mean foreground cells per frame per class (class 1 = UAV, 2 = pedestrian, 3 = vehicle; 8 of 9 sequences are pure-UAV), a label-density side effect of resolution: D8: [0.7, 0.42, 0.04], D16: [0.75, 0.44, 0.04], D32: [0.92, 0.61, 0.05]. Class 3 is nearly absent, so mIoU is capped by it at every resolution.
- Visualisation of one sample + labels at the three resolutions: `plots/preprocessing_d8_d16_d32.pdf`.

## Batch size
Physical batch 6, gradient accumulation 1, effective batch 6 for every model and resolution (config batch_size=6; the small models fit on the GPU at all three resolutions, so no accumulation was needed).

## Concurrency
```
2026-10-21 15:21:31 start: physical GPUs=1; training processes=12 -> 12.0 jobs/GPU (all on GPU 0, CUDA_VISIBLE_DEVICES=0); per-job VRAM MiB min/max=372/476; total used=5280 MiB. Wall-clock of concurrent runs is NOT a benchmark; latency must be measured in isolation.
2026-10-21 15:36:37 wave 1 queued: physical GPUs=1; training processes=0 -> 0.0 jobs/GPU (all on GPU 0, CUDA_VISIBLE_DEVICES=0); per-job VRAM MiB min/max=0/0; total used=0 MiB. Wall-clock of concurrent runs is NOT a benchmark; latency must be measured in isolation.
2026-10-21 17:27:59 wave 2 queued: physical GPUs=1; training processes=0 -> 0.0 jobs/GPU (all on GPU 0, CUDA_VISIBLE_DEVICES=0); per-job VRAM MiB min/max=0/0; total used=0 MiB. Wall-clock of concurrent runs is NOT a benchmark; latency must be measured in isolation.
2026-10-21 19:53:29 end: physical GPUs=1; training processes=0 -> 0.0 jobs/GPU (all on GPU 0, CUDA_VISIBLE_DEVICES=0); per-job VRAM MiB min/max=0/0; total used=0 MiB. Wall-clock of concurrent runs is NOT a benchmark; latency must be measured in isolation.
```

## Methodological notes
- Seed 42 is the hyper-parameter / model-selection seed (LR search, Exp B candidate choice, QAT-LR choice). Final mean ± std use seeds [43, 44, 45] only; seed-42 rows are labelled "tuning" in the CSVs.
- Checkpoint = best validation mIoU epoch; test evaluated once at that checkpoint (the trainer also stores a best-test checkpoint, which is never used). Final-epoch numbers are never reported.
- FP32 and QAT are separate stages; QAT starts from the best FP32 checkpoint of the same seed and is paired per seed.
- The 32 KB per-bank NPU constraint is applied during the co-design search (candidates over the limit are logged and not trained).
- Unequal-budget caveat: tuning budget per resolution is the same grid; feasible candidates differ by resolution because of the constraint. Exp A uses 200 epochs; the best-val epoch is logged per run (runs whose best epoch is near 200 may be under-trained).
- Several concurrent runs shared one GPU; wall-clock is not a benchmark. Measured NPU latency is still pending, so no latency claim is made.
- An earlier batch at lr 1e-4 / 100 epochs was discarded (all-background plateau) and is kept only in exp03/runs_invalid_lr1e-4_ep100 and runs_100ep_lrsearch.
