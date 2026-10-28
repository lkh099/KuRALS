# Design-point decision (step 4 of platform_model_server.md)

```text
FINAL_RD_RESOLUTION  = R64-D8
FINAL_MODEL_VARIANT  = KuRALSNetNPUSeg, width 0.5x, stem_stride 1, encoder_depth 1, bottleneck kernel 3, eltwise_act linear
                       (9,587 params, 1.57 M MACs, peak NPU buffer 16,384 B, 11,568 B weights+scale+bias)
PRECISION            = int8 QAT (fine-tuned from the best-val FP32 checkpoint)
STATUS               = provisional: measured NPU latency is still missing (see "What would change this")
```

All numbers are from the Exp03 pipeline: best-validation checkpoints, test evaluated once, final statistics over seeds 43/44/45 (seed 42 only tunes). All std values are sample std (ddof=1) over seeds 43/44/45 (exp03_summary.md uses the population std, which is 0.816x smaller). Source tables: `codesign_resolution_systems.csv`, `resolution_reference_unet_aggregate.csv`, `qat_resolution_comparison.csv`, `codesign_search_log.csv`.

## Evidence

| | R64-D8 | R64-D16 | R64-D32 |
|---|---|---|---|
| A. Same U-Net (0.5x), FP32 test mIoU | 0.557 ± 0.066 | 0.530 ± 0.062 | 0.505 ± 0.102 |
| B. Co-designed, FP32 test mIoU | 0.521 ± 0.068 | 0.465 ± 0.068 | 0.293 ± 0.015 |
| B. Co-designed, int8 QAT test mIoU | 0.597 ± 0.060 | 0.508 ± 0.093 | 0.321 ± 0.031 |
| B. FP32 continued 100 more epochs (control) | 0.518 ± 0.069 | 0.466 ± 0.064 | 0.275 ± 0.016 |
| Chosen B config | w0.5, stride 1 | w0.5, stride 1 | w0.5, stride 2 |
| MACs | 1.57 M | 3.14 M | 1.57 M |
| Peak NPU buffer (limit 32,768 B) | 16,384 B | 32,768 B (at the limit) | 16,384 B |
| Feasible candidates in the identical 12-point grid | 10 | 8 | 4 |
| DSP -> NPU input per frame (int8) | 512 B | 1,024 B | 2,048 B |

## Reasoning

1. **Accuracy does not favour a larger resolution.** With the architecture held fixed (A), accuracy decreases from D8 to D32, but the seed spread (std 0.06-0.10) is as large as the gaps, so the claim is "no evidence that D16/D32 are better", not "D8 is better". Under co-design (B), D8 is highest in both FP32 and QAT and D32 is far lower (0.29).
2. **Cost scales with resolution.** D16 doubles the MACs, sits exactly on the 32 KB buffer limit (no headroom), and doubles the DSP-to-NPU data. D32 only fits at all with stem_stride 2, which discards the resolution it paid for; that is why its accuracy collapses.
3. **Model size within D8.** In the D8 grid, w0.5 and w1.0 (stride 1) are tied on tuning-seed validation mIoU (0.5772 vs 0.5729, inside the 0.01 tie band), so the pre-registered rule picks the cheaper one: w0.5 (1.57 M vs 4.95 M MACs, 16 KB vs 32 KB buffer). w1.5 does not fit the buffer at stride 1; every stride-2 variant loses about 0.3 mIoU (0.26-0.29).
4. **Pareto position.** At D8 the chosen model costs 1.57 M MACs; NPU-UNet 0.5x/1.0x/1.5x cost 3.3/13.1/29.4 M MACs for 0.557/0.589/0.599 FP32 test mIoU, and the original KuRALS-Net costs 311 M MACs for 0.570. The chosen model is the cheapest point and about 0.04-0.08 mIoU below the U-Nets at FP32; after QAT it is at 0.597. The U-Nets were not QAT-trained, so the QAT row is **not** a like-for-like comparison with them. The proposed model is the cheapest point in MACs but is **not** Pareto-optimal on memory: NPU-UNet 0.5x and 1.0x have lower-or-equal peak NPU buffer (8 KB, 16 KB vs 16 KB) and higher mIoU. Its advantage is compute (2.1x fewer MACs than U-Net 0.5x), not buffer size.

## Caveats

- **Seed noise is large.** Seed 43 is a weak FP32 run at every resolution (best-val epoch 64 at D8). The D8-vs-D16 QAT gap (0.597 vs 0.508) is within about one standard deviation.
- **QAT mIoU above FP32 is not a measure of quantization cost, and it is not an extra-training effect either.** The matched FP32-continuation control (same start, LR, epochs, seeds) changes mIoU by -0.004 / +0.001 / -0.018 (D8/D16/D32), while QAT is 0.04-0.08 above that control; the cause is untested (a candidate is BN: QAT folds running statistics in every forward, FP32 trains on noisy batch-6 statistics). Treat the FP32 rows as a reference that may under-represent the architecture.
- **Resolution reduction is an emulation.** D8/D16/D32 inputs are block max-pooled from the native map; the real SoC computes a small FFT from raw samples. The resolution conclusions are about this emulation.
- **Class 3 is almost absent** (about 0.04 cells per frame), which caps and destabilises mIoU at every resolution.
- **The U-Net comparison rows are not legality-checked** (baselines are exempt); only the chosen model is NPU-legal (npu_check ok at 32 KB buffers, CUTPOINT 3).

## What would change this

The spec's decision rule is system-level, and two inputs are missing: measured NPU latency and FPGA/ASIC cost. If D16 turned out to cost little extra latency and the 32 KB headroom problem were solved (for example by a different stem), its accuracy would still need to beat D8 by more than the seed spread to be worth doubling the DSP-to-NPU traffic; today it does not.
