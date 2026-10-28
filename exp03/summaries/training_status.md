# Exp03 training status (2026-10-26)

**All 102/102 runs are DONE; nothing is running; no training is required for the paper.** Per-run table: `training_status.csv`.

| stage | runs | done |
|---|---|---|
| Exp A: NPU-UNet (incl. U-Net 1.0x/1.5x for Exp C) | 27 | 27 |
| Exp B: co-designed FP32 (grid + finals, incl. 1.0x finals) | 34 | 34 |
| Exp C: KuRALS-Net baseline | 8 | 8 |
| FP32-continuation control (0.5x only) | 9 | 9 |
| QAT fine-tunes (LR search + finals, 0.5x and 1.0x) | 24 | 24 |

## Covered by finished runs (checklist items 2-3)
- Software baseline table at R64-D8: U-Net 0.5x/1.0x/1.5x, KuRALS-Net, proposed FP32, proposed QAT, for both the 0.5x (1.57 M MACs) and 1.0x (4.95 M MACs) designs. Values: `paper_extract_model_server.md`.
- mIoU vs resolution (D8/D16/D32): Exp A and B, FP32 and QAT.
- mIoU vs MACs: ready. mIoU vs measured NPU latency: needs board/NPU measurements, not obtainable on this server.
- Deployable int8 packages: `exp03/model_release_v01` (0.5x), `exp03/model_release_w1.0_v01` (1.0x).

## Optional, not required
| item | cost | value |
|---|---|---|
| FP32-continuation control for the 1.0x design | 3 runs, ~1 h | shows whether the 1.0x QAT gain is also not a training effect (true for 0.5x) |
| SegFormer-B0 at R64-D8 | new model code + 3 seeds | extra transformer reference row |
| BN-fold test of why QAT > FP32 | small | tests an untested explanation |
| Diagnose export (0.6152) vs trainer (0.5947) test-mIoU gap for the 1.0x release (seed 45) | analysis only | explains a number quoted with the release |

## Not training (checklist items 1, 4-8)
Architecture figures, SISO direct-write and MIMO-vs-K board measurements, SNR-recovery plot and power are hardware or plotting work.

## Resuming in the background
All launchers skip runs that have a `DONE` file, so relaunching only restarts unfinished work:
```bash
cd ~/xsrc/KuRALS
setsid nohup ~/.venvs/kurals_train/bin/python exp03/<script>.py > exp03/<script>.out 2>&1 < /dev/null &
```
`setsid nohup` keeps the job alive after the terminal closes. Scripts: `pipeline.py`, `run_fp32_control.py` (0.5x), `run_w10_finals.py` (1.0x). Monitor with `tail exp03/<script>.log`. Background waiters started inside a Claude session do not survive the session; the scripts do.
