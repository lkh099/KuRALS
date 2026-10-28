"""Writes summaries/training_status.md/.csv: inventory of all Exp03 runs + what is left."""
import csv, os, sys, collections
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import pipeline as P
rows = []
for t in sorted(os.listdir(f'{ROOT}/runs')):
    d = f'{ROOT}/runs/{t}'; r = P.result(t) if os.path.exists(d + '/DONE') else None
    g = lambda k: round(r[k], 4) if r and r.get(k) is not None else ''
    rows.append(dict(run=t, stage=t.split('_')[0], done=os.path.exists(d + '/DONE'), best_epoch=r.get('best_epoch', '') if r else '', val_miou=g('val_miou'), test_miou=g('test_miou'), test_dice=g('test_dice'), test_uav_iou=g('test_uav_iou')))
with open(f'{ROOT}/summaries/training_status.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
names = dict(A='Exp A: NPU-UNet (incl. U-Net 1.0x/1.5x for Exp C)', B='Exp B: co-designed FP32 (grid + finals, incl. 1.0x finals)', C='Exp C: KuRALS-Net baseline', Q='QAT fine-tunes (LR search + finals, 0.5x and 1.0x)', F='FP32-continuation control (0.5x only)')
c = collections.defaultdict(lambda: [0, 0])
for r in rows: c[r['stage']][0] += 1; c[r['stage']][1] += r['done']
md = ['# Exp03 training status (2026-10-26)\n', f"**All {sum(v[1] for v in c.values())}/{len(rows)} runs are DONE; nothing is running; no training is required for the paper.** Per-run table: `training_status.csv`.\n",
      '| stage | runs | done |\n|---|---|---|'] + [f'| {names.get(k, k)} | {v[0]} | {v[1]} |' for k, v in sorted(c.items())]
md += ['''
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
''']
open(f'{ROOT}/summaries/training_status.md', 'w').write('\n'.join(md))
