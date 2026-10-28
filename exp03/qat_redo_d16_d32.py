"""Re-run the R64-D16 / R64-D32 int8 QAT fine-tunes at the two lower grid LRs.

Why: at the pipeline's chosen QAT LR (1e-4) the D16 final seeds 43 and 44 reached their best
validation mIoU at the FIRST evaluation (epoch 2 of 100) and never improved -- see
summaries/qat_stability.json (`val_miou_best_qat` == `val_miou_first_after_qat`). That is not a
collapse, so the pipeline's retry rule (best val < 0.8 x the FP32 checkpoint's val) never fired,
but no useful fine-tuning happened either. D32 seed 45 shows a milder version (best at epoch 8).

This re-runs seeds 43/44/45 at QAT LR 3e-5 and 1e-5 for both resolutions (12 runs). Nothing
existing is touched: every new run gets its own tag and directory, and the 1e-4 runs stay as they
are, so the comparison is additive and the original pre-registered result remains on record.

Selection stays the project rule: highest VALIDATION mIoU, test read once at that epoch. This
script does not pick a winner; `report_qat_redo()` prints the comparison.

  setsid nohup ~/.venvs/kurals_train/bin/python exp03/qat_redo_d16_d32.py \
      > exp03/qat_redo.out 2>&1 < /dev/null &

Resumable: runs with a DONE file are skipped, so re-running only restarts unfinished work.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from pipeline import (FINAL_SEEDS, QAT_EPOCHS, QAT_FREEZE_ITER, ckpt, job,  # noqa: E402
                      log, result, run_all, run_dir)

REDO_LRS = [3e-5, 1e-5]
# resolution -> (width_mult, stem_stride, FP32 parent LR) of the Exp B winner, from summaries/exp03_summary.md
DESIGNS = {16: (0.5, 1, 1e-3), 32: (0.5, 2, 3.33333e-4)}
BASELINE_QLR = 1e-4


def fp32_tag(res, s):
    wm, ss, lr = DESIGNS[res]
    return f'B_kuralsnet_npu_seg_wm{wm}_ss{ss}_R64-D{res}_lr{lr:g}_s{s}'


def qat_tag(res, s, qlr):
    wm, ss, _ = DESIGNS[res]
    return f'Q_kuralsnet_npu_seg_wm{wm}_ss{ss}_R64-D{res}_qlr{qlr:g}_s{s}'


def build_jobs():
    jobs = []
    for res, (wm, ss, _) in DESIGNS.items():
        for s in FINAL_SEEDS:
            src = fp32_tag(res, s)
            if not os.path.exists(run_dir(src) + '/DONE'):
                log('SKIP: FP32 parent not done:', src)
                continue
            for qlr in REDO_LRS:
                jobs.append(job('kuralsnet_npu_seg', res, s, qlr, wm=wm, ss=ss, stage='qat', exp='Q',
                                epochs=QAT_EPOCHS, finetune=ckpt(src),
                                qat=dict(quant_warmup_iters=0, quant_freeze_iters=QAT_FREEZE_ITER),
                                tag=qat_tag(res, s, qlr)))
    return jobs


def report_qat_redo():
    """Baseline (1e-4) vs the two redo LRs, per resolution and seed. Validation is the selection
    metric; `first-eval == best` marks the pathology this redo targets."""
    out = []
    for res in DESIGNS:
        for s in FINAL_SEEDS:
            for qlr in [BASELINE_QLR] + REDO_LRS:
                r = result(qat_tag(res, s, qlr))
                row = dict(resolution=f'R64-D{res}', seed=s, qat_lr=qlr,
                           val_miou=None if not r else round(r['val_miou'], 4),
                           test_miou=None if not r else round(r['test_miou'], 4),
                           best_epoch=None if not r else r['best_epoch'],
                           state='done' if r else ('running/queued' if not os.path.exists(
                               run_dir(qat_tag(res, s, qlr)) + '/DONE') else 'no result'))
                out.append(row)
    path = f'{ROOT}/summaries/qat_redo_d16_d32.json'
    json.dump(out, open(path, 'w'), indent=1)
    log('wrote', path)
    hdr = f"{'resolution':11} {'seed':>4} {'qat_lr':>8} {'val':>7} {'test':>7} {'epoch':>5}  state"
    log('\n' + hdr + '\n' + '\n'.join(
        f"{r['resolution']:11} {r['seed']:>4} {r['qat_lr']:>8.0e} "
        f"{(r['val_miou'] if r['val_miou'] is not None else float('nan')):>7.4f} "
        f"{(r['test_miou'] if r['test_miou'] is not None else float('nan')):>7.4f} "
        f"{str(r['best_epoch']):>5}  {r['state']}" for r in out))
    return out


if __name__ == '__main__':
    jobs = build_jobs()
    log(f'QAT redo: {len(jobs)} runs ({len(DESIGNS)} resolutions x {len(FINAL_SEEDS)} seeds x {len(REDO_LRS)} LRs)')
    for t, _ in jobs:
        log('  queued', t)
    failed = run_all(jobs)
    log('QAT REDO COMPLETE' + (f' with failures: {failed}' if failed else ''))
    report_qat_redo()
