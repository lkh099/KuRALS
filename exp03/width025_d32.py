"""Exp B width-grid extension at R64-D32: width_mult 0.25 with stem_stride 1 (full-resolution head).

Why: D32 is the only resolution whose Exp B pick uses stem_stride=2, so its head predicts at /2 and
nearest-upsamples. That caps per-class IoU at 1/4 for a 1-pixel target (mIoU <= ~0.4375 with 4
classes), and the nine existing D32 QAT runs sit at 0.3026 +- 0.011 -- between the trivial
all-background 0.2497 and that ceiling -- while D8/D16 (both stem_stride=1) reach ~0.58/~0.60.

stem_stride=1 was not a choice that lost: codesign_search_log.csv rejected `R64-D32, wm0.5, ss1` as
infeasible (peak 65536 B > 32768 B). But the search grid was WIDTHS = [0.5, 1.0, 1.5], and
kurals/npu_check.py reports `R64-D32, wm0.25, ss1` at 2,246,656 MACs and peak 32768 B -- feasible,
exactly at the buffer budget, and never evaluated because it is below the grid's floor.

So this tests the one untried configuration that can give D32 a full-resolution output head within
the 32 KB buffer. It trades capacity for output resolution (3323 params vs 9587, a 2.9x cut), which
is a real bet, not a free win: it could land below the incumbent.

Protocol is Exp B's, so the result is comparable to the pre-registered picks:
  S1  FP32 LR search on the tuning seed over lrs_B = [lr_A, lr_A/3] = [1e-3, 3.333e-4], 200 epochs
  S2  FP32 finals, seeds 43-45, at the winning LR
  S3  QAT LR search on the tuning seed's FP32 checkpoint over QAT_LRS, 100 epochs
  S4  QAT finals, seeds 43-45, at the winning QAT LR, each paired to its own seed's FP32 checkpoint
Selection is always highest VALIDATION mIoU; test is read once at that checkpoint.

Nothing existing is touched: the BW_/QW_ tag prefixes keep these runs in their own directories, so
the pre-registered Exp B pick for D32 stays on record as-is.

  cd ~/xsrc/KuRALS && setsid nohup ~/.venvs/kurals_train/bin/python exp03/width025_d32.py \
      > exp03/width025_d32.out 2>&1 < /dev/null &

Resumable: runs with a DONE file are skipped, so re-running only restarts unfinished work.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from pipeline import (BUF, EPOCHS, FINAL_SEEDS, QAT_EPOCHS, QAT_FREEZE_ITER, QAT_LRS,  # noqa: E402
                      TUNE_SEED, ckpt, job, log, profile, result, run_all, run_dir)

RES = 32
WM = 0.25
SS = 1
FP32_LRS = [1e-3, 1e-3 / 3]          # lrs_B = [lr_A, lr_A / 3]; lr_A = 1e-3 from pipeline_state.json
INCUMBENT = 'Q_kuralsnet_npu_seg_wm0.5_ss2_R64-D32_qlr0.0001_s44'   # the shipped model_release_d32_v01 source


def fp32_tag(seed, lr):
    return f'BW_kuralsnet_npu_seg_wm{WM}_ss{SS}_R64-D{RES}_lr{lr:g}_s{seed}'


def qat_tag(seed, qlr):
    return f'QW_kuralsnet_npu_seg_wm{WM}_ss{SS}_R64-D{RES}_qlr{qlr:g}_s{seed}'


def pick(tags):
    """Highest validation mIoU among finished runs. Returns (key, result) or (None, None)."""
    scored = [(k, r) for k, r in ((k, result(t)) for k, t in tags.items()) if r]
    if not scored:
        return None, None
    return max(scored, key=lambda kr: kr[1]['val_miou'])


def check_feasible():
    p = profile('kuralsnet_npu_seg', WM, SS, RES)
    log(f'profile wm{WM} ss{SS} R64-D{RES}: params={p["params"]} macs={p["macs"]:,} '
        f'peak_npu_buffer={p["peak_npu_buffer"]} B (budget {BUF} B) feasible={p["feasible"]} {p["rejection"] or ""}')
    if not p['feasible']:
        log('ABORT: configuration is not NPU-feasible; nothing launched.')
        sys.exit(1)
    return p


def report(profile_row):
    out = dict(design=dict(resolution=f'R64-D{RES}', width_mult=WM, stem_stride=SS, encoder_depth=1,
                           params=profile_row['params'], macs=profile_row['macs'],
                           peak_npu_buffer=profile_row['peak_npu_buffer'], buffer_budget=BUF),
               fp32_lr_search={}, fp32_finals={}, qat_lr_search={}, qat_finals={}, incumbent=None)

    for lr in FP32_LRS:
        r = result(fp32_tag(TUNE_SEED, lr))
        out['fp32_lr_search'][f'{lr:g}'] = None if not r else round(r['val_miou'], 4)
    fp32_lr, _ = pick({lr: fp32_tag(TUNE_SEED, lr) for lr in FP32_LRS})
    out['fp32_lr_chosen'] = None if fp32_lr is None else f'{fp32_lr:g}'

    if fp32_lr is not None:
        for s in FINAL_SEEDS:
            r = result(fp32_tag(s, fp32_lr))
            out['fp32_finals'][str(s)] = None if not r else dict(
                val_miou=round(r['val_miou'], 4), test_miou=round(r['test_miou'], 4), best_epoch=r['best_epoch'])

    for qlr in QAT_LRS:
        r = result(qat_tag(TUNE_SEED, qlr))
        out['qat_lr_search'][f'{qlr:g}'] = None if not r else round(r['val_miou'], 4)
    qat_lr, _ = pick({q: qat_tag(TUNE_SEED, q) for q in QAT_LRS})
    out['qat_lr_chosen'] = None if qat_lr is None else f'{qat_lr:g}'

    vals, tests = [], []
    if qat_lr is not None:
        for s in FINAL_SEEDS:
            r = result(qat_tag(s, qat_lr))
            out['qat_finals'][str(s)] = None if not r else dict(
                val_miou=round(r['val_miou'], 4), test_miou=round(r['test_miou'], 4),
                best_epoch=r['best_epoch'], test_iou_by_class=[round(x, 4) for x in r['test_iou_by_class']])
            if r:
                vals.append(r['val_miou']); tests.append(r['test_miou'])
    if vals:
        mean = lambda v: sum(v) / len(v)
        sd = lambda v: (sum((x - mean(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        out['qat_final_summary'] = dict(n=len(vals), val_mean=round(mean(vals), 4), val_sd=round(sd(vals), 4),
                                        test_mean=round(mean(tests), 4), test_sd=round(sd(tests), 4))

    inc = result(INCUMBENT)
    if inc:
        out['incumbent'] = dict(tag=INCUMBENT, val_miou=round(inc['val_miou'], 4), test_miou=round(inc['test_miou'], 4),
                                test_iou_by_class=[round(x, 4) for x in inc['test_iou_by_class']],
                                note='wm0.5 ss2, source of model_release_d32_v01; single best-val run, '
                                     'not a seed mean -- its 0.3475 is the max of a 0.3026 +- 0.011 band')

    path = f'{ROOT}/summaries/width025_d32.json'
    json.dump(out, open(path, 'w'), indent=1)
    log('wrote', path)
    log('\n' + json.dumps(out, indent=1))
    return out


def main():
    p = check_feasible()

    log('S1 FP32 LR search (seed %d, %d epochs): %s' % (TUNE_SEED, EPOCHS, [f'{lr:g}' for lr in FP32_LRS]))
    run_all([job('kuralsnet_npu_seg', RES, TUNE_SEED, lr, wm=WM, ss=SS, exp='BW', epochs=EPOCHS,
                 tag=fp32_tag(TUNE_SEED, lr)) for lr in FP32_LRS])
    fp32_lr, best = pick({lr: fp32_tag(TUNE_SEED, lr) for lr in FP32_LRS})
    if fp32_lr is None:
        log('ABORT: no FP32 LR-search run produced a result.'); report(p); sys.exit(1)
    log(f'FP32 LR = {fp32_lr:g} (val mIoU {best["val_miou"]:.4f})')

    log('S2 FP32 finals (seeds %s)' % FINAL_SEEDS)
    run_all([job('kuralsnet_npu_seg', RES, s, fp32_lr, wm=WM, ss=SS, exp='BW', epochs=EPOCHS,
                 tag=fp32_tag(s, fp32_lr)) for s in FINAL_SEEDS])

    tune_src = fp32_tag(TUNE_SEED, fp32_lr)
    if not os.path.exists(run_dir(tune_src) + '/DONE'):
        log('ABORT: tuning-seed FP32 parent missing, cannot run the QAT LR search:', tune_src)
        report(p); sys.exit(1)

    log('S3 QAT LR search (seed %d, %d epochs): %s' % (TUNE_SEED, QAT_EPOCHS, [f'{q:g}' for q in QAT_LRS]))
    run_all([job('kuralsnet_npu_seg', RES, TUNE_SEED, qlr, wm=WM, ss=SS, stage='qat', exp='QW',
                 epochs=QAT_EPOCHS, finetune=ckpt(tune_src),
                 qat=dict(quant_warmup_iters=0, quant_freeze_iters=QAT_FREEZE_ITER),
                 tag=qat_tag(TUNE_SEED, qlr)) for qlr in QAT_LRS])
    qat_lr, qbest = pick({q: qat_tag(TUNE_SEED, q) for q in QAT_LRS})
    if qat_lr is None:
        log('ABORT: no QAT LR-search run produced a result.'); report(p); sys.exit(1)
    log(f'QAT LR = {qat_lr:g} (val mIoU {qbest["val_miou"]:.4f})')

    log('S4 QAT finals (seeds %s, paired to each seed\'s own FP32 checkpoint)' % FINAL_SEEDS)
    jobs = []
    for s in FINAL_SEEDS:
        src = fp32_tag(s, fp32_lr)
        if not os.path.exists(run_dir(src) + '/DONE'):
            log('SKIP: FP32 parent not done:', src); continue
        jobs.append(job('kuralsnet_npu_seg', RES, s, qat_lr, wm=WM, ss=SS, stage='qat', exp='QW',
                        epochs=QAT_EPOCHS, finetune=ckpt(src),
                        qat=dict(quant_warmup_iters=0, quant_freeze_iters=QAT_FREEZE_ITER),
                        tag=qat_tag(s, qat_lr)))
    failed = run_all(jobs)
    log('WIDTH-0.25 D32 COMPLETE' + (f' with failures: {failed}' if failed else ''))
    report(p)


if __name__ == '__main__':
    main()
