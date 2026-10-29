"""Diagnostic: is the background-purity deficit specific to D32 seed 44, or general?

Stage P ended in the frozen stop condition (summaries/resolution_fixed_w1_convergence.md). It also
showed the original "degenerate" class held two separable failures: plateau non-escape / foreground
death (budget- and LR-limited, fixed by A1 and A2) and a background-purity deficit (train loss ~0.0120,
background IoU ~0.984 vs ~0.9996) that was invariant to every LR and budget tried and cost ~0.001 mIoU.

Only D32 s44 ever showed failure 2, on 1 of 4 seeds at that resolution. This fills in arm A2
(lr 2e-3, 200 epochs) across D8/D16/D32 x seeds 42-45 so the rate of each failure type can be read per
resolution under ONE recipe -- which is what a corrected criterion has to be written against.

Diagnostic only. Dev seeds 42-45; fresh seeds 46/47/48 stay untouched and unused. No arm is locked and
nothing here may be quoted as an accuracy result.

  cd ~/xsrc/KuRALS && setsid nohup ~/.venvs/kurals_train/bin/python exp03/resolution_a2_probe.py \
      > exp03/resolution_a2_probe.out 2>&1 < /dev/null &
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from pipeline import log, run_all, run_dir  # noqa: E402
from resolution_convergence import DEV_SEEDS, RESOLUTIONS, convergence, fp32, tag  # noqa: E402

LR, EP = 2e-3, 200          # arm A2
BG_CLEAN = 0.9985           # observed floor of every converged run's background IoU, 36 runs
BG_DEFICIT = 0.9848         # observed ceiling of every non-converged run's


def val_classes(t):
    f = glob.glob(f'{run_dir(t)}/logs/KuRALS_CW/*/*/results/val_doppler_results.json')
    if not f:
        return None
    v = json.load(open(f[0]))['val_metrics']['range_doppler']
    return v['miou'], [round(x, 4) for x in v['miou_by_class']]


def classify(iou):
    """The stage-P taxonomy, applied descriptively. Classes 1 and 2 only: class 3 is 18-23 px in the
    split and was excluded a priori by the first pre-registration."""
    bg, c1, c2 = iou[0], iou[1], iou[2]
    fg_dead = min(c1, c2) < 0.05
    bg_deficit = bg < BG_CLEAN
    return ('foreground death + background deficit' if fg_dead and bg_deficit else
            'foreground death only' if fg_dead else
            'background deficit only' if bg_deficit else 'clean')


def main():
    jobs = [fp32(r, s, LR, EP) for r in RESOLUTIONS for s in DEV_SEEDS
            if not os.path.exists(run_dir(tag(r, s, LR, EP)) + '/DONE')]
    log(f'A2 probe: lr {LR:g}, {EP} epochs, D{RESOLUTIONS} x seeds {DEV_SEEDS}; '
        f'{len(jobs)} runs to launch, {len(RESOLUTIONS) * len(DEV_SEEDS) - len(jobs)} already done')
    run_all(jobs)

    out = {}
    for r in RESOLUTIONS:
        rows = []
        for s in DEV_SEEDS:
            t = tag(r, s, LR, EP)
            c, vc = convergence(t), val_classes(t)
            if not c or not vc:
                log(f'  D{r} s{s}: NO RESULT'); continue
            miou, iou = vc
            rows.append(dict(seed=s, med10=c['med10'], converged=c['converged'],
                             escape_epoch=c['escape_epoch'], val_miou=round(miou, 4),
                             val_iou_by_class=iou, failure=classify(iou)))
            log(f'  D{r} s{s}: med10={c["med10"]:.6f} esc={c["escape_epoch"]} val={miou:.4f} '
                f'bg={iou[0]:.4f} c1={iou[1]:.4f} c2={iou[2]:.4f} -> {rows[-1]["failure"]}')
        if rows:
            out[f'D{r}'] = dict(runs=rows, n=len(rows),
                                clean=sum(1 for x in rows if x['failure'] == 'clean'),
                                background_deficit=sum(1 for x in rows if 'background deficit' in x['failure']),
                                foreground_death=sum(1 for x in rows if 'foreground death' in x['failure']))

    rep = dict(purpose='diagnostic: per-resolution failure-type rate under one recipe (arm A2)',
               arm=dict(lr=LR, epochs=EP), seeds=DEV_SEEDS,
               taxonomy=dict(foreground_death='min(val IoU class1, class2) < 0.05',
                             background_deficit=f'val background IoU < {BG_CLEAN}',
                             note='descriptive, applied after stage P; NOT a pre-registered criterion'),
               diagnostic_only='dev seeds 42-45; fresh seeds 46/47/48 untouched; no accuracy claim',
               by_resolution=out)
    p = f'{ROOT}/summaries/resolution_fixed_w1_a2_probe.json'
    json.dump(rep, open(p, 'w'), indent=1, default=float)
    log('wrote', p)
    for k, v in out.items():
        log(f'{k}: {v["clean"]}/{v["n"]} clean, {v["background_deficit"]}/{v["n"]} background deficit, '
            f'{v["foreground_death"]}/{v["n"]} foreground death')


if __name__ == '__main__':
    main()
