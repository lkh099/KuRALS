"""Fixed-1.0x stride-1 RD-resolution study: budget/LR arm selection under a pre-registered
training-loss convergence criterion.

Protocol: summaries/resolution_fixed_w1_CONVERGENCE_PROTOCOL.md (frozen before any run here exists).
Background: summaries/resolution_fixed_w1_stability.md -- the LR stop condition this answers.

The stop condition had one mechanical cause: training sits on a loss plateau at ~0.0240 and either
escapes it to ~0.0004 or does not, escape time scales as 1/LR, and the 200-epoch budget is a hard
deadline. Escaping predicts the outcome on 24 of 24 existing runs with no overlap. So the lever is
the budget and the LR, and the criterion is measurable from the TRAINING loss alone -- which is why
acting on it is not conditioning on the reported metric.

  cd ~/xsrc/KuRALS && setsid nohup ~/.venvs/kurals_train/bin/python exp03/resolution_convergence.py \
      > exp03/resolution_convergence.out 2>&1 < /dev/null &

Resumable: runs with a DONE file are skipped, in-flight runs are adopted.
"""
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from pipeline import FINAL_SEEDS, TUNE_SEED, job, log, result, run_all, run_dir  # noqa: E402

MODEL, WM, SS = 'kuralsnet_npu_seg', 1.0, 1
RESOLUTIONS = [8, 16, 32]
DEV_SEEDS = [42, 43, 44, 45]
FRESH_SEEDS = [46, 47, 48]

# Frozen arm set, in the frozen preference order: minimal deviation from the registered recipe first.
# 2e-3/3e-3 are in pipeline.py's own LR_GRID = [3e-3, 2e-3, 1e-3, 3e-4], not new free parameters.
ARMS = [('A1', 1e-3, 400), ('A2', 2e-3, 200), ('A3', 3e-3, 200)]
RESCUE = [(16, 45), (32, 44)]       # the two non-converged runs at the registered recipe
SCALING = [(16, 42), (32, 42)]      # escape-epoch scaling, to test the mechanism

MED10 = 0.005                       # frozen convergence threshold on the training loss
TRAIN_LOSS = re.compile(r'\[Epoch (\d+)/\d+, iter \d+\]: Train loss ([\d.]+)')


def tag(res, seed, lr, ep):
    return f'BC_{MODEL}_wm{WM}_ss{SS}_R64-D{res}_lr{lr:g}_e{ep}_s{seed}'


def losses(t):
    """[(epoch, train_loss)] from a run's train.log, or [] if it has not started."""
    f = f'{run_dir(t)}/train.log'
    return [(int(e), float(v)) for e, v in TRAIN_LOSS.findall(open(f).read())] if os.path.exists(f) else []


def convergence(t):
    """The frozen criterion. med10 = median of the last 10 logged train losses; converged iff < 0.005."""
    L = losses(t)
    if len(L) < 10:
        return None
    med = statistics.median(v for _, v in L[-10:])
    esc = next((e for e, v in L if v < MED10), None)
    return dict(tag=t, med10=med, converged=med < MED10, escape_epoch=esc, n_loss_points=len(L))


def row(res, seed, lr, ep):
    """Convergence + accuracy for one run, or None if unfinished."""
    t = tag(res, seed, lr, ep)
    c, r = convergence(t), result(t)
    if not c or not r:
        return None
    return dict(res=res, seed=seed, lr=lr, epochs=ep, **{k: c[k] for k in ('med10', 'converged', 'escape_epoch')},
                val_miou=r['val_miou'], test_miou=r['test_miou'], best_epoch=r['best_epoch'],
                val_iou_by_class=val_by_class(t))


def val_by_class(t):
    """Per-class VALIDATION IoU. pipeline.result() only exposes the test split's."""
    import glob
    f = glob.glob(f'{run_dir(t)}/logs/KuRALS_CW/*/*/results/val_doppler_results.json')
    if not f:
        return None
    return [round(x, 4) for x in json.load(open(f[0]))['val_metrics']['range_doppler']['miou_by_class']]


def fp32(res, seed, lr, ep):
    return job(MODEL, res, seed, lr, wm=WM, ss=SS, exp='BC', epochs=ep, tag=tag(res, seed, lr, ep))


def stage_p():
    """Each arm on the two rescue runs and the two scaling runs. 12 runs, one pool."""
    jobs = [fp32(res, seed, lr, ep) for _, lr, ep in ARMS for res, seed in RESCUE + SCALING]
    log(f'STAGE P diagnostic: {len(jobs)} runs, arms {[(n, f"{lr:g}", ep) for n, lr, ep in ARMS]}, '
        f'rescue {RESCUE}, scaling {SCALING}')
    run_all(jobs)
    for name, lr, ep in ARMS:
        for res, seed in RESCUE + SCALING:
            c = convergence(tag(res, seed, lr, ep))
            kind = 'rescue ' if (res, seed) in RESCUE else 'scaling'
            log(f'  {name} lr{lr:g} e{ep} D{res} s{seed} {kind}: ' +
                ('NO RESULT' if not c else f'med10={c["med10"]:.6f} escape_epoch={c["escape_epoch"]} '
                                           f'{"CONVERGED" if c["converged"] else "NOT CONVERGED"}'))


def lock():
    """Frozen rule: the first arm in preference order that is converged on BOTH rescue runs."""
    for name, lr, ep in ARMS:
        c = [convergence(tag(res, seed, lr, ep)) for res, seed in RESCUE]
        if all(x and x['converged'] for x in c):
            log(f'LOCKED arm {name}: lr {lr:g}, {ep} epochs -- converged on both rescue runs')
            return name, lr, ep
        log(f'  {name} lr{lr:g} e{ep} rejected: rescue convergence '
            f'{[None if not x else x["converged"] for x in c]}')
    log('STOP CONDITION: no arm converged on both rescue runs. Not adding arms, not retuning the '
        'threshold, not running fresh seeds.')
    return None


def stage_v(lr, ep):
    """The locked arm on every resolution x dev seed: a 4-seed convergence rate under one recipe."""
    jobs = [fp32(res, s, lr, ep) for res in RESOLUTIONS for s in DEV_SEEDS]
    log(f'STAGE V convergence rate: lr {lr:g}, {ep} epochs, D{RESOLUTIONS} x seeds {DEV_SEEDS}')
    run_all(jobs)


def stage_f(lr, ep):
    """The locked arm on fresh seeds. The only stage whose mIoU may be quoted as the headline."""
    jobs = [fp32(res, s, lr, ep) for res in RESOLUTIONS for s in FRESH_SEEDS]
    log(f'STAGE F fresh final seeds {FRESH_SEEDS} at the locked recipe lr {lr:g}, {ep} epochs')
    run_all(jobs)


def rate(rows):
    n = len(rows)
    k = sum(1 for r in rows if r['converged'])
    return dict(n=n, converged=k, rate=None if not n else round(k / n, 4))


def report(locked):
    out = dict(protocol='summaries/resolution_fixed_w1_CONVERGENCE_PROTOCOL.md',
               criterion=dict(statistic='median of the last 10 logged train-loss values',
                              threshold=MED10, source='training loss only; no val/test quantity'),
               arms=[dict(name=n, lr=lr, epochs=ep) for n, lr, ep in ARMS],
               preference_order=[n for n, _, _ in ARMS], locked=None, stage_p={}, stage_v={}, stage_f={})

    for name, lr, ep in ARMS:
        out['stage_p'][name] = dict(lr=lr, epochs=ep, rescue={}, scaling={})
        for res, seed in RESCUE:
            out['stage_p'][name]['rescue'][f'D{res}_s{seed}'] = row(res, seed, lr, ep)
        for res, seed in SCALING:
            out['stage_p'][name]['scaling'][f'D{res}_s{seed}'] = row(res, seed, lr, ep)

    if locked:
        name, lr, ep = locked
        out['locked'] = dict(name=name, lr=lr, epochs=ep)
        for stage, seeds in (('stage_v', DEV_SEEDS), ('stage_f', FRESH_SEEDS)):
            for res in RESOLUTIONS:
                rows = [r for r in (row(res, s, lr, ep) for s in seeds) if r]
                if not rows:
                    continue
                conv = [r for r in rows if r['converged']]
                e = dict(runs=rows, convergence=rate(rows))
                # Accuracy is summarised over CONVERGED runs only, and the protocol requires the
                # non-convergence rate to be reported beside it -- a resolution with any
                # non-converged final seed gets no headline mIoU.
                if conv:
                    v = [r['val_miou'] for r in conv]; t = [r['test_miou'] for r in conv]
                    sd = lambda x: round(statistics.stdev(x), 4) if len(x) > 1 else 0.0
                    e['converged_only'] = dict(n=len(conv), val_mean=round(statistics.mean(v), 4), val_sd=sd(v),
                                               test_mean=round(statistics.mean(t), 4), test_sd=sd(t))
                e['headline_eligible'] = len(conv) == len(rows)
                out[stage][f'D{res}'] = e

    p = f'{ROOT}/summaries/resolution_fixed_w1_convergence.json'
    json.dump(out, open(p, 'w'), indent=1, default=float)
    log('wrote', p)
    return out


def main():
    stage_p()
    locked = lock()
    if not locked:
        report(None)
        return
    _, lr, ep = locked
    stage_v(lr, ep)
    stage_f(lr, ep)
    out = report(locked)
    log('\n' + json.dumps(out.get('stage_f', {}), indent=1, default=float))
    log('FP32 STAGES COMPLETE. QAT / export / deliverables are stage Q/X of the protocol.')


if __name__ == '__main__':
    main()
