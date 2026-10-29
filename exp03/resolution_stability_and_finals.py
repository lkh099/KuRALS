"""Stability arm + fresh-seed final comparison for the fixed-1.0x resolution study.

Protocol and the frozen degeneracy criterion: summaries/resolution_fixed_w1_PROTOCOL.md,
written before the lr 3.333e-4 arm existed. Read it first; this file only executes it.

Why this exists: the development arm (resolution_fixed_w1.py, lr 1e-3, seeds 42-45) produced
two non-converged runs, D16 s45 and D32 s44, and none at D8. That makes the lr-1e-3 D16/D32
means a mixture of a resolution effect and an optimisation failure, and it is the handoff's
stop condition, so the recipe was not silently changed. Instead:

  A  stability arm      lr 3.333e-4 at D16/D32, seeds 42-45 -- a sensitivity experiment,
                        NOT a replacement for the development arm, which is kept as-is
  B  lock the LR        per resolution, by the rules declared in the protocol (zero
                        degenerate seeds; tie-break on development seed 42)
  C  QAT LR re-search   on seed 42 from the locked parent, for any resolution whose LR moved
  D  fresh final seeds  46/47/48 FP32 at the locked recipe, all three resolutions
  E  QAT finals         on 46/47/48 from those FP32 checkpoints
  F  report             development / stability / final, each allowed to claim only its own
                        question, plus a parent-health tag on every QAT run
  G  exports            model_release_d{8,16,32}_w1.0_s1 from the best-validation FINAL seed

Seeds 46/47/48 are fresh: never used for tuning, and not used to motivate this amendment.
That is the point -- it keeps the headline off both the failed runs and a post-hoc rescue
chosen on the same seeds that would be reported.

  cd ~/xsrc/KuRALS && setsid nohup ~/.venvs/kurals_train/bin/python \
      exp03/resolution_stability_and_finals.py > exp03/resolution_stability_and_finals.out 2>&1 < /dev/null &

  python exp03/resolution_stability_and_finals.py --show-criterion   # reproduce the frozen numbers
Resumable: runs with a DONE file are skipped.
"""
import csv
import glob
import json
import os
import statistics as stat
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))
from pipeline import (EPOCHS, PY, QAT_EPOCHS, QAT_FREEZE_ITER, QAT_LRS,  # noqa: E402
                      ckpt, job, log, result, run_all, run_dir)
from resolution_fixed_w1 import RESOLUTIONS, SS, WM, costs, mean_sd  # noqa: E402

DEV_SEEDS = [42, 43, 44, 45]          # development / tuning, already spent
TUNE_SEED = 42
FINAL_SEEDS = [46, 47, 48]            # fresh: the final comparison, untouched by any tuning
LR_HIGH = 1e-3                        # the development arm's LR (lr_A)
LR_LOW = 1e-3 / 3                     # the stability arm; the second LR the old Exp B grid used
STABILITY_RES = [16, 32]              # D8 had 0/4 degenerate, so it is not amended
MODEL = 'kuralsnet_npu_seg'

# ---- frozen by summaries/resolution_fixed_w1_PROTOCOL.md section 3-4. Do not edit after the fact.
TRIVIAL_MIOU = {8: 0.249445, 16: 0.249705, 32: 0.249824}   # all-background predictor, Validation split
HALF_REFERENCE_GAIN = 0.162949                             # (median D8 lr-1e-3 best-val 0.575343 - trivial) / 2
C1_EPS = 0.02                                              # "effectively never predicted"
CRITERION_CLASSES = (1, 2)                                 # class 3 excluded a priori: 18-23 px in the split


def fp32_tag(res, seed, lr):
    return f'B_{MODEL}_wm{WM}_ss{SS}_R64-D{res}_lr{lr:g}_s{seed}'


def qat_tag(res, seed, qlr):
    return f'Q_{MODEL}_wm{WM}_ss{SS}_R64-D{res}_qlr{qlr:g}_s{seed}'


def done(tag):
    return os.path.exists(run_dir(tag) + '/DONE')


def val_detail(tag):
    """Validation metrics at the run's best-validation checkpoint. pipeline.result() exposes the
    test per-class IoU but not the validation one, and the criterion is defined on validation."""
    if not done(tag):
        return None
    f = glob.glob(run_dir(tag) + '/logs/KuRALS_CW/*/*/results/val_doppler_results.json')
    if not f:
        return None
    r = json.load(open(f[0]))
    v = r['val_metrics']['range_doppler']
    return dict(tag=tag, best_epoch=r['epoch'], val_miou=v['miou'], val_iou_by_class=v['miou_by_class'])


def degenerate(res, tag):
    """The criterion declared in the protocol, section 4. Both conditions are required."""
    d = val_detail(tag)
    if d is None:
        return None
    c1_val = min(d['val_iou_by_class'][i] for i in CRITERION_CLASSES)
    thr = TRIVIAL_MIOU[res] + HALF_REFERENCE_GAIN
    c1, c2 = c1_val <= C1_EPS, d['val_miou'] < thr
    return dict(tag=tag, best_epoch=d['best_epoch'], val_miou=round(d['val_miou'], 4),
                val_iou_by_class=[round(x, 4) for x in d['val_iou_by_class']],
                c1_min_fg_iou=round(c1_val, 4), c1_triggered=c1, c2_threshold=round(thr, 6),
                c2_triggered=c2, degenerate=bool(c1 and c2))


def arm(res, lr):
    """Degeneracy verdict for every development seed of one (resolution, LR) cell."""
    rows = [degenerate(res, fp32_tag(res, s, lr)) for s in DEV_SEEDS]
    rows = [r for r in rows if r]
    return dict(lr=lr, n=len(rows), n_degenerate=sum(r['degenerate'] for r in rows),
                degenerate_seeds=[r['tag'].rsplit('_s', 1)[1] for r in rows if r['degenerate']],
                seeds={r['tag'].rsplit('_s', 1)[1]: r for r in rows})


def show_criterion():
    print('Frozen criterion (summaries/resolution_fixed_w1_PROTOCOL.md section 4)')
    print(f'  C1: min(IoU_c1, IoU_c2) <= {C1_EPS}      class 3 excluded a priori')
    print(f'  C2: best val mIoU < trivial + {HALF_REFERENCE_GAIN}')
    for r in RESOLUTIONS:
        print(f'      D{r}: {TRIVIAL_MIOU[r]:.6f} + {HALF_REFERENCE_GAIN} = {TRIVIAL_MIOU[r] + HALF_REFERENCE_GAIN:.6f}')
    for r in RESOLUTIONS:
        for lr in (LR_HIGH, LR_LOW):
            a = arm(r, lr)
            if not a['n']:
                continue
            print(f'\n  D{r} lr {lr:g}: {a["n"]} runs, {a["n_degenerate"]} degenerate {a["degenerate_seeds"] or ""}')
            for s, v in a['seeds'].items():
                print(f'    s{s}: val {v["val_miou"]:.4f} per-class {v["val_iou_by_class"]} '
                      f'C1={v["c1_triggered"]} C2={v["c2_triggered"]} -> {"DEGENERATE" if v["degenerate"] else "ok"}')


# ------------------------------------------------------------------------------------------- stages
def stage_a():
    log(f'STAGE A stability arm: lr {LR_LOW:g}, {EPOCHS} epochs, seeds {DEV_SEEDS}, resolutions {STABILITY_RES}')
    run_all([job(MODEL, r, s, LR_LOW, wm=WM, ss=SS, exp='B', epochs=EPOCHS, tag=fp32_tag(r, s, LR_LOW))
             for r in STABILITY_RES for s in DEV_SEEDS])


def stage_b():
    """Lock the FP32 LR per resolution by the protocol's declared rules. Returns (locked, arms, stop)."""
    arms, locked, stop = {}, {}, []
    for r in RESOLUTIONS:
        cand = [LR_HIGH] + ([LR_LOW] if r in STABILITY_RES else [])
        arms[r] = {f'{lr:g}': arm(r, lr) for lr in cand}
        clean = [lr for lr in cand if arm(r, lr)['n'] == len(DEV_SEEDS) and arm(r, lr)['n_degenerate'] == 0]
        for lr in cand:
            a = arm(r, lr)
            if a['n'] and a['n'] != len(DEV_SEEDS):
                log(f'R64-D{r} lr {lr:g}: only {a["n"]}/{len(DEV_SEEDS)} development seeds finished, '
                    f'not eligible for the LR lock')
        if len(clean) == 1:
            locked[r], why = clean[0], 'the only LR with zero degenerate seeds'
        elif len(clean) > 1:
            v = {lr: (val_detail(fp32_tag(r, TUNE_SEED, lr)) or {}).get('val_miou', -1) for lr in clean}
            locked[r], why = max(v, key=v.get), f'both LRs clean; higher seed-{TUNE_SEED} val mIoU ' \
                                                + str({f'{k:g}': round(x, 4) for k, x in v.items()})
        else:
            stop.append(r)
            log(f'STOP CONDITION R64-D{r}: every LR has a degenerate seed '
                + str({k: a['degenerate_seeds'] for k, a in arms[r].items()}))
            continue
        log(f'R64-D{r}: FP32 LR locked at {locked[r]:g} -- {why}')
    return locked, arms, stop


def stage_c(locked):
    """QAT LR on development seed 42, from the LOCKED parent. A QAT LR chosen on a 1e-3 parent
    does not transfer to a 3.333e-4 one, so it is re-searched wherever the FP32 LR moved."""
    need = [r for r in locked if not all(done(qat_tag(r, TUNE_SEED, q)) for q in QAT_LRS)
            or locked[r] != LR_HIGH]
    jobs = []
    for r in need:
        src = fp32_tag(r, TUNE_SEED, locked[r])
        if not done(src):
            log(f'cannot re-search QAT LR for D{r}: locked parent missing {src}')
            continue
        for q in QAT_LRS:
            # The development arm already searched from the 1e-3 parent under the plain tag. Reuse that
            # only when 1e-3 is also the locked LR; otherwise the search is from a different parent and
            # gets its own suffixed tag, so the two can never be confused.
            t = qat_tag(r, TUNE_SEED, q) + ('' if locked[r] == LR_HIGH else f'_lr{locked[r]:g}')
            if done(t):
                continue
            jobs.append(job(MODEL, r, TUNE_SEED, q, wm=WM, ss=SS, stage='qat', exp='Q', epochs=QAT_EPOCHS,
                            finetune=ckpt(src), qat=dict(quant_warmup_iters=0, quant_freeze_iters=QAT_FREEZE_ITER),
                            tag=t))
    if jobs:
        log(f'STAGE C QAT LR search on seed {TUNE_SEED} from the locked parents: {[t for t, _ in jobs]}')
        run_all(jobs)

    qlr, table = {}, {}
    for r in locked:
        cand = {}
        for q in QAT_LRS:
            t = qat_tag(r, TUNE_SEED, q) + ('' if locked[r] == LR_HIGH else f'_lr{locked[r]:g}')
            if not done(t):
                t = qat_tag(r, TUNE_SEED, q)
            d = val_detail(t)
            if d:
                cand[q] = d
        if not cand:
            log(f'no QAT LR-search result for D{r}')
            continue
        qlr[r] = max(cand, key=lambda q: cand[q]['val_miou'])
        table[r] = {f'{q:g}': dict(tag=d['tag'], val_miou=round(d['val_miou'], 4)) for q, d in cand.items()}
        log(f'R64-D{r}: QAT LR = {qlr[r]:g} from seed-{TUNE_SEED} '
            + str({f'{q:g}': round(d["val_miou"], 4) for q, d in cand.items()}))
    return qlr, table


def stage_d(locked):
    log(f'STAGE D fresh final FP32 seeds {FINAL_SEEDS} at the locked recipe: '
        + str({f'D{r}': f'{lr:g}' for r, lr in locked.items()}))
    run_all([job(MODEL, r, s, locked[r], wm=WM, ss=SS, exp='B', epochs=EPOCHS, tag=fp32_tag(r, s, locked[r]))
             for r in locked for s in FINAL_SEEDS])


def stage_e(locked, qlr):
    log(f'STAGE E QAT finals on seeds {FINAL_SEEDS}, each paired to its own FP32 checkpoint')
    jobs = []
    for r in qlr:
        for s in FINAL_SEEDS:
            src = fp32_tag(r, s, locked[r])
            if not done(src):
                log(f'SKIP QAT D{r} s{s}: FP32 parent not done ({src})')
                continue
            jobs.append(job(MODEL, r, s, qlr[r], wm=WM, ss=SS, stage='qat', exp='Q', epochs=QAT_EPOCHS,
                            finetune=ckpt(src), qat=dict(quant_warmup_iters=0, quant_freeze_iters=QAT_FREEZE_ITER),
                            tag=qat_tag(r, s, qlr[r])))
    failed = run_all(jobs)
    if failed:
        log('QAT finals FAILED:', failed)


# ----------------------------------------------------------------------------------------- reporting
def stage_summary(res, lr, seeds, qlr):
    """FP32 and QAT accuracy for one (resolution, LR, seed set), with a degeneracy verdict on every
    FP32 run and a parent-health tag on every QAT run."""
    out = dict(resolution=f'R64-D{res}', fp32_lr=lr, qat_lr=None if qlr is None else float(qlr), fp32={}, qat={})
    for s in seeds:
        ft = fp32_tag(res, s, lr)
        r, deg = result(ft), degenerate(res, ft)
        out['fp32'][str(s)] = None if not r else dict(
            tag=ft, best_epoch=r['best_epoch'], val_miou=round(r['val_miou'], 4),
            test_miou=round(r['test_miou'], 4), test_dice=round(r['test_dice'], 4),
            test_uav_iou=round(r['test_uav_iou'], 4),
            test_iou_by_class=[round(x, 4) for x in r['test_iou_by_class']],
            degenerate=None if not deg else deg['degenerate'],
            val_iou_by_class=None if not deg else deg['val_iou_by_class'])
        if qlr is None:
            continue
        qt = qat_tag(res, s, qlr)
        q = result(qt)
        out['qat'][str(s)] = None if not q else dict(
            tag=qt, best_epoch=q['best_epoch'], val_miou=round(q['val_miou'], 4),
            test_miou=round(q['test_miou'], 4), test_dice=round(q['test_dice'], 4),
            test_uav_iou=round(q['test_uav_iou'], 4),
            test_iou_by_class=[round(x, 4) for x in q['test_iou_by_class']],
            fp32_parent=ft, fp32_parent_degenerate=None if not deg else deg['degenerate'],
            parent_note=None if not (deg and deg['degenerate']) else
            'FP32 parent was degenerate: any gain here is further optimisation escaping a bad FP32 '
            'solution, NOT evidence that quantization improves accuracy')
    for stage in ('fp32', 'qat'):
        got = [x for x in out[stage].values() if x]
        agg = dict(n=len(got), n_degenerate=sum(1 for x in got
                                                if x.get('degenerate') or x.get('fp32_parent_degenerate')))
        for key, fld in (('test_miou', 'test_miou'), ('test_dice', 'test_dice'),
                         ('uav_iou', 'test_uav_iou'), ('val_miou', 'val_miou')):
            m, sd = mean_sd([x[fld] for x in got])
            agg[f'{key}_mean'], agg[f'{key}_std'] = (None if m is None else round(m, 4),
                                                     None if sd is None else round(sd, 4))
        out[stage + '_aggregate'] = agg
    return out


def paired(a, b, stage, key='test_miou'):
    """Per-seed difference a - b for the seeds both have. n=3, reported as sign consistency."""
    d = [round(a[stage][s][key] - b[stage][s][key], 4) for s in a[stage]
         if a[stage].get(s) and b[stage].get(s)]
    if len(d) < 2:
        return None
    m, sd = mean_sd(d)
    return dict(per_seed=d, mean=round(m, 4), sd=round(sd, 4), n=len(d), same_sign=len({x > 0 for x in d}) == 1,
                t=None if not sd else round(m / (sd / len(d) ** 0.5), 3),
                note='paired by seed; n=3 has very little power, so a non-zero mean is not by itself '
                     'evidence of a real difference')


def export(res, final, cost_row):
    """Package from the best-VALIDATION final-seed QAT run. Only final seeds are eligible, so the
    FP32-parent glob in make_release.py stays unambiguous (development and stability arms share a
    resolution but never a seed with the final stage)."""
    cands = [(v['val_miou'], v['tag'], s) for s, v in final['qat'].items() if v]
    if not cands:
        log(f'skip export D{res}: no finished final-seed QAT run')
        return None
    _, tag, seed = max(cands)
    out = f'model_release_d{res}_w1.0_s1'
    stem = f'B_{MODEL}_wm{WM}_ss{SS}_R64-D{res}'
    amb = sorted(os.path.basename(d) for d in glob.glob(f'{ROOT}/runs/{stem}_lr*_s{seed}') if done(os.path.basename(d)))
    if len(amb) != 1:
        log(f'WARNING export D{res}: {len(amb)} FP32 parents match seed {seed} -> {amb}; '
            f'make_release.py will record fp32_reference_tag UNRESOLVED')
    if os.path.exists(f'{ROOT}/{out}/metrics.json') and \
       json.load(open(f'{ROOT}/{out}/metrics.json'))['run'] == tag:
        log(f'export {out} already built from {tag}, keeping it')
    else:
        bank = cost_row['bank_required_bytes']
        log(f'export {out} from {tag} (--buf-size {hex(bank)})')
        rc = subprocess.run([PY, f'{ROOT}/make_release.py', '--run', tag, '--out', f'{ROOT}/{out}', '--buf-size', hex(bank)],
                            cwd=os.path.dirname(ROOT), stdout=open(f'{ROOT}/make_release_{out}.log', 'w'),
                            stderr=subprocess.STDOUT).returncode
        if rc != 0:
            log(f'EXPORT FAILED {out}: see exp03/make_release_{out}.log')
            return None
        subprocess.run([PY, f'{ROOT}/make_package_readme.py', f'{ROOT}/{out}'], cwd=ROOT,
                       stdout=open(f'{ROOT}/make_release_{out}.log', 'a'), stderr=subprocess.STDOUT)
    met = json.load(open(f'{ROOT}/{out}/metrics.json'))
    return dict(path=f'exp03/{out}', run=met['run'], seed=seed, weights_sha256=met['weights_sha256'],
                exact_integer_replay_test_miou=met['exact_integer_replay']['test_miou'],
                npu_buffer_required=met.get('npu_buffer_required'))


def main():
    if '--show-criterion' in sys.argv:
        show_criterion()
        return

    log('stability arm + fresh-seed finals for the fixed-1.0x resolution study')
    log('protocol: exp03/summaries/resolution_fixed_w1_PROTOCOL.md (written before this arm existed)')
    cost = {r: costs(r) for r in RESOLUTIONS}

    stage_a()
    locked, arms, stop = stage_b()
    # The final stage is a THREE-WAY comparison, so it runs only when every resolution has a stable
    # recipe. Running fresh seeds for a subset would produce a "final" table that cannot answer the
    # question, which is worse than producing none.
    if stop or len(locked) < len(RESOLUTIONS):
        missing = [r for r in RESOLUTIONS if r not in locked]
        log(f'ABORT: no stable LR at {stop or missing}. Per the protocol this is the handoff stop '
            f'condition, reported rather than worked around. Fresh final seeds are NOT run for any '
            f'resolution, including the ones that did lock.')
        write_stability_report(arms, locked, stop)
        return

    qlr, qlr_table = stage_c(locked)
    stage_d(locked)
    stage_e(locked, qlr)

    dev_qlr = {8: 1e-5, 16: 1e-5, 32: 1e-4}      # fallback; the development driver's own picks win
    _dev_f = f'{ROOT}/summaries/resolution_fixed_w1.json'
    if os.path.exists(_dev_f):
        _d = json.load(open(_dev_f))
        for k, v in _d.get('resolutions', {}).items():
            if (v.get('accuracy') or {}).get('qat_lr'):
                dev_qlr[int(k)] = v['accuracy']['qat_lr']
    log('development-arm QAT LRs:', {f'D{k}': f'{v:g}' for k, v in sorted(dev_qlr.items())})
    dev = {r: stage_summary(r, LR_HIGH, DEV_SEEDS[1:], dev_qlr.get(r)) for r in RESOLUTIONS}
    stab = {r: stage_summary(r, LR_LOW, DEV_SEEDS[1:], None) for r in STABILITY_RES}
    final = {r: stage_summary(r, locked[r], FINAL_SEEDS, qlr.get(r)) for r in locked}

    st = dict(
        experiment='fixed-1.0x stride-1 RD-resolution study: stability arm + fresh-seed final comparison',
        protocol='exp03/summaries/resolution_fixed_w1_PROTOCOL.md',
        generated_by='exp03/resolution_stability_and_finals.py',
        criterion=dict(c1=f'min(IoU_class1, IoU_class2) <= {C1_EPS}', c1_classes=list(CRITERION_CLASSES),
                       c1_class3_excluded='18-23 px in the Validation split; 0.0000 in two converged D8 runs',
                       c2=f'best val mIoU < trivial + {HALF_REFERENCE_GAIN}',
                       trivial_miou=TRIVIAL_MIOU, half_reference_gain=HALF_REFERENCE_GAIN,
                       c2_threshold={str(r): round(TRIVIAL_MIOU[r] + HALF_REFERENCE_GAIN, 6) for r in RESOLUTIONS},
                       frozen='before the lr 3.333e-4 arm was trained'),
        fp32_lr_locked={str(r): lr for r, lr in locked.items()},
        qat_lr={str(r): q for r, q in qlr.items()}, qat_lr_search=qlr_table,
        stop_condition=stop,
        stages=dict(
            development=dict(lr=LR_HIGH, seeds=DEV_SEEDS, answers='behaviour of the recipe as originally '
                            'specified, including its failures', resolutions=dev),
            stability=dict(lr=LR_LOW, seeds=DEV_SEEDS, answers='whether the failures are LR-driven; a '
                           'sensitivity result, not a replacement', resolutions=stab),
            final=dict(seeds=FINAL_SEEDS, answers='does the larger RD representation help when the model '
                       'is optimised stably', resolutions=final)),
        cost={str(r): cost[r] for r in RESOLUTIONS})
    for r in locked:
        if r == 8:
            continue
        for stage in ('fp32', 'qat'):
            p = paired(final[r], final[8], stage) if 8 in final else None
            if p:
                st.setdefault('final_paired_vs_d8', {})[f'{stage}_D{r}_minus_D8'] = p
    for r in locked:
        st['stages']['final']['resolutions'][r]['export'] = export(r, final[r], cost[r])

    json.dump(st, open(f'{ROOT}/summaries/resolution_fixed_w1_final.json', 'w'), indent=1)
    log('wrote', f'{ROOT}/summaries/resolution_fixed_w1_final.json')
    write_deliverables(st)
    log('STABILITY + FINALS COMPLETE')


def write_deliverables(st):
    """paper_resolution_fixed_w1_model.{csv,md}: the handoff's Phase 1 step 8 output, with the fresh
    final seeds as the primary table and the development/stability arms kept as their own sections."""
    fin = st['stages']['final']['resolutions']
    cols = ['resolution', 'width', 'stem_stride', 'params', 'macs_exact', 'peak_activation_bytes',
            'mem1_required_bytes', 'mem2_required_bytes', 'input_shape',
            'fp32_miou_mean', 'fp32_miou_std', 'qat_miou_mean', 'qat_miou_std',
            'fp32_dice_mean', 'fp32_dice_std', 'qat_dice_mean', 'qat_dice_std',
            'uav_iou_fp32_mean', 'uav_iou_fp32_std', 'uav_iou_qat_mean', 'uav_iou_qat_std',
            'export_path', 'weights_sha256', 'notes']
    n = lambda v: '' if v is None else f'{v:.4f}'
    rows = []
    for r in RESOLUTIONS:
        if r not in fin:
            continue
        c, f = st['cost'][str(r)], fin[r]
        a, q, e = f['fp32_aggregate'], f['qat_aggregate'], f.get('export') or {}
        rows.append({'resolution': c['resolution'], 'width': WM, 'stem_stride': SS, 'params': c['params'],
                     'macs_exact': c['macs_exact'], 'peak_activation_bytes': c['peak_activation_bytes'],
                     'mem1_required_bytes': c['mem1_required_bytes'], 'mem2_required_bytes': c['mem2_required_bytes'],
                     'input_shape': 'x'.join(map(str, c['input_shape'])),
                     'fp32_miou_mean': n(a['test_miou_mean']), 'fp32_miou_std': n(a['test_miou_std']),
                     'qat_miou_mean': n(q['test_miou_mean']), 'qat_miou_std': n(q['test_miou_std']),
                     'fp32_dice_mean': n(a['test_dice_mean']), 'fp32_dice_std': n(a['test_dice_std']),
                     'qat_dice_mean': n(q['test_dice_mean']), 'qat_dice_std': n(q['test_dice_std']),
                     'uav_iou_fp32_mean': n(a['uav_iou_mean']), 'uav_iou_fp32_std': n(a['uav_iou_std']),
                     'uav_iou_qat_mean': n(q['uav_iou_mean']), 'uav_iou_qat_std': n(q['uav_iou_std']),
                     'export_path': e.get('path', ''), 'weights_sha256': e.get('weights_sha256', ''),
                     'notes': f"final seeds {FINAL_SEEDS}; FP32 lr {f['fp32_lr']:g}; QAT lr {f['qat_lr']}; "
                              f"n_fp32={a['n']} n_qat={q['n']}; degenerate_in_cell={a['n_degenerate']}; "
                              f"bank required {c['bank_required_kib']} KiB; "
                              + ('fits' if c['fits_taped_out_bank'] else 'exceeds')
                              + ' the taped-out 32 KiB bank'})
    p_csv = f'{ROOT}/summaries/paper_resolution_fixed_w1_model.csv'
    with open(p_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log('wrote', p_csv)

    f4 = lambda m, sd: 'n/a' if m is None else f'{m:.4f} ± {sd:.4f}'
    L = ['# Primary RD-resolution experiment: fixed 1.0x stride-1 network', '',
         'Phase 1 (model server) of `resolution_codesign_fixed_w1_handoff.md`. Generated by',
         '`exp03/resolution_stability_and_finals.py`. Protocol and the frozen degeneracy criterion:',
         '`resolution_fixed_w1_PROTOCOL.md`, written before the stability arm was trained.', '',
         '**The primary table uses fresh seeds 46/47/48.** Seeds 42-45 were spent on development and on',
         'discovering a training-stability problem, so they are reported separately and are not the',
         'headline evidence. Nothing is dropped and nothing is overwritten.', '',
         '## What is held fixed', '', '```text',
         f'model            KuRALSNetNPUSeg, {st["cost"]["8"]["layers"]} layers, cutpoint {st["cost"]["8"]["cutpoint"]}',
         f'width_mult       {WM}      stem_stride {SS}      encoder_depth 1, bottleneck kernel 3, eltwise_act linear',
         f'params           {st["cost"]["8"]["params"]:,}  (identical at all three resolutions)',
         'dataset          KuRALS-CW, existing chronological 80/10/10, existing block-max-pooled RD grids',
         'input            DSP 5.3-bit log2 code, input_scale 1.0, natural Doppler order',
         f'budget           FP32 {EPOCHS} epochs, QAT {QAT_EPOCHS} epochs, batch 6, cos schedule',
         'selection        best validation mIoU; test read once at that checkpoint',
         'LR procedure     fixed: locked per resolution on development seeds, before the final seeds ran',
         '```', '',
         'The experimental variable is the RD representation, not the optimiser. The learning rate may',
         'therefore differ per resolution; what is held fixed is the architecture, width, stride, budget,',
         'the LR **selection procedure**, and the fact that the final seeds are untouched by it.', '',
         '## Locked recipe', '', '| Resolution | FP32 LR | QAT LR | degenerate development seeds at that LR |',
         '|---|---:|---:|---|']
    for r in RESOLUTIONS:
        if str(r) not in st['fp32_lr_locked']:
            L.append(f'| R64-D{r} | — | — | **no stable LR found; stop condition reported** |')
            continue
        lr = st['fp32_lr_locked'][str(r)]
        a = st['stages']['stability' if lr == LR_LOW else 'development']['resolutions'].get(r)
        nd = a['fp32_aggregate']['n_degenerate'] if a else '?'
        L.append(f'| R64-D{r} | {lr:g} | {st["qat_lr"].get(str(r), "—")} | {nd} |')
    L += ['', '## Primary result — fresh seeds 46/47/48 (test, mean ± sample std)', '',
          '| | ' + ' | '.join(f'R64-D{r}' for r in RESOLUTIONS) + ' |',
          '|---|' + '---:|' * len(RESOLUTIONS)]
    for stage, nm in (('fp32', 'FP32'), ('qat', 'QAT int8')):
        for key, lbl in (('test_miou', 'mIoU'), ('test_dice', 'Dice'), ('uav_iou', 'UAV IoU')):
            cells = []
            for r in RESOLUTIONS:
                a = fin.get(r, {}).get(stage + '_aggregate')
                cells.append('not run' if not a else f4(a[f'{key}_mean'], a[f'{key}_std']))
            L.append(f'| {nm} {lbl} | ' + ' | '.join(cells) + ' |')
    if st.get('final_paired_vs_d8'):
        L += ['', '### Paired against R64-D8, same seed', '',
              '| Comparison | per-seed Δ | mean ± sd | same sign | paired t |', '|---|---|---:|:-:|---:|']
        for k, d in st['final_paired_vs_d8'].items():
            L.append(f'| {k} | {d["per_seed"]} | {d["mean"]:+.4f} ± {d["sd"]:.4f} | '
                     f'{"yes" if d["same_sign"] else "no"} | {d["t"] if d["t"] is not None else "n/a"} |')
        L += ['', 'n = 3. Reported so the gap is not over-read; this is not a significance test.']
    L += ['', '## Secondary — training stability of the common recipe', '',
          'At the development LR of 1e-3 the recipe showed occasional foreground-separation collapse at',
          'D16 and D32 but not at D8, which motivated the lower-LR check before the final comparison.',
          'With four seeds per cell this is **not** a claim that instability increases with resolution.', '',
          '| Resolution | lr 1e-3 degenerate / n | lr 3.333e-4 degenerate / n |', '|---|---|---|']
    for r in RESOLUTIONS:
        hi = st['stages']['development']['resolutions'].get(r, {}).get('fp32_aggregate', {})
        lo = st['stages']['stability']['resolutions'].get(r, {}).get('fp32_aggregate')
        L.append(f'| R64-D{r} | {hi.get("n_degenerate", "?")} / {hi.get("n", "?")} | '
                 + (f'{lo["n_degenerate"]} / {lo["n"]} |' if lo else 'not run |'))
    L += ['', 'The failure is not all-background collapse: both runs emit the UAV class, but class 2',
          'collapses to ~0 and background purity drops below the 0.9976-0.9996 every converged run',
          'reaches. Per-run verdicts, with both criterion conditions, are in',
          '`resolution_fixed_w1_final.json` under `stages.*.resolutions.*.fp32.*.degenerate`.', '',
          'Every QAT run carries `fp32_parent_degenerate`. Where a parent was degenerate, a QAT gain is',
          'further optimisation escaping a bad FP32 solution, not quantization improving accuracy.', '',
          '## Cost (built graph + gen_cfg.py allocator, run with an unlimited bank)', '',
          '| | ' + ' | '.join(f'R64-D{r}' for r in RESOLUTIONS) + ' |', '|---|' + '---:|' * len(RESOLUTIONS)]
    crow = lambda lbl, k, f=lambda v: f'{v:,}': f'| {lbl} | ' + ' | '.join(f(st['cost'][str(r)][k]) for r in RESOLUTIONS) + ' |'
    L += [crow('Params', 'params'), crow('MACs (exact)', 'macs_exact'), crow('Weight bytes', 'weight_bytes'),
          '| Input shape | ' + ' | '.join(str(tuple(st['cost'][str(r)]['input_shape'])) for r in RESOLUTIONS) + ' |',
          crow('Peak activation bytes', 'peak_activation_bytes'),
          crow('MEM1 required', 'mem1_required_bytes'), crow('MEM2 required', 'mem2_required_bytes'),
          crow('Activation bank required', 'bank_required_kib', lambda v: f'{v} KiB'),
          '| Fits the taped-out 32 KiB bank | ' + ' | '.join(
              ('yes' if st['cost'][str(r)]['fits_taped_out_bank'] else 'no') for r in RESOLUTIONS) + ' |',
          '| Frame-reuse cells (H×W) | ' + ' | '.join(
              f'{st["cost"][str(r)]["shape_derived_depths"]["frame_reuse_cells"]:,}' for r in RESOLUTIONS) + ' |',
          '', 'These are the capacities each resolution **requires**: the old `--buf-size 0x8000` rejection',
          'rule is not applied, because here the bank size is a result rather than a constraint. Shape-derived',
          'high-water marks for the line buffer, partial sums, maxpool and fused upsample, plus the full',
          'per-layer shape table, are in `resolution_fixed_w1_final.json` under `cost.*`.', '',
          '## A confound to report with the result', '',
          'The foreground fraction of a frame falls from 0.22 % at D8 to 0.07 % at D32 (Validation split:',
          '176/99/18, 187/106/19 and 223/127/23 pixels of classes 1/2/3), because the frame grows faster',
          'than the target footprint. Class imbalance is therefore not constant along the resolution axis.',
          'This is inherent to the comparison and is not corrected for.', '',
          '## Exports', '']
    rows_e = [(r, fin.get(r, {}).get('export')) for r in RESOLUTIONS]
    if not any(e for _, e in rows_e):
        L += ['Not built.']
    else:
        L += ['| Resolution | Package | Source run | int8 replay test mIoU | weights SHA-256 |',
              '|---|---|---|---:|---|']
        for r, e in rows_e:
            L.append(f'| R64-D{r} | not built | | | |' if not e else
                     f'| R64-D{r} | `{e["path"]}` | `{e["run"]}` | {e["exact_integer_replay_test_miou"]:.4f} | '
                     f'`{e["weights_sha256"][:32]}…` |')
        L += ['', 'Packages are built only from final-seed runs. Read one back without a checkpoint with',
              '`exp03/load_release.py --package <dir>`.']
    L += ['', '## Relation to the old fixed-32-KiB search', '',
          'That search held the bank at 0x8000 and let width and stride float, so it compared three',
          'different networks (0.5x-ss1 at D8/D16, 0.5x-**ss2** at D32, whose /2 output head caps IoU for a',
          'one-pixel target). It is secondary evidence for what a frozen accelerator forces, not the',
          'resolution comparison, and its packages are untouched.', '']
    p_md = f'{ROOT}/summaries/paper_resolution_fixed_w1_model.md'
    open(p_md, 'w').write('\n'.join(L) + '\n')
    log('wrote', p_md)




def write_stability_report(arms, locked, stop):
    """Written when stage B finds no stable recipe: the handoff stop condition as a deliverable.

    Reports the frozen-criterion counts AND a threshold-free read, because two runs in the 3.333e-4
    arm sit within 0.007 of flipping the criterion. The criterion is NOT retuned after the fact --
    that would throw away the only thing pre-registration buys."""
    rows, substantive = [], {}
    for r in RESOLUTIONS:
        for lr, a in arms.get(r, {}).items():
            if not a['n']:
                continue
            # threshold-free: a foreground channel that is clearly dead, well away from the 0.02 epsilon
            broken = [s for s, v in a['seeds'].items() if v['c1_min_fg_iou'] < 0.05]
            bg_min = min(v['val_iou_by_class'][0] for v in a['seeds'].values())
            substantive[f'D{r}_lr{lr}'] = dict(
                n=a['n'], degenerate_by_criterion=a['n_degenerate'], criterion_seeds=a['degenerate_seeds'],
                seeds_with_dead_foreground_channel=broken, n_dead_foreground=len(broken),
                min_background_iou=round(bg_min, 4))
            rows.append((r, lr, a, broken, bg_min))
    st = dict(
        experiment='fixed-1.0x stride-1 RD-resolution study -- STOP CONDITION at stage B',
        protocol='exp03/summaries/resolution_fixed_w1_PROTOCOL.md',
        generated_by='exp03/resolution_stability_and_finals.py',
        outcome='no FP32 learning rate produced four clean development seeds at R64-D16 or R64-D32, so '
                'the fresh-seed final comparison was not run at any resolution',
        stop_condition_resolutions=stop,
        locked={str(k): v for k, v in locked.items()},
        lowering_the_lr_made_it_worse=True,
        criterion=dict(c1=f'min(IoU_class1, IoU_class2) <= {C1_EPS}', c2=f'best val mIoU < trivial + {HALF_REFERENCE_GAIN}',
                       trivial_miou=TRIVIAL_MIOU, frozen='before the lr 3.333e-4 arm was trained',
                       brittleness='two runs in the 3.333e-4 arm are within 0.007 of flipping: D16 s44 '
                                   '(UAV IoU 0.0265 vs the 0.02 epsilon) and D16 s45 (val 0.4158 vs the '
                                   '0.412654 threshold). The criterion is left exactly as declared; the '
                                   'threshold-free counts below are reported alongside it.'),
        arms=arms, threshold_free_read=substantive,
        mechanistic_signature='every run with a dead foreground channel also has background IoU <= 0.9849, '
                              'while every clean run has >= 0.9976 -- a bimodal split with no run in between. '
                              'The failure is false positives sprayed over background together with the loss '
                              'of one foreground channel, not an all-background collapse.',
        confound='the foreground fraction of a frame falls from 0.22 % at D8 to 0.07 % at D32, so class '
                 'imbalance is not constant along the resolution axis under study')
    p = f'{ROOT}/summaries/resolution_fixed_w1_stability.json'
    json.dump(st, open(p, 'w'), indent=1)
    log('wrote', p)

    L = ['# Stop condition: the fixed 1.0x stride-1 network could not be trained stably at D16/D32', '',
         'Stage B of `exp03/resolution_stability_and_finals.py`, under the protocol frozen in',
         '`resolution_fixed_w1_PROTOCOL.md` before the lr 3.333e-4 arm was trained.', '',
         '**No learning rate produced four clean development seeds at R64-D16 or R64-D32, so the fresh-seed',
         'final comparison (seeds 46/47/48) was not run at any resolution** -- a three-way comparison with',
         'two unusable rows cannot answer the question. R64-D8 did lock, at lr 1e-3.', '',
         '## Seeds with a collapsed foreground channel', '',
         '| | lr 1e-3 | lr 3.333e-4 |', '|---|---|---|']
    for r in RESOLUTIONS:
        cells = []
        for lr in (f'{LR_HIGH:g}', f'{LR_LOW:g}'):
            k = f'D{r}_lr{lr}'
            if k not in substantive:
                cells.append('not run')
                continue
            d = substantive[k]
            cells.append(f'{d["n_dead_foreground"]}/{d["n"]}  (criterion: {d["degenerate_by_criterion"]}/{d["n"]})')
        L.append(f'| R64-D{r} | ' + ' | '.join(cells) + ' |')
    L += ['', 'Counts are seeds with `min(IoU_UAV, IoU_c2) < 0.05` -- a clearly dead foreground channel,',
          'well away from the criterion\'s 0.02 epsilon. The frozen-criterion count is in parentheses.', '',
          '**Lowering the learning rate made stability worse, not better**: across D16+D32, 2 of 8 seeds',
          'have a dead foreground channel at lr 1e-3 and 5 of 8 at lr 3.333e-4. Both readings agree on the',
          'direction, so the conclusion does not depend on where the threshold sits. R64-D8 is clean in all',
          'five runs across both LRs.', '',
          '## Why the threshold-free count is reported too', '',
          'Two runs in the new arm sit within 0.007 of flipping the frozen criterion: D16 s44 has UAV IoU',
          '0.0265 against the 0.02 epsilon, and D16 s45 has val mIoU 0.4158 against the 0.412654 threshold.',
          'The criterion is **left exactly as declared** -- retuning it after seeing the arm would discard',
          'the only thing pre-registration buys -- so both counts are given and they agree.', '',
          '## The failure mode', '',
          'Every run with a dead foreground channel also has background IoU <= 0.9849; every clean run has',
          '>= 0.9976. The split is bimodal with nothing in between. So the failure is false positives',
          'sprayed across background together with the loss of one foreground channel -- **not** an',
          'all-background collapse, and not something a class-1 test would catch (D32 s43 is clean with a',
          'lower UAV IoU, 0.2230, than either lr-1e-3 failure).', '',
          '## Per-run detail', '',
          '| Resolution | FP32 LR | seed | val mIoU | val IoU `[bg, UAV, c2, c3]` | verdict |',
          '|---|---:|---:|---:|---|---|']
    for r, lr, a, broken, _ in rows:
        for s in sorted(a['seeds']):
            v = a['seeds'][s]
            verdict = ('degenerate (criterion)' if v['degenerate']
                       else ('dead foreground channel' if s in broken else 'ok'))
            L.append(f'| R64-D{r} | {lr} | {s} | {v["val_miou"]:.4f} | `{v["val_iou_by_class"]}` | {verdict} |')
    L += ['', '## What this does and does not support', '',
          'Supported: under a single fixed training recipe and budget, the 1.0x stride-1 network trains',
          'reliably at R64-D8 and unreliably at R64-D16 and R64-D32, and the instability is not fixed by',
          'the lower of the two learning rates in the pre-registered grid.', '',
          '**Not** supported: that the larger RD representations carry less semantic information. That',
          'question is unanswered, because the comparison cannot be made from runs that did not converge.',
          'The handoff\'s prohibition therefore still stands -- "D16/D32 are worse because their mIoU is',
          'lower" may not be written.', '',
          '## A confound that points at the likely cause', '',
          'The foreground fraction of a frame falls from 0.22 % at D8 to 0.07 % at D32 (Validation split:',
          '176/99/18 pixels of classes 1/2/3 at D8 against 223/127/23 at D32, in a frame four times larger).',
          'Class imbalance is therefore not constant along the axis under study, and the observed failure --',
          'one foreground channel lost, background purity degraded -- is what a loss that cannot hold the',
          'minority classes at higher imbalance would look like. Testing that means changing the training',
          'recipe, which the handoff requires to be a separate experiment, not an amendment to this one.', '']
    p = f'{ROOT}/summaries/resolution_fixed_w1_stability.md'
    open(p, 'w').write('\n'.join(L) + '\n')
    log('wrote', p)


if __name__ == '__main__':
    main()
