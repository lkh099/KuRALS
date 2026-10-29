"""Phase 1 final: locked-recipe FP32 + QAT + int8 exports + the handoff's deliverables.

Protocol: summaries/resolution_fixed_w1_FINAL_PROTOCOL.md (frozen before this ran).
Locked recipe: lr 2e-3, 200 epochs, identical at R64-D8/D16/D32 (arm A2).

Reuses exp03/resolution_fixed_w1.py wholesale -- costs(), accuracy(), paired_vs_d8(), export(),
write_csv(), write_md(), link_into_paper_results() are all recipe-agnostic and read their recipe from
module globals, so this module rebinds those globals rather than duplicating 500 lines. The tag
schemes are rebound too, so nothing here can collide with the superseded lr-1e-3 runs.

Two tables, per protocol section 5: PRIMARY on fresh seeds 46/47/48, SECONDARY on the handoff's
declared 43/44/45 (whose FP32 runs already exist from the A2 probe). No exclusion rule -- every
launched run appears with its med10 diagnostic.

  cd ~/xsrc/KuRALS && setsid nohup ~/.venvs/kurals_train/bin/python exp03/resolution_final.py \
      > exp03/resolution_final.out 2>&1 < /dev/null &

Resumable: DONE runs are skipped, in-flight runs adopted.
"""
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

import resolution_fixed_w1 as R                                             # noqa: E402
from pipeline import QAT_EPOCHS, QAT_FREEZE_ITER, QAT_LRS, log, result, run_all  # noqa: E402
from resolution_convergence import convergence                              # noqa: E402

LOCKED_LR, LOCKED_EPOCHS = 2e-3, 200
PRIMARY = [46, 47, 48]          # untouched by every selection; the headline
SECONDARY = [43, 44, 45]        # the handoff's declared report set; selection-contaminated
TUNE = 42

# ---- rebind the recipe onto the reused module -------------------------------------------------
R.FP32_LR = LOCKED_LR
R.EPOCHS = LOCKED_EPOCHS
R.FINAL_SEEDS = PRIMARY
R.SUMMARY = f'{ROOT}/summaries/resolution_fixed_w1_final.json'
# BC_/QC_ keep these runs disjoint from the superseded B_/Q_ lr-1e-3 families. The FP32 scheme is
# exactly resolution_convergence.py's, so the A2 probe's seeds 42-45 are reused rather than retrained.
R.fp32_tag = lambda res, seed: f'BC_{R.MODEL}_wm{R.WM}_ss{R.SS}_R{R.RANGE_BINS}-D{res}_lr{LOCKED_LR:g}_e{LOCKED_EPOCHS}_s{seed}'
R.qat_tag = lambda res, seed, qlr: f'QC_{R.MODEL}_wm{R.WM}_ss{R.SS}_R{R.RANGE_BINS}-D{res}_qlr{qlr:g}_s{seed}'


def retire_stale_exports():
    """A d8 package from the superseded recipe would otherwise be kept by export() and silently
    enter the deliverable. Move it aside rather than delete it."""
    for res in R.RESOLUTIONS:
        p = f'{ROOT}/model_release_d{res}_w1.0_s1'
        m = f'{p}/metrics.json'
        if not os.path.exists(m):
            continue
        run = json.load(open(m))['run']
        if run.startswith('QC_'):
            continue
        dst = f'{p}_SUPERSEDED_{time.strftime("%Y%m%d")}'
        shutil.move(p, dst)
        log(f'retired stale export {os.path.basename(p)} (built from {run}) -> {os.path.basename(dst)}')


def fp32_jobs(seeds):
    return [R.job(R.MODEL, r, s, LOCKED_LR, wm=R.WM, ss=R.SS, exp='BC', epochs=LOCKED_EPOCHS,
                  tag=R.fp32_tag(r, s)) for r in R.RESOLUTIONS for s in seeds]


def qat_jobs(res_seeds):
    out = []
    for r, s, q in res_seeds:
        if not R.done(R.fp32_tag(r, s)):
            log(f'SKIP QAT D{r} s{s}: FP32 parent not done'); continue
        out.append(R.qjob(r, s, q))
    return out


def diagnostics(seeds):
    """The med10 column the protocol requires beside every reported run. Never a filter."""
    d = {}
    for r in R.RESOLUTIONS:
        for s in seeds:
            c = convergence(R.fp32_tag(r, s))
            d[f'D{r}_s{s}'] = None if not c else dict(
                med10=round(c['med10'], 6), converged=c['converged'], escape_epoch=c['escape_epoch'])
    return d


def table(qlr, seeds, label):
    """accuracy() reads R.FINAL_SEEDS, so swap it for the duration."""
    keep, R.FINAL_SEEDS = R.FINAL_SEEDS, seeds
    try:
        acc = {r: R.accuracy(r, qlr[r]) for r in R.RESOLUTIONS}
        paired = R.paired_vs_d8(acc)
    finally:
        R.FINAL_SEEDS = keep
    log(f'--- {label} table, seeds {seeds}')
    for r in R.RESOLUTIONS:
        a = acc[r]
        log(f'  D{r}: FP32 mIoU {a["fp32"].get("test_miou_mean")} (n={a["fp32"]["n"]})  '
            f'QAT mIoU {a["qat"].get("test_miou_mean")} (n={a["qat"]["n"]})')
    return acc, paired


def main():
    log('PHASE 1 FINAL: locked recipe lr %g, %d epochs, R64-D8/D16/D32' % (LOCKED_LR, LOCKED_EPOCHS))
    log('protocol summaries/resolution_fixed_w1_FINAL_PROTOCOL.md; primary seeds %s, secondary %s'
        % (PRIMARY, SECONDARY))
    retire_stale_exports()

    cost = {r: R.costs(r) for r in R.RESOLUTIONS}
    bad = {r: c['handoff_expectation'] for r, c in cost.items() if c['handoff_expectation'] != 'confirmed'}
    if bad:
        log('WARNING: graph disagrees with the handoff, recorded and continuing:', bad)

    log('STAGE F FP32: fresh %s + tune seed %d + secondary %s (existing runs reused)'
        % (PRIMARY, TUNE, SECONDARY))
    run_all(fp32_jobs(PRIMARY + [TUNE] + SECONDARY))

    log('STAGE Q1 QAT LR search on seed %d over %s' % (TUNE, [f'{q:g}' for q in QAT_LRS]))
    run_all(qat_jobs([(r, TUNE, q) for r in R.RESOLUTIONS for q in QAT_LRS]))

    qlr, searches = {}, {}
    for r in R.RESOLUTIONS:
        v = {q: x['val_miou'] for q, x in ((q, result(R.qat_tag(r, TUNE, q))) for q in QAT_LRS) if x}
        searches[str(r)] = {f'{k:g}': round(x, 4) for k, x in v.items()}
        qlr[r] = max(v, key=v.get) if v else None
        log(f'  D{r}: QAT lr = {qlr[r]} from {searches[str(r)]}')

    log('STAGE Q2 QAT finals: primary %s and secondary %s, each paired to its own FP32 seed'
        % (PRIMARY, SECONDARY))
    run_all(qat_jobs([(r, s, qlr[r]) for r in R.RESOLUTIONS if qlr[r]
                      for s in PRIMARY + SECONDARY]))

    acc, paired = table(qlr, PRIMARY, 'PRIMARY (headline, untouched seeds)')
    acc2, paired2 = table(qlr, SECONDARY, 'SECONDARY (handoff report set, selection-contaminated)')

    st = dict(
        experiment='primary RD-resolution study, fixed 1.0x stride-1 KuRALSNetNPUSeg',
        handoff=os.path.expanduser('~/resolution_codesign_fixed_w1_handoff.md'),
        generated_by='exp03/resolution_final.py',
        protocol=dict(model=R.MODEL, width_mult=R.WM, stem_stride=R.SS, encoder_depth=R.ED,
                      fp32_epochs=LOCKED_EPOCHS, fp32_lr=LOCKED_LR,
                      fp32_lr_source='arm A2, locked in summaries/resolution_fixed_w1_FINAL_PROTOCOL.md; '
                                     'in pipeline.py LR_GRID; identical at all three resolutions',
                      qat_epochs=QAT_EPOCHS, qat_lr_grid=QAT_LRS, qat_freeze_iters=QAT_FREEZE_ITER,
                      qat_lr_selection=f'per resolution on seed {TUNE} by validation mIoU',
                      tune_seed=TUNE, final_seeds=PRIMARY, secondary_seeds=SECONDARY,
                      exclusion_rule='NONE -- every launched run is reported; med10 is a diagnostic column',
                      selection='best validation mIoU; test read once at that checkpoint',
                      miou='4-class mean including background',
                      dataset='KuRALS-CW, existing chronological 80/10/10, existing block-max-pooled RD grids'),
        design=dict(params=cost[8]['params'], layers=cost[8]['layers'], cutpoint=cost[8]['cutpoint']),
        supersedes='the fixed-32-KiB co-design search (varied width/stride) and the lr-1e-3 arm',
        resolutions={}, paired_vs_d8=paired,
        secondary=dict(seeds=SECONDARY, accuracy=acc2, paired_vs_d8=paired2,
                       caveat='arm A2 was selected using seeds 42-45; this table is a consistency '
                              'check, not a headline'),
        convergence_diagnostics=dict(primary=diagnostics(PRIMARY), secondary=diagnostics(SECONDARY),
                                     criterion='median of last 10 train losses < 0.005; diagnostic only'))
    for r in R.RESOLUTIONS:
        st['resolutions'][str(r)] = dict(cost=cost[r], qat_lr_search=searches[str(r)],
                                         accuracy=acc[r], export=R.export(r, acc[r], cost[r]))
    st['qat_stability'] = [x for x in (R.qat_status(R.qat_tag(r, s, qlr[r]), R.fp32_tag(r, s))
                                       for r in R.RESOLUTIONS if qlr[r] for s in PRIMARY) if x]

    json.dump(st, open(R.SUMMARY, 'w'), indent=1, default=float)
    log('wrote', R.SUMMARY)

    rows = []
    for r in R.RESOLUTIONS:
        d, a, e = (st['resolutions'][str(r)][k] for k in ('cost', 'accuracy', 'export'))
        note = (f'FP32 lr {LOCKED_LR:g} / {LOCKED_EPOCHS} ep; QAT lr {a["qat_lr"] or "n/a"}; '
                f'n_fp32={a["fp32"]["n"]} n_qat={a["qat"]["n"]}; seeds {PRIMARY}; '
                f'bank required {d["bank_required_kib"]} KiB; '
                + ('fits the taped-out 32 KiB bank' if d['fits_taped_out_bank']
                   else 'exceeds the taped-out 32 KiB bank'))
        rows.append(dict(resolution=d['resolution'], width=R.WM, stem_stride=R.SS, params=d['params'],
                         macs_exact=d['macs_exact'], peak_activation_bytes=d['peak_activation_bytes'],
                         mem1_required_bytes=d['mem1_required_bytes'],
                         mem2_required_bytes=d['mem2_required_bytes'],
                         input_shape='x'.join(map(str, d['input_shape'])),
                         fp32_miou_mean=R.num(a['fp32'].get('test_miou_mean')),
                         fp32_miou_std=R.num(a['fp32'].get('test_miou_std')),
                         qat_miou_mean=R.num(a['qat'].get('test_miou_mean')),
                         qat_miou_std=R.num(a['qat'].get('test_miou_std')),
                         fp32_dice_mean=R.num(a['fp32'].get('test_dice_mean')),
                         fp32_dice_std=R.num(a['fp32'].get('test_dice_std')),
                         qat_dice_mean=R.num(a['qat'].get('test_dice_mean')),
                         qat_dice_std=R.num(a['qat'].get('test_dice_std')),
                         uav_iou_fp32_mean=R.num(a['fp32'].get('uav_iou_mean')),
                         uav_iou_fp32_std=R.num(a['fp32'].get('uav_iou_std')),
                         uav_iou_qat_mean=R.num(a['qat'].get('uav_iou_mean')),
                         uav_iou_qat_std=R.num(a['qat'].get('uav_iou_std')),
                         export_path='' if not e else e['path'],
                         weights_sha256='' if not e else e['weights_sha256'], notes=note))

    csv_p = f'{ROOT}/summaries/paper_resolution_fixed_w1_model.csv'
    md_p = f'{ROOT}/summaries/paper_resolution_fixed_w1_model.md'
    R.write_csv(csv_p, rows)
    R.write_md(md_p, st)
    R.link_into_paper_results(csv_p, md_p, R.SUMMARY)
    log('PHASE 1 COMPLETE -- deliverables in exp03/summaries/ and ~/paper_results/')


if __name__ == '__main__':
    main()
