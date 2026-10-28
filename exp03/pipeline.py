"""Exp03 autonomous pipeline v2 (resumable; finished runs carry a DONE file; running runs are adopted).

  nohup setsid python exp03/pipeline.py > exp03/pipeline.log 2>&1 < /dev/null &

Protocol (see exp03/summaries/exp03_summary.md "Methodological notes"):
  * seed 42 = tuning / model-selection seed; seeds 43,44,45 = final evaluation seeds (mean +- std only over these)
  * FP32: 200 epochs, DSP log-code input, best-val-mIoU checkpoint, test evaluated once at that checkpoint
  * S1  U-Net 0.5x LR search (seed 42): pick ONE lr (best mean val mIoU over D8/D16/D32)
  * S2  Exp A finals: U-Net 0.5x, seeds 43-45, all resolutions, that lr
  * S3  Exp B: kuralsnet_npu_seg over width x stem_stride x 2 LRs, identical grid at every resolution (seed 42);
        every candidate is logged with params/MACs/peak NPU buffer/feasible/reason; infeasible (>32 KB/bank) are not trained;
        pick = best val mIoU, ties within 0.01 -> fewer MACs; finals seeds 43-45
  * S4  Exp C (R64-D8): U-Net 1.0x/1.5x finals; KuRALS-Net (algorithmic reference, NOT NPU-legal) LR search + finals
  * S5  QAT: LR search on seed-42 FP32 ckpt per resolution; chosen LR applied to each final FP32 seed (paired); stability
        check + one predefined retry (next smaller LR) on failure
  * S6  report.py
"""
import glob, json, os, subprocess, sys, itertools, time, re
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '..'))
PY = os.path.expanduser('~/.venvs/kurals_train/bin/python')
EPOCHS = 200
MAXJOBS = 12
LR_GRID = [3e-3, 2e-3, 1e-3, 3e-4]
KURALSNET_LR_GRID = LR_GRID + [1e-4]          # + the baseline's own published lr
WIDTHS = [0.5, 1.0, 1.5]
STEM_STRIDES = [1, 2]
QAT_LRS = [1e-4, 3e-5, 1e-5]
TUNE_SEED = 42
FINAL_SEEDS = [43, 44, 45]
RES = [8, 16, 32]
BUF = 0x8000
QAT_EPOCHS = 100
QAT_FREEZE_ITER = 24000
QAT_FAIL_RATIO = 0.8        # QAT run is "failed" if its best val mIoU < 0.8 x the FP32 checkpoint's val mIoU
QAT_COLLAPSE_RATIO = 0.5    # "unstable" if the final-eval val mIoU < 0.5 x the run's own best (late collapse; best ckpt unaffected)
BATCH = dict(physical=6, accumulation=1, effective=6)   # identical for every resolution/model (cfg batch_size=6, no accumulation)


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def run_dir(tag):
    return f'{ROOT}/runs/{tag}'


def result(tag):
    f = glob.glob(run_dir(tag) + '/logs/KuRALS_CW/*/*/results/val_doppler_results.json')
    if not f or not os.path.exists(run_dir(tag) + '/DONE'):
        return None
    r = json.load(open(f[0])); v = r['val_metrics']['range_doppler']; t = r['test_metrics']['range_doppler']
    return dict(tag=tag, best_epoch=r['epoch'], val_miou=v['miou'], val_dice=v['dice'], test_miou=t['miou'], test_dice=t['dice'],
                test_iou_by_class=t['miou_by_class'], test_uav_iou=t['miou_by_class'][1], test_dice_by_class=t['dice_by_class'])


def ckpt(tag):
    return os.path.realpath(glob.glob(run_dir(tag) + '/logs/KuRALS_CW/*/*/results/val_doppler_model.pt')[0])


def job(model, res, seed, lr, wm=1.0, ss=1, stage='fp32', exp='X', epochs=EPOCHS, finetune=None, qat=None, tag=None, extra=None):
    tag = tag or f'{exp}_{model}_wm{wm}_ss{ss}_R64-D{res}_lr{lr:g}_s{seed}'
    cmd = [PY, f'{ROOT}/launch.py', '--wait', '--exp', exp, '--stage', stage, '--model', model, '--res', str(res), '--seed', str(seed),
           '--width-mult', str(wm), '--lr', str(lr), '--epochs', str(epochs), '--stem-stride', str(ss), '--tag', tag]
    if finetune: cmd += ['--finetune', finetune]
    if qat: cmd += ['--qat', json.dumps(qat)]
    if extra: cmd += extra
    return tag, cmd


def running(tag):
    return subprocess.run(['pgrep', '-f', f'launch.py --wait.* --tag {tag}( |$)'], capture_output=True).returncode == 0


def run_all(jobs):
    """Run jobs MAXJOBS at a time. Finished (DONE) runs are skipped; runs already executing (adopted from an earlier
    orchestrator) are waited on. One automatic retry on failure. Returns the tags that failed."""
    def one(j):
        tag, cmd = j
        while running(tag):
            time.sleep(30)
        if os.path.exists(run_dir(tag) + '/DONE'):
            return None
        for attempt in (1, 2):
            if subprocess.run(cmd).returncode == 0:
                log('done', tag); return None
            log('FAILED attempt', attempt, tag)
        return tag
    with ThreadPoolExecutor(MAXJOBS) as ex:
        failed = [t for t in ex.map(one, jobs) if t]
    if failed: log('FAILED RUNS', failed)
    return failed


def concurrency_log(stage):
    """Record how many GPUs / jobs / VRAM per job (methodological note: concurrent training is fine, timing is not)."""
    q = lambda args: subprocess.run(args, capture_output=True, text=True).stdout.strip()
    gpus = q(['nvidia-smi', '-L']).splitlines()
    apps = [l.split(',') for l in q(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader']).splitlines() if l]
    n_jobs = len(apps)
    mem = [int(re.sub(r'\D', '', a[1])) for a in apps] or [0]
    line = (f'{time.strftime("%F %T")} {stage}: physical GPUs={len(gpus)}; training processes={n_jobs} -> '
            f'{n_jobs / max(1, len(gpus)):.1f} jobs/GPU (all on GPU 0, CUDA_VISIBLE_DEVICES=0); per-job VRAM MiB min/max={min(mem)}/{max(mem)}; '
            f'total used={sum(mem)} MiB. Wall-clock of concurrent runs is NOT a benchmark; latency must be measured in isolation.')
    os.makedirs(f'{ROOT}/summaries', exist_ok=True)
    open(f'{ROOT}/summaries/concurrency_check.log', 'a').write(line + '\n'); log(line)


def profile(model, wm, ss, res):
    from kurals.npu_check import build, npu_profile
    return npu_profile(build(model, wm, stem_stride=ss, encoder_depth=1), res, 64, BUF)


# ---------------------------------------------------------------------------------------------------------------
def qat_status(qtag, ftag):
    """Stability analysis of one QAT run vs. its FP32 source (also written to summaries/qat_stability.json)."""
    q, f = result(qtag), result(ftag)
    if not q or not f:
        return None
    vals = [float(x) for x in re.findall(r'\] Val mIoU: RD=([0-9.eE+-]+)', open(run_dir(qtag) + '/train.log').read())]
    losses = [float(x) for x in re.findall(r'Train loss ([0-9.eE+-]+)', open(run_dir(qtag) + '/train.log').read())]
    flosses = [float(x) for x in re.findall(r'Train loss ([0-9.eE+-]+)', open(run_dir(ftag) + '/train.log').read())]
    cfg = json.load(open(run_dir(qtag) + '/config.json'))
    best = max(vals) if vals else None
    failed = q['val_miou'] < QAT_FAIL_RATIO * f['val_miou']
    collapsed = bool(vals) and vals[-1] < QAT_COLLAPSE_RATIO * best
    info = dict(qat_tag=qtag, fp32_tag=ftag, qat_start='iteration 0 of the fine-tune (no float warm-up; fine-tune starts from the best FP32 checkpoint)',
                lr_at_qat_start=cfg['lr'], loss_before_qat_fp32_last20=sum(flosses[-20:]) / max(1, len(flosses[-20:])),
                loss_after_qat_start_first3=sum(losses[:3]) / max(1, len(losses[:3])), val_miou_before_qat=f['val_miou'],
                val_miou_first_after_qat=vals[0] if vals else None, val_miou_best_qat=q['val_miou'], val_miou_last_qat=vals[-1] if vals else None,
                observer_freeze_iter=cfg.get('quant_freeze_iters'), bn_freeze='none: BN is folded from running statistics at every forward; BN not frozen',
                fake_quant='int8 symmetric per-tensor activations (EMA observers), per-channel weights with hardware w_eff=2*w+1, int16 scale/bias, '
                           'truncating shifts bias_shift in [5,20], act_shift in [0,7]; stem input scale fixed to 1.0 (DSP log code)',
                failed=failed, late_collapse=collapsed, status='failed' if failed else ('unstable' if collapsed else 'stable'))
    return info


def main():
    state_f = f'{ROOT}/pipeline_state.json'
    st = json.load(open(state_f)) if os.path.exists(state_f) else {}
    st['protocol'] = dict(tune_seed=TUNE_SEED, final_seeds=FINAL_SEEDS, epochs=EPOCHS, batch=BATCH, lr_grid=LR_GRID, buf_bytes=BUF)
    save = lambda: json.dump(st, open(state_f, 'w'), indent=1)
    concurrency_log('start')

    # ---- S1: LR search (tuning seed)
    log('S1 LR search')
    run_all([job('npu_unet', r, TUNE_SEED, lr, wm=0.5, exp='A') for r in RES for lr in LR_GRID])
    if 'lr_A' not in st:
        ok = {}
        for lr in LR_GRID:
            v = [(result(f'A_npu_unet_wm0.5_ss1_R64-D{r}_lr{lr:g}_s{TUNE_SEED}') or {}).get('val_miou') for r in RES]
            if all(x is not None for x in v): ok[lr] = sum(v) / len(v)
        st['lr_A'] = max(ok, key=ok.get); st['lr_A_table'] = {str(k): v for k, v in ok.items()}; save()
    lrA = st['lr_A']; log('LR_A =', lrA, st['lr_A_table'])

    # ---- candidate table for Exp B (same grid at every resolution; infeasible kept in the log)
    lrs_B = [lrA, lrA / 3]
    if 'B_candidates' not in st:
        cands = []
        for r, wm, ss, lr in itertools.product(RES, WIDTHS, STEM_STRIDES, lrs_B):
            p = profile('kuralsnet_npu_seg', wm, ss, r)
            cands.append(dict(res=r, width=wm, stem_stride=ss, lr=lr, params=p['params'], macs=p['macs'], weight_bytes=p['param_bytes'],
                              peak_npu_buffer=p['peak_npu_buffer'], feasible=p['feasible'], rejection=p['rejection']))
        st['B_candidates'] = cands; save()
    cands = st['B_candidates']

    # ---- wave 1: everything that only needs LR_A (independent, so one pool keeps the GPU busy)
    log('wave 1: Exp A finals, Exp B search, Exp C U-Nets, KuRALS-Net LR search')
    w1 = [job('npu_unet', r, s, lrA, wm=0.5, exp='A') for r in RES for s in FINAL_SEEDS]
    w1 += [job('kuralsnet_npu_seg', c['res'], TUNE_SEED, c['lr'], wm=c['width'], ss=c['stem_stride'], exp='B') for c in cands if c['feasible']]
    w1 += [job('npu_unet', 8, s, lrA, wm=wm, exp='A') for wm in (1.0, 1.5) for s in FINAL_SEEDS]
    w1 += [job('kuralsnet', 8, TUNE_SEED, lr, exp='C', extra=['--generic-size']) for lr in KURALSNET_LR_GRID]
    concurrency_log('wave 1 queued'); run_all(w1)

    picks = st.get('B_picks', {})
    for r in RES:
        scored = []
        for c in cands:
            if c['res'] != r or not c['feasible']: continue
            x = result(f"B_kuralsnet_npu_seg_wm{c['width']}_ss{c['stem_stride']}_R64-D{r}_lr{c['lr']:g}_s{TUNE_SEED}")
            if x: scored.append((x['val_miou'], c, x))
        if not scored: continue
        top = max(s[0] for s in scored)
        near = [s for s in scored if s[0] >= top - 0.01]                  # within 0.01 mIoU of the best -> lowest MACs wins
        _, c, x = min(near, key=lambda s: s[1]['macs'])
        picks[str(r)] = dict(wm=c['width'], ss=c['stem_stride'], lr=c['lr'], val_miou=x['val_miou'], macs=c['macs'], n_feasible=len(scored))
    st['B_picks'] = picks
    if 'kn_lr' not in st:
        kn = {lr: (result(f'C_kuralsnet_wm1.0_ss1_R64-D8_lr{lr:g}_s{TUNE_SEED}') or {}).get('val_miou') for lr in KURALSNET_LR_GRID}
        kn = {k: v for k, v in kn.items() if v is not None}
        if kn: st['kn_lr'] = max(kn, key=kn.get); st['kn_lr_table'] = {str(k): v for k, v in kn.items()}
    save(); log('B picks', picks, 'KuRALS-Net lr', st.get('kn_lr'))

    # ---- wave 2: finals for the picks
    log('wave 2: Exp B finals, KuRALS-Net finals')
    w2 = [job('kuralsnet_npu_seg', int(r), s, p['lr'], wm=p['wm'], ss=p['ss'], exp='B') for r, p in picks.items() for s in FINAL_SEEDS]
    if 'kn_lr' in st:
        w2 += [job('kuralsnet', 8, s, st['kn_lr'], exp='C', extra=['--generic-size']) for s in FINAL_SEEDS]
    # QAT LR search can already start from the seed-42 FP32 checkpoints
    def src_tag(r, p, s):
        return f"B_kuralsnet_npu_seg_wm{p['wm']}_ss{p['ss']}_R64-D{r}_lr{p['lr']:g}_s{s}"
    def qat_job(r, p, s, qlr, suffix=''):
        src = src_tag(r, p, s)
        return job('kuralsnet_npu_seg', int(r), s, qlr, wm=p['wm'], ss=p['ss'], stage='qat', exp='Q', epochs=QAT_EPOCHS, finetune=ckpt(src),
                   qat=dict(quant_warmup_iters=0, quant_freeze_iters=QAT_FREEZE_ITER),
                   tag=f"Q_kuralsnet_npu_seg_wm{p['wm']}_ss{p['ss']}_R64-D{r}_qlr{qlr:g}_s{s}{suffix}")
    w2 += [qat_job(r, p, TUNE_SEED, q) for r, p in picks.items() if os.path.exists(run_dir(src_tag(r, p, TUNE_SEED)) + '/DONE') for q in QAT_LRS]
    concurrency_log('wave 2 queued'); run_all(w2)

    # ---- wave 3: QAT finals (paired by seed) with the QAT LR chosen on seed 42
    qlr_pick = st.get('qat_lr', {})
    for r, p in picks.items():
        v = {q: (result(f"Q_kuralsnet_npu_seg_wm{p['wm']}_ss{p['ss']}_R64-D{r}_qlr{q:g}_s{TUNE_SEED}") or {}).get('val_miou') for q in QAT_LRS}
        v = {k: x for k, x in v.items() if x is not None}
        if v: qlr_pick[str(r)] = max(v, key=v.get)
    st['qat_lr'] = qlr_pick; save(); log('QAT lr per resolution (chosen on seed 42):', qlr_pick)
    log('wave 3: QAT finals')
    run_all([qat_job(r, picks[r], s, qlr_pick[r]) for r in qlr_pick for s in FINAL_SEEDS
             if os.path.exists(run_dir(src_tag(r, picks[r], s)) + '/DONE')])

    # ---- QAT stability check + the predefined single retry (next smaller QAT LR), all attempts kept
    stab, retry = [], []
    for r in qlr_pick:
        p = picks[r]
        for s in [TUNE_SEED] + FINAL_SEEDS:
            qlr = qlr_pick[r] if s != TUNE_SEED else None
            for q in ([qlr] if qlr else QAT_LRS):
                qt = f"Q_kuralsnet_npu_seg_wm{p['wm']}_ss{p['ss']}_R64-D{r}_qlr{q:g}_s{s}"
                info = qat_status(qt, src_tag(r, p, s))
                if not info: continue
                stab.append(info)
                smaller = [x for x in QAT_LRS if x < q]
                if info['failed'] and s != TUNE_SEED and smaller:
                    retry.append(qat_job(r, p, s, max(smaller), suffix='_retry'))
    if retry:
        log('QAT retries (policy: one retry at next smaller lr):', [t for t, _ in retry]); run_all(retry)
        for r in qlr_pick:
            p = picks[r]
            for s in FINAL_SEEDS:
                for q in QAT_LRS:
                    qt = f"Q_kuralsnet_npu_seg_wm{p['wm']}_ss{p['ss']}_R64-D{r}_qlr{q:g}_s{s}_retry"
                    info = qat_status(qt, src_tag(r, p, s))
                    if info: info['retry'] = True; stab.append(info)
    json.dump(stab, open(f'{ROOT}/summaries/qat_stability.json', 'w'), indent=1)

    concurrency_log('end'); log('S6 report')
    subprocess.run([PY, f'{ROOT}/report.py'])
    log('PIPELINE COMPLETE')


if __name__ == '__main__':
    main()
