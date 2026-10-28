"""Aggregate finished Exp03 runs into the deliverable CSVs, plots and summaries/exp03_summary.md (safe to run any time).
Final statistics = seeds 43/44/45 only; seed 42 (tuning seed) rows are labelled and never enter mean/std."""
import csv, glob, json, os, sys, statistics as stt
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(ROOT, '..')); sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import pipeline as P
from kurals.npu_check import build, npu_profile, generic_cost
from kurals.models import KuRALSNet

st = json.load(open(f'{ROOT}/pipeline_state.json')) if os.path.exists(f'{ROOT}/pipeline_state.json') else {}
os.makedirs(f'{ROOT}/plots', exist_ok=True); os.makedirs(f'{ROOT}/summaries', exist_ok=True)
FIN, TUNE = P.FINAL_SEEDS, P.TUNE_SEED
_cache = {}
def prof(model, wm, ss, res):
    k = (model, wm, ss, res)
    if k not in _cache:
        if model == 'kuralsnet':
            c = generic_cost(KuRALSNet(4, 1, generic_size=True), res, 64, five_d=True)
            _cache[k] = dict(params=c['params'], macs=c['macs'], peak_act_bytes=c['peak_act_bytes'], peak_npu_buffer=None, feasible=False, param_bytes=None)
        else:
            _cache[k] = npu_profile(build(model, wm, stem_stride=ss, encoder_depth=1), res, 64, P.BUF)
    return _cache[k]
def ms(vals):
    return (stt.mean(vals), stt.pstdev(vals) if len(vals) > 1 else 0.0) if vals else (float('nan'), float('nan'))
def fmt(t): return f'{t[0]:.4f} ± {t[1]:.4f}'
def write(name, rows):
    if rows:
        with open(f'{ROOT}/{name}', 'w', newline='') as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
def agg(tags):
    rs = [x for x in (P.result(t) for t in tags) if x]
    if not rs: return None
    d = dict(n=len(rs), seeds=[int(x['tag'].rsplit('_s', 1)[1].split('_')[0]) for x in rs])
    for k in ('val_miou', 'test_miou', 'test_dice', 'test_uav_iou'): d[k] = ms([x[k] for x in rs])
    d['iou_by_class'] = [float(np.mean([x['test_iou_by_class'][c] for x in rs])) for c in range(4)]
    return d
SEEDNOTE = f'tuning seed {TUNE}; final seeds {FIN}'
lrA = st.get('lr_A')
md = ['# Exp03 summary (auto-generated; regenerate with `python exp03/report.py`)\n']

# ----------------------------------------------------------------------------------- 1. resolution-only
rowsA, aggA, ptsA = [], [], []
if lrA:
    for r in P.RES:
        pf = prof('npu_unet', 0.5, 1, r)
        for s in [TUNE] + FIN:
            x = P.result(f'A_npu_unet_wm0.5_ss1_R64-D{r}_lr{lrA:g}_s{s}')
            if x: rowsA.append(dict(resolution=f'R64-D{r}', seed=s, role='tuning (excluded from mean/std)' if s == TUNE else 'final', LR=lrA,
                                    best_val_epoch=x['best_epoch'], best_val_miou=round(x['val_miou'], 4), test_miou=round(x['test_miou'], 4),
                                    test_dice=round(x['test_dice'], 4), test_uav_iou=round(x['test_uav_iou'], 4), params=pf['params'], MACs=pf['macs'],
                                    peak_activation_memory=pf['peak_act_bytes'], peak_npu_buffer=pf['peak_npu_buffer'],
                                    physical_batch=P.BATCH['physical'], effective_batch=P.BATCH['effective']))
        a = agg([f'A_npu_unet_wm0.5_ss1_R64-D{r}_lr{lrA:g}_s{s}' for s in FIN])
        if a: aggA.append(dict(resolution=f'R64-D{r}', cells=64 * r, n_final_seeds=a['n'], val_miou=fmt(a['val_miou']), test_miou=fmt(a['test_miou']),
                               test_dice=fmt(a['test_dice']), test_uav_iou=fmt(a['test_uav_iou'])));  ptsA.append((r, a))
write('resolution_reference_unet.csv', rowsA); write('resolution_reference_unet_aggregate.csv', aggA)
if ptsA:
    plt.figure(figsize=(4, 3)); plt.errorbar([64 * r for r, _ in ptsA], [a['test_miou'][0] for _, a in ptsA], [a['test_miou'][1] for _, a in ptsA], marker='o', capsize=3)
    plt.xscale('log', base=2); plt.xticks([64 * r for r, _ in ptsA], [f'D{r}\n{64*r}' for r, _ in ptsA]); plt.xlabel('RD cells'); plt.ylabel('test mIoU (mean ± std, 3 seeds)')
    plt.title('NPU-UNet 0.5x, FP32'); plt.tight_layout(); plt.savefig(f'{ROOT}/plots/unet_resolution_accuracy.pdf')
md.append('## 1. Resolution-only reference (Experiment A)\n')
md.append(f'Common model NPU-UNet 0.5x, ONE LR for all resolutions chosen on the tuning seed: LR = {lrA} (mean val mIoU over D8/D16/D32 per candidate: {st.get("lr_A_table")}). '
          f'Final statistics over seeds {FIN}.\n')
md += [f"- {r['resolution']} ({r['cells']} cells): test mIoU {r['test_miou']}, test Dice {r['test_dice']}, UAV IoU {r['test_uav_iou']} (n={r['n_final_seeds']})" for r in aggA] or ['- (pending)']
md.append('\nSupports statements about resolution itself (same architecture, optimiser, loss, schedule, augmentation, selection rule, evaluation code, batch).\n')

# ----------------------------------------------------------------------------------- 2. co-design
rowsB, searchB, ptsB = [], [], []
for c in st.get('B_candidates', []):
    x = P.result(f"B_kuralsnet_npu_seg_wm{c['width']}_ss{c['stem_stride']}_R64-D{c['res']}_lr{c['lr']:g}_s{TUNE}") if c['feasible'] else None
    pk = st.get('B_picks', {}).get(str(c['res']))
    sel = bool(pk and pk['wm'] == c['width'] and pk['ss'] == c['stem_stride'] and abs(pk['lr'] - c['lr']) < 1e-12)
    searchB.append(dict(resolution=f"R64-D{c['res']}", width=c['width'], stem_stride=c['stem_stride'], LR=c['lr'], params=c['params'], MACs=c['macs'],
                        peak_npu_buffer=c['peak_npu_buffer'], feasible='yes' if c['feasible'] else 'no', rejection_reason=c['rejection'] or '',
                        val_miou_seed42=round(x['val_miou'], 4) if x else '', trained='yes' if x else ('no (infeasible)' if not c['feasible'] else 'pending'), selected='yes' if sel else ''))
write('codesign_search_log.csv', searchB)
for r, p in st.get('B_picks', {}).items():
    pf = prof('kuralsnet_npu_seg', p['wm'], p['ss'], int(r))
    for s in [TUNE] + FIN:
        x = P.result(f"B_kuralsnet_npu_seg_wm{p['wm']}_ss{p['ss']}_R64-D{r}_lr{p['lr']:g}_s{s}")
        if x: rowsB.append(dict(resolution=f'R64-D{r}', width=p['wm'], stem_stride=p['ss'], LR=p['lr'], seed=s, role='tuning (excluded)' if s == TUNE else 'final',
                                params=pf['params'], MACs=pf['macs'], peak_npu_buffer=pf['peak_npu_buffer'], feasible='yes' if pf['feasible'] else 'no',
                                best_val_epoch=x['best_epoch'], best_val_miou=round(x['val_miou'], 4), test_miou=round(x['test_miou'], 4), test_dice=round(x['test_dice'], 4)))
    a = agg([f"B_kuralsnet_npu_seg_wm{p['wm']}_ss{p['ss']}_R64-D{r}_lr{p['lr']:g}_s{s}" for s in FIN])
    if a: ptsB.append((pf, a, f'D{r}', p))
write('codesign_resolution_systems.csv', rowsB)
if ptsB:
    plt.figure(figsize=(4.2, 3))
    for pf, a, l, _ in ptsB: plt.errorbar(pf['macs'] / 1e6, a['test_miou'][0], a['test_miou'][1], marker='o', capsize=3, label=l)
    plt.xlabel('MACs (M)'); plt.ylabel('test mIoU'); plt.legend(); plt.tight_layout(); plt.savefig(f'{ROOT}/plots/codesign_accuracy_vs_macs.pdf')
md.append('## 2. Resolution-specific co-designed systems (Experiment B)\n')
md.append('Maximise validation mIoU subject to peak NPU buffer <= 32 KB per bank (see docs/NPU_BUFFER_MODEL.md), tie within 0.01 mIoU -> fewer MACs. '
          'Identical search grid at every resolution (width x stem_stride x 2 LRs), chosen on the tuning seed; every candidate is in `codesign_search_log.csv`.\n')
for pf, a, l, p in ptsB:
    md.append(f"- {l}: width {p['wm']}, stem_stride {p['ss']}, lr {p['lr']:g}: test mIoU {fmt(a['test_miou'])}, Dice {fmt(a['test_dice'])}, UAV IoU {fmt(a['test_uav_iou'])}; "
              f"{pf['params']} params, {pf['macs']/1e6:.2f} M MACs, peak NPU buffer {pf['peak_npu_buffer']} B (n={a['n']})")
n_feas = {r: sum(1 for c in st.get('B_candidates', []) if c['res'] == r and c['feasible']) for r in P.RES}
md.append(f"- Feasible candidates per resolution (of {len(P.WIDTHS)*len(P.STEM_STRIDES)*2}): {n_feas}. Fewer feasible options at higher resolution is a direct consequence of the buffer limit, "
          'not of a smaller search budget (the grid is identical; infeasible candidates are logged, not trained).')
md.append('\nSupports statements about co-design; does NOT isolate resolution (architecture and training differ across rows).\n')

# ----------------------------------------------------------------------------------- 3. D8 comparison
rowsC, ptsC = [], []
def addC(name, model, wm, ss, tags, legal, reason):
    a = agg(tags)
    if not a: return
    pf = prof(model, wm, ss, 8)
    rowsC.append(dict(model=name, n_final_seeds=a['n'], test_miou=fmt(a['test_miou']), test_dice=fmt(a['test_dice']), test_uav_iou=fmt(a['test_uav_iou']),
                      per_class_iou_mean=[round(v, 4) for v in a['iou_by_class']], params=pf['params'], MACs=pf['macs'], peak_activation_memory=pf['peak_act_bytes'],
                      peak_npu_buffer=pf['peak_npu_buffer'] if pf['peak_npu_buffer'] is not None else 'N/A', npu_legal=legal, npu_feasible_32KB=('yes' if pf['feasible'] else 'no'),
                      npu_latency='N/A' if model == 'kuralsnet' else 'pending', note=reason))
    ptsC.append((pf['macs'], a, name))
if lrA:
    for wm in (0.5, 1.0, 1.5):
        addC(f'NPU-UNet {wm}x', 'npu_unet', wm, 1, [f'A_npu_unet_wm{wm}_ss1_R64-D8_lr{lrA:g}_s{s}' for s in FIN], 'not checked (baseline)', 'LR from Exp A search')
p = st.get('B_picks', {}).get('8')
if p: addC('Proposed (D8 co-designed KuRALSNetNPUSeg)', 'kuralsnet_npu_seg', p['wm'], p['ss'], [f"B_kuralsnet_npu_seg_wm{p['wm']}_ss{p['ss']}_R64-D8_lr{p['lr']:g}_s{s}" for s in FIN], 'yes', 'Exp B pick')
if st.get('kn_lr'):
    addC('KuRALS-Net (algorithmic reference)', 'kuralsnet', 1.0, 1, [f"C_kuralsnet_wm1.0_ss1_R64-D8_lr{st['kn_lr']:g}_s{s}" for s in FIN], 'not checked (baseline)',
         'Original KuRALS-Net as an accuracy reference; NPU legality is not required for baselines and was not checked. Run with size-adaptive global pooling (identical to the original at native size).')
write('r64d8_model_comparison.csv', rowsC)
if ptsC:
    plt.figure(figsize=(4.4, 3))
    for m, a, l in ptsC: plt.errorbar(m / 1e6, a['test_miou'][0], a['test_miou'][1], marker='o', capsize=3, label=l[:26])
    plt.xscale('log'); plt.xlabel('MACs (M)'); plt.ylabel('test mIoU'); plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(f'{ROOT}/plots/r64d8_model_pareto.pdf')
md.append('## 3. R64-D8 model comparison (Experiment C)\n')
md += [f"- {r['model']}: test mIoU {r['test_miou']}, UAV IoU {r['test_uav_iou']}, {r['params']} params, {r['MACs']/1e6:.2f} M MACs, peak act {r['peak_activation_memory']} B, NPU legal: {r['npu_legal']}, latency: {r['npu_latency']}" for r in rowsC] or ['- (pending)']
md.append('\nOnly the proposed model is required to be NPU-legal; baselines (U-Net, KuRALS-Net) are not legality-checked. Claims here are about accuracy-complexity operating points, not latency.\n')

# ----------------------------------------------------------------------------------- 4. QAT
rowsQ, pairs = [], {}
stab = {i['qat_tag']: i for i in (json.load(open(f'{ROOT}/summaries/qat_stability.json')) if os.path.exists(f'{ROOT}/summaries/qat_stability.json') else [])}
for r, p in st.get('B_picks', {}).items():
    for s in [TUNE] + FIN:
        f32t = f"B_kuralsnet_npu_seg_wm{p['wm']}_ss{p['ss']}_R64-D{r}_lr{p['lr']:g}_s{s}"; f = P.result(f32t)
        qs = [(q, sfx) for q in P.QAT_LRS for sfx in ('', '_retry')]
        for q, sfx in qs:
            qt = f"Q_kuralsnet_npu_seg_wm{p['wm']}_ss{p['ss']}_R64-D{r}_qlr{q:g}_s{s}{sfx}"; x = P.result(qt)
            if f and x:
                status = stab.get(qt, {}).get('status', P.qat_status(qt, f32t)['status'] if P.qat_status(qt, f32t) else 'n/a')
                chosen = (s != TUNE and abs(st.get('qat_lr', {}).get(r, -1) - q) < 1e-12 and sfx == '') or (sfx == '_retry')
                rowsQ.append(dict(resolution=f'R64-D{r}', seed=s, role='tuning (QAT-LR search)' if s == TUNE else 'final', fp32_checkpoint=f32t, fp32_test_miou=round(f['test_miou'], 4),
                                  QAT_LR=q, QAT_best_epoch=x['best_epoch'], QAT_test_miou=round(x['test_miou'], 4), quantization_delta=round(f['test_miou'] - x['test_miou'], 4),
                                  status=status, retry='yes' if sfx else '', used_for_final_stats='yes' if (chosen and s != TUNE) else ''))
write('qat_resolution_comparison.csv', rowsQ)
md.append('## 4. QAT (separate from FP32)\n')
for r in st.get('B_picks', {}):
    d = [x['quantization_delta'] for x in rowsQ if x['resolution'] == f'R64-D{r}' and x['used_for_final_stats'] == 'yes' and x['status'] != 'failed']
    nf = [x for x in rowsQ if x['resolution'] == f'R64-D{r}' and x['used_for_final_stats'] == 'yes' and x['status'] == 'failed']
    md.append(f"- D{r}: paired quantization delta (FP32 - QAT test mIoU) {fmt(ms(d))} over {len(d)} seeds; failed runs: {[(x['seed'], x['QAT_LR']) for x in nf]}")
pq = {}
for x in rowsQ:
    if x['used_for_final_stats'] == 'yes': pq.setdefault(x['resolution'], []).append(x)
if pq:
    plt.figure(figsize=(4.4, 3))
    for i, (res, xs) in enumerate(sorted(pq.items())):
        for x in xs: plt.plot([i - 0.15, i + 0.15], [x['fp32_test_miou'], x['QAT_test_miou']], marker='o', color='C%d' % i, alpha=0.7)
    plt.xticks(range(len(pq)), sorted(pq)); plt.ylabel('test mIoU'); plt.title('FP32 (left) vs QAT (right), paired by seed'); plt.tight_layout(); plt.savefig(f'{ROOT}/plots/qat_fp32_vs_int.pdf')
md.append('\nQAT starts from the best-validation FP32 checkpoint of the same seed; QAT LR chosen on seed 42. Stability details: summaries/qat_stability.json.\n')

# ----------------------------------------------------------------------------------- 5/6 feasibility, pending
md.append('## 5. NPU feasibility\n')
md.append('Per-candidate buffer demand is in `codesign_search_log.csv` (peak_npu_buffer, feasible, rejection_reason). Definition of the 32 KB constraint: `docs/NPU_BUFFER_MODEL.md` '
          '(per-bank allocator watermark from the NPU8 cfg generator; weights stream from DRAM in the frame-reuse region; line buffers/accumulators excluded).\n')
md.append('## 6. Pending: measured NPU latency\n')
md.append('No NPU cycle/latency numbers exist yet. MACs, parameter count, peak NPU buffer and activation memory are implementation proxies only; this report makes no latency claim. '
          'When cycle counts arrive, add latency-vs-mIoU plots (do not replace the MAC plots) and rerun the Pareto analysis.\n')

# ----------------------------------------------------------------------------------- preprocessing / batch / concurrency
md.append('## Data preparation of R64-D8 / D16 / D32\n')
md.append('- Source: KuRALS-CW native range-Doppler maps, 124 Doppler x 2048 range bins (Doppler axis = raw bins [2:126] of 128; the crop is identical for all three).')
md.append('- Resolution reduction (`kurals/dataset_process/resize_extracted_dataset.py`, same code for all three): **block max-pooling** of the linear-magnitude map over contiguous, near-equal bin groups '
          '(`np.array_split`): range 2048 -> 64 = exactly 32 bins per cell; Doppler 124 -> {8,16,32} = groups of 15/16, 7/8, 3/4 bins (not an integer ratio, so group sizes differ by one).')
md.append('- Labels: not resampled as images; every foreground cell of the native one-hot dense mask is remapped through the same bin lookup, background = complement (dilate_radius = 0, so no label dilation at any resolution).')
md.append('- The inputs are produced **after** the native FFT by pooling, i.e. an emulation; the real SoC computes its own small FFT directly from raw samples (not reproducible here, no raw IQ). Accuracy differences between resolutions therefore include this emulation effect.')
md.append('- Normalisation (same for all): DSP 5.3-bit log2 code, `q = clamp(log8(floor(v) >> 16) - 0, 0, 127)` (npu_riscv proxy bridge, LOG_SCALE = 0), fed with input scale 1.0; Doppler axis ifftshifted to natural FFT order for both input and labels.')
md.append('- Augmentation: random horizontal/vertical flip (identical in every run).')
try:
    import glob as _g
    dens = {}
    for r in P.RES:
        fs = sorted(_g.glob(os.path.expanduser(f'~/datasets_local/KuRALS_CW_{r}x64/KuRALS_CW/*/annotations/dense/*/range_doppler.npy')))[::10]
        dens[r] = np.mean([np.load(f)[1:].sum(axis=(1, 2)) for f in fs], axis=0).round(2).tolist()
    md.append(f'- Mean foreground cells per frame per class (class 1 = UAV, 2 = pedestrian, 3 = vehicle; 8 of 9 sequences are pure-UAV), a label-density side effect of resolution: ' + ', '.join(f'D{r}: {v}' for r, v in dens.items()) + '. Class 3 is nearly absent, so mIoU is capped by it at every resolution.')
    base = os.path.expanduser('~/datasets_local/KuRALS_CW_%dx64/KuRALS_CW')
    ref = sorted(_g.glob(base % 8 + '/*/annotations/dense/*/range_doppler.npy'))
    pick = next((f for f in ref[::7] if np.load(f)[1].sum() > 0), ref[0])
    seq, frame = pick.split('/KuRALS_CW/')[1].split('/annotations/dense/'); frame = frame.split('/')[0]
    import torch; from kurals.utils.functions import dsp_log_quantize
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.6))
    for a_, r in zip(ax, P.RES):
        root = base % r
        rd = np.load(f'{root}/{seq}/range_doppler_numpy/{frame}.npy'); m = np.load(f'{root}/{seq}/annotations/dense/{frame}/range_doppler.npy')
        a_.imshow(dsp_log_quantize(torch.from_numpy(rd)).numpy(), aspect='auto', cmap='viridis')
        ys, xs = np.where(m[1:].sum(0) > 0); a_.scatter(xs, ys, s=60, facecolors='none', edgecolors='r'); a_.set_title(f'R64-D{r} (stored fftshifted order)')
    plt.tight_layout(); plt.savefig(f'{ROOT}/plots/preprocessing_d8_d16_d32.pdf')
    md.append('- Visualisation of one sample + labels at the three resolutions: `plots/preprocessing_d8_d16_d32.pdf`.')
except Exception as e:
    md.append(f'- (visualisation/density not generated: {e!r})')
md.append(f"\n## Batch size\nPhysical batch {P.BATCH['physical']}, gradient accumulation {P.BATCH['accumulation']}, effective batch {P.BATCH['effective']} for every model and resolution (config batch_size=6; the small models fit on the GPU at all three resolutions, so no accumulation was needed).")
md.append('\n## Concurrency\n```\n' + (open(f'{ROOT}/summaries/concurrency_check.log').read() if os.path.exists(f'{ROOT}/summaries/concurrency_check.log') else '(none logged)') + '```')
md.append('\n## Methodological notes\n'
          f'- Seed {TUNE} is the hyper-parameter / model-selection seed (LR search, Exp B candidate choice, QAT-LR choice). Final mean ± std use seeds {FIN} only; seed-{TUNE} rows are labelled "tuning" in the CSVs.\n'
          '- Checkpoint = best validation mIoU epoch; test evaluated once at that checkpoint (the trainer also stores a best-test checkpoint, which is never used). Final-epoch numbers are never reported.\n'
          '- FP32 and QAT are separate stages; QAT starts from the best FP32 checkpoint of the same seed and is paired per seed.\n'
          '- The 32 KB per-bank NPU constraint is applied during the co-design search (candidates over the limit are logged and not trained).\n'
          f'- Unequal-budget caveat: tuning budget per resolution is the same grid; feasible candidates differ by resolution because of the constraint. Exp A uses {P.EPOCHS} epochs; the best-val epoch is logged per run (runs whose best epoch is near {P.EPOCHS} may be under-trained).\n'
          '- Several concurrent runs shared one GPU; wall-clock is not a benchmark. Measured NPU latency is still pending, so no latency claim is made.\n'
          '- An earlier batch at lr 1e-4 / 100 epochs was discarded (all-background plateau) and is kept only in exp03/runs_invalid_lr1e-4_ep100 and runs_100ep_lrsearch.')
open(f'{ROOT}/summaries/exp03_summary.md', 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
