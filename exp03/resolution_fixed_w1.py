"""PRIMARY RD-resolution experiment: one fixed 1.0x stride-1 KuRALSNetNPUSeg at R64-D8/D16/D32.

Driven by ~/resolution_codesign_fixed_w1_handoff.md (Phase 1, MODEL SERVER). The question is

    if the semantic network is held fixed, what accuracy / compute / memory does each RD
    resolution cost?

so width, stride, topology, operator set, quantization and training recipe are frozen and only the
RD grid -- and the activation-bank capacity that grid requires -- may change.

This REPLACES the old fixed-32-KiB co-design search as the main D8/D16/D32 comparison. That search
held the bank at 0x8000 and let width/stride float, which picked 0.5x-ss1 / 0.5x-ss1 / 0.5x-ss2 and
therefore compared three *different* networks; it survives as a secondary stress test only (see
codesign_search_log.csv and summaries/design_point_decision.md). Nothing here reads or rewrites it.

Protocol, identical at all three resolutions (pipeline.py's Exp B / run_w10_finals.py):
  S1  FP32, 200 epochs, lr 1e-3 = lr_A (pipeline_state.json), seeds 42 + 43/44/45
  S2  QAT LR search on the seed-42 FP32 checkpoint over QAT_LRS = {1e-4, 3e-5, 1e-5}, 100 epochs
  S3  QAT finals, seeds 43/44/45, at the seed-42 winner, each paired to its OWN seed's FP32 parent
  S4  costs from the built graph + gen_cfg.py's allocator, with NO 32-KiB rejection rule
  S5  exports model_release_d{8,16,32}_w1.0_s1, each checked at the bank size it actually needs
  S6  paper_resolution_fixed_w1_model.{md,csv}
Selection is always highest VALIDATION mIoU; test is read once at that checkpoint.

R64-D8 is already trained (run_w10_finals.py, 2026-10-21) under exactly these tags and this recipe,
so it is reused rather than retrained -- that is what makes the three rows comparable. Only D16 and
D32 are new. Resumable: runs with a DONE file are skipped.

  cd ~/xsrc/KuRALS && setsid nohup ~/.venvs/kurals_train/bin/python exp03/resolution_fixed_w1.py \
      > exp03/resolution_fixed_w1.out 2>&1 < /dev/null &
"""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))
from pipeline import (EPOCHS, FINAL_SEEDS, PY, QAT_EPOCHS, QAT_FREEZE_ITER, QAT_LRS,  # noqa: E402
                      TUNE_SEED, ckpt, job, log, qat_status, result, run_all, run_dir)

MODEL = 'kuralsnet_npu_seg'
WM = 1.0                      # frozen: the primary comparison may not change width
SS = 1                        # frozen: full-resolution output head at every resolution
ED = 1                        # encoder_depth, as in the hardware-paper model
RANGE_BINS = 64
RESOLUTIONS = [8, 16, 32]
FP32_LR = 1e-3                # lr_A, the Exp B winner; the same value D8's 1.0x finals used
TAPED_OUT_BANK = 0x8000       # the shipped NPU8 capacity -- reported for reference, NOT a filter

# From the handoff, to be confirmed against the built graph rather than trusted.
EXPECTED = {'params': 31139, 'macs': {8: 4952576, 16: 9905152, 32: 19810304},
            'bank_bytes': {8: 32768, 16: 65536, 32: 131072},
            'frame_reuse_cells': {8: 512, 16: 1024, 32: 2048}}

SUMMARY = f'{ROOT}/summaries/resolution_fixed_w1.json'
PAPER_DIR = os.path.expanduser('~/paper_results')


def fp32_tag(res, seed):
    return f'B_{MODEL}_wm{WM}_ss{SS}_R{RANGE_BINS}-D{res}_lr{FP32_LR:g}_s{seed}'


def qat_tag(res, seed, qlr):
    return f'Q_{MODEL}_wm{WM}_ss{SS}_R{RANGE_BINS}-D{res}_qlr{qlr:g}_s{seed}'


def done(tag):
    return os.path.exists(run_dir(tag) + '/DONE')


def mean_sd(v):
    """Mean and SAMPLE std (n-1), the convention used everywhere else in exp03."""
    if not v:
        return None, None
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    return m, sd


# --------------------------------------------------------------------------------- S4: graph costs
def shape_walk(topo, h, w, pf=8):
    """Per-layer in/out shapes of the NPU layer list, with channels padded to the PF=8 datapath.

    This is the same traversal kurals/npu_check.py's cost() does, kept separately here because the
    hardware server needs the shapes themselves (handoff Phase 1 step 6 / Phase 2 step 2), not just
    the aggregate it returns."""
    pad = lambda c: -(-c // pf) * pf
    rows, hh, ww = [], h, w
    for e in topo:
        k, ci, co, st = e['kernel_size'], e['in_ch'], e['out_ch'], e['stride']
        oh, ow = -(-hh // st), -(-ww // st)
        row = dict(name=e['name'], kernel=k, stride=st, depthwise=e['depthwise'],
                   in_ch=ci, in_ch_padded=pad(ci), out_ch=co, out_ch_padded=pad(co),
                   in_hw=[hh, ww], out_hw=[oh, ow],
                   in_cells=hh * ww, out_cells=oh * ow,
                   in_bytes=pad(ci) * hh * ww, out_bytes=pad(co) * oh * ow,
                   concat_operand_bytes=(pad(ci) * hh * ww if 'route_source' in e else 0),
                   maxpool_after=bool(e.get('maxpool_after')), upsample_after=bool(e.get('upsample_after')))
        rows.append(row)
        hh, ww = oh, ow
        if e.get('maxpool_after'):
            hh, ww = hh // 2, ww // 2
        if e.get('upsample_after'):
            hh, ww = hh * 2, ww * 2
    return rows


def shape_derived_depths(rows, h, w):
    """High-water marks the hardware structures must cover, read off the layer shapes.

    These are MINIMA implied by the network's geometry, not RTL depths: mapping them onto MEM1/MEM2,
    the partial-sum banks, the output writer, the line buffer, the maxpool buffer and the fused
    upsample buffer is the NPU server's job (handoff Phase 2 step 2), which is why each number below
    says what it counts rather than naming an RTL parameter."""
    mp = [r for r in rows if r['maxpool_after']]
    up = [r for r in rows if r['upsample_after']]
    return dict(
        frame_cells=h * w,
        frame_reuse_cells=h * w,
        max_layer_out_cells_per_channel=max(r['out_cells'] for r in rows),
        max_layer_in_cells_per_channel=max(r['in_cells'] for r in rows),
        partial_sum_cells_per_channel=max(r['out_cells'] for r in rows),
        line_buffer_cells=max((r['kernel'] - 1) * r['in_hw'][1] for r in rows),
        line_buffer_rows=max(r['kernel'] - 1 for r in rows),
        maxpool_input_row_cells=max((r['in_hw'][1] for r in mp), default=0),
        maxpool_input_cells=max((r['in_cells'] for r in mp), default=0),
        upsample_input_cells=max((r['out_cells'] for r in up), default=0),
        upsample_output_cells=max((r['out_cells'] * 4 for r in up), default=0),
        definition=dict(
            cells='feature-map positions for ONE channel; multiply by the padded channel count for bytes at 1 B/activation',
            partial_sum_cells_per_channel='largest single-layer output map, i.e. the accumulation high-water mark',
            line_buffer_cells='(kernel-1) input rows of the widest layer input',
            upsample_output_cells='cells after the fused 2x nearest upsample of the deepest hop that actually strides',
            frame_reuse_cells='H*W input cells of one RD frame, the frame-reuse working set',
            handoff_phase2='frame_reuse_cells is the handoff\'s MAX_OBUFF_SZ (512/1024/2048) and '
                           'upsample_input_cells is its upsample depth (128/256/512)'))


def costs(res):
    """Everything the handoff's step 6 asks for, from the graph and from gen_cfg.py's allocator.

    The allocator is run with an effectively unlimited bank (npu_profile), so what comes back is the
    capacity this resolution REQUIRES. The old `--buf-size 0x8000` rejection rule is deliberately
    not applied: here the bank size is a result, not a constraint."""
    from kurals.npu_check import build, npu_profile, topology
    net = build(MODEL, WM, stem_stride=SS, encoder_depth=ED)
    p = npu_profile(net, res, RANGE_BINS, TAPED_OUT_BANK)
    if not p['ok']:
        raise SystemExit(f'gen_cfg.py could not allocate R{RANGE_BINS}-D{res} at all: {p.get("error")}')
    rows = shape_walk(topology(net), res, RANGE_BINS)
    bank = p['peak_npu_buffer']
    c = dict(resolution=f'R{RANGE_BINS}-D{res}', doppler_bins=res, range_bins=RANGE_BINS,
             width_mult=WM, stem_stride=SS, encoder_depth=ED, layers=p['layers'], cutpoint=p['cutpoint'],
             params=p['params'], macs_exact=p['macs'], weight_bytes=p['param_bytes'],
             input_shape=[1, res, RANGE_BINS], input_elements=res * RANGE_BINS,
             peak_activation_bytes=p['peak_act_bytes'],
             mem1_required_bytes=p['buf1'], mem2_required_bytes=p['buf2'], mem3_required_bytes=p['buf3'],
             bank_required_bytes=bank, bank_required_kib=bank // 1024,
             fits_taped_out_bank=p['feasible'], taped_out_bank_bytes=TAPED_OUT_BANK,
             shape_derived_depths=shape_derived_depths(rows, res, RANGE_BINS), layer_shapes=rows)
    mismatch = []
    if p['params'] != EXPECTED['params']:
        mismatch.append(f'params {p["params"]} != handoff {EXPECTED["params"]}')
    if p['macs'] != EXPECTED['macs'][res]:
        mismatch.append(f'macs {p["macs"]} != handoff {EXPECTED["macs"][res]}')
    if bank != EXPECTED['bank_bytes'][res]:
        mismatch.append(f'bank {bank} != handoff {EXPECTED["bank_bytes"][res]}')
    if c['shape_derived_depths']['frame_reuse_cells'] != EXPECTED['frame_reuse_cells'][res]:
        mismatch.append('frame_reuse_cells != handoff')
    c['handoff_expectation'] = 'confirmed' if not mismatch else '; '.join(mismatch)
    log(f'R{RANGE_BINS}-D{res}: params={p["params"]} macs={p["macs"]:,} bank_required={bank:,} B '
        f'({bank // 1024} KiB) peak_act={p["peak_act_bytes"]:,} B fits_0x8000={p["feasible"]} '
        f'-> handoff {c["handoff_expectation"]}')
    return c


# ------------------------------------------------------------------------------ S1-S3: the training
def qjob(res, seed, qlr):
    return job(MODEL, res, seed, qlr, wm=WM, ss=SS, stage='qat', exp='Q', epochs=QAT_EPOCHS,
               finetune=ckpt(fp32_tag(res, seed)),
               qat=dict(quant_warmup_iters=0, quant_freeze_iters=QAT_FREEZE_ITER),
               tag=qat_tag(res, seed, qlr))


def train(todo):
    """FP32 -> QAT LR search -> QAT finals for the resolutions in `todo`.

    One pool per stage across all resolutions, not one resolution at a time: run_all already caps
    concurrency at MAXJOBS, and queueing D16 and D32 together keeps the GPU full. Returns the chosen
    QAT LR per resolution."""
    log(f'S1 FP32: lr {FP32_LR:g}, {EPOCHS} epochs, seeds {[TUNE_SEED] + FINAL_SEEDS}, resolutions {todo}')
    run_all([job(MODEL, r, s, FP32_LR, wm=WM, ss=SS, exp='B', epochs=EPOCHS, tag=fp32_tag(r, s))
             for r in todo for s in [TUNE_SEED] + FINAL_SEEDS])

    ready = [r for r in todo if done(fp32_tag(r, TUNE_SEED))]
    for r in set(todo) - set(ready):
        log(f'ABORT R{RANGE_BINS}-D{r}: seed-{TUNE_SEED} FP32 parent missing, no QAT LR search possible')
    log(f'S2 QAT LR search on seed {TUNE_SEED} over {[f"{q:g}" for q in QAT_LRS]}, resolutions {ready}')
    run_all([qjob(r, TUNE_SEED, q) for r in ready for q in QAT_LRS])

    qlr = {}
    for r in ready:
        v = {q: x['val_miou'] for q, x in ((q, result(qat_tag(r, TUNE_SEED, q))) for q in QAT_LRS) if x}
        if not v:
            log(f'ABORT R{RANGE_BINS}-D{r}: no QAT LR-search run produced a result')
            continue
        qlr[r] = max(v, key=v.get)
        log(f'R{RANGE_BINS}-D{r}: QAT LR = {qlr[r]:g} from seed-{TUNE_SEED} val mIoU '
            + str({f'{k:g}': round(x, 4) for k, x in v.items()}))

    log(f'S3 QAT finals, seeds {FINAL_SEEDS}, each paired to its own seed\'s FP32 checkpoint')
    failed = run_all([qjob(r, s, qlr[r]) for r in qlr for s in FINAL_SEEDS if done(fp32_tag(r, s))])
    if failed:
        log('QAT finals FAILED:', failed)
    return qlr


# ------------------------------------------------------------------------------------- aggregation
def accuracy(res, qlr):
    """FP32 and QAT accuracy over the final seeds. mIoU/Dice are the 4-class means including
    background; UAV IoU is class 1. Test is whatever the best-validation checkpoint scored."""
    out = dict(fp32_lr=FP32_LR, qat_lr=None if qlr is None else float(qlr), fp32={}, qat={})
    for q, qq in (('fp32', None), ('qat', qlr)):
        per = {}
        for s in FINAL_SEEDS:
            tag = fp32_tag(res, s) if q == 'fp32' else (qat_tag(res, s, qq) if qq is not None else None)
            r = result(tag) if tag else None
            per[str(s)] = None if not r else dict(
                tag=tag, best_epoch=r['best_epoch'], val_miou=r['val_miou'], test_miou=r['test_miou'],
                test_dice=r['test_dice'], test_uav_iou=r['test_uav_iou'],
                test_iou_by_class=r['test_iou_by_class'], test_dice_by_class=r['test_dice_by_class'])
        got = [x for x in per.values() if x]
        agg = dict(n=len(got))
        for key, field in (('test_miou', 'test_miou'), ('test_dice', 'test_dice'), ('uav_iou', 'test_uav_iou'),
                           ('val_miou', 'val_miou')):
            m, sd = mean_sd([x[field] for x in got])
            agg[f'{key}_mean'], agg[f'{key}_std'] = m, sd
        out[q] = dict(per_seed=per, **agg)
    best = None
    if qlr is not None:
        cands = [(r['val_miou'], s) for s in FINAL_SEEDS for r in [result(qat_tag(res, s, qlr))] if r]
        if cands:
            _, s = max(cands)
            best = dict(seed=s, tag=qat_tag(res, s, qlr),
                        rule='highest VALIDATION mIoU among the QAT final seeds; test read once at that checkpoint')
    out['release_checkpoint'] = best
    return out


def paired_vs_d8(acc):
    """Per-seed differences against D8 at the same seed. n=3, so this is a sanity check on sign
    consistency, not a significance test; reported so the paper cannot over-read a 0.0x gap."""
    out = {}
    for stage in ('fp32', 'qat'):
        base = acc[8][stage]['per_seed']
        for res in (16, 32):
            d = []
            for s in map(str, FINAL_SEEDS):
                a, b = acc[res][stage]['per_seed'].get(s), base.get(s)
                if a and b:
                    d.append(round(a['test_miou'] - b['test_miou'], 4))
            if len(d) < 2:
                continue
            m, sd = mean_sd(d)
            out[f'{stage}_D{res}_minus_D8'] = dict(
                per_seed=d, mean=round(m, 4), sd=round(sd, 4), n=len(d),
                t=None if not sd else round(m / (sd / len(d) ** 0.5), 3),
                same_sign=len({x > 0 for x in d}) == 1,
                note='paired by seed; n=3 has almost no power, so a non-zero mean is not evidence of a real gap')
    return out


# ---------------------------------------------------------------------------------- S5: the exports
def export(res, acc_res, cost_res):
    """Build model_release_d{res}_w1.0_s1 from the best-validation QAT final seed, checked against
    the bank size this resolution requires rather than the taped-out 0x8000."""
    rel = acc_res.get('release_checkpoint')
    if not rel:
        log(f'skip export R{RANGE_BINS}-D{res}: no finished QAT final seed')
        return None
    out = f'model_release_d{res}_w1.0_s1'
    if os.path.exists(f'{ROOT}/{out}/metrics.json'):
        log(f'export {out} already present, keeping it')
    else:
        bank = cost_res['bank_required_bytes']
        log(f'export {out} from {rel["tag"]} (--buf-size {hex(bank)})')
        rc = subprocess.run([PY, f'{ROOT}/make_release.py', '--run', rel['tag'], '--out', f'{ROOT}/{out}',
                             '--buf-size', hex(bank)],
                            cwd=os.path.dirname(ROOT),
                            stdout=open(f'{ROOT}/make_release_{out}.log', 'w'),
                            stderr=subprocess.STDOUT).returncode
        if rc != 0:
            log(f'EXPORT FAILED {out}: see exp03/make_release_{out}.log')
            return None
        subprocess.run([PY, f'{ROOT}/make_package_readme.py', f'{ROOT}/{out}'], cwd=ROOT,
                       stdout=open(f'{ROOT}/make_release_{out}.log', 'a'), stderr=subprocess.STDOUT)
    met = json.load(open(f'{ROOT}/{out}/metrics.json'))
    return dict(path=f'exp03/{out}', run=met['run'], weights_sha256=met['weights_sha256'],
                exact_integer_replay_test_miou=met['exact_integer_replay']['test_miou'],
                npu_buffer_required=met.get('npu_buffer_required'))


# ----------------------------------------------------------------------------- S6: the deliverables
def write_csv(path, rows):
    cols = ['resolution', 'width', 'stem_stride', 'params', 'macs_exact', 'peak_activation_bytes',
            'mem1_required_bytes', 'mem2_required_bytes', 'input_shape',
            'fp32_miou_mean', 'fp32_miou_std', 'qat_miou_mean', 'qat_miou_std',
            'fp32_dice_mean', 'fp32_dice_std', 'qat_dice_mean', 'qat_dice_std',
            'uav_iou_fp32_mean', 'uav_iou_fp32_std', 'uav_iou_qat_mean', 'uav_iou_qat_std',
            'export_path', 'weights_sha256', 'notes']
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, '') for c in cols})
    log('wrote', path)


def num(v, nd=4):
    """CSV cell: fixed-point, or empty when the run that would fill it does not exist yet."""
    return '' if v is None else f'{v:.{nd}f}'


def write_md(path, st):
    f4 = lambda m, sd: 'not trained' if m is None else f'{m:.4f} ± {sd:.4f}'
    L = ['# Primary RD-resolution experiment: fixed 1.0x stride-1 network',
         '',
         'Model server output for Phase 1 of `resolution_codesign_fixed_w1_handoff.md`. Generated by',
         f'`{st.get("generated_by", "exp03/resolution_fixed_w1.py")}`; every number below is read from a '
         'training run or from the built graph, none are retyped by hand.',
         '',
         '## What is held fixed',
         '',
         '```text',
         f'model                 KuRALSNetNPUSeg, {st["design"]["layers"]} layers, cutpoint {st["design"]["cutpoint"]}',
         f'width_mult            {WM}',
         f'stem_stride           {SS}   (full-resolution output head at every resolution)',
         f'encoder_depth         {ED}, bottleneck kernel 3, dropout 0.1, eltwise_act linear',
         f'params                {st["design"]["params"]:,}  (identical at all three resolutions)',
         'dataset               KuRALS-CW, existing chronological 80/10/10, block-max-pooled RD grids',
         'input                 DSP 5.3-bit log2 code, input_scale 1.0, natural Doppler order',
         'optimizer/scheduler   same trainer, cos schedule, batch 6 (physical 6, no accumulation)',
         f'FP32                  {EPOCHS} epochs, lr {FP32_LR:g}',
         f'QAT                   {QAT_EPOCHS} epochs from the best-val FP32 checkpoint, observers frozen at iter {QAT_FREEZE_ITER}',
         'selection             best validation mIoU; test read once at that checkpoint',
         f'seeds                 {TUNE_SEED} tunes, {FINAL_SEEDS} report',
         '```',
         '',
         'Only the RD grid changes, and with it the activation-bank capacity the grid requires.',
         '',
         '## Learning rate',
         '',
         f'FP32 uses one learning rate at every resolution (lr = {FP32_LR:g}, {EPOCHS} epochs), so the',
         'FP32 recipe is not tuned per resolution. Source: '
         + st['protocol'].get('fp32_lr_source', 'pipeline.py LR_GRID') + '.',
         '',
         f'The QAT learning rate IS selected per resolution, on seed {TUNE_SEED} only, from',
         '{' + ', '.join(f'{q:g}' for q in QAT_LRS) + '} -- the same procedure the D8 1.0x run used,',
         'which is what makes the three rows comparable. Selected values:',
         '']
    L += [f'- **R64-D{r}**: QAT lr {st["resolutions"][str(r)]["accuracy"]["qat_lr"] or "n/a"}'
          + ('' if not st['resolutions'][str(r)].get('qat_lr_search') else
             '  (seed-42 val mIoU ' + ', '.join(f'{k}: {v:.4f}' for k, v in st['resolutions'][str(r)]['qat_lr_search'].items()) + ')')
          for r in RESOLUTIONS]
    seed_txt = '/'.join(str(s) for s in FINAL_SEEDS)
    L += ['', f'## Accuracy (test, mean ± sample std over seeds {seed_txt})', '',
          '| | R64-D8 | R64-D16 | R64-D32 |', '|---|---:|---:|---:|']
    def arow(label, stage, key):
        cells = []
        for r in RESOLUTIONS:
            a = st['resolutions'][str(r)]['accuracy'][stage]
            cells.append(f4(a.get(f'{key}_mean'), a.get(f'{key}_std')))
        return f'| {label} | ' + ' | '.join(cells) + ' |'
    for stage, nm in (('fp32', 'FP32'), ('qat', 'QAT int8')):
        L += [arow(f'{nm} mIoU', stage, 'test_miou'),
              arow(f'{nm} Dice', stage, 'test_dice'),
              arow(f'{nm} UAV IoU', stage, 'uav_iou')]
    L += ['', '### Per-seed differences against R64-D8 (paired by seed)', '']
    if not st['paired_vs_d8']:
        L += ['Not available until all three resolutions have final seeds.']
    else:
        L += ['| Comparison | per-seed Δ test mIoU | mean ± sd | same sign | paired t |', '|---|---|---:|:-:|---:|']
        for k, d in st['paired_vs_d8'].items():
            L.append(f'| {k} | {d["per_seed"]} | {d["mean"]:+.4f} ± {d["sd"]:.4f} | '
                     f'{"yes" if d["same_sign"] else "no"} | {d["t"] if d["t"] is not None else "n/a"} |')
        L += ['', 'n = 3. This table exists so the gap is not over-read: with three seeds a paired t of',
              'this size is not evidence of a real accuracy difference, and the handoff forbids claiming',
              'one that the seed variance does not support.']
    L += ['', '## Cost (from the built graph and gen_cfg.py\'s allocator)', '',
          '| | R64-D8 | R64-D16 | R64-D32 |', '|---|---:|---:|---:|']
    def crow(label, key, f=lambda v: f'{v:,}'):
        return f'| {label} | ' + ' | '.join(f(st['resolutions'][str(r)]['cost'][key]) for r in RESOLUTIONS) + ' |'
    L += [crow('Params', 'params'),
          crow('MACs (exact)', 'macs_exact'),
          crow('Weight bytes', 'weight_bytes'),
          f'| Input shape | ' + ' | '.join(str(tuple(st['resolutions'][str(r)]['cost']['input_shape'])) for r in RESOLUTIONS) + ' |',
          crow('Input elements', 'input_elements'),
          crow('Peak activation bytes', 'peak_activation_bytes'),
          crow('MEM1 required', 'mem1_required_bytes'),
          crow('MEM2 required', 'mem2_required_bytes'),
          crow('Activation bank required', 'bank_required_kib', lambda v: f'{v} KiB'),
          f'| Fits the taped-out 32 KiB bank | ' + ' | '.join(
              ('yes' if st['resolutions'][str(r)]['cost']['fits_taped_out_bank'] else 'no') for r in RESOLUTIONS) + ' |',
          f'| Frame-reuse cells (H×W) | ' + ' | '.join(
              f"{st['resolutions'][str(r)]['cost']['shape_derived_depths']['frame_reuse_cells']:,}" for r in RESOLUTIONS) + ' |',
          '',
          'The allocator was run with an effectively unlimited bank, so these are the capacities each',
          'resolution **requires**. The old `--buf-size 0x8000` rejection rule is not applied here: in',
          'this experiment the bank size is a result, not a constraint. The last row is reported only so',
          'the secondary fixed-32-KiB experiment stays legible.',
          '',
          'Shape-derived high-water marks for every hardware structure the handoff lists (line buffer,',
          f'partial sums, maxpool, fused upsample) are in `summaries/{os.path.basename(SUMMARY)}` under',
          '`shape_derived_depths`, together with the full per-layer shape table under `layer_shapes`.',
          'They are minima implied by the network geometry; mapping them onto RTL depths is Phase 2.',
          '',
          '## Exports', '']
    rows = [st['resolutions'][str(r)].get('export') for r in RESOLUTIONS]
    if not any(rows):
        L += ['Not built yet.']
    else:
        L += ['| Resolution | Package | Source run | int8 replay test mIoU | weights SHA-256 |', '|---|---|---|---:|---|']
        for r, e in zip(RESOLUTIONS, rows):
            if not e:
                L.append(f'| R64-D{r} | not built | | | |')
            else:
                L.append(f'| R64-D{r} | `{e["path"]}` | `{e["run"]}` | '
                         f'{e["exact_integer_replay_test_miou"]:.4f} | `{e["weights_sha256"][:32]}…` |')
        L += ['', 'Each package carries `architecture.json`, `weights_int.bin`, `scales.bin`, `biases.bin`,',
              '`quantization.json`, `preprocessing.json`, `class_map.json`, `sample_inputs/`,',
              '`golden_outputs/`, `golden_layer_outputs/` and `metrics.json`. `metrics.json` records the',
              'allocator result at the bank size that resolution needs, plus `npu_buffer_required`.',
              'Read a package back without a checkpoint with `exp03/load_release.py --package <dir>`.']
    if st.get('secondary', {}).get('accuracy'):
        sec, ss = st['secondary'], '/'.join(str(s) for s in st['secondary']['seeds'])
        L += ['', f'## Secondary consistency check (seeds {ss})', '',
              sec.get('caveat', ''), '',
              '| | R64-D8 | R64-D16 | R64-D32 |', '|---|---:|---:|---:|']
        for stage, nm in (('fp32', 'FP32'), ('qat', 'QAT int8')):
            for key, lab in (('test_miou', 'mIoU'), ('test_dice', 'Dice'), ('uav_iou', 'UAV IoU')):
                get = lambda r: sec['accuracy'].get(str(r), sec['accuracy'].get(r, {})).get(stage, {})
                cells = [f4(get(r).get(f'{key}_mean'), get(r).get(f'{key}_std')) for r in RESOLUTIONS]
                L.append(f'| {nm} {lab} | ' + ' | '.join(cells) + ' |')
    L += ['', '## Relation to the old fixed-32-KiB search', '',
          'That search held the bank at 0x8000 and let width and stride float, so it compared three',
          'different networks (0.5x-ss1 at D8 and D16, 0.5x-ss2 at D32). It is secondary evidence for',
          'what a frozen accelerator forces, not the resolution comparison. Its packages',
          '(`model_release_v01`, `model_release_d16_v01`, `model_release_d32_v01`,',
          '`model_release_d32_w0.25_v01`) are untouched; the primary experiment uses only the three',
          '`model_release_d*_w1.0_s1` packages above.',
          '']
    open(path, 'w').write('\n'.join(L) + '\n')
    log('wrote', path)


def link_into_paper_results(*paths):
    """paper_results/ carries symlinks to the canonical copies in exp03/summaries (its convention)."""
    if not os.path.isdir(PAPER_DIR):
        return
    for p in paths:
        dst = os.path.join(PAPER_DIR, os.path.basename(p))
        rel = os.path.relpath(p, PAPER_DIR)
        if os.path.islink(dst) or os.path.exists(dst):
            if os.path.islink(dst) and os.readlink(dst) == rel:
                continue
            os.remove(dst)
        os.symlink(rel, dst)
        log('linked', dst, '->', rel)


# ---------------------------------------------------------------------------------------------- main
def main():
    log('fixed-1.0x stride-1 resolution experiment: R64-D8 / D16 / D32')
    cost = {r: costs(r) for r in RESOLUTIONS}
    bad = {r: c['handoff_expectation'] for r, c in cost.items() if c['handoff_expectation'] != 'confirmed'}
    if bad:
        log('WARNING: the graph disagrees with the handoff; training anyway, mismatch recorded:', bad)

    def complete(r):
        """True when this resolution already has every run the protocol asks for -- which is how
        R64-D8 (trained by run_w10_finals.py under exactly these tags) is reused rather than redone."""
        if not all(done(fp32_tag(r, s)) for s in [TUNE_SEED] + FINAL_SEEDS):
            return False
        v = {q: x for q, x in ((q, result(qat_tag(r, TUNE_SEED, q))) for q in QAT_LRS) if x}
        if len(v) < len(QAT_LRS):
            return False
        best = max(v, key=lambda q: v[q]['val_miou'])
        return all(done(qat_tag(r, s, best)) for s in FINAL_SEEDS)

    qlr = {}
    for r in RESOLUTIONS:
        if complete(r):
            v = {q: x['val_miou'] for q, x in ((q, result(qat_tag(r, TUNE_SEED, q))) for q in QAT_LRS) if x}
            qlr[r] = max(v, key=v.get)
            log(f'R{RANGE_BINS}-D{r}: already trained under this protocol, reusing it (QAT lr {qlr[r]:g})')
    todo = [r for r in RESOLUTIONS if r not in qlr]
    if todo:
        qlr.update(train(todo))
    for r in RESOLUTIONS:
        qlr.setdefault(r, None)

    acc = {r: accuracy(r, qlr[r]) for r in RESOLUTIONS}
    st = dict(
        experiment='primary RD-resolution study, fixed 1.0x stride-1 KuRALSNetNPUSeg',
        handoff=os.path.expanduser('~/resolution_codesign_fixed_w1_handoff.md'),
        generated_by='exp03/resolution_fixed_w1.py',
        protocol=dict(model=MODEL, width_mult=WM, stem_stride=SS, encoder_depth=ED,
                      fp32_epochs=EPOCHS, fp32_lr=FP32_LR, fp32_lr_source='lr_A from pipeline_state.json, not tuned per resolution',
                      qat_epochs=QAT_EPOCHS, qat_lr_grid=QAT_LRS, qat_freeze_iters=QAT_FREEZE_ITER,
                      qat_lr_selection=f'per resolution on seed {TUNE_SEED} by validation mIoU, documented per the handoff',
                      tune_seed=TUNE_SEED, final_seeds=FINAL_SEEDS,
                      selection='best validation mIoU; test read once at that checkpoint',
                      miou='4-class mean including background',
                      dataset='KuRALS-CW, existing chronological 80/10/10, existing block-max-pooled RD grids (not regenerated)'),
        design=dict(params=cost[8]['params'], layers=cost[8]['layers'], cutpoint=cost[8]['cutpoint']),
        supersedes='the fixed-32-KiB co-design search, which varied width/stride and is now secondary only',
        resolutions={}, paired_vs_d8=paired_vs_d8(acc))
    for r in RESOLUTIONS:
        v = {f'{q:g}': x['val_miou'] for q, x in ((q, result(qat_tag(r, TUNE_SEED, q))) for q in QAT_LRS) if x}
        st['resolutions'][str(r)] = dict(cost=cost[r], qat_lr_search=v, accuracy=acc[r],
                                         export=export(r, acc[r], cost[r]))
    st['qat_stability'] = [x for x in (qat_status(qat_tag(r, s, qlr[r]), fp32_tag(r, s))
                                       for r in RESOLUTIONS if qlr[r] for s in FINAL_SEEDS) if x]

    os.makedirs(f'{ROOT}/summaries', exist_ok=True)
    json.dump(st, open(SUMMARY, 'w'), indent=1)
    log('wrote', SUMMARY)

    rows = []
    for r in RESOLUTIONS:
        d, a, e = st['resolutions'][str(r)]['cost'], st['resolutions'][str(r)]['accuracy'], st['resolutions'][str(r)]['export']
        g = lambda stage, k: a[stage].get(k)
        n_fp32, n_qat = a['fp32']['n'], a['qat']['n']
        note = (f'FP32 lr {FP32_LR:g}; QAT lr {a["qat_lr"] or "n/a"}; n_fp32={n_fp32} n_qat={n_qat}; '
                f'bank required {d["bank_required_kib"]} KiB; '
                + ('fits the taped-out 32 KiB bank' if d['fits_taped_out_bank'] else 'exceeds the taped-out 32 KiB bank')
                + ('' if d['handoff_expectation'] == 'confirmed' else '; HANDOFF MISMATCH: ' + d['handoff_expectation']))
        rows.append(dict(resolution=d['resolution'], width=WM, stem_stride=SS, params=d['params'],
                         macs_exact=d['macs_exact'], peak_activation_bytes=d['peak_activation_bytes'],
                         mem1_required_bytes=d['mem1_required_bytes'], mem2_required_bytes=d['mem2_required_bytes'],
                         input_shape='x'.join(map(str, d['input_shape'])),
                         fp32_miou_mean=num(g('fp32', 'test_miou_mean')), fp32_miou_std=num(g('fp32', 'test_miou_std')),
                         qat_miou_mean=num(g('qat', 'test_miou_mean')), qat_miou_std=num(g('qat', 'test_miou_std')),
                         fp32_dice_mean=num(g('fp32', 'test_dice_mean')), fp32_dice_std=num(g('fp32', 'test_dice_std')),
                         qat_dice_mean=num(g('qat', 'test_dice_mean')), qat_dice_std=num(g('qat', 'test_dice_std')),
                         uav_iou_fp32_mean=num(g('fp32', 'uav_iou_mean')), uav_iou_fp32_std=num(g('fp32', 'uav_iou_std')),
                         uav_iou_qat_mean=num(g('qat', 'uav_iou_mean')), uav_iou_qat_std=num(g('qat', 'uav_iou_std')),
                         export_path=(e or {}).get('path', ''), weights_sha256=(e or {}).get('weights_sha256', ''),
                         notes=note))
    csv_path = f'{ROOT}/summaries/paper_resolution_fixed_w1_model.csv'
    md_path = f'{ROOT}/summaries/paper_resolution_fixed_w1_model.md'
    write_csv(csv_path, rows)
    write_md(md_path, st)
    link_into_paper_results(csv_path, md_path)
    log('PHASE 1 COMPLETE')


if __name__ == '__main__':
    main()
