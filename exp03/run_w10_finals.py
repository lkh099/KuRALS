"""Final-seed FP32 + QAT for the 1.0x-width R64-D8 design (the 4.95 M-MAC network the hardware paper uses).
Same protocol as pipeline.py for the Exp B picks: LR 1e-3 (the better of the two Exp B LRs on the tuning seed, already trained for seed 42),
FP32 200 epochs on seeds 43/44/45; QAT 100 epochs from each seed's best-val checkpoint, QAT LR chosen on seed 42 from {1e-4, 3e-5, 1e-5}.
Tags follow the pipeline naming, so paper_extract.py/report.py pick them up. Resumable (DONE files).
  python exp03/run_w10_finals.py      # log: exp03/w10_finals.log
"""
import json, os, sys, time
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import pipeline as P

LOG = f'{ROOT}/w10_finals.log'
P.log = lambda *a: open(LOG, 'a').write(time.strftime('%F %T ') + ' '.join(map(str, a)) + '\n')
WM, SS, LR, R = 1.0, 1, 1e-3, 8
src = lambda s: f'B_kuralsnet_npu_seg_wm{WM}_ss{SS}_R64-D{R}_lr{LR:g}_s{s}'
qtag = lambda q, s: f'Q_kuralsnet_npu_seg_wm{WM}_ss{SS}_R64-D{R}_qlr{q:g}_s{s}'
qjob = lambda q, s: P.job('kuralsnet_npu_seg', R, s, q, wm=WM, ss=SS, stage='qat', exp='Q', epochs=P.QAT_EPOCHS, finetune=P.ckpt(src(s)),
                          qat=dict(quant_warmup_iters=0, quant_freeze_iters=P.QAT_FREEZE_ITER), tag=qtag(q, s))
assert P.result(src(P.TUNE_SEED)), 'seed-42 FP32 checkpoint of the 1.0x design is missing'

P.log('wave 1: FP32 finals + QAT LR search (seed 42)')
fp32 = [P.job('kuralsnet_npu_seg', R, s, LR, wm=WM, ss=SS, exp='B') for s in P.FINAL_SEEDS]
P.run_all(fp32 + [qjob(q, P.TUNE_SEED) for q in P.QAT_LRS])

v = {q: (P.result(qtag(q, P.TUNE_SEED)) or {}).get('val_miou') for q in P.QAT_LRS}
v = {k: x for k, x in v.items() if x is not None}
best = max(v, key=v.get)
P.log('QAT LR search (seed 42 val mIoU):', v, '-> picked', best)
P.log('wave 2: QAT finals')
failed = P.run_all([qjob(best, s) for s in P.FINAL_SEEDS if os.path.exists(P.run_dir(src(s)) + '/DONE')])

stab = [x for x in (P.qat_status(qtag(q, s), src(s)) for q, s in [(q, P.TUNE_SEED) for q in P.QAT_LRS] + [(best, s) for s in P.FINAL_SEEDS]) if x]
json.dump(dict(qat_lr=best, qat_lr_search_val_miou=v, runs=stab), open(f'{ROOT}/summaries/qat_stability_w1.0.json', 'w'), indent=1)
P.log('W10 FINALS COMPLETE' if not failed else f'FAILED: {failed}')
