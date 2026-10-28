"""Matched FP32-continuation control for the QAT stage.

QAT = 100 more epochs of fine-tuning from the best-val FP32 checkpoint, so QAT-vs-FP32 deltas mix the int8 effect with
extra training. This control does the identical fine-tune (same start checkpoint, LR, epochs, cosine schedule, seeds,
batch, augmentation, selection rule) with quantization never enabled (no quant_* keys -> float forward).
  python exp03/run_fp32_control.py            # runs all 9 (D8/D16/D32 x seeds 43/44/45) concurrently, resumable via DONE files
Tags: F_kuralsnet_npu_seg_wm0.5_ss<ss>_R64-D<res>_flr0.0001_s<seed>
"""
import glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.expanduser('~/.venvs/kurals_train/bin/python')
st = json.load(open(f'{ROOT}/pipeline_state.json'))
LR, EPOCHS, SEEDS = 1e-4, 100, [43, 44, 45]   # LR = the QAT LR chosen on the tuning seed at every resolution


def src_tag(r, p, s):
    return f"B_kuralsnet_npu_seg_wm{p['wm']}_ss{p['ss']}_R64-D{r}_lr{p['lr']:g}_s{s}"


def job(r, p, s):
    src = src_tag(r, p, s)
    ckpt = glob.glob(f'{ROOT}/runs/{src}/logs/*/*/*/results/val_doppler_model.pt')[0]
    tag = f"F_kuralsnet_npu_seg_wm{p['wm']}_ss{p['ss']}_R64-D{r}_flr{LR:g}_s{s}"
    cmd = [PY, f'{ROOT}/launch.py', '--wait', '--exp', 'F', '--stage', 'fp32cont', '--model', 'kuralsnet_npu_seg', '--res', str(r), '--seed', str(s),
           '--width-mult', str(p['wm']), '--lr', str(LR), '--epochs', str(EPOCHS), '--stem-stride', str(p['ss']), '--tag', tag, '--finetune', ckpt]
    rc = subprocess.run(cmd).returncode
    print(tag, 'rc', rc, flush=True)
    return rc


jobs = [(r, p, s) for r, p in st['B_picks'].items() for s in SEEDS]
with ThreadPoolExecutor(len(jobs)) as ex:
    rcs = list(ex.map(lambda j: job(*j), jobs))
print('FP32 CONTROL COMPLETE' if not any(rcs) else f'FAILED: {rcs}', flush=True)
