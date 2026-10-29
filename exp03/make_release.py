"""Step 7 of platform_model_server.md: build model_release_vXX/ for a frozen R<range>-D<doppler> design point.

  python exp03/make_release.py --run Q_kuralsnet_npu_seg_wm0.5_ss1_R64-D8_qlr0.0001_s44 --out model_release_v01

The RD grid is resolution-agnostic. By default the range/Doppler bin counts come from the run tag
(`R<range>-D<doppler>`) via exp03/configs/rd_r64_d<doppler>.yaml; `--range-bins` / `--doppler-bins`
override them for a grid that has no preset config, and `--dataset-root` overrides the dataset path.
Nothing below assumes 8x64.

The release checkpoint is chosen by the caller (the pipeline summary picks the final-seed QAT run with the best
VALIDATION mIoU). Everything below is derived from that one checkpoint:
  * int8 export (export_int8.py) -> weights_int.bin / biases.bin / scales.bin / architecture.json / quantization.json
  * exact-integer replay (int8_inference.py) over the whole Test split -> must reproduce the QAT test mIoU
  * golden outputs + per-layer golden outputs for a few Test frames
  * NPU legality (npu_check.py -> gen_cfg.py) of the released network at its own RD grid with 32 KB buffers
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True, help='tag of the QAT run under exp03/runs')
ap.add_argument('--out', required=True)
ap.add_argument('--n-samples', type=int, default=8)
ap.add_argument('--version', default='v01')
ap.add_argument('--range-bins', type=int, default=None, help='override W (range bins); default: from the run tag / rd config')
ap.add_argument('--doppler-bins', type=int, default=None, help='override H (Doppler bins); default: from the run tag / rd config')
ap.add_argument('--dataset-root', default=None, help='override the dataset root for a grid with no preset rd config')
# The activation-bank size the released network is checked against. 0x8000 is the taped-out NPU8
# capacity and stays the default; the fixed-1.0x resolution study (exp03/resolution_fixed_w1.py)
# passes the capacity its own resolution actually requires, because there the bank size is an
# output of the experiment rather than a constraint on it.
ap.add_argument('--buf-size', type=lambda v: int(v, 0), default=0x8000)
a = ap.parse_args()

run_dir = f'{ROOT}/runs/{a.run}'
cfg = json.load(open(f'{run_dir}/config.json'))

# ---------------------------------------------------------------- RD grid (no 8x64 assumption)
m = re.search(r'R(\d+)-D(\d+)', a.run)
W = a.range_bins or (int(m.group(1)) if m else None)        # range bins   (tensor width)
H = a.doppler_bins or (int(m.group(2)) if m else None)      # Doppler bins (tensor height)
if W is None or H is None:
    raise SystemExit(f'cannot infer the RD grid from run tag {a.run!r}; pass --range-bins and --doppler-bins')
RES_NAME = f'R{W}-D{H}'
rd_path = f'{ROOT}/configs/rd_r{W}_d{H}.yaml'
dataset_root = a.dataset_root
if dataset_root is None:
    if not os.path.exists(rd_path):
        raise SystemExit(f'no rd config at {rd_path}; pass --dataset-root for this grid')
    rd_cfg = open(rd_path).read()
    dataset_root = [l.split(':', 1)[1].strip() for l in rd_cfg.splitlines() if l.startswith('dataset_root')][0]
if not os.path.isdir(dataset_root):
    raise SystemExit(f'dataset root does not exist: {dataset_root}')
print(f'release grid: {RES_NAME}  (H={H} Doppler x W={W} range)  dataset={dataset_root}')
os.environ['KURALS_CWR'] = dataset_root

import numpy as np
import torch
from torch.utils.data import DataLoader

import kurals.int8_inference as q8
from kurals.export_int8 import export_kuralsnet_npu_seg
from kurals.loaders.dataloaders import KuRALSDataset, SequenceDataset
from kurals.loaders.dataset import KuRALS_CW
from kurals.npu_check import check as npu_check, npu_profile
from kurals.utils.checkpoint_io import load_segmentation_model
from kurals.utils.functions import normalize, to_natural_doppler
from kurals.utils.metrics import Evaluator
from kurals.utils.paths import Paths

res = glob.glob(f'{run_dir}/logs/*/*/*/results')[0]
ckpt = f'{res}/val_doppler_model.pt'
cfg['dataset'], cfg['nb_classes'] = 'KuRALS_CW', 4
net, _ = load_segmentation_model(cfg, ckpt, torch.device('cpu'), quant_mode='on')
arrays, manifest = export_kuralsnet_npu_seg(net)
topo = manifest['topology']

out = os.path.abspath(a.out)
for sub in ('sample_inputs', 'golden_outputs', 'golden_layer_outputs'):
    os.makedirs(f'{out}/{sub}', exist_ok=True)

# ---------------------------------------------------------------- weights / biases / scales
layers, w_off, b_off = [], 0, 0
wbuf, bbuf, sbuf = bytearray(), bytearray(), bytearray()
for L in topo:
    n = L['name']
    if f'{n}.weight_int8' not in arrays:        # fused op entries (pool/upsample/route) have no parameters
        layers.append(dict(L))
        continue
    w = arrays[f'{n}.weight_int8'].astype(np.int8)
    s = arrays[f'{n}.scale_int16'].astype('<i2')
    b = arrays[f'{n}.bias_int16'].astype('<i2')
    layers.append(dict(L, weight_shape=list(w.shape), weight_offset=w_off, weight_bytes=w.size,
                       scale_offset=len(sbuf), n_scale=int(s.size), bias_offset=b_off, n_bias=int(b.size)))
    wbuf += w.tobytes(); sbuf += s.tobytes(); bbuf += b.tobytes()
    w_off += w.size; b_off += b.size * 2
open(f'{out}/weights_int.bin', 'wb').write(bytes(wbuf))
open(f'{out}/biases.bin', 'wb').write(bytes(bbuf))
open(f'{out}/scales.bin', 'wb').write(bytes(sbuf))   # not in the template tree: int16 per-channel multipliers, kept separate from biases
sha = hashlib.sha256(bytes(wbuf) + bytes(sbuf) + bytes(bbuf)).hexdigest()

json.dump(dict(model='kuralsnet_npu_seg', input_shape=[manifest['n_frames'], H, W], n_classes=manifest['n_classes'],
               stride=manifest['stride'], align=manifest['align'], final_upsample=manifest['final_upsample'],
               input_assembly=manifest['input_assembly'], layers=layers,
               file_layout=dict(weights_int_bin='int8, per layer in `layers` order, OIHW as exported (stored byte w; hardware uses 2w+1)',
                                scales_bin='int16 little-endian, per output channel', biases_bin='int16 little-endian, per output channel')),
          open(f'{out}/architecture.json', 'w'), indent=1)
json.dump(dict(scheme='symmetric int8, no zero point (always 0)', weight_precision='int8, per-channel; effective weight 2*w+1',
               activation_precision='int8 signed, per-tensor (EMA observers frozen after quant_freeze_iters)',
               accumulator_precision='int32 (replayed exactly in float64)', scale_bias_precision='int16', mode='QAT (fine-tuned from the best FP32 checkpoint)',
               clipping='int8 saturation [-128,127]; truncating shifts bias_shift in [5,20], act_shift in [0,7]',
               input_scale=manifest['input_scale'], output_scale=manifest['output_scale'], output_unsigned=manifest['output_unsigned'],
               per_layer=[{k: L[k] for k in L if k in ('name', 'shift1', 'shift2', 'in_scale', 'out_scale', 'act_name')} for L in topo]),
          open(f'{out}/quantization.json', 'w'), indent=1)
json.dump(dict(input_encoding='dsp_log', norm_type=cfg['norm_type'],
               contract='uint32(floor(RD magnitude)) >> 16 -> 5.3-bit log2 code log8(P) -> clamp(code - LOG_SCALE(0), 0, 127); fed as int8 with input_scale 1.0',
               doppler_order='natural FFT order (zero-Doppler at index 0); dataset stores it fftshifted, so ifftshift on the Doppler axis for input and labels',
               source_resolution='native 124 Doppler x 2048 range', resolution_reduction=f'block max-pooling of the linear-magnitude map to {H} Doppler x {W} range (emulation)',
               tensor_order=f'(n_frames={manifest["n_frames"]}, Doppler={H}, Range={W}), row-major int8',
               reference='npu_riscv docs/integration/radar_dsp_npu_input.md'), open(f'{out}/preprocessing.json', 'w'), indent=1)
json.dump(dict(order=['background', 'class1 (UAV)', 'class2', 'class3'], index_to_name={0: 'background', 1: 'UAV', 2: 'class2', 3: 'class3'},
               note='classes 2/3 are the non-UAV (pedestrian/vehicle) channels of KuRALS_CW; class 3 is nearly absent in the data',
               decode='argmax over the 4 logits per pixel'), open(f'{out}/class_map.json', 'w'), indent=1)

# ---------------------------------------------------------------- exact integer replay over Test
paths = Paths().get()
data = KuRALS_CW().get('Test')
seq_loader = DataLoader(SequenceDataset(data), batch_size=1, shuffle=False, num_workers=0)
ev_int, ev_qat = Evaluator(num_class=4), Evaluator(num_class=4)
net.eval(); net.set_quant_enabled(True); net.freeze_observers()
agree = total = nframe = 0
samples, record = [], {}
orig_conv = q8.conv_layer


def rec_conv(x, arrs, meta, out_scale_override=None):
    y = orig_conv(x, arrs, meta, out_scale_override)
    if record is not None and 'on' in record:
        record.setdefault('layers', {})[meta['name']] = y.detach().numpy().astype(np.float32)
    return y


q8.conv_layer = rec_conv
for seq_name, seq in seq_loader:
    ds = KuRALSDataset(seq, cfg.get('annot_type', 'dense'), paths['KuRALS_CW'] / seq_name[0], cfg['process_signal'], cfg['nb_input_channels'], None, False)
    for frame in DataLoader(ds, shuffle=False, batch_size=1, num_workers=0):
        x = to_natural_doppler(normalize(frame['rd_matrix'].float(), 'KuRALS_CW', 'range_doppler', norm_type=cfg['norm_type']), cfg['norm_type'])
        y = to_natural_doppler(frame['rd_mask'].float(), cfg['norm_type']).argmax(dim=1)
        x_int8 = x[0].numpy().astype(np.int8)              # DSP code in [0,127], input_scale 1.0
        assert np.array_equal(x_int8.astype(np.float32), x[0].numpy()), 'input is not integer-valued'
        nframe += 1
        want = len(samples) < a.n_samples and (y > 0).any() and nframe % 5 == 0   # frames that contain a target
        record.clear()
        if want:
            record['on'] = True
        logits = q8.kuralsnet_npu_seg_int8_forward(x_int8, manifest, arrays)
        with torch.no_grad():
            sim = net(x)[0].argmax(0).numpy()
        pred = logits.argmax(0)
        ev_int.add_batch(y, torch.from_numpy(pred[None]).long()); ev_qat.add_batch(y, torch.from_numpy(sim[None]).long())
        agree += int((pred == sim).sum()); total += pred.size
        if want:
            i = len(samples)
            np.save(f'{out}/sample_inputs/sample_{i:02d}_int8.npy', x_int8)
            np.save(f'{out}/golden_outputs/sample_{i:02d}_logits_f32.npy', logits)
            np.save(f'{out}/golden_outputs/sample_{i:02d}_argmax_u8.npy', pred.astype(np.uint8))
            np.save(f'{out}/golden_outputs/sample_{i:02d}_label_u8.npy', y[0].numpy().astype(np.uint8))
            np.savez(f'{out}/golden_layer_outputs/sample_{i:02d}.npz', **record['layers'])
            samples.append(i)
q8.conv_layer = orig_conv

miou_int, iou_int = ev_int.get_miou_class()
miou_sim, iou_sim = ev_qat.get_miou_class()
test_json = json.load(open(f'{res}/val_doppler_results.json'))['test_metrics']['range_doppler']   # test at the best-VAL epoch (test_doppler_results.json is the best-test checkpoint: never used)
val_json = json.load(open(f'{res}/val_doppler_results.json'))['val_metrics']['range_doppler']

# ---------------------------------------------------------------- legality + metrics
leg = npu_check(net, H, W, a.buf_size)
# What the allocator would need if nothing capped it -- the number the hardware server sizes MEM1/MEM2
# from. Equal to the bank size above whenever this design fits the taped-out NPU; larger when it does not.
req = npu_profile(net, H, W, a.buf_size)
state = json.load(open(f'{ROOT}/pipeline_state.json'))
# The FP32 parent is whichever B_ run shares this design and seed. Globbing it (instead of
# templating a hard-coded LR, as this line used to) keeps the tag correct at every resolution and
# width: the D16 winner trained at lr 1e-3 and the 1.0x design at lr 1e-3, not at the D8 winner's 3.33e-4.
_seed = a.run.rsplit('_s', 1)[1]
# Q_ -> B_, and any suffixed variant alongside it (QW_ -> BW_, for the width-grid extension
# runs in width025_d32.py), so a QAT tag from outside the pre-registered grid still resolves.
_stem = re.sub(r'^Q([A-Z]*)_', r'B\1_', a.run).split('_qlr')[0]
_cands = sorted(d for d in glob.glob(f'{ROOT}/runs/{_stem}_lr*_s{_seed}') if os.path.exists(f'{d}/DONE'))
if len(_cands) == 1:
    fp32_tag = os.path.basename(_cands[0])
else:
    fp32_tag = f'UNRESOLVED ({len(_cands)} candidates for {_stem}_lr*_s{_seed})'
    print('WARNING: fp32_reference_tag', fp32_tag)
metrics = dict(checkpoint=os.path.relpath(ckpt, ROOT), run=a.run, seed=cfg['torch_seed'], best_val_epoch=json.load(open(f'{res}/val_doppler_results.json'))['epoch'],
               selection_rule='best validation mIoU among final seeds 43/44/45 of the frozen configuration; test evaluated once',
               qat_float_sim=dict(test_miou=float(miou_sim), test_iou_by_class=[float(v) for v in iou_sim]),
               exact_integer_replay=dict(test_miou=float(miou_int), test_iou_by_class=[float(v) for v in iou_int],
                                         pixel_agreement_with_qat_sim=agree / total),
               recorded_by_trainer=dict(val_miou=val_json['miou'], test_miou=test_json['miou'], test_iou_by_class=test_json['miou_by_class']),
               weight_bytes=len(wbuf), n_params_conv=int(sum(arrays[k].size for k in arrays if k.endswith('weight_int8'))),
               macs=leg['macs'], params=leg['params'], peak_activation_bytes=leg['peak_act_bytes'], npu_check=leg,
               npu_buffer_required=dict(bank_bytes=req['peak_npu_buffer'], buf1=req['buf1'], buf2=req['buf2'], buf3=req['buf3'],
                                        checked_against_bytes=a.buf_size, fits=req['feasible']),
               fp32_reference_tag=fp32_tag, weights_sha256=sha)
json.dump(metrics, open(f'{out}/metrics.json', 'w'), indent=1, default=str)

shutil.copy(f'{run_dir}/config.json', f'{out}/train_config.json')
import yaml
yaml.safe_dump(dict(name=f"{RES_NAME} KuRALSNetNPUSeg {cfg['width_mult']}x (QAT int8)", version=a.version, rd_resolution=RES_NAME,
                    range_bins=W, doppler_bins=H, model='kuralsnet_npu_seg',
                    width_mult=cfg['width_mult'], stem_stride=cfg['stem_stride'], encoder_depth=cfg.get('encoder_depth'), bottleneck_kernel_size=cfg['bottleneck_kernel_size'],
                    eltwise_act=cfg['eltwise_act'], n_classes=4, norm_type=cfg['norm_type'], npu_buffer_bytes=hex(a.buf_size), source_run=a.run, seed=cfg['torch_seed']),
               open(f'{out}/config.yaml', 'w'), sort_keys=False)
print(json.dumps({k: metrics[k] for k in ('exact_integer_replay', 'qat_float_sim', 'recorded_by_trainer', 'weight_bytes', 'macs')}, indent=1))
print('npu_check:', json.dumps(leg))
print('samples:', len(samples), 'sha256:', sha[:16])
