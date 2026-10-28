"""Launch one Exp03 FP32 run in the background.
  python exp03/launch.py --stage fp32 --exp A --model npu_unet --res 8 --seed 42 [--width-mult 0.5] [--lr 1e-4] [--epochs 100]
Writes exp03/runs/<tag>/{config.json,train.log}; the trainer's own run dir lands under runs/<tag>/logs."""
import argparse, json, os, subprocess, yaml
ROOT = os.path.dirname(os.path.abspath(__file__)); KUR = os.path.join(ROOT, '..', 'kurals')
ap = argparse.ArgumentParser()
ap.add_argument('--stage', default='fp32'); ap.add_argument('--exp', required=True)
ap.add_argument('--model', required=True); ap.add_argument('--res', type=int, required=True)
ap.add_argument('--seed', type=int, default=42); ap.add_argument('--width-mult', type=float, default=1.0)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--epochs', type=int, default=100)
ap.add_argument('--norm', default=None, help='override input encoding (diagnostics)'); ap.add_argument('--no-roll', action='store_true'); ap.add_argument('--tag', default=None); ap.add_argument('--finetune', default=None)
ap.add_argument('--stem-stride', type=int, default=1); ap.add_argument('--encoder-depth', type=int, default=1)
ap.add_argument('--generic-size', action='store_true', help='kuralsnet: size-adaptive global pooling')
ap.add_argument('--wait', action='store_true', help='run in foreground and write DONE on success')
ap.add_argument('--qat', default=None, help='json dict of quant_warmup_iters etc for the QAT stage')
a = ap.parse_args()
rd = yaml.safe_load(open(f'{ROOT}/configs/rd_r64_d{a.res}.yaml'))
cfg = dict(model=a.model, dataset='KuRALS_CW', process_signal=True, annot_format='light_frame', annot_type='dense',
           w_size=256, h_size=256, nb_classes=4, nb_input_channels=1, nb_epochs=a.epochs, batch_size=6, lr=a.lr,
           lr_step=20, schedular='cos', Tmax=a.epochs, loss_step=100, val_step=600, viz_step=4000,
           torch_seed=a.seed, numpy_seed=a.seed, version=0, device='cuda', custom_loss='nbs1',
           transformations='hflip,vflip', norm_type=a.norm or rd['input_encoding'], shuffle=True, num_workers=1,
           width_mult=a.width_mult, select_metric='miou')
if a.model == 'kuralsnet_npu_seg':
    cfg.update(stem_stride=a.stem_stride, encoder_depth=a.encoder_depth, bottleneck_kernel_size=3, dropout_rate=0.1, eltwise_act='linear')
if a.generic_size: cfg['generic_size'] = True
if a.qat: cfg.update(json.loads(a.qat))
tag = a.tag or f"{a.exp}_{a.stage}_{a.model}_wm{a.width_mult}_{rd['name']}_lr{a.lr:g}_s{a.seed}"
cfg['comments'] = f"Exp03 {a.exp} {a.stage} {tag}; {rd['name']}; dsp_log input; best-val checkpoint"
cfg['name_exp'] = tag
d = f'{ROOT}/runs/{tag}'; os.makedirs(d, exist_ok=True)
json.dump(cfg, open(f'{d}/config.json', 'w'), indent=1)
env = dict(os.environ, **({'KURALS_NO_ROLL': '1'} if a.no_roll else {}), KURALS_CWR=rd['dataset_root'], KURALS_LOGS=f'{d}/logs', CUDA_VISIBLE_DEVICES='0')
cmd = ['nohup', os.path.expanduser('~/.venvs/kurals_train/bin/python'), 'train.py', '--cfg', f'{d}/config.json', '--dataset', 'KuRALS_CW']
if a.finetune: cmd += ['--finetune', a.finetune]
if a.wait:
    if os.path.exists(f'{d}/DONE'): print('skip (done)', tag); raise SystemExit(0)
    rc = subprocess.run(cmd, cwd=KUR, env=env, stdout=open(f'{d}/train.log', 'w'), stderr=subprocess.STDOUT).returncode
    if rc == 0: open(f'{d}/DONE', 'w').write('ok')
    raise SystemExit(rc)
subprocess.Popen(cmd, cwd=KUR, env=env, stdout=open(f'{d}/train.log', 'w'), stderr=subprocess.STDOUT, start_new_session=True)
print('launched', tag)
