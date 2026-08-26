# Usage

All commands below assume the environment is set up (see the root [`README.md`](../README.md)) and are run from inside the `kurals/` directory, matching the existing `train.sh`/`test.sh`-style scripts' own convention:

```bash
cd kurals/
```

## Training

### Current best model: `kuralsnet_npu_seg`

The NPU-legal (128-MAC-array-deployable) segmentation model, `KuRALSNetNPUSeg`, with quantization-aware training built in:

```bash
python train.py --cfg config_files/kuralsnet_npu_seg.json --dataset KuRALS_CW
```

Key config fields (`config_files/kuralsnet_npu_seg.json`): `bottleneck_ch` (default 64, the final architecture -- see `CLAUDE.md` for why widening to 96 was tried and rejected), `quant_warmup_iters`/`quant_freeze_iters` (float32-then-QAT schedule), `custom_loss: "nbs1"`. Checkpoints and per-split metrics land under `<logs>/KuRALS_CW/kuralsnet_npu_seg/<run_name>/results/` (the `logs` path from `config.ini`), written on every validation step (`val_doppler_model.pt`/`test_doppler_model.pt`/`latest_model.pt` + matching `*_results.json`).

### Baselines

The original non-NPU-constrained baselines (`kuralsnet` and its ASPP/ADA/PKC/AdaPKC variants, FCN8s, U-Net, DeepLabv3+, HRNet, RSSNet, SegFormer, Swin Transformer) train the same way, e.g.:

```bash
python train.py --cfg config_files/kuralsnet.json --dataset KuRALS_CW
# or: bash train.sh
```

## Pretrained checkpoints

`checkpoints/` at the repo root ships the curated best-per-resolution checkpoints
directly (see `checkpoints/README.md` for what's in each directory and its val/test
numbers) -- no separate download or archive needed. E.g. to run the tracker demo against
the native best model without training anything yourself:

```bash
python demo_tracker.py \
    --model-path ../checkpoints/native_best_0.5144/results/val_doppler_model.pt \
    --dataset KuRALS_CW --split Test --sequence 两个无人机 --output tracker_demo.gif
```

(You still need the dataset itself -- see "SoC 8x64 buffer dataset" below, or the root
`README.md` for the native dataset.)

## Evaluating against CFAR

`test_kuralsnet_vs_cfar.py` scores a trained segmentation model (any of the baselines or `kuralsnet_npu_seg`) against CFAR using the same binary foreground/background convention (`Evaluator(num_class=2)`, ground truth = "not background"):

```bash
python test_kuralsnet_vs_cfar.py \
    --cfg config_files/kuralsnet_npu_seg.json \
    --model-path /path/to/val_doppler_model.pt \
    --dataset KuRALS_CW --split Test \
    --cfar-alphas 0.5 1.0 1.5
```

For `kuralsnet_npu_seg` specifically, this always evaluates with QAT enabled (int8-simulated forward), not the float32-equivalent network -- see the script's own comment for why.

For a per-class (UAV vs non-UAV) breakdown instead of binary foreground/background, use `test_class_discrimination.py` (also supports both the retired CenterNet model via `kurals/legacy/` and the current `kuralsnet_npu_seg`, auto-selected from `cfg['model']`):

```bash
python test_class_discrimination.py \
    --cfg config_files/kuralsnet_npu_seg.json \
    --model-path /path/to/val_doppler_model.pt \
    --dataset KuRALS_CW --split Test
```

## SoC 8x64 buffer dataset

The 8x64-buffer branch (see `CLAUDE.md`) trains against a block-max-pooled emulation of
a smaller deployment RD buffer, not the repo's native 124x2048 resolution. Neither the
native nor the 8x64 dataset is committed to this repo (`.gitignore` excludes `*.npy`),
so regenerate the 8x64 one from a native-resolution extracted dataset (produced by
`kuralscw_processing.py` from raw `.mat` data):

```bash
python -m kurals.dataset_process.resize_extracted_dataset \
    --src /path/to/native/KuRALS_CW --dst /path/to/KuRALS_CW_8x64/KuRALS_CW \
    --doppler-bins 8 --range-bins 64
```

Then point `config.ini` at the new dataset root:

```bash
python kurals/utils/set_paths.py --cwr /path/to/KuRALS_CW_8x64/KuRALS_CW --pdr /path/to/KuRALS_PD --logs /path/to/logs
```

Train the current best 8x64 result the same way as the native model, but with the
8x64-specific config and `--finetune` from a native checkpoint (partial-transfer init
via `expand_kernel3.py`, since the bottleneck's kernel size differs between the native
and 8x64 configs):

```bash
python expand_kernel3.py --ckpt /path/to/native/val_doppler_model.pt --out native_ckpt_kernel3_init.pt
python train.py --cfg config_files/kuralsnet_npu_seg_8x64_kernel3_finetune_dropout.json \
    --dataset KuRALS_CW --finetune native_ckpt_kernel3_init.pt
```

## Tracker demo

`demo_tracker.py` runs a trained `kuralsnet_npu_seg` checkpoint over one real recorded sequence, extracts per-frame UAV blob centroids (`scipy.ndimage.label`/`center_of_mass`), tracks them with a constant-velocity Kalman filter + Hungarian assignment (`kurals/utils/tracker.py`), and renders an annotated GIF:

```bash
python demo_tracker.py \
    --model-path /path/to/val_doppler_model.pt \
    --dataset KuRALS_CW --split Test \
    --sequence 两个无人机 \
    --output tracker_demo.gif
```

`--sequence` is matched as a substring against sequence names to pick which recorded sequence to run (default is one of the multi-UAV recordings). Tracker parameters (`--max-match-dist`, `--confirm-hits`, `--max-misses`) are standard SORT-style knobs, tunable per dataset/frame-rate.

## Exporting for NPU deployment (int8)

Four steps, each a separate script, chained via files on disk:

1. **Export** a trained (QAT) checkpoint to int8 weights + per-layer dequantization parameters:
   ```bash
   python export_int8.py \
       --cfg config_files/kuralsnet_npu_seg.json \
       --model-path /path/to/val_doppler_model.pt \
       --out kuralsnet_npu_seg_int8   # writes <out>.npz and <out>.json
   ```

2. **Get a raw RD map** to quantize -- either random noise for a pure pipeline smoke test, or one real recorded frame:
   ```bash
   python dummy_ifm.py --random --out dummy_ifm.npy                       # random noise, no dataset needed
   python dummy_ifm.py --dataset KuRALS_CW --split Test --out real_ifm.npy  # one real frame
   ```

3. **Quantize** the raw RD map to the exact int8 bytes the NPU's first conv layer consumes:
   ```bash
   python quantize_input.py dummy_ifm.npy --manifest kuralsnet_npu_seg_int8.json --out dummy_ifm.int8.npy
   ```

4. **Replay and benchmark** the exported model under literal integer arithmetic against a full dataset split, to sanity-check the export end-to-end (not just that training produced a good loss curve):
   ```bash
   python benchmark_int8_pipeline.py \
       --manifest kuralsnet_npu_seg_int8.json --arrays kuralsnet_npu_seg_int8.npz \
       --dataset KuRALS_CW --split Test --max-frames 500
   ```
   Prints Acc/Prec/Pd/FAR/mIoU/mDice on the same binary foreground/background convention as `test_kuralsnet_vs_cfar.py`, so int8 numbers are directly comparable to the float32 ones.

`kurals/int8_inference.py`'s `conv_layer`/`residual_block`/`upconv_block`/`refine_block` helpers do the actual exact-integer replay (accumulated in `float64`, exact for this model's weight/activation bit widths -- see that module's docstring); `kuralsnet_npu_seg_int8_forward` chains them in `KuRALSNetNPUSeg.forward()`'s exact order.

## Retired CenterNet-head pipeline

`kurals/legacy/` holds the original anchor-free detection-head approach (`KuRALSNetNPU`, `train_detect.py`, etc.), superseded by `kuralsnet_npu_seg`. See `kurals/legacy/README.md`.
