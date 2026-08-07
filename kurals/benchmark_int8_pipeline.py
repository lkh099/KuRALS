"""The full deployment-simulation loop: for each dataset sample, quantize the
raw RD map to int8 (kurals.quantize_input), run the exact-integer replay of
the exported model (kurals.int8_inference, driven by an export_int8.py
manifest -- NOT the live training-time model object), argmax-decode the
per-pixel logits into a binary foreground mask, and benchmark against the
same ground truth and the same Evaluator(num_class=2) that
test_kuralsnet_vs_cfar.py uses, so the numbers are directly comparable.

This is the thing to run to sanity-check that what got exported actually
works under literal integer arithmetic, not just that the training-time
simulation produced a good loss curve.
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from kurals.loaders.dataset import KuRALS_CW, KuRALS_PD
from kurals.loaders.dataloaders import SequenceDataset, KuRALSDataset
from kurals.int8_inference import kuralsnet_npu_seg_int8_forward
from kurals.quantize_input import quantize_input
from kurals.utils.metrics import Evaluator
from kurals.utils.paths import Paths


def run_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help="export_int8.py's <out>.json")
    parser.add_argument('--arrays', required=True, help="export_int8.py's <out>.npz")
    parser.add_argument('--dataset', default='KuRALS_CW')
    parser.add_argument('--split', default='Test')
    parser.add_argument('--norm-type', dest='norm_type', default='tvt')
    parser.add_argument('--max-frames', dest='max_frames', type=int, default=None,
                         help='Stop after this many frames total (the pure-integer replay is one sample '
                              'at a time in plain PyTorch ops, not batched/GPU-accelerated, so this is '
                              'useful for a quick check on a large split).')
    args = parser.parse_args()

    with open(args.manifest, 'r') as fp:
        manifest = json.load(fp)
    arrays = np.load(args.arrays)

    paths = Paths().get()
    data = (KuRALS_CW() if args.dataset == 'KuRALS_CW' else KuRALS_PD()).get(args.split)
    seq_loader = DataLoader(SequenceDataset(data), batch_size=1, shuffle=False, num_workers=0)

    evaluator = Evaluator(num_class=2)
    n_frames_seen = 0

    for seq_name, seq in seq_loader:
        path_to_frames = paths[args.dataset] / seq_name[0]
        kurals_dataset = KuRALSDataset(seq, 'dense', path_to_frames, True, manifest['n_frames'], None, False)
        frame_dataloader = DataLoader(kurals_dataset, shuffle=False, batch_size=1, num_workers=0)

        for frame in frame_dataloader:
            rd_matrix = frame['rd_matrix'][0].numpy()  # (n_frames, H, W)
            rd_mask = frame['rd_mask'][0].numpy()       # (n_classes, H, W)
            gt_fg = torch.from_numpy((1 - rd_mask[0]).astype(np.int64))  # "not background" == foreground

            x_int8 = quantize_input(rd_matrix, args.dataset, args.norm_type, manifest['input_scale'])
            logits = kuralsnet_npu_seg_int8_forward(x_int8, manifest, arrays)  # (n_classes, H, W)
            pred_fg = torch.from_numpy((logits.argmax(axis=0) != 0).astype(np.int64))

            evaluator.add_batch(gt_fg.unsqueeze(0), pred_fg.unsqueeze(0))

            n_frames_seen += 1
            if args.max_frames is not None and n_frames_seen >= args.max_frames:
                break
        if args.max_frames is not None and n_frames_seen >= args.max_frames:
            break

    acc, _ = evaluator.get_pixel_acc_class()
    prec, _ = evaluator.get_pixel_prec_class()
    _, recall_by_class = evaluator.get_pixel_recall_class()
    _, far_by_class = evaluator.get_pixel_far_class()
    miou, _ = evaluator.get_miou_class()
    dice, _ = evaluator.get_dice_class()
    print(f'\n=== int8-exported kuralsnet_npu_seg on {args.dataset} / {args.split} ({n_frames_seen} frames) ===')
    print(f'  Acc={acc:.4f}  Prec={prec:.4f}  Pd={recall_by_class[1]:.4f}  '
          f'FAR={far_by_class[1]:.4f}  mIoU={miou:.4f}  mDice={dice:.4f}')


if __name__ == '__main__':
    run_pipeline()
