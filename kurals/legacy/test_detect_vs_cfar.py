"""Benchmark KuRALSNetNPU against CFAR using the exact same evaluation this
repo already uses for CFAR (see test_cfar.py): both methods are reduced to a
binary foreground/background decision per pixel and scored with
kurals.utils.metrics.Evaluator(num_class=2) against the same ground truth
((1 - rd_mask[:, 0]) -- "not background"), so precision / recall (= Pd,
detection probability) / FAR / mIoU / mDice are directly comparable numbers,
not a different metric that merely looks similar.

KuRALSNetNPU only predicts on an 8x-downsampled grid with a sub-cell (x, y)
offset, so getting a full-resolution binary mask out of it requires a decode
step: sigmoid the heatmap logits (off-NPU, same as the excluded 'region'/
'yolo' Darknet decode layers), take the per-cell max over foreground classes
(CFAR doesn't distinguish classes either, so this keeps the comparison fair),
threshold it (this script's analogue of CFAR's alpha), and for every cell that
passes, decode a single full-resolution point using the offset head. That
single point -- not a splatted neighborhood -- is marked positive in the
output mask: CFAR's own decisions are one pixel at a time, so splatting would
inflate this model's apparent recall relative to CFAR rather than measuring
the same thing.
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from kurals.loaders.dataset import KuRALS_CW, KuRALS_PD
from kurals.loaders.dataloaders import SequenceDataset, KuRALSDataset, _scan_num
from kurals.models.cfar import CFAR2D_Parallel
from kurals.legacy.checkpoint_io import load_detection_model
from kurals.legacy.decode import decode_points, points_to_mask
from kurals.utils.functions import normalize
from kurals.utils.metrics import Evaluator
from kurals.utils.paths import Paths


def run_temporal_consistency(net, device, cfg, args, seq_loader, paths, stride):
    """Postprocessing-only variant of the main detection loop: a decoded point
    only survives if a point also exists within --tc-radius full-resolution
    pixels in a scan-adjacent neighbor frame (up to --temporal-consistency
    frames away in either direction). No retraining, no architecture/loss
    change -- this targets the demo's false-alarm rate directly on top of
    whatever checkpoint is passed in, on the premise that a real target
    persists frame-to-frame while a false alarm mostly doesn't.

    "Scan-adjacent" reuses KuRALSDataset.MAX_SCAN_GAP (see loaders/dataloaders.py)
    so this can't silently treat two frames on opposite sides of a
    sequence-list splice/gap as temporal neighbors -- the same
    windowing-safety guard that protects multi-frame training inputs also
    protects this eval-time filter.

    Class-agnostic by design (matches decode_points / this script's own
    max-over-classes Pd/FAR convention, not decode_points_with_class) -- a
    detection only needs a nearby detection of *any* foreground class in the
    neighbor frame to survive, since CFAR's own comparison never distinguishes
    classes either.

    Effect measured on the final CenterNet checkpoint: narrow and
    threshold-dependent, not a universal win -- helps recall/FAR at low
    thresholds (where baseline false-alarm rate is real) at the cost of some
    recall, but doesn't unlock a new best operating point beyond the
    unfiltered high-threshold precision ceiling. Useful as an optional
    postprocessing knob, not a fix for the underlying threshold-plateau
    problem that motivated moving to the segmentation head.
    """
    max_gap = KuRALSDataset.MAX_SCAN_GAP
    radius_sq = args.tc_radius ** 2
    tc_evaluators = {t: Evaluator(num_class=2) for t in args.thresholds}

    for seq_name, seq in seq_loader:
        path_to_frames = paths[args.dataset] / seq_name[0]
        kurals_dataset = KuRALSDataset(seq, cfg.get('annot_type', 'dense'), path_to_frames,
                                        cfg['process_signal'], cfg['nb_input_channels'], None, False)
        n = len(kurals_dataset)
        scan_nums = [_scan_num(kurals_dataset.dataset[i][0]) for i in range(n)]

        # Buffer every frame's raw model output for this sequence once (cheap: Hs x Ws
        # is small), so each threshold below doesn't re-run the forward pass.
        frame_infos = []
        loader = DataLoader(kurals_dataset, shuffle=False, batch_size=1, num_workers=0)
        for frame in loader:
            rd_data = frame['rd_matrix'].float().to(device)
            rd_mask = frame['rd_mask'].float().to(device)
            rd_data = normalize(rd_data, args.dataset, 'range_doppler', norm_type=cfg['norm_type'])
            gt_mask = (1 - rd_mask[:, 0, :, :]).int()
            with torch.no_grad():
                output = net(rd_data)
            n_fg = output.shape[1] - 2
            heatmap_prob = torch.sigmoid(output[:, :n_fg]).cpu().numpy()[0]
            offset = output[:, n_fg:].cpu().numpy()[0]
            frame_infos.append({'gt_mask': gt_mask.cpu()[0], 'heatmap_prob': heatmap_prob, 'offset': offset})

        for threshold, evaluator in tc_evaluators.items():
            per_frame_points = [
                decode_points(info['heatmap_prob'], info['offset'], threshold, stride, nms_kernel=args.nms_kernel)
                for info in frame_infos
            ]
            for i, info in enumerate(frame_infos):
                points = per_frame_points[i]
                if not points:
                    h, w = info['gt_mask'].shape
                    evaluator.add_batch(info['gt_mask'].unsqueeze(0),
                                        torch.zeros(1, h, w, dtype=torch.int32))
                    continue
                pts_arr = np.asarray(points)  # (P, 2) [cy, cx]
                matched_mask = np.zeros(len(points), dtype=bool)
                for delta in range(1, args.temporal_consistency + 1):
                    if matched_mask.all():
                        break
                    for j in (i - delta, i + delta):
                        if not (0 <= j < len(frame_infos)):
                            continue
                        if scan_nums[i] is None or scan_nums[j] is None \
                                or abs(scan_nums[i] - scan_nums[j]) > max_gap:
                            continue
                        neighbor_pts = per_frame_points[j]
                        if not neighbor_pts:
                            continue
                        npts_arr = np.asarray(neighbor_pts)  # (Q, 2)
                        # (P, Q) pairwise squared distance, vectorized -- the naive
                        # nested-Python-loop version times out on low thresholds
                        # (hundreds of candidate points/frame), see task history.
                        dist_sq = ((pts_arr[:, None, :] - npts_arr[None, :, :]) ** 2).sum(axis=2)
                        matched_mask |= (dist_sq <= radius_sq).any(axis=1)
                kept = [p for p, m in zip(points, matched_mask) if m]
                h, w = info['gt_mask'].shape
                pred_mask = torch.from_numpy(points_to_mask(kept, h, w)).unsqueeze(0)
                evaluator.add_batch(info['gt_mask'].unsqueeze(0), pred_mask)

    return tc_evaluators


def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='kurals/legacy/kuralsnet_npu.json', help='Training config of the model to test.')
    parser.add_argument('--dataset', default='KuRALS_CW')
    parser.add_argument('--split', default='Test', help='Train, Validation or Test')
    parser.add_argument('--model-path', dest='model_path', required=True, help='Path to a checkpoint (dict w/ "net") or bare state_dict.')
    parser.add_argument('--quant', choices=['auto', 'on', 'off'], default='auto',
                         help='Force the quantized (QAT/int8-simulated) or float forward path, or infer from the checkpoint.')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.3, 0.5, 0.7, 0.9],
                         help='Detection-probability thresholds to sweep, analogous to CFAR alpha.')
    parser.add_argument('--cfar-alphas', dest='cfar_alphas', type=float, nargs='+', default=[0.5, 1.0, 1.5],
                         help='CFAR alpha values to sweep for the same comparison.')
    parser.add_argument('--cfar-type', dest='cfar_type', default='CA')
    parser.add_argument('--skip-cfar', dest='skip_cfar', action='store_true', help='Only evaluate the detector.')
    parser.add_argument('--nms-kernel', dest='nms_kernel', type=int, default=1,
                         help='Local-peak NMS window passed to decode_points (see kurals/utils/decode.py). '
                              'Default 1 (off): this per-pixel-recall metric rewards multiple adjacent-cell '
                              'detections sweeping a target\'s 9-pixel ground-truth blob (see decode.py\'s '
                              'docstring), and CFAR\'s own Pd gets no such deduplication either, so NMS-off '
                              'is the fair comparison basis. Pass 5 explicitly to study NMS\'s effect.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--temporal-consistency', dest='temporal_consistency', type=int, default=0,
                         help='If > 0, additionally evaluate a temporally-filtered variant on top of '
                              'the same checkpoint: a decoded point only survives if a point also exists '
                              'within --tc-radius pixels in a scan-adjacent neighbor frame, up to this '
                              'many frames away in either direction. Pure postprocessing, no retraining. '
                              '0 (default) = off, report baseline only.')
    parser.add_argument('--tc-radius', dest='tc_radius', type=float, default=6.0,
                         help='Full-resolution pixel radius for matching a point to a neighbor frame\'s '
                              'point when --temporal-consistency is set.')
    args = parser.parse_args()

    with open(args.cfg, 'r') as fp:
        cfg = json.load(fp)
    cfg['dataset'] = args.dataset
    cfg['nb_classes'] = 4 if args.dataset == 'KuRALS_CW' else 5

    paths = Paths().get()
    data = (KuRALS_CW() if args.dataset == 'KuRALS_CW' else KuRALS_PD()).get(args.split)
    seq_loader = DataLoader(SequenceDataset(data), batch_size=1, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    net, quant_enabled = load_detection_model(cfg, args.model_path, device, args.quant)
    print(f'Loaded {args.model_path} (quant_enabled={quant_enabled})')
    stride = net.stride

    det_evaluators = {t: Evaluator(num_class=2) for t in args.thresholds}
    cfar_evaluators = {a: Evaluator(num_class=2) for a in args.cfar_alphas} if not args.skip_cfar else {}
    cfar_models = {a: CFAR2D_Parallel(cfar_type=args.cfar_type, alpha=a, ref_cells=1, guard_cells=1)
                   for a in args.cfar_alphas} if not args.skip_cfar else {}

    for seq_name, seq in seq_loader:
        path_to_frames = paths[args.dataset] / seq_name[0]
        kurals_dataset = KuRALSDataset(seq, cfg.get('annot_type', 'dense'), path_to_frames,
                                        cfg['process_signal'], cfg['nb_input_channels'], None, False)
        frame_dataloader = DataLoader(kurals_dataset, shuffle=False,
                                       batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

        for frame in frame_dataloader:
            rd_data = frame['rd_matrix'].float().to(device)
            rd_mask = frame['rd_mask'].float().to(device)
            rd_data = normalize(rd_data, args.dataset, 'range_doppler', norm_type=cfg['norm_type'])
            gt_mask = (1 - rd_mask[:, 0, :, :]).int()  # "not background" == foreground, same as test_cfar.py
            b, h, w = gt_mask.shape

            with torch.no_grad():
                output = net(rd_data)
            n_fg = output.shape[1] - 2
            heatmap_prob = torch.sigmoid(output[:, :n_fg]).cpu().numpy()
            offset = output[:, n_fg:].cpu().numpy()

            for threshold, evaluator in det_evaluators.items():
                pred_mask = torch.zeros(b, h, w, dtype=torch.int32)
                for i in range(b):
                    points = decode_points(heatmap_prob[i], offset[i], threshold, stride, nms_kernel=args.nms_kernel)
                    pred_mask[i] = torch.from_numpy(points_to_mask(points, h, w))
                evaluator.add_batch(gt_mask.cpu(), pred_mask)

            for alpha, evaluator in cfar_evaluators.items():
                # CFAR2D_Parallel precomputes its sampling-index tensors on CPU and was
                # never made device-aware (test_cfar.py never moves rd_data to a device
                # either) -- match that existing behavior rather than porting it to CUDA.
                # CFAR is a single-frame algorithm and was always evaluated on exactly
                # one frame -- rd_data now carries nb_input_channels frames (temporal
                # fusion, see kuralsnet_npu.py's v8), so explicitly take only the most
                # recent one (last channel -- see KuRALSDataset.__getitem__: frames are
                # stacked oldest-first, and the ground-truth mask is drawn from the last
                # one in the window), keeping CFAR's own comparison unchanged rather than
                # accidentally handing it frame_t-1 as an extra "channel" it was never
                # designed to interpret.
                cfar_out = cfar_models[alpha].filter(rd_data[:, -1:, :, :].cpu())
                evaluator.add_batch(gt_mask.cpu(), cfar_out.squeeze(1))

    def report(name, key, evaluator):
        acc, _ = evaluator.get_pixel_acc_class()
        # Foreground-only (class 1), matching Pd/FAR below -- get_pixel_prec_class()'s
        # first return value is the mean over both classes (background+foreground),
        # which is not what Pd/FAR ever reported (they've always used class[1] directly).
        _, prec_by_class = evaluator.get_pixel_prec_class()
        recall, recall_by_class = evaluator.get_pixel_recall_class()
        far, far_by_class = evaluator.get_pixel_far_class()
        miou, _ = evaluator.get_miou_class()
        dice, _ = evaluator.get_dice_class()
        print(f'{name:>18s} {key!s:>8s}  Acc={acc:.4f}  Prec={prec_by_class[1]:.4f}  '
              f'Pd={recall_by_class[1]:.4f}  FAR={far_by_class[1]:.4f}  mIoU={miou:.4f}  mDice={dice:.4f}')

    print(f'\n=== KuRALSNetNPU vs CFAR on {args.dataset} / {args.split} ===')
    for threshold, evaluator in det_evaluators.items():
        report('KuRALSNetNPU', threshold, evaluator)
    for alpha, evaluator in cfar_evaluators.items():
        report(f'CFAR({args.cfar_type})', alpha, evaluator)

    if args.temporal_consistency > 0:
        seq_loader_tc = DataLoader(SequenceDataset(data), batch_size=1, shuffle=False, num_workers=0)
        tc_evaluators = run_temporal_consistency(net, device, cfg, args, seq_loader_tc, paths, stride)
        print(f'\n=== KuRALSNetNPU +temporal-consistency(window={args.temporal_consistency}, '
              f'radius={args.tc_radius}) on {args.dataset} / {args.split} ===')
        for threshold, evaluator in tc_evaluators.items():
            report('KuRALSNetNPU+TC', threshold, evaluator)


if __name__ == '__main__':
    run_benchmark()
