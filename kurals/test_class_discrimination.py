"""Class-aware companion to test_detect_vs_cfar.py: instead of collapsing
KuRALSNetNPU's n_fg_classes heatmap channels into a single "any foreground"
decision, this keeps each decoded point's most likely class (see
kurals.utils.decode.decode_points_with_class) and scores per-class Pd/Prec
against the ground-truth dense mask's own channel identity.

For KuRALS_CW, foreground channel 1 = UAV, channels 2-3 = pedestrian/vehicle
(see kurals/dataset_process/kuralscw_processing.py's get_mask()). Rather than
report a 4x4 confusion matrix (background + 3 raw classes, most of it noise
given the class imbalance -- 8 of 9 recorded sequences are pure-UAV, and the
only non-UAV signal at all comes from a single sequence's pedestrian/vehicle
targets), everything that isn't the UAV class is merged into one "non-UAV"
class, giving a 3-class {background, UAV, non-UAV} confusion matrix per
threshold. --uav-class-idx (default 1, i.e. channel 1) selects which
ground-truth mask value counts as UAV, so this stays usable on a dataset with
a different channel convention if ever pointed at one.

CAVEAT worth keeping in mind reading these numbers: since only one recording
session ever contains a non-UAV target, whatever "non-UAV" precision/recall
comes out of this is largely measuring "does this one scene's pedestrian/
vehicle signature look different from a UAV's", not a signature that's been
shown to generalize across radar geometries or environments.

Also supports kuralsnet_npu_seg (kurals/models/kuralsnet_npu_seg.py), the
segmentation-formulated NPU-legal model -- structurally different enough
(direct per-pixel argmax(logits) decode, no heatmap/offset split, no
threshold to sweep, bare-state_dict checkpoint via kurals/learners/model.py's
_save_results instead of DetectionModel's dict-with-'net') that it gets its
own loading/decode branch below rather than being forced through
decode_points_with_class. merge_uav_nonuav and the Evaluator/reporting shape
are shared across both paths so the two are directly comparable.
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from kurals.loaders.dataset import KuRALS_CW, KuRALS_PD
from kurals.loaders.dataloaders import SequenceDataset, KuRALSDataset
from kurals.models import KuRALSNetNPUSeg
from kurals.legacy.checkpoint_io import load_detection_model
from kurals.legacy.decode import decode_points_with_class, points_to_class_mask
from kurals.utils.functions import normalize
from kurals.utils.metrics import Evaluator
from kurals.utils.paths import Paths

CLASS_NAMES = ('background', 'UAV', 'non-UAV')


def merge_uav_nonuav(class_mask, uav_value):
    """class_mask: int array, 0=background, else ground-truth mask channel
    index (1..n_fg_classes). Returns the same shape with 0 unchanged,
    uav_value -> 1, every other nonzero value -> 2."""
    merged = np.zeros_like(class_mask)
    merged[class_mask == uav_value] = 1
    merged[(class_mask != 0) & (class_mask != uav_value)] = 2
    return merged


def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config_files/kuralsnet_npu_seg.json', help='Training config of the model to test.')
    parser.add_argument('--dataset', default='KuRALS_CW')
    parser.add_argument('--split', default='Test', help='Train, Validation or Test')
    parser.add_argument('--model-path', dest='model_path', required=True, help='Path to a checkpoint (dict w/ "net") or bare state_dict.')
    parser.add_argument('--quant', choices=['auto', 'on', 'off'], default='auto')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument('--uav-class-idx', dest='uav_class_idx', type=int, default=1,
                         help='Ground-truth mask value (= dense-mask channel index) that counts as UAV. '
                              'Default 1, matching KuRALS_CW\'s get_mask() convention (channel 1 = UAV).')
    parser.add_argument('--nms-kernel', dest='nms_kernel', type=int, default=1,
                         help='See test_detect_vs_cfar.py -- default off (1), same per-pixel-vs-9-pixel-blob '
                              'reasoning applies here, now applied per class instead of any-class.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    with open(args.cfg, 'r') as fp:
        cfg = json.load(fp)
    cfg['dataset'] = args.dataset
    cfg['nb_classes'] = 4 if args.dataset == 'KuRALS_CW' else 5

    paths = Paths().get()
    data = (KuRALS_CW() if args.dataset == 'KuRALS_CW' else KuRALS_PD()).get(args.split)
    seq_loader = DataLoader(SequenceDataset(data), batch_size=1, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    is_seg_model = cfg['model'] == 'kuralsnet_npu_seg'

    if is_seg_model:
        # Bare state_dict (kurals/learners/model.py's _save_results), no
        # 'iteration'/'cfg' quant metadata -- unlike load_detection_model's
        # 'auto' mode, force QAT on: this checkpoint was always trained with
        # it, and evaluating without it would silently score the
        # float32-equivalent network instead of real int8-simulated behavior.
        net = KuRALSNetNPUSeg(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'],
                               dataset_type=cfg['dataset'], bottleneck_ch=cfg.get('bottleneck_ch', 64))
        state = torch.load(args.model_path, map_location='cpu')
        net.load_state_dict(state, strict=True)
        net.to(device)
        net.eval()
        net.set_quant_enabled(True)
        net.freeze_observers()
        print(f'Loaded {args.model_path} (quant_enabled=True)')
        # No threshold to sweep -- argmax is the model's only operating point.
        thresholds = [None]
    else:
        net, quant_enabled = load_detection_model(cfg, args.model_path, device, args.quant)
        print(f'Loaded {args.model_path} (quant_enabled={quant_enabled})')
        stride = net.stride
        thresholds = args.thresholds

    evaluators = {t: Evaluator(num_class=3) for t in thresholds}

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
            gt_class = rd_mask.argmax(dim=1).int().cpu().numpy()  # (B, H, W), 0=bg, else channel index
            b, h, w = gt_class.shape
            gt_merged = merge_uav_nonuav(gt_class, args.uav_class_idx)

            with torch.no_grad():
                output = net(rd_data)

            if is_seg_model:
                # output: (B, n_classes, H, W) raw logits including background
                # at index 0 -- argmax decode, same convention as
                # test_kuralsnet_vs_cfar.py.
                pred_class = output.argmax(dim=1).int().cpu().numpy()
                pred_merged = merge_uav_nonuav(pred_class, args.uav_class_idx)
                evaluators[None].add_batch(torch.from_numpy(gt_merged), torch.from_numpy(pred_merged))
            else:
                n_fg = output.shape[1] - 2
                heatmap_prob = torch.sigmoid(output[:, :n_fg]).cpu().numpy()
                offset = output[:, n_fg:].cpu().numpy()
                for threshold, evaluator in evaluators.items():
                    pred_merged = np.zeros((b, h, w), dtype=np.int32)
                    for i in range(b):
                        points, classes = decode_points_with_class(
                            heatmap_prob[i], offset[i], threshold, stride, nms_kernel=args.nms_kernel)
                        pred_class = points_to_class_mask(points, classes, h, w)
                        pred_merged[i] = merge_uav_nonuav(pred_class, args.uav_class_idx)
                    evaluator.add_batch(torch.from_numpy(gt_merged), torch.from_numpy(pred_merged))

    print(f'\n=== {cfg["model"]} UAV vs non-UAV discrimination on {args.dataset} / {args.split} ===')
    print(f'(uav_class_idx={args.uav_class_idx}; non-UAV merges every other foreground channel)\n')
    for threshold, evaluator in evaluators.items():
        _, prec_by_class = evaluator.get_pixel_prec_class()
        _, recall_by_class = evaluator.get_pixel_recall_class()
        cm = evaluator.confusion_matrix
        print(f'threshold={"argmax" if threshold is None else threshold}')
        for c in (1, 2):
            print(f'  {CLASS_NAMES[c]:>8s}  Prec={prec_by_class[c]:.4f}  Pd(recall)={recall_by_class[c]:.4f}  '
                  f'(gt cells={int(cm[c].sum())}, pred cells={int(cm[:, c].sum())})')
        print(f'  confusion matrix (rows=gt, cols=pred) [{", ".join(CLASS_NAMES)}]:')
        print(f'    {cm.astype(int)}')


if __name__ == '__main__':
    run_benchmark()
