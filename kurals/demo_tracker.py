"""End-to-end demo: KuRALSNetNPUSeg (phase 1, /2-resolution, argmax decode) ->
UAV-class centroid extraction -> kurals.utils.tracker.SortTracker -> an
animated overlay of confirmed track IDs and trails on the raw range-Doppler
sequence.

Runs frame-by-frame (batch_size=1, shuffle=False) over a single named
sequence so the tracker sees frames in true temporal order -- unlike the
evaluation scripts (test_kuralsnet_vs_cfar.py, test_class_discrimination.py)
which only aggregate per-frame metrics and don't care about order. Default
sequence is Test split's own two-drone recording ("...两个无人机..."), chosen
because it is the only Test-split sequence with more than one simultaneous
target, making it the most informative case for the tracker's Hungarian
association (a single-target sequence never has to make an association
decision at all).

Same checkpoint-loading convention as test_class_discrimination.py's seg
branch: bare state_dict (kurals/learners/model.py's _save_results), QAT
forced on + observers frozen (this checkpoint was always trained that way;
evaluating without it would score the float32-equivalent network instead of
the real int8-simulated one).
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from torch.utils.data import DataLoader

from kurals.loaders.dataset import KuRALS_CW, KuRALS_PD
from kurals.loaders.dataloaders import SequenceDataset, KuRALSDataset
from kurals.models import KuRALSNetNPUSeg
from kurals.utils.functions import normalize
from kurals.utils.paths import Paths
from kurals.utils.tracker import SortTracker, extract_centroids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config_files/kuralsnet_npu_seg.json')
    parser.add_argument('--model-path', dest='model_path', required=True)
    parser.add_argument('--dataset', default='KuRALS_CW')
    parser.add_argument('--split', default='Test')
    parser.add_argument('--sequence', default='两个无人机',
                         help='Substring matched against sequence names to pick which one to run.')
    parser.add_argument('--uav-class-idx', dest='uav_class_idx', type=int, default=1)
    parser.add_argument('--max-match-dist', dest='max_match_dist', type=float, default=15.0)
    parser.add_argument('--confirm-hits', dest='confirm_hits', type=int, default=3)
    parser.add_argument('--max-misses', dest='max_misses', type=int, default=5)
    parser.add_argument('--max-frames', dest='max_frames', type=int, default=200,
                         help='Cap the number of frames rendered (sequences can be long).')
    parser.add_argument('--output', default='tracker_demo.gif')
    parser.add_argument('--title', default=None,
                         help='Plot title (ASCII only -- the default DejaVu Sans font used for '
                              'GIF rendering has no CJK glyphs, so raw sequence names render as '
                              'tofu boxes). Defaults to a generic label if omitted.')
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    with open(args.cfg, 'r') as fp:
        cfg = json.load(fp)
    cfg['dataset'] = args.dataset
    cfg['nb_classes'] = 4 if args.dataset == 'KuRALS_CW' else 5

    device = torch.device(args.device)
    net = KuRALSNetNPUSeg(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'],
                           dataset_type=cfg['dataset'])
    state = torch.load(args.model_path, map_location='cpu')
    net.load_state_dict(state, strict=True)
    net.to(device)
    net.eval()
    net.set_quant_enabled(True)
    net.freeze_observers()
    print(f'Loaded {args.model_path} (quant_enabled=True, observers frozen)')

    paths = Paths().get()
    data = (KuRALS_CW() if args.dataset == 'KuRALS_CW' else KuRALS_PD()).get(args.split)
    seq_loader = DataLoader(SequenceDataset(data), batch_size=1, shuffle=False, num_workers=0)

    seq_name = None
    seq = None
    for name, s in seq_loader:
        if args.sequence in name[0]:
            seq_name, seq = name[0], s
            break
    if seq_name is None:
        available = [name[0] for name, _ in seq_loader]
        raise SystemExit(f'No sequence matching {args.sequence!r} in {args.split}. '
                          f'Available: {available}')
    print(f'Running on sequence: {seq_name}')
    if args.title is None:
        args.title = f'{args.dataset} / {args.split} sequence (UAV-class tracking demo)'

    path_to_frames = paths[args.dataset] / seq_name
    kurals_dataset = KuRALSDataset(seq, cfg['annot_type'], path_to_frames,
                                    cfg['process_signal'], cfg['nb_input_channels'], None, False)
    frame_dataloader = DataLoader(kurals_dataset, shuffle=False, batch_size=1, num_workers=0)

    tracker = SortTracker(max_match_dist=args.max_match_dist, confirm_hits=args.confirm_hits,
                           max_misses=args.max_misses)

    frames_rd, frames_pred_mask, frames_gt_mask, frames_tracks = [], [], [], []
    with torch.no_grad():
        for i, frame in enumerate(frame_dataloader):
            if i >= args.max_frames:
                break
            rd_data = frame['rd_matrix'].float().to(device)
            rd_mask = frame['rd_mask'].float()
            rd_data_norm = normalize(rd_data, args.dataset, 'range_doppler', norm_type=cfg['norm_type'])

            logits = net(rd_data_norm)
            pred_class = logits.argmax(dim=1)[0].cpu().numpy()
            gt_class = rd_mask.argmax(dim=1)[0].numpy()

            uav_mask = pred_class == args.uav_class_idx
            centroids = extract_centroids(uav_mask)
            confirmed = tracker.step(centroids)

            frames_rd.append(rd_data[0, 0].cpu().numpy())
            frames_pred_mask.append(uav_mask)
            frames_gt_mask.append(gt_class == args.uav_class_idx)
            frames_tracks.append([(t.id, t.kf.position, list(t.history)) for t in confirmed])

    n = len(frames_rd)
    print(f'Rendered {n} frames. Max concurrent confirmed tracks: '
          f'{max((len(t) for t in frames_tracks), default=0)}. '
          f'Total unique track IDs ever confirmed: '
          f'{len({tid for fr in frames_tracks for tid, _, _ in fr})}')

    vmin = min(f.min() for f in frames_rd)
    vmax = max(f.max() for f in frames_rd)

    fig, ax = plt.subplots(figsize=(10, 6))

    def render(i):
        ax.clear()
        ax.imshow(frames_rd[i], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
        gt_ys, gt_xs = np.nonzero(frames_gt_mask[i])
        if len(gt_ys):
            ax.scatter(gt_xs, gt_ys, s=4, c='white', alpha=0.3, label='GT UAV pixels')
        for tid, (py, px), history in frames_tracks[i]:
            hy = [p[0] for p in history[-15:]]
            hx = [p[1] for p in history[-15:]]
            ax.plot(hx, hy, '-', color='red', linewidth=1.5, alpha=0.8)
            ax.scatter([px], [py], s=60, facecolors='none', edgecolors='red', linewidths=2)
            ax.text(px + 3, py + 3, f'ID {tid}', color='red', fontsize=9, weight='bold')
        ax.set_title(f'{args.title}  |  frame {i+1}/{n}')
        ax.set_xlabel('Doppler bin')
        ax.set_ylabel('Range bin')

    anim = animation.FuncAnimation(fig, render, frames=n, interval=1000 / args.fps)
    anim.save(args.output, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f'Saved demo animation to {args.output}')


if __name__ == '__main__':
    main()
