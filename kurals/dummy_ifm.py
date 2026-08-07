"""Produce one raw RD map .npy to feed into kurals/quantize_input.py, either
as pure random noise (a smoke test of the export+replay pipeline that needs
no dataset at all) or pulled from a real recorded frame.

Output shape matches range_doppler_numpy/*.npy exactly: (H, W) for a single
frame, real-valued (no channel/frame-stacking dim -- quantize_input.py adds
that itself for n_frames>1 by being called once per frame).
"""
import argparse

import numpy as np


def random_ifm(h=124, w=2048, seed=None):
    """Pure random-noise RD map, roughly matching real data's dynamic range
    (KuRALS_CW's raw range_doppler_numpy/*.npy are float64, O(1e0-1e3) --
    exact scale doesn't matter for a pipeline smoke test since normalize()
    rescales using dataset-wide stats regardless)."""
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=50.0, size=(h, w)).astype(np.float64)


def real_ifm(dataset_type, split, seq_index=0, frame_index=0):
    """Pull one real frame's raw RD map from the dataset (first sequence,
    first frame by default)."""
    from kurals.loaders.dataset import KuRALS_CW, KuRALS_PD
    from kurals.utils.paths import Paths

    paths = Paths().get()
    data = (KuRALS_CW() if dataset_type == 'KuRALS_CW' else KuRALS_PD()).get(split)  # dict keyed by sequence name
    seq_name = list(data.keys())[seq_index]
    seq_dir = paths[dataset_type] / seq_name / 'range_doppler_numpy'
    frame_path = sorted(seq_dir.glob('*.npy'))[frame_index]
    return np.load(frame_path), str(frame_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='dummy_ifm.npy')
    parser.add_argument('--random', action='store_true', help='Generate random noise instead of pulling a real frame.')
    parser.add_argument('--h', type=int, default=124, help='(--random only) RD map height.')
    parser.add_argument('--w', type=int, default=2048, help='(--random only) RD map width.')
    parser.add_argument('--seed', type=int, default=None, help='(--random only)')
    parser.add_argument('--dataset', default='KuRALS_CW')
    parser.add_argument('--split', default='Test')
    parser.add_argument('--seq-index', dest='seq_index', type=int, default=0)
    parser.add_argument('--frame-index', dest='frame_index', type=int, default=0)
    args = parser.parse_args()

    if args.random:
        rd_matrix = random_ifm(args.h, args.w, args.seed)
        print(f'Generated random IFM {rd_matrix.shape} -> {args.out}')
    else:
        rd_matrix, frame_path = real_ifm(args.dataset, args.split, args.seq_index, args.frame_index)
        print(f'Pulled real IFM {rd_matrix.shape} from {frame_path} -> {args.out}')

    np.save(args.out, rd_matrix)


if __name__ == '__main__':
    main()
