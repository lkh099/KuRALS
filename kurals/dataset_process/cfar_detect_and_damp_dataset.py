"""Contrast-preserving alternative to plain block max-pooling: detect real peaks at
native resolution via CFAR (its actual intended function -- a binary detection
decision, not the continuous signal-to-clutter ratio kurals/models/cfar.py's
get_scr() computes, which was tried separately and found to destroy dynamic range),
then flatten every non-detected pixel to a flat background baseline before pooling.

Diagnosed problem (see kurals/dataset_process/resize_extracted_dataset.py's
docstring): block max-pooling amplifies background bins via order statistics (max of
~500 native pixels), collapsing measured target-vs-background contrast from ~19.4x
(native) to ~6.7x (pooled). An oracle version of this script (using ground-truth
target locations instead of CFAR to decide what to preserve) recovers the full 19.4x
-- confirming the mechanism. A real CFAR detector (alpha=5.0, ref_cells=1,
guard_cells=1) gets to the same ~19.4x contrast at ~89% recall on the pixel used for
the contrast measurement.

Caveat: the ~11% of targets this detector misses aren't just noisier after this
transform, they're erased entirely (flattened to the same baseline as real
background) -- structurally unrecoverable by any downstream model regardless of
architecture. This is a real trade against the contrast improvement, not a free win;
only training on it settles whether it nets out positive.

Masks (annotations/dense/) and dataset-root JSON/txt files are copied unchanged --
only range_doppler_numpy/*.npy is transformed. Run resize_extracted_dataset.py on
this script's output to get the final small SoC-buffer-sized dataset.
"""
import argparse
import os
import shutil

import numpy as np
import torch

from kurals.models.cfar import CFAR2D_Parallel


def estimate_background_baseline(src, n_sample=200, seed=0):
    """Median native pixel value across a random sample of frames -- the flat value
    non-detected pixels get set to."""
    import glob
    rd_files = glob.glob(os.path.join(src, '*', 'range_doppler_numpy', '*.npy'))
    rng = np.random.RandomState(seed)
    rng.shuffle(rd_files)
    sample = rd_files[:n_sample]
    vals = np.concatenate([np.load(f).ravel() for f in sample])
    return float(np.median(vals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True, help='Root of the already-extracted native-resolution dataset.')
    parser.add_argument('--dst', required=True, help='Output root (still native resolution).')
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--ref-cells', type=int, default=1)
    parser.add_argument('--guard-cells', type=int, default=1)
    args = parser.parse_args()

    cfar = CFAR2D_Parallel(cfar_type='CA', alpha=args.alpha, ref_cells=args.ref_cells, guard_cells=args.guard_cells)
    baseline = estimate_background_baseline(args.src)
    print(f'Background baseline (median native pixel, n=200 frame sample): {baseline:.2f}')

    os.makedirs(args.dst, exist_ok=True)
    for name in os.listdir(args.src):
        src_path = os.path.join(args.src, name)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(args.dst, name))

    seq_names = [n for n in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, n))]
    n_flagged_total, n_pixels_total = 0, 0
    for i, seq_name in enumerate(seq_names):
        print(f'[{i+1}/{len(seq_names)}] {seq_name}')
        src_rd_dir = os.path.join(args.src, seq_name, 'range_doppler_numpy')
        dst_rd_dir = os.path.join(args.dst, seq_name, 'range_doppler_numpy')
        os.makedirs(dst_rd_dir, exist_ok=True)
        for fname in os.listdir(src_rd_dir):
            data = np.load(os.path.join(src_rd_dir, fname))
            x = torch.from_numpy(data).float().view(1, 1, *data.shape)
            detected = cfar.filter(x)[0, 0].numpy().astype(bool)
            damped = np.where(detected, data, baseline)
            n_flagged_total += detected.sum()
            n_pixels_total += detected.size
            np.save(os.path.join(dst_rd_dir, fname), damped)

        src_dense_dir = os.path.join(args.src, seq_name, 'annotations', 'dense')
        dst_dense_dir = os.path.join(args.dst, seq_name, 'annotations', 'dense')
        if os.path.isdir(src_dense_dir):
            shutil.copytree(src_dense_dir, dst_dense_dir, dirs_exist_ok=True)

    print(f'Done. Fraction of pixels flagged as peaks (preserved): {n_flagged_total / n_pixels_total:.5f}')


if __name__ == '__main__':
    main()
