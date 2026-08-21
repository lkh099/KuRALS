"""Ceiling check for kurals/dataset_process/cfar_detect_and_damp_dataset.py's whole
approach family (detect real peaks at native resolution, flatten everything else to a
flat background baseline before pooling): use ground-truth dense annotation masks
instead of a CFAR detector to decide what to preserve. This is not a real preprocessing
pipeline -- it cheats with labels no real detector (CFAR, Gaussian, or otherwise) can
recover -- so it isn't deployable. Its only purpose is to measure the best possible
downstream dice any detect-then-flatten scheme could reach: if training on this still
loses to plain max-pooling, the contrast-dilution theory isn't what's gating downstream
dice, and no smarter detector (Gaussian peak-finding included) would help either.

annotations/dense/<frame>/range_doppler.npy is a (4, H, W) one-hot mask, channel 0 =
background, channels 1-3 = foreground classes (confirmed by inspection: channel 0 is
~99.996% of pixels in a sample frame). A pixel is "detected" here iff any of channels
1-3 is 1 at that pixel.

Masks and dataset-root JSON/txt files are copied unchanged -- only
range_doppler_numpy/*.npy is transformed. Run resize_extracted_dataset.py on this
script's output to get the final small SoC-buffer-sized dataset.
"""
import argparse
import os
import shutil

import numpy as np


def estimate_background_baseline(src, n_sample=200, seed=0):
    """Median native pixel value across a random sample of frames -- the flat value
    non-detected pixels get set to. Matches cfar_detect_and_damp_dataset.py exactly so
    the two are comparable."""
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
    args = parser.parse_args()

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
        src_dense_dir = os.path.join(args.src, seq_name, 'annotations', 'dense')
        os.makedirs(dst_rd_dir, exist_ok=True)
        for fname in os.listdir(src_rd_dir):
            data = np.load(os.path.join(src_rd_dir, fname))
            stem = fname[:-4]
            mask_path = os.path.join(src_dense_dir, stem, 'range_doppler.npy')
            if os.path.isfile(mask_path):
                mask = np.load(mask_path)
                detected = mask[1:].sum(axis=0) > 0
            else:
                # No ground-truth mask for this frame -- can't cheat, so preserve nothing
                # detectable (matches how a real detector would have to treat it: no
                # signal to key off). Flatten the whole frame.
                detected = np.zeros_like(data, dtype=bool)
            damped = np.where(detected, data, baseline)
            n_flagged_total += detected.sum()
            n_pixels_total += detected.size
            np.save(os.path.join(dst_rd_dir, fname), damped)

        dst_dense_dir = os.path.join(args.dst, seq_name, 'annotations', 'dense')
        if os.path.isdir(src_dense_dir):
            shutil.copytree(src_dense_dir, dst_dense_dir, dirs_exist_ok=True)

    print(f'Done. Fraction of pixels flagged as ground-truth targets (preserved): {n_flagged_total / n_pixels_total:.6f}')


if __name__ == '__main__':
    main()
