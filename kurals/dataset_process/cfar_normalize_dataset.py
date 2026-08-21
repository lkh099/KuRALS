"""Contrast-normalize an already-extracted KuRALS dataset's RD maps at native
resolution, before any downsampling.

Diagnosed problem (measured on 400 sampled targets from KuRALS_CW): block
max-pooling a raw-magnitude RD map down to a small SoC buffer (e.g. 8x64)
correctly preserves a target's own peak value 94% of the time, but it also
inflates each BACKGROUND bin's value the same way (every bin grabs its own
local peak from ~500 native pixels), so the target-vs-background contrast a
detector needs collapses -- median contrast drops from ~19.4x at native
resolution to ~6.7x after pooling (68% relative loss).

This script replaces raw magnitude with local signal-to-clutter ratio (SCR,
via kurals.models.cfar.CFAR2D_Parallel.get_scr -- the same CA-CFAR machinery
test_kuralsnet_vs_cfar.py uses for the baseline comparison, just exposing the
continuous noise-floor ratio instead of its final binary decision) computed
at native resolution, so the discriminative signal survives being pooled down
to a small buffer instead of getting drowned by inflated background peaks.

NOTE: this only makes sense as a training-time preprocessing step run against
this repo's dataset -- there is no equivalent native-resolution buffer on the
target deployment SoC to compute this against (confirmed: no such buffer
exists there), so a model trained on CFAR-normalized input cannot be deployed
as-is against raw-magnitude hardware input. This is a diagnostic/ceiling
experiment: does the accuracy gap close if the contrast-collapse problem is
fixed, and is that worth pursuing a hardware-side CFAR front end for.

Masks (annotations/dense/) and dataset-root JSON/txt files are copied
unchanged -- only range_doppler_numpy/*.npy is transformed. Run
resize_extracted_dataset.py on this script's output to get the final small
SoC-buffer-sized dataset (its block max-pooling is unchanged; it now just
pools SCR values instead of raw magnitude).
"""
import argparse
import os
import shutil

import numpy as np
import torch

from kurals.models.cfar import CFAR2D_Parallel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True, help='Root of the already-extracted native-resolution dataset.')
    parser.add_argument('--dst', required=True, help='Output root for the CFAR-normalized dataset (still native resolution).')
    parser.add_argument('--ref-cells', type=int, default=1)
    parser.add_argument('--guard-cells', type=int, default=1)
    args = parser.parse_args()

    cfar = CFAR2D_Parallel(cfar_type='CA', ref_cells=args.ref_cells, guard_cells=args.guard_cells)
    # CFAR2D_Parallel's internal grid buffers (self.prf) are built on CPU with no device
    # transfer support -- run on CPU (this is a batch preprocessing job, not latency-critical).
    device = torch.device('cpu')

    os.makedirs(args.dst, exist_ok=True)
    for name in os.listdir(args.src):
        src_path = os.path.join(args.src, name)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(args.dst, name))

    seq_names = [n for n in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, n))]
    for i, seq_name in enumerate(seq_names):
        print(f'[{i+1}/{len(seq_names)}] {seq_name}')
        src_rd_dir = os.path.join(args.src, seq_name, 'range_doppler_numpy')
        dst_rd_dir = os.path.join(args.dst, seq_name, 'range_doppler_numpy')
        os.makedirs(dst_rd_dir, exist_ok=True)
        for fname in os.listdir(src_rd_dir):
            data = np.load(os.path.join(src_rd_dir, fname))
            x = torch.from_numpy(data).float().to(device).view(1, 1, *data.shape)
            scr = cfar.get_scr(x)[0, 0].cpu().numpy()
            np.save(os.path.join(dst_rd_dir, fname), scr)

        src_dense_dir = os.path.join(args.src, seq_name, 'annotations', 'dense')
        dst_dense_dir = os.path.join(args.dst, seq_name, 'annotations', 'dense')
        if os.path.isdir(src_dense_dir):
            shutil.copytree(src_dense_dir, dst_dense_dir, dirs_exist_ok=True)

    print('Done.')


if __name__ == '__main__':
    main()
