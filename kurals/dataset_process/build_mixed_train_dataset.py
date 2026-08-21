"""Build a dataset where only the TRAIN split's frames get a privileged (label-requiring)
transform applied, while Validation/Test stay on the realistic, deployable distribution.

Motivation: kuralsnet_npu_seg_8x64_oracle_damp_kernel3_finetune_dropout.json's 0.4821 val
dice used the oracle-damp transform (kurals/dataset_process/oracle_detect_and_damp_dataset.py)
on train AND val/test alike. That's a valid ceiling-check (is there real headroom at all?)
but not a deployable result or even a valid estimate of one, because real deployment never
has ground truth to oracle-clean its input -- val/test evaluated on oracle-cleaned data
measures performance in a world that won't occur at inference.

Using oracle (or any label-requiring transform) to build the TRAIN set alone is completely
legitimate, though -- training data may use privileged information the model won't have at
inference, same as e.g. teacher forcing. What's untested is whether a model trained on the
clean oracle distribution transfers to the noisy realistic one it will actually see. This
script builds that experiment's dataset: base = plain max-pooled --realistic-src (the
deployable distribution, used for Val/Test), frames belonging to Train (per
kurals.loaders.dataset.KuRALS_CW's own _split(), imported directly so the split is
byte-identical to what train.py will use) get overwritten with the corresponding
--privileged-src frame instead.

--realistic-src and --privileged-src must be two already-pooled (8x64) dataset variants
built from the same underlying native extraction (same sequence/frame IDs), e.g. the plain
max-pooled dataset and the oracle-damped dataset.
"""
import argparse
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--realistic-src', required=True, help='Deployable-distribution pooled dataset (e.g. plain max-pooled). Used as-is for Val/Test, and as the base for Train frames not overridden.')
    parser.add_argument('--privileged-src', required=True, help='Label-requiring pooled dataset (e.g. oracle-damped). Only its Train-split frames get copied in.')
    parser.add_argument('--dst', required=True)
    args = parser.parse_args()

    # KuRALS_CW's _split() only needs light_dataset_frame_oriented.json, which is identical
    # across variants derived from the same native extraction -- read it directly rather than
    # going through kurals.utils.paths.Paths()/config.ini to avoid clobbering the caller's
    # config.ini mid-run.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import json
    import numpy as np

    with open(os.path.join(args.realistic_src, 'light_dataset_frame_oriented.json'), encoding='GBK') as fp:
        annotations = json.load(fp)

    ratio_split = [0.8, 0.1, 0.1]
    cum_ratio = np.cumsum(ratio_split)
    train_frames = {}
    for sequence in annotations.keys():
        if '机场人车' not in sequence:
            num_frames = len(annotations[sequence])
            train_frames[sequence] = set(annotations[sequence][0:int(num_frames * cum_ratio[0])])
        else:
            num_frames = len(annotations[sequence]) - 200
            train_frames[sequence] = set(
                annotations[sequence][0:int(200 * cum_ratio[0])] +
                annotations[sequence][200:200 + int(num_frames * cum_ratio[0])]
            )

    print(f'Copying base (realistic/deployable) dataset {args.realistic_src} -> {args.dst} ...')
    shutil.copytree(args.realistic_src, args.dst, dirs_exist_ok=True)

    n_overridden, n_missing = 0, 0
    for sequence, frame_ids in train_frames.items():
        dst_rd_dir = os.path.join(args.dst, sequence, 'range_doppler_numpy')
        priv_rd_dir = os.path.join(args.privileged_src, sequence, 'range_doppler_numpy')
        if not os.path.isdir(priv_rd_dir):
            print(f'  [!] no privileged-src sequence dir for {sequence}, skipping its {len(frame_ids)} train frames')
            n_missing += len(frame_ids)
            continue
        for frame_id in frame_ids:
            fname = frame_id + '.npy'
            priv_path = os.path.join(priv_rd_dir, fname)
            dst_path = os.path.join(dst_rd_dir, fname)
            if os.path.isfile(priv_path):
                shutil.copy2(priv_path, dst_path)
                n_overridden += 1
            else:
                n_missing += 1
        print(f'[{sequence}] {len(frame_ids)} train frames overridden with privileged-src')

    print(f'Done. {n_overridden} train frames overridden with privileged-src, {n_missing} missing (left as realistic-src). Val/Test untouched (still realistic-src).')


if __name__ == '__main__':
    main()
