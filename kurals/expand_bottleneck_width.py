"""One-off checkpoint transform: adapt a trained KuRALSNetNPUSeg checkpoint
(bottleneck_ch=64, the native-resolution model) for finetuning into a model with a
different bottleneck_ch (and/or bottleneck_kernel_size, since both commonly change
together for the 8x64 SoC-buffer branch -- see expand_kernel3.py) instead of training
that variant from scratch.

Generalizes expand_kernel3.py's approach: rather than hardcoding which keys are
channel-count-dependent, build a fresh target-shaped model and copy every key whose
shape matches the source checkpoint; any key whose shape doesn't match (down2.pw,
bott_c/bott_d1/bott_d2's dw+pw+bn, upconv_b's dw+pw -- see
kurals/models/kuralsnet_npu_seg.py's __init__ for which layers bottleneck_ch touches)
is left at its fresh random initialization. Everything outside the bottleneck path
(stem, stageA, down1, dec_a, head_out) is bottleneck_ch-independent and transfers
unchanged regardless of target width.
"""
import argparse

import torch

from kurals.models import KuRALSNetNPUSeg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True, help='Path to a bottleneck_ch=64 checkpoint.')
    parser.add_argument('--n-classes', type=int, default=4)
    parser.add_argument('--n-frames', type=int, default=1)
    parser.add_argument('--bottleneck-ch', type=int, required=True)
    parser.add_argument('--bottleneck-kernel-size', type=int, default=3)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    state_dict = torch.load(args.ckpt, map_location='cpu')
    fresh = KuRALSNetNPUSeg(n_classes=args.n_classes, n_frames=args.n_frames,
                             bottleneck_ch=args.bottleneck_ch,
                             bottleneck_kernel_size=args.bottleneck_kernel_size)
    fresh_sd = fresh.state_dict()

    out = {}
    n_reinit = 0
    for key, fresh_val in fresh_sd.items():
        src_val = state_dict.get(key)
        if src_val is not None and src_val.shape == fresh_val.shape:
            out[key] = src_val
        else:
            out[key] = fresh_val
            n_reinit += 1

    torch.save(out, args.out)
    print(f'Wrote {args.out} ({len(out)} keys, {n_reinit} randomly re-initialized '
          f'for bottleneck_ch={args.bottleneck_ch}, bottleneck_kernel_size={args.bottleneck_kernel_size})')


if __name__ == '__main__':
    main()
