"""One-off checkpoint transform: adapt a trained KuRALSNetNPUSeg checkpoint
(bottleneck_kernel_size=5, the native-resolution model) for finetuning into a
bottleneck_kernel_size=3 model instead of training that variant from scratch.

Only bott_c/bott_d1/bott_d2's depthwise conv weight has a kernel-size-dependent
shape (kurals/models/quant.py's QuantResidualDWSeparableBlock: only .dw.conv.weight
depends on dw_kernel_size, .dw.bn/.pw/observers do not -- see
kurals/models/kuralsnet_npu_seg.py's bottleneck_kernel_size docstring). Those three
keys are randomly initialized (from a fresh kernel3 model); everything else --
including the rest of the bottleneck's BN/observer state -- transfers unchanged.
"""
import argparse
import sys

import torch

sys.path.insert(0, '/home/lkh099/xsrc/KuRALS')
from kurals.models import KuRALSNetNPUSeg

MISMATCHED_KEYS = ['bott_c.dw.conv.weight', 'bott_d1.dw.conv.weight', 'bott_d2.dw.conv.weight']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True, help='Path to a bottleneck_kernel_size=5 checkpoint.')
    parser.add_argument('--n-classes', type=int, default=4)
    parser.add_argument('--n-frames', type=int, default=1)
    parser.add_argument('--bottleneck-ch', type=int, default=64)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    state_dict = torch.load(args.ckpt, map_location='cpu')
    fresh = KuRALSNetNPUSeg(n_classes=args.n_classes, n_frames=args.n_frames,
                             bottleneck_ch=args.bottleneck_ch, bottleneck_kernel_size=3)
    fresh_sd = fresh.state_dict()

    out = dict(state_dict)
    for key in MISMATCHED_KEYS:
        assert state_dict[key].shape != fresh_sd[key].shape, f'{key} unexpectedly already matches'
        out[key] = fresh_sd[key]

    torch.save(out, args.out)
    print(f'Wrote {args.out} ({len(out)} keys, {len(MISMATCHED_KEYS)} randomly re-initialized)')


if __name__ == '__main__':
    main()
