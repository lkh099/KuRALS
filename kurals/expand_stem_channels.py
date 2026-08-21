"""One-off checkpoint transform: expand a trained KuRALSNetNPUSeg checkpoint's
stem from n_frames=1 to n_frames=2 (or more), so it can be loaded via
train.py --finetune into a wider-stem model instead of training from scratch.

Only stem.dw/stem.pw touch the input-channel count -- every other layer's
shape is unaffected by n_frames (see kurals/models/kuralsnet_npu_seg.py) and
is copied through unchanged. The new stem channels are initialized by
duplicating the existing single-frame filter, with stem.pw's weights halved
so that two identical input frames reproduce the original single-frame
forward pass at initialization (a standard trick for expanding a conv's
input channel count without discarding pretrained behavior).
"""
import argparse
import torch


def expand_stem(state_dict, n_frames):
    out = dict(state_dict)
    dw_w = state_dict['stem.dw.conv.weight']  # (1, 1, kH, kW)
    out['stem.dw.conv.weight'] = dw_w.repeat(n_frames, 1, 1, 1)
    for key in ('stem.dw.bn.weight', 'stem.dw.bn.bias', 'stem.dw.bn.running_mean', 'stem.dw.bn.running_var'):
        out[key] = state_dict[key].repeat(n_frames)
    pw_w = state_dict['stem.pw.conv.weight']  # (16, 1, 1, 1)
    out['stem.pw.conv.weight'] = pw_w.repeat(1, n_frames, 1, 1) / n_frames
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True, help='Path to a single-frame (n_frames=1) checkpoint.')
    parser.add_argument('--n-frames', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    state_dict = torch.load(args.ckpt, map_location='cpu')
    out = expand_stem(state_dict, args.n_frames)
    torch.save(out, args.out)
    print(f'Wrote {args.out} (stem expanded to n_frames={args.n_frames}, {len(out)} keys)')


if __name__ == '__main__':
    main()
