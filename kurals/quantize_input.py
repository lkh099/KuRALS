"""Convert a raw RD map (as loaded from range_doppler_numpy/*.npy) into the
int8 byte array the NPU actually consumes as its first-layer input, using the
exact same preprocessing (normalize()) the model was trained on, then
quantizing with the exported stem input_scale from export_int8.py's manifest.
"""
import argparse
import json

import numpy as np
import torch

from kurals.utils.functions import normalize


def quantize_input(rd_matrix, dataset_type, norm_type, input_scale):
    """
    PARAMETERS
    ----------
    rd_matrix: np.ndarray or torch.Tensor, shape (n_frames, H, W) or (H, W)
        Raw RD map(s) exactly as loaded from range_doppler_numpy/*.npy (real-
        valued; process_signal=True in the dataset pipeline already applied
        whatever power/log transform is used upstream of this).
    dataset_type: 'KuRALS_CW' or 'KuRALS_PD'
    norm_type: str
        Must match what the model was trained with (e.g. 'tvt') -- normalize()
        uses dataset-wide min/max stats from cwr_rd_stats_all.json /
        pdr_rd_stats_all.json for that.
    input_scale: float
        The exported manifest's "input_scale" (export_int8.py's <out>.json),
        i.e. the stem's calibrated int8 quantization scale.

    RETURNS
    -------
    np.ndarray, same shape as input, dtype int8: the exact bytes the NPU's
    first conv layer consumes. This format is symmetric (no zero-point)
    throughout, so quantization is just round(normalized / input_scale),
    clamped to [-128, 127].
    """
    was_numpy = not torch.is_tensor(rd_matrix)
    x = torch.as_tensor(rd_matrix).float() if was_numpy else rd_matrix.float()

    needs_channel_dim = x.dim() == 2
    if needs_channel_dim:
        x = x.unsqueeze(0)
    x = x.unsqueeze(0)  # normalize() expects a batch dim

    normalized = normalize(x, dataset_type, 'range_doppler', norm_type=norm_type)
    quantized = torch.clamp(torch.round(normalized / input_scale), -128, 127).to(torch.int8)

    quantized = quantized.squeeze(0)
    if needs_channel_dim:
        quantized = quantized.squeeze(0)
    return quantized.numpy() if was_numpy else quantized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rd_npy', help='Path to a raw RD map .npy file (e.g. from range_doppler_numpy/).')
    parser.add_argument('--manifest', required=True, help="export_int8.py's <out>.json (for input_scale).")
    parser.add_argument('--dataset', default='KuRALS_CW')
    parser.add_argument('--norm-type', dest='norm_type', default='tvt')
    parser.add_argument('--out', default=None, help='Output .npy path (default: <rd_npy>.int8.npy).')
    args = parser.parse_args()

    with open(args.manifest, 'r') as fp:
        manifest = json.load(fp)

    rd_matrix = np.load(args.rd_npy)
    quantized = quantize_input(rd_matrix, args.dataset, args.norm_type, manifest['input_scale'])

    out_path = args.out or (args.rd_npy.rsplit('.npy', 1)[0] + '.int8.npy')
    np.save(out_path, quantized)
    print(f'{args.rd_npy} {rd_matrix.shape} {rd_matrix.dtype} -> {out_path} {quantized.shape} {quantized.dtype}')


if __name__ == '__main__':
    main()
