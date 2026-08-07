"""Detection-target dataset wrapper for KuRALSNetNPU.

Every sample produced by KuRALSDataset (annot_type='dense') already carries
exactly one annotated target: 'rd_mask' is a one-hot [n_classes, H, W] array
with a fixed 3x3 blob at that target's single (range, doppler) pixel (see
get_mask()/get_dense() in kurals/dataset_process/kuralscw_processing.py).
Rather than depending on a separately-generated box raster (the 'box'
annotation folder is described in the README but is not produced by
kuralscw_processing.py, only 'dense' is), this wrapper derives the detection
target directly from that same dense mask, so it reuses the exact same
ground-truth pixel already validated by the segmentation training pipeline.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from kurals.loaders.dataloaders import KuRALSDataset
from kurals.legacy.kuralsnet_npu import KuRALSNetNPU


def _gaussian_2d(radius, sigma):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    return g


def _draw_gaussian_splat(channel, center_y, center_x, radius):
    """In-place max-splat a 2D Gaussian (CenterNet/CornerNet convention:
    sigma = diameter / 6) centered at (center_y, center_x), clipped to the
    channel's bounds. Uses np.maximum rather than assignment so overlapping
    splats (not possible here, one target per sample, but standard practice)
    don't overwrite each other's peak."""
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    gaussian = _gaussian_2d(radius, sigma)
    h, w = channel.shape

    left, right = min(center_x, radius), min(w - center_x, radius + 1)
    top, bottom = min(center_y, radius), min(h - center_y, radius + 1)

    splat_region = channel[center_y - top:center_y + bottom, center_x - left:center_x + right]
    gaussian_region = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(splat_region.shape) > 0 and min(gaussian_region.shape) > 0:
        np.maximum(splat_region, gaussian_region, out=splat_region)


def encode_detection_target(rd_mask, out_h, out_w, stride, n_fg_classes, gaussian_radius=2):
    """Convert a dense one-hot [n_classes, H, W] mask (channel 0 = background)
    into CenterNet-style training targets at the head's output resolution.

    `out_h`/`out_w` must come from KuRALSNetNPU.output_hw(H, W) -- not a plain
    H // stride -- so the target shape always matches the model's actual
    (alignment-padded) output shape.

    The heatmap target is a Gaussian splat (not a single hard pixel): with
    exactly one positive cell out of thousands, a hard target gives gradient
    from only one location per sample and treats a one-cell-off prediction as
    a complete miss. A Gaussian bump gives denser supervision and softly
    discounts near-misses instead of punishing them as hard negatives. This
    is the standard CenterNet/CornerNet target encoding; DetectionLoss's
    penalty-reduced focal loss already expects soft [0,1] targets ((1-pos)**
    beta on the negative term is what makes cells near the peak count less as
    negatives), so no loss-side change is needed to consume it. offset/mask
    stay a single hard pixel at the true center -- sub-pixel offset
    regression only makes sense evaluated there.

    radius history: 1 -> 2 was part of the same fix as DetectionLoss's beta
    parameter (see that module's "v4-fix" docstring note): run 1's
    radius=1/sigma=0.5 splat put almost no weight beyond the immediate 4
    neighbors (corner cells got ~exp(-4)=0.02), barely denser than a hard
    single-pixel target in practice. 2 -> 3 (KuRALSNetNPU v6) followed the
    STRIDE 8->4 change (v5): the output grid went from 16x256 to 32x512 (4x
    more cells for the same 1-3 positives per frame), so the same radius=2
    splat now covered proportionally less of the grid than it did before --
    radius=3/sigma=7/6 was meant to restore a comparable relative footprint.
    v6 regressed hard (Pd 18.2% -> ~4.5%), and inspecting the actual raw RD
    magnitude around real annotated targets showed why: the physically
    detectable signal is confined to ~1 input pixel (adjacent cells already
    back near the noise floor -- see conversation record, not reproduced
    here), while radius=3 at STRIDE=4 spreads non-trivial positive weight out
    to +/-2 output cells = +/-8-12 input pixels, i.e. a wide ring of pure
    noise being labeled "plausibly positive". The dense annotation's "3x3
    blob" (get_dense() in kuralscw_processing.py) is a fixed-size drawing
    convention, not a measured target extent, and should not have been used
    to justify a bigger splat. 3 -> 1 (with the dual-scale head still in
    place) recovered some Pd (~4.5% -> 7.6%) but stayed far below v5's
    18.2%, isolating the dual-scale head itself -- not the radius -- as the
    dominant regression (see kuralsnet_npu.py's docstring; the head was
    reverted to v5's single-scale version). With that single-scale head,
    radius=1 (matching the physical signal exactly) gave much cleaner
    precision (up to 0.91 vs v5's 0.51-0.66 ceiling) but *lower* recall
    (Pd 12.8% vs v5's 18.2%) -- radius is a real recall/precision trade, not
    simply "smaller is more correct": the physically-oversized radius=2
    splat gives denser/more robust gradient during training and wins on Pd
    despite being spatially loose. 1 -> 2 restored v5's original, still the best
    Pd result overall at STRIDE=4/2; radius=1's precision profile was noted as
    worth remembering if a deployment ever prioritizes fewer false alarms over
    raw detection rate.

    2 -> 1 (this revision, kuralsnet_npu.py's STRIDE=1 has been in place since
    v12 -- this radius default was never revisited despite every stride cut
    since (v10, v12) explicitly flagging it as an open caveat in
    kuralsnet_npu.py's docstring). At STRIDE=4 (where radius=2 was chosen),
    one output cell = 4x4 native pixels, so a radius=2 splat already only
    loosely corresponded to real pixels; at STRIDE=1, one output cell IS one
    native pixel, so the same radius=2 now means labeling a ~5x5 native-pixel
    neighborhood "positive" around a signal whose measured physical extent is
    ~1 native pixel (see this docstring's history above) -- a much larger,
    more direct mismatch than radius=2 ever had at coarser strides. This is a
    plausible, previously-untested contributor to the discrete/coarse
    confidence-output plateau that motivated a whole session of quantization-
    calibration and architecture experiments (all on kuralsnet_npu.py's v14
    baseline, all regressed or collapsed -- see that file's docstring):
    training the network to output high confidence over a blob 4-5x wider
    than the real signal gives focal-BCE every incentive to learn a small
    number of "blob-average" response levels rather than a sharp, continuous
    gradient. radius=1 at STRIDE=1 is an exact match to the measured physical
    extent for the first time in this project's history -- worth testing on
    its own merits (a real recall/precision trade per the history above) and
    as a completely different axis from every previous plateau investigation.

    Result: a regression. Pd 33.0% at 0.05/0.1 (vs radius=2/v14's 76.4%), 1.05% at
    0.3/0.5, 0% beyond -- worse detection rate at every threshold, though the
    plateau's *shape* reverted to v14's own pattern (two pairs of identical
    thresholds, 0.05=0.1 and 0.3=0.5) rather than getting worse the way v17's
    architecture change did. 14 unique sigmoid values, close to v14's 19 -- so
    output granularity wasn't meaningfully different either way. So the
    signal/label mismatch hypothesis didn't pan out as a fix: matching the label
    to the measured physical extent exactly made the detection task harder to
    learn, not easier -- echoing the same "physically-oversized splat gives
    denser/more robust gradient and wins on Pd despite being spatially loose"
    lesson from the STRIDE=4 (v6/v7) history above, just re-confirmed at STRIDE=1.
    Reverted to 2.

    RETURNS
    -------
    heatmap: np.ndarray, shape (n_fg_classes, out_h, out_w)
        Gaussian-splatted around the true center cell for that class, peak 1.0.
    offset: np.ndarray, shape (2, out_h, out_w)
        Sub-cell (x, y) offset in [0, 1) of the target within its grid cell,
        valid only where `mask` is 1.
    mask: np.ndarray, shape (1, out_h, out_w)
        1.0 at the single true-center grid cell, used to gate the offset loss
        and to normalize the heatmap loss (see DetectionLoss).
    """
    n_classes, _, _ = rd_mask.shape
    assert n_classes == n_fg_classes + 1, (
        f"rd_mask has {n_classes} channels but n_fg_classes={n_fg_classes} "
        "implies n_classes - 1 foreground channels"
    )
    heatmap = np.zeros((n_fg_classes, out_h, out_w), dtype=np.float32)
    offset = np.zeros((2, out_h, out_w), dtype=np.float32)
    mask = np.zeros((1, out_h, out_w), dtype=np.float32)

    for c in range(1, n_classes):
        ys, xs = np.nonzero(rd_mask[c])
        if ys.size == 0:
            continue
        cy = float(ys.mean())
        cx = float(xs.mean())
        gy = min(max(int(cy // stride), 0), out_h - 1)
        gx = min(max(int(cx // stride), 0), out_w - 1)
        _draw_gaussian_splat(heatmap[c - 1], gy, gx, gaussian_radius)
        heatmap[c - 1, gy, gx] = 1.0  # guarantee the exact peak is 1.0, not just approximately via the splat
        offset[0, gy, gx] = (cx / stride) - gx
        offset[1, gy, gx] = (cy / stride) - gy
        mask[0, gy, gx] = 1.0

    return heatmap, offset, mask


class KuRALSDetectionDataset(Dataset):
    """Wraps KuRALSDataset (annot_type='dense') and converts each sample's
    dense mask into detection targets for KuRALSNetNPU.

    PARAMETERS
    ----------
    dataset: SequenceCarradaDataset-style object (sequence of frame entries)
    path_to_frames: str
    process_signal: boolean
    n_frames: int
    n_fg_classes: int
        nb_classes - 1 (foreground classes only, background excluded).
    transformations: list of functions (optional)
    """

    def __init__(self, dataset, path_to_frames, process_signal, n_frames,
                 n_fg_classes, transformations=None):
        self.inner = KuRALSDataset(dataset, 'dense', path_to_frames,
                                    process_signal, n_frames,
                                    transformations, add_temp=False)
        self.n_fg_classes = n_fg_classes

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        frame = self.inner[idx]
        rd_matrix = frame['rd_matrix']  # (n_frames, H, W)
        rd_mask = frame['rd_mask']      # (n_classes, H, W)

        h, w = rd_mask.shape[-2:]
        out_h, out_w = KuRALSNetNPU.output_hw(h, w)
        heatmap, offset, mask = encode_detection_target(
            rd_mask, out_h, out_w, KuRALSNetNPU.STRIDE, self.n_fg_classes)

        return {
            'rd_matrix': torch.from_numpy(np.ascontiguousarray(rd_matrix)),
            'heatmap': torch.from_numpy(heatmap),
            'offset': torch.from_numpy(offset),
            'mask': torch.from_numpy(mask),
        }
