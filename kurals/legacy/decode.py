"""Shared decode logic: turn KuRALSNetNPU's per-cell heatmap+offset output
into full-resolution point detections, then into a binary mask comparable to
CFAR's (used by test_detect_vs_cfar.py). Retired along with the CenterNet
head -- see kurals/legacy/README.md.

NMS: the Gaussian-splat training target (see dataloaders_detect.py) means
several adjacent cells around a true target legitimately score high, not just
the single peak cell -- left alone, decode_points emits multiple nearby
points per real target, one per cell that clears threshold. This is
CenterNet-style point NMS (max-pool + keep-only-local-peaks), not the
box-IoU NMS used by anchor-based detectors -- there's no bounding box here to
compute overlap on, just a peak-pick over the heatmap. This never touches the
NPU: sigmoid decode already happens off-chip, and this is a continuation of
that same off-chip step.

Default is `nms_kernel=1` (off): this repo's comparison-vs-CFAR metric
(test_detect_vs_cfar.py) is a *per-pixel* recall against each target's 9-pixel
(3x3) ground-truth blob, and points_to_mask marks only one pixel per decoded
point, so enabling NMS collapses several adjacent-cell detections landing on
different pixels within the same blob down to one -- inflating precision at
the cost of recall on this specific metric, not reflecting real target-finding
rate. Since CFAR's own Pd gets no such deduplication either (a raw per-pixel
threshold, no clustering), leaving NMS off is the fairer apples-to-apples
comparison. Pass nms_kernel=5 (or any odd k>1) for real single-point-per-target
deployment postprocessing.
"""
import numpy as np
import torch
import torch.nn.functional as F


def decode_points(heatmap_prob, offset, threshold, stride, nms_kernel=1):
    """
    PARAMETERS
    ----------
    heatmap_prob: np.ndarray, shape (n_fg_classes, Hs, Ws)
        sigmoid(heatmap logits) -- already decoded off-NPU.
    offset: np.ndarray, shape (2, Hs, Ws)
        Raw (x, y) sub-cell offset regression output.
    threshold: float
    stride: int
    nms_kernel: int
        Side length of the local-max window used to suppress non-peak cells
        before thresholding (see module docstring). Set to 1 to disable.

    RETURNS
    -------
    list of (y, x) full-resolution points, one per surviving local-peak grid
    cell whose max-over-classes probability clears `threshold`.
    """
    combined = heatmap_prob.max(axis=0)  # (Hs, Ws), any-class foreground probability

    if nms_kernel > 1:
        t = torch.from_numpy(combined).unsqueeze(0).unsqueeze(0)
        pooled = F.max_pool2d(t, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)
        is_peak = (t == pooled).squeeze(0).squeeze(0).numpy()
        combined = np.where(is_peak, combined, 0.0)

    gy_idx, gx_idx = np.nonzero(combined >= threshold)
    points = []
    for gy, gx in zip(gy_idx.tolist(), gx_idx.tolist()):
        ox = float(offset[0, gy, gx])
        oy = float(offset[1, gy, gx])
        cy = (gy + oy) * stride
        cx = (gx + ox) * stride
        points.append((cy, cx))
    return points


def points_to_mask(points, h, w):
    """points: list of (y, x) full-resolution points -> (h, w) int32 binary mask.
    Marks the single nearest pixel per point (not a splatted neighborhood) --
    see test_detect_vs_cfar.py's module docstring for why."""
    mask = np.zeros((h, w), dtype=np.int32)
    for cy, cx in points:
        iy, ix = int(round(cy)), int(round(cx))
        if 0 <= iy < h and 0 <= ix < w:
            mask[iy, ix] = 1
    return mask


def decode_points_with_class(heatmap_prob, offset, threshold, stride, nms_kernel=1):
    """Like decode_points, but also returns each surviving point's most likely
    class (argmax channel of heatmap_prob at that cell, 0-indexed among
    n_fg_classes). Peak selection and thresholding are identical to
    decode_points (still keyed on the max-over-classes probability at each
    cell) -- this only adds a label, it never changes which cells survive, so
    it stays consistent with decode_points's own Pd/Prec numbers for anyone
    who needs both.

    RETURNS
    -------
    (points, classes): points as in decode_points; classes is a same-length
    list of 0-indexed foreground-class ids, one per point.
    """
    combined = heatmap_prob.max(axis=0)  # (Hs, Ws), any-class foreground probability
    class_idx = heatmap_prob.argmax(axis=0)  # (Hs, Ws), which class achieved that max

    if nms_kernel > 1:
        t = torch.from_numpy(combined).unsqueeze(0).unsqueeze(0)
        pooled = F.max_pool2d(t, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)
        is_peak = (t == pooled).squeeze(0).squeeze(0).numpy()
        combined = np.where(is_peak, combined, 0.0)

    gy_idx, gx_idx = np.nonzero(combined >= threshold)
    points = []
    classes = []
    for gy, gx in zip(gy_idx.tolist(), gx_idx.tolist()):
        ox = float(offset[0, gy, gx])
        oy = float(offset[1, gy, gx])
        cy = (gy + oy) * stride
        cx = (gx + ox) * stride
        points.append((cy, cx))
        classes.append(int(class_idx[gy, gx]))
    return points, classes


def points_to_class_mask(points, classes, h, w):
    """points/classes as returned by decode_points_with_class -> (h, w) int32
    mask. 0 = no detection; else (class index + 1), matching the dense
    ground-truth mask's own channel convention (mask value v came from
    rd_mask channel v, so foreground class c sits at value c+1) -- lets a
    decoded mask and a ground-truth rd_mask.argmax(0) be compared directly."""
    mask = np.zeros((h, w), dtype=np.int32)
    for (cy, cx), c in zip(points, classes):
        iy, ix = int(round(cy)), int(round(cx))
        if 0 <= iy < h and 0 <= ix < w:
            mask[iy, ix] = c + 1
    return mask
