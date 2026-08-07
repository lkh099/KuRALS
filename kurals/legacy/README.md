# Legacy: CenterNet-style detection head

This directory holds the retired NPU-legal detection pipeline: `KuRALSNetNPU`
(`kuralsnet_npu.py`), a heatmap+offset head trained via `train_detect.py` /
`model_detect.py` on `dataloaders_detect.py` with `detection_loss.py`, decoded via
`decode.py`, and evaluated by `test_detect_vs_cfar.py`.

It was superseded by the segmentation-head model at the repo root
(`kurals/models/kuralsnet_npu_seg.py`), which reproduces baseline `kuralsnet`'s task
formulation on the same NPU-legal backbone and reaches substantially better
precision/recall — see that file's module docstring for why. This code is kept for
reference only; nothing in the live training/eval/export pipeline imports from here.
