# model_release_v01 -- R64-D8 KuRALSNetNPUSeg 0.5x, int8 QAT

Built by `exp03/make_release.py` from run `Q_kuralsnet_npu_seg_wm0.5_ss1_R64-D8_qlr0.0001_s44` (QAT fine-tune of the best-validation FP32 checkpoint, seed 44; best validation mIoU of the three final seeds, 0.598). Design-point rationale: `exp03/summaries/design_point_decision.md`.

| Item | Value |
|---|---|
| RD resolution | R64-D8: 8 Doppler x 64 range |
| Tensor order | `(n_frames=1, Doppler=8, Range=64)`, row-major, Doppler in natural FFT order (zero-Doppler at index 0) |
| Input dtype / scaling | int8 in [0,127]; DSP 5.3-bit log2 code `clamp(log8(uint32(floor(P)) >> 16) - 0, 0, 127)`; input_scale = 1.0, no zero point |
| Output dtype | dequantized float32 logits `(4, 8, 64)` from an int8 head (output_scale 0.1663, signed); decode with per-pixel argmax |
| Class ordering | 0 background, 1 UAV, 2 and 3 non-UAV channels (see `class_map.json`) |
| Weights | int8, per-channel; hardware weight is `2*w+1`; scales and biases int16 |
| Model size | 9,587 params; 8,585 B weights + 1,002 B scales + 1,002 B biases |
| Complexity | 1.57 M MACs; peak NPU buffer 16,384 B per bank (limit 32,768 B); CUTPOINT 3 |
| Checksum | SHA-256 of `weights_int.bin` + `scales.bin` + `biases.bin`: see `metrics.json` -> `weights_sha256` |

## Expected software output

`exact_integer_replay` in `metrics.json`: `kurals/int8_inference.py` over the whole R64-D8 Test split gives **test mIoU 0.6002**, per-class IoU [0.9988, 0.6623, 0.4731, 0.2667]. The QAT training-time simulation gives 0.6017 and agrees with the integer replay on 99.9993 % of pixels. Any hardware run should match the integer replay, not the float simulation.

## Files

- `architecture.json` -- layer list in execution order with fused ops, shapes and byte offsets into the three `.bin` files.
- `weights_int.bin`, `scales.bin`, `biases.bin` -- int8 weights, int16 scales, int16 biases (little-endian). `scales.bin` is an addition to the template, kept apart from `biases.bin`.
- `quantization.json`, `preprocessing.json`, `class_map.json`, `config.yaml`, `train_config.json`, `metrics.json`.
- `sample_inputs/sample_NN_int8.npy` -- 8 Test frames that contain a target, shape (1, 8, 64), int8, exactly what the NPU reads.
- `golden_outputs/` -- `logits_f32` (4,8,64), `argmax_u8` and `label_u8` (8,64) per sample.
- `golden_layer_outputs/sample_NN.npz` -- every layer's dequantized output (float32, real units) from the integer replay, keyed by layer name.

## Known limits

- Inputs are block max-pooled from the native map; the SoC computes its own small FFT, so accuracy on silicon data is not established.
- Class 3 is nearly absent in the data (per-class IoU 0.27 on very few cells).
- No measured NPU cycles or latency yet.
- The NPU feasibility check passes for the network at R64-D8 with 32 KB buffers; the `cfg_gen.py` chain was exercised through `npu_check.py` only, not through a board run.
