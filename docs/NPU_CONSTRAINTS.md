# NPU constraints (consolidated)

Every model in this repo that is meant to run on the NPU must satisfy everything below.
This file collects constraints that were scattered across `CLAUDE.md`, `kurals/cfg_gen.py`,
`kurals/models/quant.py`, `export_int8.py`, and the earlier design conversations. Items are
tagged with how they were established:

- **[spec]** visible in `cfg_gen.py` / the register format
- **[HW]** confirmed by the hardware owner or against RTL / silicon, *not* visible in `cfg_gen.py`
- **[user]** stated by the user during design sessions
- **[guard]** enforced by an assert in code (file named)

Hardware target: 128-MAC array @ 100 MHz, 20 Hz frame rate, i.e. a hard budget of
`128 * 100e6 * 0.050 s = 640 M MAC/inference` (see CLAUDE.md for why `thop` MACs and this
accounting may differ).

## 1. Operator set

| Constraint | Source |
|---|---|
| Conv2d kernel 1 or 3 (any), stride 1 or 2 | [spec][guard quant.py] |
| Depthwise conv kernel 1/3/5; **5x5 only for depthwise** (5x5 dense unverified, excluded) | [spec][guard quant.py] |
| Stride is a single register bit: 1 or 2 only | [spec][guard quant.py] |
| Depthwise forces `out_ch == in_ch` | [spec][guard quant.py] |
| No dilation, no transposed conv, no 3D conv, no bilinear upsample, no attention/LayerNorm/softmax on-NPU | [spec][user] |
| Activations: `leaky` / `relu` / `linear` only (leaky slope 2^-3 = 0.125) | [spec][HW] |
| Maxpool: **2x2 only**, stride 1 or 2 (no 5x5 maxpool; CPU must do any larger NMS pool) | [user] |
| Upsample: nearest 2x only, **fused onto the preceding conv** (no standalone upsample, no pass-through conv exists) | [spec][user] |
| No pass-through / identity conv exists; "free" ops must ride on a real conv's output pipeline | [user] |
| Temporal: at most 2 stacked frames as input channels (via route/concat); frame differencing `x_t - x_{t-1}` unsupported; n_frames > 2 regressed and is not supported | [user] |
| Padding: zero-padding only (no alternative pad mode found) | [spec] |
| Channel parallelism factor PF = 16 (Ti = To); channels are processed in groups of 16 | [spec] |

## 2. Fused ops need exactly one consumer

- Fusion tags `eltwise_add_with`, `maxpool_after`, `upsample_after` assume the raw conv output has
  a single consumer. A conv cannot be both a residual/shortcut operand **and** a route/concat
  source. **[user][guard export_int8.py `_SINGLE_CONSUMER_FUSIONS`]**
- Use only one of residual or concat per conv ("no_concat_with_residual"). [user]

## 3. Eltwise-add (`[shortcut]`)

- **`add -> activation` is illegal.** The fused eltwise-add has no usable activation stage; an add
  must be followed by a conv, and that conv's `act_type` supplies the nonlinearity. Only
  `[shortcut] activation=linear` is legal. The `act_elt` register bit-field exists but is
  unexercised (tiny-yolov2, the only silicon-verified workload, has no shortcut). **[HW]
  [guard export_int8.py `LEGAL_ELTWISE_ACTS`, manifest_to_cfg_input.py]**
  - Model-side form: `eltwise_act='linear'` (pre-activation ResNet ordering).
- An eltwise-add may **never cross the row-reuse -> frame-reuse transition**; both operands must
  be on the same side. **[HW]**

## 4. Row-reuse / frame-reuse dataflow (`CUTPOINT`)

`cfg_gen.py` splits the layer sequence at exactly one `CUTPOINT` (an `npu_idx` threshold):
layers before it are **row-reuse** (`dataflow_en=0`, weights resident, IFM streamed), layers at or
after it are **frame-reuse** (`dataflow_en=1`, IFM resident, weights streamed).

- The row-reuse -> frame-reuse transition happens **at most once**. [HW]
- `[route]` (concat) is legal **only in the frame-reuse domain**. [HW]
  - Consequence: the earliest operand of every route must already be past the CUTPOINT, so
    `CUTPOINT <=` the npu_idx of the earliest route source. For `KuRALSNetNPUSeg` this pins
    `CUTPOINT=3`.
  - A model with several skip concats (e.g. a U-Net) therefore has its CUTPOINT pinned by the
    **earliest skip source**, which pushes almost the whole network into frame-reuse.
- `[shortcut]` may not cross the transition (see sec. 3). [HW]
- **A strided depthwise conv is illegal in the frame-reuse domain.** Downsample there with a
  stride-1 depthwise + fused `maxpool(k=2)` (`downsample='maxpool'` in `QuantDepthwiseSeparableBlock`).
  [HW] This changes the function (strided conv != conv then maxpool), so checkpoints trained with
  the strided form must be retrained.
- Route is a 2-source concat only; concat has **no rescale registers**, so both sources must share a
  quantization grid (the model forces the upconv output onto the skip's grid via
  `out_scale_override`). [spec][HW]

## 5. Quantization datapath

- int8 activations and weights; int16 per-channel scale and bias registers; **no rounding term,
  truncating arithmetic right-shift**. [user]
- Chain: `acc * scale16 -> >> bias_shift -> + bias16 -> >> act_shift -> activation`.
- **Naming trap (verified against RTL `batch_norm_quant_act.sv`):** register field `bias_shift`
  is the shift applied *before* the bias add (`shift1` in quant.py, legal range **[5, 20]**);
  `act_shift` is applied *after* it (`shift2`, range **[0, 7]**). Swapping them saturates every
  layer to -128 on hardware. Reference: tiny_yolov2 uses `bias_shift=19`, `act_shift=4`.
  **[HW][guard quant.py `SHIFT1_RANGE`, `SHIFT2_RANGE`]**
- **Weight multiplier datapath: `w_eff = 2*w_stored + 1`** (always odd, never zero; effective
  range [-255, 255]). Training-time fake-quant and export must use it, `w_scale` calibrated against
  255, not 127. It injects an unlearnable `sum(x)` term into every layer's accumulator. [HW]
- Quantization must be bit-exact: the QAT `forward()` and the pure-integer `int8_inference.py` replay
  of the exported manifest must agree (0.0 max abs diff on random input). [guard: test by replay]
- Activation quantization is per-tensor (CLAUDE.md attributes decoder activation-magnitude collapse risk at native resolution to this).

## 6. Memory / buffers

- Three on-chip activation buffers, each `buf_size = 0x12200 = 74,240 B`. Buffer allocation raises
  `Buf{1,2,3} Allocation error` if a layer's IFM/OFM does not fit. [spec]
- Activation storage likely uses PF=16 channel grouping (channels padded to multiples of 16) -- **inferred, not verified**; see sec. 9.
- Weight + scale/bias storage per layer: `k*k*in*out` (or `k*k*in` depthwise) bytes weights, plus
  `2*out + 2*out` bytes scale/bias. [spec]
- Stride-1 stem at 8x64 costs ~32 KB peak activation (vs ~8 KB for stride 2); still fits.

## 7. Input / output interface

- Input is a single-frame (or 2-frame) RD magnitude map quantized to int8 (`quantize_input.py`,
  `dummy_ifm.py`); fixed global min/max normalization (`norm_type='tvt'`).
- On-NPU output is raw per-pixel logits; argmax / NMS / tracker run on the CPU. [user]
- Real SoC computes its small RD grid with its own small FFT; the repo's block-maxpooled 8x64
  data is an *emulation* only (CLAUDE.md "SoC 8x64 buffer emulation").

## 7b. DSP -> NPU input contract (from npu_riscv `radar-dsp-integration`, docs/integration/radar_dsp_npu_input.md)

- The NPU input is the DSP's int8 code, **input_scale == 1.0** (stem input scale pinned, `fixed_in_scale`):
  `q = clamp(log8(P) - LOG_SCALE, 0, 127)`, `log8(P) = {leading_one_index, 3 bits below it}` (5.3-bit log2,
  0.376 dB/code, 47.7 dB window), `LOG_SCALE` is a uint8 APB register.
- Natural FFT Doppler order (zero-Doppler at index 0), Doppler-major layout, 4 B/pixel (only byte 0 used).
- Implemented in training as `norm_type: "dsp_log[:proxy_shift[:log_scale]]"` (utils/functions.py), bit-exact against
  npu_riscv `dsp_multiscale.log8_uint32`; labels are ifftshifted together with the input.
- The newer NPU8 build in npu_riscv uses Ti=To=8 (PF8), 3x 74 KB buffers replaced by a 2048x128b MEM2 macro (32 KB
  activation span); **the PF=16 / 0x12200 figures in sec. 6 are from this repo's older cfg_gen.py and must be re-checked
  against npu_riscv `cfg_generator/gen_cfg.py` before sizing D16/D32 models.**

## 8. Model-design consequences for the generic baselines (U-Net, KuRALS-Net, ...)

The spec for Exp 03 requires U-Net / KuRALS-Net to be NPU-executable. Stock implementations are
not (`models/unet.py`: bilinear upsample, 2x BN+ReLU double conv with non-fused `cat`;
`models/kuralsnet.py`: dilated ASPP, ConvTranspose2d, Conv3d). The NPU-legal U-Net therefore:

1. uses only conv 3x3/1x1 (+folded BN) with leaky/relu/linear,
2. downsamples with fused `maxpool(2)` (not strided conv in frame-reuse),
3. upsamples with fused nearest-2x on the preceding conv,
4. skip connections via 2-source route only, each skip source's conv having no other fused op,
5. places every route (hence the CUTPOINT) per sec. 4,
6. keeps every activation tensor inside the 74,240 B buffers.

## 9. Open / unverified

- Whether a dense 5x5 conv is legal (representable in the filter_size field, no reference workload).
- Exact PF-padding rule for non-multiple-of-16 channel counts (e.g. the 4-class head, 24-ch layers).
- Whether the `QuantDepthwiseSeparableBlock` depthwise+pointwise pairing is required by the
  scheduler or merely convention (the reference MobileNet/EfficientNet inputs use it).
- Real-hardware behaviour of the combined constraints at R64-D16 / R64-D32 (larger IFMs).
  `cfg_gen.py` buffer allocation is the practical check.
