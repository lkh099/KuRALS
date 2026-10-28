# What the "32 KB NPU buffer" constraint means

Sources: `npu_riscv` (branch `radar-dsp-integration`) `cfg_generator/gen_cfg.py` (allocator, read in full for this
note), `docs/integration/kurals_mp_npu8_integration.md`, `CLAUDE.md` (memory section). Items marked **[confirm]** are my
reading of the code, not yet confirmed by the NPU/RTL owners; resolve them before the paper states the constraint as hardware fact.

## What is limited
- `gen_cfg.py` models on-chip scratch as **Buffer1 and Buffer2**, each `buf_size` bytes (a first-fit free-list allocator).
  Buffer3 is declared but the generator only ever allocates in buffers 1 and 2.
- NPU8 physical sizing: MEM1 workspace `[0x6000_0000, +0x8000)` and MEM2 activation span `[0x6001_0000, +0x8000)`, i.e.
  **32 KB per bank, two banks (64 KB total)**. The limit applies **per bank**, not to the sum. We pass `--buf-size 0x8000`.
- **Peak NPU buffer** (our metric, `kurals/npu_check.py::npu_profile`) = the largest address watermark the allocator reaches in
  either bank (`Buffer1 requirement` / `Buffer2 requirement` printed by gen_cfg.py), run with an unlimited buffer so infeasible
  candidates still get a number. **Feasible iff peak <= 32768 B.** (Cross-checked: the pass/fail of a run at 0x8000 equals peak <= 0x8000.)

## What lives in the buffers (from the allocator)
| Phase | Allocated on-chip | Not on-chip |
|---|---|---|
| Row-reuse layers (before `CUTPOINT`) | that layer's **weights + scale/bias** (`param_size`, resident while the layer runs, freed after) in Buffer1; outputs that feed a later residual/route in the frame-reuse region | IFM/OFM stream through DRAM |
| Cut layer (`CUTPOINT-1`) | its OFM (becomes the first resident frame) | |
| Frame-reuse layers | the **whole IFM and whole OFM** of the current layer, ping-ponged between Buffer1/Buffer2 (so each must be <= 32 KB and they are in different banks), plus any tensor kept alive for a later **eltwise-add** or **concat** consumer until it is consumed | **weights stream from DRAM** (not counted); last-layer output goes to DRAM |

Tensors that must coexist at a layer: input, output, and every pending residual/concat source (route sources before the cut are
reloaded from DRAM and are not resident). Sizes are `H * W * C_padded` bytes (int8), channels rounded **up to a multiple of PF=8**
(depthwise included; see gen_cfg.py `new_filters`/`out_ch` padding). A model with many 4-channel or 1-channel tensors is therefore
charged for 8.

## Double buffering
No explicit double buffering of activations is modelled: the "ping-pong" is the IFM-in-one-bank / OFM-in-the-other allocation above.
The weight SRAM (`weight_buffer_sram.sv`, double-buffered) is a **separate macro** and is not part of this limit.

## Not included in the 32 KB figure
Line buffers and maxpool/upsample row buffers (separate macros, `NPU_KURALS_BUFFERS`), accumulators / temporary convolution state
inside the PE array, `weight_buffer_sram`, DSP staging pages (time-shared inside the concat half of the workspace for the radar
demo, `src/dsp/radar_e2e_memory.h`), and DRAM traffic. **[confirm]** that the radar DSP staging genuinely cannot shrink the usable
span below 32 KB for D16/D32-sized frames (the integration doc says staging is time-shared, so DSP and NPU do not overlap frames).

## How this enters the paper
Experiment B is posed as: maximise validation mIoU subject to `peak_npu_buffer <= 32 KB` (per bank), tie-break on MACs.
Candidates over the limit are logged with their peak and a rejection reason, never silently dropped. The statement to make is
"fits the NPU8 activation banks as modelled by the generator", not "32 KB total on-chip memory".

## Practical consequence observed (kuralsnet_npu_seg, stem_stride=1, enc_depth=1)
width 1.0x: D8 fits (32768 B, exactly the limit), D16 needs 65536 B, D32 more; stem_stride=2 or narrower widths shrink the full-resolution
tensors that dominate the peak.
