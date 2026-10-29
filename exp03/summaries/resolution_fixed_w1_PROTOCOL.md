# Protocol amendment and pre-registration — fixed-1.0x resolution study

Written **2026-10-29, before the lr 3.333e-4 arm existed**. Nothing below may be
restated after those runs are read. The frozen numbers are reproduced by
`exp03/resolution_stability_and_finals.py --show-criterion`.

## 1. What happened

The development stage trained the fixed 1.0x stride-1 network at lr 1e-3, 200 epochs,
seeds 42/43/44/45, at R64-D8/D16/D32 (`summaries/resolution_fixed_w1.json`). Two of the
twelve runs failed to converge: **D16 s45** and **D32 s44**. D8 had none.

This is the handoff's stop condition ("a fixed 1.0x stride-1 model cannot be trained
stably"), so the recipe was **not** silently changed. The 1e-3 runs are kept exactly as
they are and reported as the behaviour of the recipe as originally specified.

Two consequences for the claim, both of which this amendment exists to avoid:

* the D16/D32 means at lr 1e-3 mix a resolution effect with an optimisation failure, so
  they cannot carry "larger RD representations do not pay for themselves";
* dropping the two runs post hoc would be outcome-conditioned selection, and averaging
  only the survivors answers a different question than the one asked.

## 2. The failure mode, stated correctly

It is **not** all-background collapse. Both runs do emit the UAV class. What fails is the
separation of the foreground channels, together with a loss of background purity:

| run | val mIoU | per-class val IoU `[bg, UAV, c2, c3]` |
|---|---:|---|
| D16 s45 | 0.3274 | `[0.9847, 0.3249, 0.0000, 0.0000]` |
| D32 s44 | 0.3534 | `[0.9683, 0.4343, 0.0108, 0.0004]` |
| every converged run | 0.43–0.59 | bg in `[0.9976, 0.9996]`, c2 in `[0.31, 0.53]` |

Their validation trajectories are flat for all 200 epochs (D16 s45 quartile means 0.271,
0.283, 0.278, 0.274; 0 of 112 evaluations above 0.40).

## 3. Trivial reference, computed from the data

All-background predictor on the Validation split (258 frames, one-hot `(4, H, W)` labels,
exactly one class per pixel), `IoU = [bg_fraction, 0, 0, 0]`:

| | bg fraction | trivial mIoU | class-1 px | class-2 px | class-3 px |
|---|---:|---:|---:|---:|---:|
| R64-D8 | 0.997782 | **0.249445** | 176 | 99 | 18 |
| R64-D16 | 0.998819 | **0.249705** | 187 | 106 | 19 |
| R64-D32 | 0.999294 | **0.249824** | 223 | 127 | 23 |

**Class 3 is excluded from the criterion a priori**: 18–23 pixels in the whole split, and
IoU exactly 0.0000 in two of the four converged D8 runs. It cannot distinguish a failure
from a healthy run.

Note also that the foreground fraction falls from 0.22 % to 0.07 % as Doppler resolution
rises, because the frame grows faster than the target footprint. Class imbalance is
therefore not constant along the resolution axis; this is inherent to the comparison and
is reported with the result, not corrected for.

## 4. Declared degeneracy criterion

A run is **degenerate** if, at its best-validation checkpoint, **both** hold:

* **C1 — behaviour.** `min(IoU_class1, IoU_class2) <= 0.02`: at least one well-represented
  foreground class is effectively never predicted.
* **C2 — trajectory against the trivial reference.** the run's best validation mIoU is
  below `trivial_miou(resolution) + half_reference_gain`, where `half_reference_gain` is
  half the gain over trivial achieved by the **median** of the D8 lr-1e-3 arm — the
  configuration already published and known to converge.

Frozen values, from the D8 lr-1e-3 arm (best-val mIoU 0.5729, 0.5778, 0.5627, 0.5937;
median 0.575343):

```text
half_reference_gain = (0.575343 - 0.249445) / 2 = 0.162949
C2 threshold:  D8 0.412394    D16 0.412654    D32 0.412773
```

Both conditions are required, so a merely weak run is not flagged. Applied to the
development arm this marks exactly D16 s45 and D32 s44, and nothing else. Closest
healthy run: D32 s43, which clears C1 by a wide margin (0.2230 vs 0.02) and C2 by 0.018.

## 5. Declared decision rules

**LR lock**, applied per resolution once the 3.333e-4 arm is complete:

1. count degenerate seeds among 42/43/44/45 at each LR;
2. lock the LR with **zero** degenerate seeds;
3. if both LRs are clean, lock the one with the higher **seed-42** validation mIoU — the
   development seed, as in the original protocol;
4. if both LRs have a degenerate seed, **stop**: report the handoff stop condition and do
   not run final seeds;
5. D8 is not amended and stays at 1e-3 (0 of 4 degenerate).

The LR may differ per resolution. The experimental variable is the RD representation, not
the optimiser; what is held fixed is the architecture, width, stride, training budget, the
LR **selection procedure**, and the fact that the final seeds are untouched.

**QAT LR** is selected per resolution on development seed 42 only, from {1e-4, 3e-5, 1e-5},
by validation mIoU, from the locked FP32 recipe's own seed-42 checkpoint. D8 keeps the
1e-5 already selected this way. Re-selected for D16/D32 if their locked LR changes, since
a QAT LR chosen on a 1e-3 parent does not transfer to a 3.333e-4 parent.

**Final comparison seeds are 46, 47, 48** — fresh, never used for any tuning or for this
amendment — run for all three resolutions at the locked recipe, then QAT from those FP32
checkpoints. Seeds 42–45 become the development record and are reported as such.

## 6. What each stage is allowed to claim

| stage | seeds | answers |
|---|---|---|
| development (lr 1e-3) | 42–45 | how the recipe **as originally specified** behaves at each resolution, including its failures |
| stability arm (lr 3.333e-4) | 42–45 | whether the failures are LR-driven — a sensitivity result, **not** a replacement |
| final | 46–48 | does the larger RD representation help when the model is optimised stably |

Every QAT run is tagged with whether its FP32 parent was degenerate. If QAT lifts a
degenerate parent, that is additional optimisation escaping a bad FP32 solution — it is
**not** evidence that quantization improves accuracy.

The instability itself is reported as a secondary observation, in these terms: the common
training recipe exhibited occasional foreground-separation collapse at D16 and D32 but not
at D8, which motivated a lower-LR stability check before the final comparison. With four
seeds per cell this is not a claim that instability increases with resolution.
