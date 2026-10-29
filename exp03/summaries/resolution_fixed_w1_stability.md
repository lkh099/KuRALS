# Stop condition: the fixed 1.0x stride-1 network could not be trained stably at D16/D32

Stage B of `exp03/resolution_stability_and_finals.py`, under the protocol frozen in
`resolution_fixed_w1_PROTOCOL.md` before the lr 3.333e-4 arm was trained.

**No learning rate produced four clean development seeds at R64-D16 or R64-D32, so the fresh-seed
final comparison (seeds 46/47/48) was not run at any resolution** -- a three-way comparison with
two unusable rows cannot answer the question. R64-D8 did lock, at lr 1e-3.

## Seeds with a collapsed foreground channel

| | lr 1e-3 | lr 3.333e-4 |
|---|---|---|
| R64-D8 | 0/4  (criterion: 0/4) | not run |
| R64-D16 | 1/4  (criterion: 1/4) | 3/4  (criterion: 1/4) |
| R64-D32 | 1/4  (criterion: 1/4) | 2/4  (criterion: 2/4) |

Counts are seeds with `min(IoU_UAV, IoU_c2) < 0.05` -- a clearly dead foreground channel,
well away from the criterion's 0.02 epsilon. The frozen-criterion count is in parentheses.

**Lowering the learning rate made stability worse, not better**: across D16+D32, 2 of 8 seeds
have a dead foreground channel at lr 1e-3 and 5 of 8 at lr 3.333e-4. Both readings agree on the
direction, so the conclusion does not depend on where the threshold sits. R64-D8 is clean in all
five runs across both LRs.

## Why the threshold-free count is reported too

Two runs in the new arm sit within 0.007 of flipping the frozen criterion: D16 s44 has UAV IoU
0.0265 against the 0.02 epsilon, and D16 s45 has val mIoU 0.4158 against the 0.412654 threshold.
The criterion is **left exactly as declared** -- retuning it after seeing the arm would discard
the only thing pre-registration buys -- so both counts are given and they agree.

## The failure mode

Every run with a dead foreground channel also has background IoU <= 0.9849; every clean run has
>= 0.9976. The split is bimodal with nothing in between. So the failure is false positives
sprayed across background together with the loss of one foreground channel -- **not** an
all-background collapse, and not something a class-1 test would catch (D32 s43 is clean with a
lower UAV IoU, 0.2230, than either lr-1e-3 failure).

## Per-run detail

| Resolution | FP32 LR | seed | val mIoU | val IoU `[bg, UAV, c2, c3]` | verdict |
|---|---:|---:|---:|---|---|
| R64-D8 | 0.001 | 42 | 0.5729 | `[0.9992, 0.7641, 0.5282, 0.0]` | ok |
| R64-D8 | 0.001 | 43 | 0.5778 | `[0.9993, 0.7635, 0.4627, 0.0857]` | ok |
| R64-D8 | 0.001 | 44 | 0.5627 | `[0.9991, 0.745, 0.5065, 0.0]` | ok |
| R64-D8 | 0.001 | 45 | 0.5937 | `[0.9994, 0.7538, 0.4926, 0.129]` | ok |
| R64-D16 | 0.001 | 42 | 0.5647 | `[0.9995, 0.6818, 0.4218, 0.1556]` | ok |
| R64-D16 | 0.001 | 43 | 0.4857 | `[0.9986, 0.4496, 0.4351, 0.0595]` | ok |
| R64-D16 | 0.001 | 44 | 0.5526 | `[0.9976, 0.7361, 0.4621, 0.0145]` | ok |
| R64-D16 | 0.001 | 45 | 0.3274 | `[0.9847, 0.3249, 0.0, 0.0]` | degenerate (criterion) |
| R64-D16 | 0.000333333 | 42 | 0.5678 | `[0.9995, 0.6598, 0.4453, 0.1667]` | ok |
| R64-D16 | 0.000333333 | 43 | 0.2453 | `[0.9371, 0.0117, 0.0299, 0.0023]` | degenerate (criterion) |
| R64-D16 | 0.000333333 | 44 | 0.3380 | `[0.9682, 0.0265, 0.3571, 0.0]` | dead foreground channel |
| R64-D16 | 0.000333333 | 45 | 0.4158 | `[0.9849, 0.6537, 0.0181, 0.0067]` | dead foreground channel |
| R64-D32 | 0.001 | 42 | 0.5053 | `[0.9996, 0.6081, 0.3886, 0.025]` | ok |
| R64-D32 | 0.001 | 43 | 0.4305 | `[0.9986, 0.223, 0.352, 0.1481]` | ok |
| R64-D32 | 0.001 | 44 | 0.3534 | `[0.9683, 0.4343, 0.0108, 0.0004]` | degenerate (criterion) |
| R64-D32 | 0.001 | 45 | 0.5264 | `[0.9996, 0.6343, 0.3147, 0.1569]` | ok |
| R64-D32 | 0.000333333 | 42 | 0.4913 | `[0.9996, 0.6364, 0.3091, 0.02]` | ok |
| R64-D32 | 0.000333333 | 43 | 0.2826 | `[0.9692, 0.0145, 0.1466, 0.0003]` | degenerate (criterion) |
| R64-D32 | 0.000333333 | 44 | 0.3301 | `[0.9684, 0.0151, 0.3366, 0.0002]` | degenerate (criterion) |
| R64-D32 | 0.000333333 | 45 | 0.4938 | `[0.9994, 0.6308, 0.3349, 0.0102]` | ok |

## What this does and does not support

Supported: under a single fixed training recipe and budget, the 1.0x stride-1 network trains
reliably at R64-D8 and unreliably at R64-D16 and R64-D32, and the instability is not fixed by
the lower of the two learning rates in the pre-registered grid.

**Not** supported: that the larger RD representations carry less semantic information. That
question is unanswered, because the comparison cannot be made from runs that did not converge.
The handoff's prohibition therefore still stands -- "D16/D32 are worse because their mIoU is
lower" may not be written.

## A confound that points at the likely cause

The foreground fraction of a frame falls from 0.22 % at D8 to 0.07 % at D32 (Validation split:
176/99/18 pixels of classes 1/2/3 at D8 against 223/127/23 at D32, in a frame four times larger).
Class imbalance is therefore not constant along the axis under study, and the observed failure --
one foreground channel lost, background purity degraded -- is what a loss that cannot hold the
minority classes at higher imbalance would look like. Testing that means changing the training
recipe, which the handoff requires to be a separate experiment, not an amendment to this one.

