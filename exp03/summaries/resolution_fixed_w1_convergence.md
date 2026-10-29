# Convergence / budget-LR arm selection: result and criterion failure

Protocol: `resolution_fixed_w1_CONVERGENCE_PROTOCOL.md` (frozen 2026-10-29 before any run below existed).
Data: `resolution_fixed_w1_convergence.json`. Driver: `exp03/resolution_convergence.py`.
Stage P ran 2026-10-29 04:48-05:43, 12 runs.

## Outcome under the frozen rule: STOP CONDITION

No arm was CONVERGED on both rescue runs, so no arm was locked, stages V and F did not run, and no
fresh seed was consumed.

    A1  lr 1e-3, 400 ep   D16 s45 CONVERGED (escape ep 211)   D32 s44 not converged   -> rejected
    A2  lr 2e-3, 200 ep   D16 s45 CONVERGED (escape ep 138)    D32 s44 not converged   -> rejected
    A3  lr 3e-3, 200 ep   D16 s45 not converged                D32 s44 not converged   -> rejected

## The pre-registered predictions

**Escape-time scaling: direction confirmed, magnitude wrong.** Seed 42, escape epoch by LR:

    D16   lr 1e-3 -> 19     lr 2e-3 -> 12     lr 3e-3 -> 10     (predicted 20 / 10 / 7)
    D32   lr 1e-3 -> 21     lr 2e-3 -> 12     lr 3e-3 ->  9     (predicted 21 / 11 / 7)

Monotone in the predicted direction 4/4, but escape scales as ~LR^-0.7, not LR^-1.0 (fitted
exponents 0.59, 0.66, 0.77, 0.81). The 1/LR estimate was taken from the downward arm only and was
too strong. Recorded as a partially failed prediction.

**The A1 window prediction hit.** A1 predicted D16 s45 would escape in epochs 200-400 or never. It
escaped at **epoch 211** and finished at val mIoU 0.5612 against seed 42's 0.5617 on the same arm --
indistinguishable. A seed that sat on the plateau for all 200 epochs of the registered recipe became
an ordinary seed when only the deadline moved. The budget was the binding constraint, as claimed.

**LR is non-monotonic in outcome.** A3 (3e-3) failed to rescue D16 s45 where A2 (2e-3) succeeded.
Raising the LR buys earlier escape and then introduces its own damage. Escaping sooner is not the
same as training better.

## The criterion failed out of sample, and was not retuned

The criterion still separates perfectly on its own terms -- across all 36 runs (24 original + 12
stage P), converged runs have background IoU >= 0.9985 and non-converged runs <= 0.9848, with
nothing between. It is a faithful measure of **background purity**.

What broke is its construct validity: background purity stopped predicting whether the model is
usable. Two runs the criterion rejects:

    D32 s44 @ 2e-3   med10 0.012085   val 0.4996   [0.9840, 0.6834, 0.3305, 0.0004]
    D32 s44 @ 3e-3   med10 0.012124   val 0.4916   [0.9840, 0.6413, 0.3409, 0.0004]

against the clean seed on the same arm:

    D32 s42 @ 2e-3   med10 0.000271   val 0.5009   [0.9996, 0.6691, 0.3349, 0.0000]

0.4996 versus 0.5009. The rejected run's foreground is healthy -- UAV 0.6834 against the clean run's
0.6691, class 2 0.3305 against 0.3349. Both exceed 0.4305, the lowest val mIoU of any run the
criterion accepted. Background is ~99.93 % of pixels, so a 1.5-point background IoU deficit dominates
the training loss while costing ~0.001 mIoU.

The threshold was not moved. Retuning a frozen criterion after seeing the arm it was built to judge
would discard the only thing pre-registration buys, and the same discipline was applied to the
previous criterion when two runs sat within 0.007 of flipping it.

## What stage P actually established

The original "degenerate" class contained **two distinct failures**, and the criterion conflated them
because in the 24 original runs they always co-occurred:

1. **Plateau non-escape / foreground death.** Training loss parked at ~0.0240, one or both foreground
   channels dead. This is budget- and LR-limited and stage P fixed it: D16 s45 escaped at epoch 211
   (A1) and epoch 138 (A2), reaching 0.5612 and 0.4805. D32 s44's foreground recovered under A2,
   class 1 0.4343 -> 0.6834 and class 2 0.0108 -> 0.3305.
2. **Background purity deficit.** Training loss parked at ~0.0120, background IoU ~0.984 against
   ~0.9996. Invariant to everything tried: present for D32 s44 at lr 1e-3/2e-3/3e-3 and at both 200
   and 400 epochs. It costs almost no mIoU.

So the honest answer to "can per-resolution LR/budget tuning solve this" is **yes for failure 1, no
for failure 2, and failure 2 does not appear to matter for accuracy.** Under A2 both former
degenerate seeds produce models level with the clean seed on the same arm: D16 s45 0.4805 and
D32 s44 0.4996 against seed 42's 0.5254 and 0.5009.

## What this does NOT establish

* No accuracy result. Every stage-P run uses seeds 42, 44 or 45. Seeds 44 and 45 are report seeds and
  the protocol designates stage P diagnostic-only. Fresh seeds 46/47/48 remain untouched and unused.
* Nothing about whether convergence failure increases with resolution. Still 4 seeds and still not
  significant.
* The handoff prohibition stands: "D16/D32 are worse because their mIoU is lower" may not be written.

## To proceed, a new pre-registration is required

Using A2 means selecting an arm the frozen rule rejected, so it cannot be done under this document.
A new pre-registration would have to declare, before any further run, a criterion on foreground
channel viability rather than background purity, and state that it was designed with stage P's
taxonomy in hand. Because no fresh seed has been spent, the headline comparison on seeds 46/47/48
would still be clean: arm selection would rest on diagnostic seeds that the protocol already excluded
from reporting.
