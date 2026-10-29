# Pre-registration: convergence criterion and budget/LR arm selection (fixed 1.0x stride-1 RD-resolution study)

Written 2026-10-29, BEFORE any run in the arm set below exists. Companion to
`resolution_fixed_w1_PROTOCOL.md` (the degeneracy pre-registration) and
`resolution_fixed_w1_stability.md` (the stop condition that made this necessary).

## 1. Why this document exists

The first pre-registration declared a *degeneracy* criterion on validation behaviour, and the lr
3.333e-4 sensitivity arm hit its stop condition: no LR in the registered grid `[1e-3, 1e-3/3]` gave
zero degenerate seeds at D16 or D32.

Reading the training logs afterwards showed the stop condition had a single mechanical cause, and
that the degeneracy criterion had been measuring it indirectly. Across all 24 existing fixed-1.0x
runs the training loss sits on a plateau at ~0.0240 and then either drops sharply to ~0.0004 or
never does. Final training loss is strictly bimodal:

    escaped the plateau    median-of-last-10 train loss  0.000248 .. 0.001788
    never escaped                                        0.011360 .. 0.047452

and the partition predicts the outcome perfectly, 24 runs out of 24:

    escaped      best val mIoU  0.4305 .. 0.5937
    never escaped               0.2453 .. 0.4158

with no overlap. Every run my earlier criterion called degenerate is a run that did not escape, and
the two 3.333e-4 runs that sat within 0.007 of flipping that criterion are both unambiguous
non-escapes by training loss. So "degenerate" was a validation-side shadow of a training-side event.

Escape time scales with the inverse of the learning rate. Paired on the same seed, cutting the LR by
3x delayed escape by 2.1x to 4.1x in 4 of 4 available pairs (D8 s42 15->32, D16 s42 16->41,
D32 s42 18->41, D32 s45 15->61, measured on validation loss < 0.01). The budget is 200 epochs and is
a hard deadline. That is the whole explanation of the stop condition: **the lower LR did not
destabilise training, it pushed escape past the deadline.** 6 of 8 D16+D32 runs escaped at lr 1e-3;
3 of 8 at lr 3.333e-4.

Post-escape epochs then matter, and more so at high resolution. D32 at lr 1e-3, by escape epoch:
ep 15 -> 0.5264, ep 18 -> 0.5053, ep 84 -> 0.4305, monotone decreasing. D8 is flat over the same
range (ep 15 -> 0.5729 ... ep 87 -> 0.5937): it is saturated with budget to spare. The 200-epoch
budget was calibrated where it has ~10x headroom and is marginal where it does not.

## 2. The criterion (frozen)

For a finished run, let `med10` be the MEDIAN of the last 10 logged `Train loss` values in
`train.log` (logged every 100 iterations; the median, not the last value, so one noisy minibatch
cannot decide it).

    CONVERGED(run)  iff  med10 < 0.005

`ESCAPE_EPOCH(run)` = the first epoch carrying a logged train loss < 0.005, or NEVER.

Threshold provenance, stated plainly: 0.005 was chosen by looking at the 24 existing runs, and sits
in the empty gap between the two modes -- 2.8x above the highest converged run and 6.4x below the
lowest non-converged one. It is calibrated on data I had already seen. What makes it usable is that
it is frozen here before any run in the arm set exists, and that it is computed from the TRAINING
loss only. No validation or test quantity enters it.

That last property is the point. Acting on this criterion -- extending, re-running, or reporting a
run as non-converged -- is not conditioning on the reported metric. It is categorically different
from the healthy-only validation averaging rejected earlier.

## 3. Arm set (frozen)

    A1  lr 1e-3, 400 epochs      LR unchanged from the exp03 pick; budget doubled
    A2  lr 2e-3, 200 epochs      budget unchanged; LR one step up exp03's own LR_GRID
    A3  lr 3e-3, 200 epochs      budget unchanged; LR at the top of exp03's own LR_GRID

`pipeline.py` has carried `LR_GRID = [3e-3, 2e-3, 1e-3, 3e-4]` since exp03 began, so 2e-3 and 3e-3
are in-grid values that the Exp-B sub-grid `[lr_A, lr_A/3]` happened to search downward from. They
are not new free parameters invented for this problem.

No other arm will be added. If none of A1-A3 satisfies section 5, that is a stop condition and gets
reported as one, exactly as the LR stop condition was.

## 4. Stages

Seed 42 CANNOT serve as the probe: it converges in every existing arm at every resolution, so it has
no power to discriminate. The probe therefore runs the seeds that actually failed.

* **Stage P (diagnostic).** Each of A1-A3 on:
  - the two known non-converged runs at the registered recipe, `D16 s45` and `D32 s44` -- rescue test;
  - `D16 s42` and `D32 s42` -- escape-epoch scaling, to test the mechanism against section 1's
    prediction.
  12 runs, one pool.
* **Stage L (lock).** Apply the selection rule in section 5. Produces one arm, or a stop condition.
* **Stage V (convergence rate).** The locked arm on D8, D16, D32 x seeds 42-45, filling in whatever
  stage P did not already run. Gives a 4-seed convergence rate per resolution under one recipe.
* **Stage F (final, FP32).** The locked arm on FRESH seeds 46/47/48 at all three resolutions.
* **Stage Q / Stage X.** QAT LR search on seed 42 from the locked parent, QAT finals on 46/47/48,
  int8 exports, and the `paper_resolution_fixed_w1_model.{md,csv}` deliverables, by the procedure
  already registered in `resolution_fixed_w1_PROTOCOL.md`.

The recipe is IDENTICAL at all three resolutions in stages V, F, Q. If the locked arm is not
(1e-3, 200), the existing D8 seeds 46/47/48 runs do not match it and are re-run under it; they are
not mixed into the final table.

## 5. Selection rule (frozen, a priori ordered)

Preference order is fixed now, by minimality of deviation from the registered recipe, NOT by result:

    A1 (1e-3, 400)  >  A2 (2e-3, 200)  >  A3 (3e-3, 200)

A1 is first because it leaves the selected learning rate alone and changes only the budget, which
section 1 identifies as the binding constraint.

**Rule.** Lock the FIRST arm in that order which is CONVERGED on both stage-P rescue runs
(`D16 s45`, `D32 s44`). If no arm converges on both, stop and report; do not add arms, do not retune
the threshold, do not proceed to fresh seeds.

**Rule.** If the locked arm produces any non-converged run in stage V or stage F, it is reported as
a non-convergence with the rate, and is NOT dropped from the table. A resolution whose final seed
set contains a non-converged run does not get a headline mIoU; it gets a convergence rate and a
statement that its accuracy under this budget is not established.

## 6. Predictions (falsifiable, recorded before the runs)

From escape time ~ 1/LR, with the lr 1e-3 escape epochs 20 (D16 s42) and 21 (D32 s42) measured on
train loss < 0.005:

    A2 (2e-3)   D16 s42 ~ 10,  D32 s42 ~ 11
    A3 (3e-3)   D16 s42 ~  7,  D32 s42 ~  7
    A1 (1e-3, 400 ep)  D16 s42 ~ 20, D32 s42 ~ 21   (unchanged; only the deadline moves)

For the rescue runs, A1 predicts escape somewhere in epochs 200-400 if the plateau is escapable at
lr 1e-3 at all, and NEVER otherwise -- A1 is the arm that can fail cleanly. A2/A3 predict escape
inside 200 epochs.

If escape epochs do not move roughly inversely with LR, the mechanism in section 1 is wrong, and
that is recorded as a failed prediction whatever the convergence outcome.

## 7. What each stage may claim

* Stage P may claim: whether the plateau escape is budget-limited and LR-rate-limited. It may NOT
  claim any accuracy result -- it reuses seeds 44/45, which are report seeds.
* Stage V may claim a convergence rate per resolution under a fixed recipe, on 4 seeds. With 4 seeds
  it may NOT claim that convergence failure increases with resolution. For the record, the existing
  evidence for that is not significant: 0/8 non-convergences at D8 against 2/8 at D16+D32 at
  lr 1e-3 is Fisher's exact p ~ 0.47.
* Stage F is the only stage whose mIoU may be quoted as the headline, because its seeds are
  untouched by every selection above.
* The prohibition from the handoff stands regardless of outcome: "D16/D32 are worse because their
  mIoU is lower" may not be written unless the fixed-1.0x training supports it and seed variance
  does too, on converged runs only.
