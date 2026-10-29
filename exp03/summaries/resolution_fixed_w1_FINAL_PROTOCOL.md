# DRAFT pre-registration: locked recipe and final comparison (fixed 1.0x stride-1 RD resolution)

Status: APPROVED 2026-10-29, frozen before stage F ran. Third and final pre-registration, after `resolution_fixed_w1_PROTOCOL.md` (degeneracy criterion, stopped) and
`resolution_fixed_w1_CONVERGENCE_PROTOCOL.md` (budget/LR arms, stopped).

## 1. What the two stopped protocols established

The first protocol's "degenerate seed" class held two separable failures. They always co-occurred in
the original 24 runs at lr <= 1e-3, which is why one criterion appeared to capture both.

**Failure 1, plateau non-escape / foreground death.** Training loss parked at ~0.0240, one or both
foreground channels dead. Budget- and LR-limited: escape time scales as ~LR^-0.7 and the 200-epoch
budget was a hard deadline calibrated where it had ~10x headroom (D8) rather than where it was
marginal. Demonstrated by direct intervention -- D16 s45, stuck for all 200 epochs at lr 1e-3,
escaped at **epoch 211** when only the deadline moved, finishing at 0.5612 against seed 42's 0.5617.

**Failure 2, background-purity deficit.** Training loss parked at ~0.0121, background IoU ~0.984
against ~0.9996. The A2 probe (`resolution_fixed_w1_a2_probe.json`) pinned it down: it is **a property
of seed 44, flat in resolution**.

    arm A2 (lr 2e-3, 200 ep), seed 44      D8  bg 0.9836  med10 0.012104
                                           D16 bg 0.9838  med10 0.012071
                                           D32 bg 0.9840  med10 0.012085
    all other seeds, all resolutions            bg 0.9973-0.9997  med10 0.000271-0.001053

Three resolutions, three near-identical values, an 11x gap to every other run. Every resolution is
3/4 converged with the same failing seed. It costs little accuracy: seed 44 reaches val 0.4969 /
0.4435 / 0.4996 against converged means of 0.5599 / 0.5173 / 0.5179.

**Under arm A2 there are zero foreground deaths: 0 of 12 runs, 0/4 at every resolution.**

## 2. The correction this forces to the earlier reading

"D8 trains reliably and D16/D32 do not" was an artifact of the budget. Under one adequate recipe all
three resolutions behave identically -- same convergence count, same failing seed, and the one
remaining failure appears at D8 too, which it never did at lr 1e-3. No statement in this study may
claim that training reliability depends on resolution. The handoff prohibition also stands: "D16/D32
are worse because their mIoU is lower" may not be written.

## 3. Locked recipe

    KuRALSNetNPUSeg, width_mult 1.0, stem_stride 1, encoder_depth 1, bottleneck kernel 3,
    dropout 0.1, eltwise_act linear, 23 layers, CUTPOINT 3
    FP32: lr 2e-3, 200 epochs, best-validation checkpoint          <- arm A2
    identical at R64-D8, R64-D16 and R64-D32

2e-3 is in `pipeline.py`'s own `LR_GRID = [3e-3, 2e-3, 1e-3, 3e-4]`, carried since exp03 began.
A2 is chosen because it is the only arm tried that removes failure 1 at every resolution. A1
(1e-3, 400 ep) also removes it at D16 but left D32 s44's class 2 at 0.0473; A3 (3e-3) reintroduced
foreground death at D16. LR is non-monotonic in outcome, so the middle value is not a compromise.

**Selection provenance, stated plainly.** A2 was selected using dev seeds 42-45, which both earlier
protocols designated diagnostic-only and barred from reporting. The frozen criterion in protocol 2
rejected A2, on failure 2; that rejection is recorded in
`resolution_fixed_w1_convergence.md` and is not being reinterpreted here. This protocol does not
re-use that criterion as a gate.

## 4. No exclusion rule -- the final table reports every run

This is the central design choice and it is what makes the comparison clean. There is **no
convergence gate, no degeneracy criterion and no exclusion of any kind** on the final seeds. Every
run that is launched appears in the table with its val/test mIoU, its per-class IoU and its `med10`.

That removes outcome-conditioning entirely rather than arguing about it. Nothing is dropped after
being seen, so there is no selection to defend. `med10` is reported as a diagnostic column, never as
a filter.

If a fresh-seed run shows failure 1 or failure 2, it stays in the mean and the text says so. The risk
is quantified: 0 of 12 foreground deaths and 3 of 12 background deficits under A2 on dev seeds, all
three the same seed.

## 5. Final comparison, and the seed-set reconciliation

The handoff specifies **seeds 43/44/45 report, 42 tunes**. The fresh set 46/47/48 comes from the
review instruction that arm selection must not be made on the seeds that are reported. Both are
honoured, and the conflict is resolved by reporting both rather than choosing:

* **PRIMARY / headline: seeds 46, 47, 48.** Untouched by every selection in all three protocols.
  This is the table the paper quotes.
* **SECONDARY / consistency check: seeds 43, 44, 45** -- the handoff's declared report set. Reported
  beside the primary table, with the disclosure that arm A2 was selected using seeds 42-45, so these
  three cannot be a clean headline. If the two tables agree, arm selection did not bias the result;
  if they disagree, that is reported as the finding.

Both sets run under the identical locked recipe, so they are directly comparable. The seed-42 runs
remain tuning-only and are excluded from both tables.

Best-validation checkpoint selection; **test read once** at that checkpoint. mIoU is the 4-class mean
including background. Comparison is paired by seed across resolutions.

**Disclosure on the fresh seeds.** 46/47/48 are untouched by every selection above, but they are not
wholly unobserved: three runs `B_..._R64-D8_lr0.001_s46/47/48` exist from the earlier aborted stage,
at D8 only and under the old recipe (lr 1e-3, 200 ep), and all three converged (med10 0.000343 /
0.000387 / 0.000481, val 0.5592 / 0.5738 / 0.5703). They were launched before any of this analysis and
were not chosen for being good. They do not match the locked recipe, are re-run under it, and the old
runs are not mixed into either table.

**Disclosure on seed 44.** Seed 44 carries the background-purity deficit at all three resolutions and
is in the handoff's report set. Under the no-exclusion rule of section 4 it stays in the secondary
table with its `med10` column showing the deficit. It is not in the primary set.

Class 3 is excluded from interpretation a priori, as in protocol 1: 18-23 px in the Validation split.
It still enters the 4-class mIoU as the handoff requires.

**Stale artefact.** `exp03/model_release_d8_w1.0_s1` currently holds a package built from
`Q_..._R64-D8_qlr1e-05_s45`, a dev seed under the superseded lr 1e-3 recipe. It is moved aside, not
reused, so it cannot silently enter the deliverable.

## 6. Stages

    F   FP32, locked recipe, seeds 46/47/48 x {D8, D16, D32}            9 runs
        (seeds 42-45 at this recipe already exist from the A2 probe and are reused, not retrained)
    Q   QAT LR search on seed 42 from the locked parent, per resolution, then QAT on 46/47/48
        (primary) and on 43/44/45 (secondary), each paired to its own seed's FP32 checkpoint
        (QAT_LRS, 100 epochs, freeze 24000)
    X   int8 export + exact-integer replay, model_release_d{8,16,32}_w1.0_s1
    D   deliverables paper_resolution_fixed_w1_model.{md,csv}, 24-column schema per the handoff

Minimum required activation-bank capacity is computed and reported per resolution, as the handoff
requires. `gen_cfg.py --buf-size 0x8000` is NOT a rejection rule here: bank size is an output of this
study, not a constraint on it.

## 7. What may be claimed

Only stage F and the QAT runs built on it may be quoted as accuracy. Permitted: a paired comparison of
mIoU across RD resolution under one fixed network, one fixed recipe and one fixed budget, with every
run reported and the failure-type rates stated. Not permitted: any claim that training reliability
depends on resolution, and the handoff's prohibited sentence.
