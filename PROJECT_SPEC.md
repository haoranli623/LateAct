# Frozen project specification

## Research question

Can a new external action that arrives after denoising of the current block has
started still be incorporated into that block, without resetting the latent?

For a three-evaluation sampler, define switch position `s` as old action for the
first `s` denoiser evaluations and new action thereafter. Thus `s=0` is the NEW
oracle and `s=3` is the OLD oracle. Initial block noise and the two scheduler
re-noising tensors are explicitly materialized once and reused bit-for-bit.

The primary intervention is universal-mode mouse yaw, in both directions:
`mouse_left -> mouse_right` and `mouse_right -> mouse_left`. Keyboard action is
zero and unchanged.

## Frozen execution order

1. Phase -1 static audit.
2. Engineering smoke: official scenes 0000 and 0001, all switch positions,
   both directions, and same-action negative control.
3. Only after the smoke passes, Gate 0: official scenes 0000 through 0007,
   both directions and all switch positions.
4. Stop. No training, repair, rollback, or foundation-model changes.

## Primary metric

Signed horizontal camera motion is the sum of robust median horizontal
Lucas-Kanade feature-track displacement from the final prefix frame through the
generated block, measured at 320x176. Dense Farneback median horizontal flow is
a predeclared evaluator-stability check.

For a direction with old score `y_old`, new score `y_new`, and switch score
`y_s`, the new-action fraction is:

`p_s = (y_s - y_old) / (y_new - y_old)`.

The endpoints must be exact (`p_0=1`, `p_3=0` by construction). Values are not
clipped, so overshoot and evaluator failures remain visible.

## Frozen Gate 0 verdict rule

A direction is evaluator-valid only if its two oracle videos differ, its oracle
motion gap exceeds ten times the same-action control range (with a `1e-6`
floor), at least 20 valid feature tracks contribute per transition in median,
and the feature-track and dense-flow oracle gaps have the same sign.

- **A. FREE-LATE-BINDING:** at least 12/16 valid direction-scene pairs have
  `p_2 >= 0.8`, and at least 12/16 are monotone within tolerance 0.10.
- **B. COMMITMENT-CURVE STRONG GO:** A is false; at least 12/16 are valid and
  monotone, the median `p_1 - p_2 >= 0.15`, and at least 12/16 have both
  `p_1` and `p_2` within `[-0.10, 1.10]`.
- **C. HARD-COMMITMENT CONDITIONAL GO:** A and B are false; at least 12/16 are
  valid, but at least 12/16 have `p_1 <= 0.2` (the current block is effectively
  committed after its first evaluation).
- **D. NO-GO:** endpoint coupling, negative controls, cache isolation, metric
  stability, or the criteria above fail.

The smoke is engineering-only and cannot change these thresholds or select the
eight Gate 0 scenes.

## Frozen Gate 1 protocol

Gate 0 is immutable at **B. COMMITMENT-CURVE STRONG GO**. Gate 1 tests arrival
after two OLD NFEs. Minimal rollback restores the exact boundary state entering
NFE2 (the re-noised latent plus current-block cache slices after OLD NFE1), then
executes NFE2 and NFE3 under NEW. Direct continuation executes only NFE3 under
NEW. Full restart resets to the original initial noisy latent and prefix-cache
indices and executes all three NFEs under NEW.

All eight Gate 0 scenes, both yaw directions, seeds, evaluator, and actions are
unchanged. The frozen strong-GO thresholds are those in `config/gate1.yaml` and
the user-approved Gate 1 specification. “No systematic visual/temporal
corruption” is operationalized before execution as: median future-frame SSIM
against NEW oracle at least 0.90, and temporal-consistency SSIM no more than
0.05 below NEW on at least 14/16 pairs. These are secondary safeguards and do
not replace the frozen signed-yaw response metric.

The early-arrival control requires no extra condition: Gate 1 minimal rollback
from the after-NFE1 checkpoint is exactly the zero-rollback direct trajectory
for an action arriving after NFE1. Its benefit is compared with the already
generated full-restart/NEW trajectory.

If strong GO passes, work stops after proposing—not running—a Gate 2 serving
policy.

## Frozen Gate 2 protocol

The project claim is now **Commitment-Aware Asynchronous Control for Interactive
Video World Models**. Gate 2 compares NEXT-BLOCK, DIRECT-LATE-BIND,
FULL-RESTART, and LATEACT without training.

Primary evaluation uses 32 fresh rollout contexts: official universal source
images 0008-0015, each with four fixed prefix-noise variants. Both mouse-yaw
directions are retained. These contexts are fixed before execution; invalid
ones are reported rather than replaced. Secondary keyboard strafe transfer uses
one variant of the same eight fresh source images only if primary Gate 2 is
positive.

Each direction receives 16 deterministic stratified-uniform arrival times over
the measured three-NFE OLD block interval (8 for secondary transfer). An arrival
is actionable at the next completed NFE boundary. Context write and VAE decode
are excluded from arrival support but included in action-to-pixel latency.
Arrivals during NFE1 bind at NFE2; arrivals during NFE2 bind at NFE3; arrivals
during NFE3 have no remaining current-block evaluation for DIRECT.

LATEACT binds directly after NFE1 and restores the after-NFE1 boundary for
arrivals after NFE2 or NFE3. The Gate 0/1 boundary is not refit. Success and
quality thresholds are frozen in `config/gate2.yaml`.

Gate 2 also validates an exact checkpoint optimization. Gate 1 stored the
entering-NFE2 latent and all current-block K/V slices. Static tracing shows that
NFE2 overwrites every current-block visual and action K/V slice before reading
it, while the immutable prefix and cross state are shared. Gate 2 therefore
stores the entering latent plus defensive cache indices, with no lossy
compression. Every optimized rollback must exactly match an independently
generated OLD-NFE1/NEW-NFE2-NFE3 trajectory.

## Frozen Gate 2 outcome

The 32-context mouse-yaw primary passed every frozen success condition. LATEACT
had median response 0.9999; all 676 late arrivals were within 0.10 of restart,
used fewer redone NFEs, and saved median 0.3069 s. All 348 early arrivals used
bit-exact direct binding with zero rollback. The visual/temporal and exact-state
audits passed. The exact optimized checkpoint is 339,360 bytes versus the Gate
1 implementation's 649,330,080 bytes.

The predeclared keyboard secondary did not transfer: median response was
0.7635, only 30.6% of late cases were within 0.10 of restart, and quality failed.
At the Gate 2 stage this motivated the action-dependent-boundary hypothesis.
Phase 3 subsequently rejects that interpretation while preserving these frozen
measurements: the response failure was evaluator mismatch and the quality
failure reproduced. No training or new method follows automatically.

## Frozen Phase 3 protocol

Phase 3 tests the mechanistic interpretation that denoising-time action
commitment can depend on the action family. Gate 0-2 results and the mouse
boundary remain immutable. Phase 3A reproduces every three-NFE switch position
for native universal-mode keyboard A/D on a new calibration split.

The keyboard primary response is not the mouse yaw score. It is the accumulated
horizontal displacement of the image center under a robust partial-affine fit
to forward/backward-consistent Lucas-Kanade tracks. Median track translation is
an independent oracle-sign check. The evaluator was engineered on already
existing Gate 2 oracle videos and frozen before Phase 3 calibration. Exact
validity, curve, quality, boundary, and A/B/C/D rules are in
`config/phase3a.yaml`.

Only official image `0016` remains unused, so ten independently seeded rollouts
from that image form calibration. This is fresh runtime context but limited
visual-source diversity and must be reported as such. A stable boundary must be
the latest switch with median response at least 0.90, bootstrap lower bound at
least 0.80, and at least 80% individual response at least 0.80; the following
switch must have median at most 0.75, median drop at least 0.15, and at least
75% of individual curves must bracket the boundary. Phase 3B is forbidden
unless Phase 3A returns A or B.

## Frozen Phase 3A outcome and Phase 3B confirmation

Phase 3A returned **B. SAME BOUNDARY**. Across 20/20 valid keyboard directions,
median response retention for switch positions 0/1/2/3 was
`1.0000/0.9957/0.1240/0.0000`. The latest safe switch is after NFE1, identical
to mouse; the following median collapses by 0.8717 and all 20 individual curves
bracket the frozen boundary. Quality and exact-state audits passed.

Phase 3B therefore cannot establish a different keyboard boundary. Its purpose
is to confirm on 24 disjoint stochastic contexts that the mouse-boundary and
action-specific policies are bit-exact, and to test keyboard serving with the
proper lateral-translation evaluator. The five policies, arrival distribution,
quality checks, and thresholds are frozen in `config/phase3b.yaml`. A positive
confirmation supports a shared boundary for these two action families and
diagnoses the Gate 2 keyboard negative as evaluator mismatch; it does not
support the proposed action-dependent-boundary claim.

## Frozen Phase 3 final outcome

Phase 3B confirmed keyboard response recovery but failed its frozen quality
safeguard. On 24 disjoint contexts and 48 valid directions, both equal-boundary
LateAct policies were bit-exact, had median response 1.0188, brought all 512
late arrivals within 0.10 of restart, and saved median 0.3105 s. Median future
SSIM to NEW was only 0.6726 against the frozen 0.90 threshold, although temporal
quality passed 48/48 and videos remained coherent.

The action-dependent-boundary hypothesis is rejected for these data: mouse yaw
and keyboard A/D share the after-NFE1 boundary. The correct keyboard metric
repairs the Gate 2 response diagnosis but not its quality failure. The main
claim remains mouse-only, and the project stops without training or new method.
