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

