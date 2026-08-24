# LateAct Phase 3 — keyboard commitment mechanism and serving confirmation

## Final conclusion

**Phase 3A: B. SAME BOUNDARY.**

**Phase 3B: CONFIRMATION FAILED on the frozen pixel-fidelity safeguard.**

The proposed hypothesis that keyboard has a different denoising-time
commitment boundary is not supported. With a keyboard-specific lateral-motion
evaluator, keyboard A/D has a clean curve: median NEW-action retention is
0.9957 after one OLD NFE and 0.1240 after two. Its latest safe switch is after
NFE1, exactly the frozen mouse boundary.

On 24 disjoint confirmation rollouts, both identically bounded LateAct policies
recover keyboard lateral response (median 1.0188), remain near restart on
512/512 late arrivals, save median 0.3105 s, and are bit-exact to each other.
But the precommitted median future-SSIM-to-NEW threshold fails badly: 0.6726
versus the required 0.90. Temporal consistency passes 48/48 and inspected
videos are coherent, so this is exact trajectory divergence rather than
generic collapse; it nevertheless remains a frozen gate failure.

The Gate 2 keyboard result therefore had two causes:

1. its response failure was an evaluator mismatch—the mouse yaw score was not
   an appropriate primary A/D locomotion metric;
2. its quality failure is real and reproduces with the correct keyboard metric.

The paper claim should remain **mouse-yaw only**. Keyboard can be reported as a
mechanistic same-boundary result and an operational quality limitation, but not
as successful action-specific LateAct expansion. No training or additional
action search follows.

## Fresh splits and scope limitation

The official Matrix-Game universal assets contain images `0000`-`0016` only.
Gates 0-2 had already used `0000`-`0015`; `0016` was the sole unused official
source. Phase 3 therefore uses fresh stochastic runtime contexts from `0016`,
not broad new image-level diversity. This limitation was identified and frozen
before calibration.

- Calibration: `0016_c00`-`0016_c09`, prefix variants 0-9, prefix/future seeds
  `120000+variant` and `130000+variant`.
- Confirmation: `0016_h10`-`0016_h33`, variants 10-33, prefix/current/next
  seeds `140000+variant`, `150000+variant`, and `160000+variant`.

Calibration and confirmation seeds do not overlap. All contexts and both
`keyboard_left -> keyboard_right` and reverse directions were retained; none
were replaced.

## Official keyboard semantics and frozen evaluator

Official universal mode maps A/D to keyboard one-hot indices 2/3 and describes
them as left/right movement. Mouse conditioning was neutral and unchanged.

The primary metric is robust signed lateral scene translation:

1. decode at 320x176 from the final prefix frame through the future block;
2. track up to 800 Shi-Tomasi features with pyramidal Lucas-Kanade;
3. reject tracks with forward/backward error >=1.5 pixels;
4. fit a partial affine transform with 2-pixel RANSAC tolerance;
5. accumulate the fitted horizontal displacement of the image center.

Center displacement avoids coordinate-origin artifacts from small fitted
rotations. The accumulated median LK translation is an independent oracle-sign
check, not the primary score. Before Phase 3 calibration, this evaluator was
tested on the already-existing Gate 2 keyboard oracle videos: all 16 directions
had gaps above 8 pixels, primary/stability signs agreed 16/16, and minimum
median RANSAC inlier support was 255.5 tracks. Exact thresholds were then frozen
in `config/phase3a.yaml`.

A Phase 3 direction was valid only if oracle latents differed, absolute affine
gap exceeded 2 pixels, median oracle inlier support was at least 20, affine and
median-LK gap signs agreed, official A/D direction sign was correct, only
`keyboard_cond` differed, noise matched, and all historical cache audits passed.

## Phase 3A complete commitment curve

Phase 3A generated OLD/NEW oracles and every switch position for ten contexts:
OLD for `s` NFEs and NEW for the remaining `3-s`. All 60 runs passed condition,
noise, cache-history, cross-state, and decode audits. All 20 directions were
evaluator-valid. Absolute oracle gap was 61.26-68.14 pixels (median 64.94), and
median oracle inlier support was 227.0-260.5 tracks.

`R_keyboard(s) = (y_s - y_old) / (y_new - y_old)` was not clipped.

| OLD NFEs `s` | Mean retention | Median | Bootstrap median 95% CI | Range |
|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | [1.0000, 1.0000] | 1.0000-1.0000 |
| 1 | 1.0054 | **0.9957** | **[0.9827, 1.0203]** | 0.9446-1.1049 |
| 2 | 0.1197 | **0.1240** | **[0.0401, 0.1711]** | -0.0044-0.3144 |
| 3 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000-0.0000 |

Direction medians:

| Direction | `s=0` | `s=1` | `s=2` | `s=3` |
|---|---:|---:|---:|---:|
| left -> right | 1.0000 | 0.9870 | 0.1917 | 0.0000 |
| right -> left | 1.0000 | 1.0103 | 0.0332 | 0.0000 |

Raw affine signed translations are shown below. `L` and `R` are oracle scores;
`LR1/LR2` and `RL1/RL2` are the two switch outputs.

| Context | L | R | LR1 | LR2 | RL1 | RL2 |
|---|---:|---:|---:|---:|---:|---:|
| c00 | 29.731 | -31.532 | -33.490 | 14.100 | 32.870 | -29.013 |
| c01 | 30.018 | -38.122 | -34.350 | 19.115 | 31.218 | -28.854 |
| c02 | 31.951 | -34.518 | -33.134 | 11.056 | 30.865 | -33.757 |
| c03 | 33.055 | -32.480 | -33.835 | 25.320 | 31.798 | -31.603 |
| c04 | 32.679 | -32.232 | -33.725 | 16.363 | 32.876 | -29.689 |
| c05 | 30.801 | -32.741 | -31.899 | 22.180 | 32.072 | -32.331 |
| c06 | 30.263 | -34.381 | -32.909 | 21.858 | 29.510 | -30.133 |
| c07 | 32.796 | -32.758 | -34.040 | 20.853 | 31.088 | -33.047 |
| c08 | 30.491 | -34.471 | -33.284 | 16.361 | 37.308 | -28.527 |
| c09 | 31.374 | -32.052 | -31.238 | 18.610 | 33.430 | -30.326 |

Nineteen of 20 curves were monotone within 0.10 and bounded in
`[-0.10,1.10]`. The retained exception was `c08` right-to-left with `R(1)=1.1049`.
All 40 switch-direction quality checks passed. Temporal difference from the
worse oracle ranged -0.0402 to +0.0627, and boundary-jump excess ranged -1.17
to +4.40 pixels.

### Frozen boundary decision

The precommitted latest-safe criterion required median >=0.90, bootstrap lower
bound >=0.80, and at least 80% individual retention >=0.80. The following
position had to have median <=0.75, median drop >=0.15, and at least 75% of
individual curves had to bracket the boundary.

- `s=1` safe: median 0.9957, lower CI 0.9827, individual near-oracle 20/20;
- `s=2` unsafe: median 0.1240, individual near-oracle 0/20;
- median boundary drop: 0.8717;
- individual bracketing: 20/20.

Keyboard's latest safe switch is after NFE1. Mouse's frozen latest safe switch
is also after NFE1. Phase 3A therefore returns **B. SAME BOUNDARY**.

## Phase 3B frozen serving confirmation

Phase 3B was permitted by verdict B and used 24 disjoint confirmation contexts,
48 directions, and 16 deterministic stratified-uniform arrivals per direction
(768 total; seed family `20260825`). Arrival was actionable at the next
completed NFE boundary. Prefix, three explicit future noise tensors, sampler,
weights, all non-action state, action pair, and arrival timestamp were paired.

Policies were NEXT-BLOCK, DIRECT-LATE-BIND, FULL-RESTART, MOUSE-BOUNDARY
LATEACT, and ACTION-SPECIFIC LATEACT. Because both boundaries are after NFE1,
the last two are the same algorithm and were required to be bit-exact. They
were identical on 768/768 arrival records; response difference was exactly
zero, and no case improved by 0.10.

All 48 directions were evaluator-valid. All optimized checkpoint replays were
bit-exact to independently generated OLD-NFE1/NEW-NFE2-NFE3 trajectories.

| Policy | Response mean | Response median | Latency mean | Latency median |
|---|---:|---:|---:|---:|
| NEXT-BLOCK, current | 0.0000 | 0.0000 | 5.1609 s | 5.1412 s |
| DIRECT-LATE-BIND | 0.3778 | 0.0985 | 3.0259 s | 2.2728 s |
| FULL-RESTART | 1.0000 | 1.0000 | 2.6033 s | 2.5806 s |
| MOUSE-BOUNDARY LATEACT | 1.0207 | 1.0188 | 2.2928 s | 2.2790 s |
| ACTION-SPECIFIC LATEACT | 1.0207 | 1.0188 | 2.2928 s | 2.2790 s |

LateAct direction medians were 1.0298 left-to-right and 1.0140 right-to-left;
all 48 direction responses were within 0.10 of restart. Direct-late medians
were 0.1702 and 0.0451.

For 512 late arrivals:

- within 0.10 of restart: 512/512;
- fewer redone NFEs than restart: 512/512;
- latency saved: mean 0.3121 s, median 0.3105 s, range 0.2650-0.3630 s;
- mean latency-saving fraction: 12.00%.

All 256 early arrivals used direct binding with zero restore, discard, or
redone NFE. Mean/median LateAct discarded and redone NFEs over all arrivals
were 0.9961/1 versus restart's 1.9961/2.

The paired direction-level LateAct-minus-DIRECT response improvement was mean
0.9098, median 0.9088, bootstrap mean 95% CI `[0.8839,0.9357]`, and one-sided
Wilcoxon `p=3.55e-15` (`n=48`). This supports the runtime response mechanism,
but not a different keyboard boundary.

## Frozen quality failure

Phase 3B required median future-frame SSIM to NEW >=0.90 and temporal deficit
no worse than 0.05 on at least 90% of directions.

- future SSIM to NEW: mean 0.7247, median **0.6726**, range 0.5989-0.9139;
- left-to-right/right-to-left median SSIM: 0.6593/0.6807;
- temporal difference from NEW: median +0.00048, range -0.00764 to +0.01086;
- temporal pass: 48/48;
- all decodes finite and nonconstant.

The first confirmation montage shows a coherent LateAct frame with correct
lateral response, not visible generic corruption. Retaining OLD NFE1 changes
the exact generated scene trajectory enough to fail pixel SSIM. The threshold
was frozen before confirmation and is not weakened: **Phase 3B fails**.

## Runtime and memory

- Phase 3A: 60 runs, 73.35 generation GPU-seconds plus 68.91 decode
  GPU-seconds; mean generation 1.222 s/run.
- Phase 3B: 48 audit bundles, 296.41 generation GPU-seconds plus 417.69 decode
  GPU-seconds; mean total measured work 14.88 s/bundle.
- Hardware: 2 x NVIDIA GeForce RTX 3090.
- Phase 3B peak allocated VRAM: 9,565,199,360 bytes (8.908 GiB).
- Exact rollback checkpoint: 339,360 bytes; logical minimum 337,928 bytes;
  host checkpoint 0 bytes.

## Artifacts

Calibration:

- `/mnt/NAS/data/hl5757/generated_artifacts/lateact/phase3/calibration/phase3a_summary.json`
- `/mnt/NAS/data/hl5757/generated_artifacts/lateact/phase3/calibration/phase3a_shard_0.json`
- `/mnt/NAS/data/hl5757/generated_artifacts/lateact/phase3/calibration/phase3a_shard_1.json`
- `/mnt/NAS/data/hl5757/generated_artifacts/lateact/phase3/calibration/phase3a_keyboard_curve_montage.png`

Confirmation:

- `/mnt/NAS/data/hl5757/generated_artifacts/lateact/phase3/confirmation/phase3b_summary.json`
- `/mnt/NAS/data/hl5757/generated_artifacts/lateact/phase3/confirmation/phase3b_shard_0.json`
- `/mnt/NAS/data/hl5757/generated_artifacts/lateact/phase3/confirmation/phase3b_shard_1.json`
- `/mnt/NAS/data/hl5757/generated_artifacts/lateact/phase3/confirmation/phase3b_keyboard_serving_montage.png`

The Phase 3 artifact tree contains 348 videos, 358 latent/prefix tensors, all
per-run audits and all 768 paired arrival rows (612 MiB at report time).

## Recommendation and stop

Action-dependent commitment is **not supported**: keyboard and mouse share the
same measured NFE1 boundary. The keyboard response mechanism reproduces under
the correct evaluator, but the frozen confirmation quality criterion fails and
visual-source diversity is limited to the only unused official image.

Keep the main paper claim mouse-only. Report keyboard transparently as evidence
that metric/action alignment matters, that the same commitment curve can exist
for A/D, and that response recovery need not imply oracle-level pixel
trajectory preservation. Do not begin second-model replication, training,
learned scheduling, adaptive per-scene commitment, or another action search.

Result commit: pending.
