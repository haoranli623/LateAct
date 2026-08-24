# LateAct Gate 2 — asynchronous serving evaluation

## Verdict

**STRONG PROJECT RESULT for mouse-yaw control, with action-dependent transfer.**

The frozen primary gate passes every criterion on 32 fresh contexts, 64 action
directions, and 1,024 paired arrivals. LATEACT preserves a median 0.9999
NEW-action response, is within 0.10 of FULL-RESTART on 676/676 late arrivals,
and saves a median 0.3069 s (11.9% by the frozen mean-fraction test) while
redoing one fewer NFE than restart. Early arrivals use direct binding with
exactly zero rollback. The visual/temporal safeguard and all checkpoint-state
audits pass.

The predeclared keyboard secondary does **not** transfer cleanly. Its median
response is 0.7635, only 30.6% of late cases are within 0.10 of restart, and
the quality safeguard fails. The result therefore supports commitment-aware
asynchronous control for the measured mouse-yaw action family, not a universal
or action-agnostic commitment boundary.

No training, weight edit, learned predictor, lossy checkpoint compression, or
boundary refit was performed. Work stops at Gate 2.

## Frozen evaluation

The primary set was fixed before execution as official universal images
`0008`-`0015`, each with four deterministic prefix-noise variants `v0`-`v3`.
All 32 contexts and both `mouse_left -> mouse_right` and
`mouse_right -> mouse_left` directions were retained. All 64 directions were
evaluator-valid; there were no replacements.

After the positive primary result, the frozen secondary used the same eight
images with variant `v0` and both `keyboard_left <-> keyboard_right`
directions. All eight contexts and 16 directions were retained and valid.

For every matched comparison, prefix, initial current-block latent, the two
explicit re-noising tensors, next-block noise, weights, sampler, historical
visual/action state, non-action conditioning, action pair, and arrival time
were identical. The primary used 16 deterministic stratified-uniform arrivals
per direction and the secondary used eight, seed `20260824`. Arrival support
was the measured wall interval of the three OLD NFEs. An arrival became
actionable at the next completed NFE boundary. Context writes and VAE decoding
were excluded from arrival support but included in action-to-pixel latency.

Arrival classes in the primary were 348 during NFE1, 338 during NFE2, and 338
during NFE3. The secondary counts were 43, 45, and 40. The same schedules were
used for every policy.

## Serving policies

- **NEXT-BLOCK:** finish the current OLD block and generate the next block
  under NEW.
- **DIRECT-LATE-BIND:** use NEW at the next available NFE; an arrival during
  NFE3 cannot affect the current block.
- **FULL-RESTART:** discard current progress and execute all three current-block
  NFEs under NEW.
- **LATEACT:** arrival during NFE1 binds NEW for NFE2-NFE3 with no restore;
  arrival during NFE2 or NFE3 restores the entering-NFE2 checkpoint and replays
  NFE2-NFE3 under NEW.

This is exactly the Gate 0/1 commitment boundary; it was not estimated again
on Gate 2 scenes.

## Exact cache hook and rollback implementation

Matrix-Game routes causal self-attention K/V insertion through
`branchsafe.cache_ops.append_and_read` at
`Matrix-Game-2/wan/modules/causal_model.py:163`. The hook receives the live
cache plus `roped_key`, value, `current_start`, sink-token count, and maximum
attention size immediately before attention reads the returned K/V. Gate 2
uses the already-audited hook; LateAct makes no additional foundation-model
edit.

The Gate 1 checkpoint copied the entering-NFE2 BF16 latent, visual current K/V
slices in all 30 layers, mouse current K/V slices in active layers 0-14,
keyboard current K/V slices in active layers 0-14, and all cache indices. Static
and runtime tracing showed that NFE2 overwrites every current-block visual and
action K/V position before it can be read. Historical prefix K/V and
cross-attention K/V are immutable and remain live by reference.

Gate 2 therefore stores only:

| Exact rollback payload | Bytes |
|---|---:|
| entering-NFE2 BF16 latent `[1,16,3,44,80]` | 337,920 |
| defensive visual/mouse/keyboard cache indices | 1,440 |
| **implemented exact checkpoint** | **339,360** |
| mathematical minimum: latent plus step | **337,928** |

The previous Gate 1 implementation was 649,330,080 bytes. The new exact
representation is 1,913x smaller; it removes redundant overwritten K/V
payloads, not numerical information required by rollback. Scheduler state is
stateless for these fixed tensors, RNG does not advance because noise is
materialized before branching, and no host checkpoint is used.

Every one of the 80 scene/direction executions independently generated
OLD-NFE1/NEW-NFE2-NFE3 and required bit-exact equality with optimized rollback.
Historical-prefix, cross-attention, scheduler, and CPU/CUDA RNG audits all
passed. Median checkpoint save and restore times in the primary were 1.456 ms
and 0.878 ms, respectively.

## Primary mouse-yaw result

Response is the frozen signed Lucas-Kanade yaw response normalized to OLD=0
and NEW/FULL-RESTART=1. Values are not clipped.

| Policy | Response mean | Response median | Latency mean | Latency median (p05-p95) |
|---|---:|---:|---:|---:|
| NEXT-BLOCK, current block | 0.0000 | 0.0000 | 5.1770 s | 5.1554 (4.5564-5.9151) s |
| DIRECT-LATE-BIND | 0.3987 | 0.1605 | 3.0405 s | 2.2925 (1.8108-5.2609) s |
| FULL-RESTART | 1.0000 | 1.0000 | 2.6127 s | 2.5900 (2.3663-2.9269) s |
| **LATEACT** | **0.9995** | **0.9999** | **2.3022 s** | **2.2843 (2.0666-2.5993) s** |

The median LATEACT responses were 0.9964 left-to-right and 1.0013
right-to-left. NEXT-BLOCK eventually reached its NEW next-block endpoint in
all cases, but its current-block response was zero and its action-to-pixel
latency included finishing OLD plus generating/decoding the next block.

For the 676 late arrivals:

- within 0.10 of FULL-RESTART: **676/676 (100%)**;
- fewer redone NFEs than FULL-RESTART: **676/676 (100%)**;
- latency saved versus restart: mean 0.3117 s, median 0.3069 s, range
  0.2708-0.3540 s;
- mean latency-saving fraction: **11.93%**;
- near-restart response with lower latency: **100%**.

For all 348 early arrivals, LATEACT and direct binding were exactly the same
trajectory, checkpoint use was false, and redone/discarded NFE counts were
zero. All were within 0.10 of restart (median absolute gap 0.0049; maximum
0.0221) while saving a median 0.3055 s.

Across all arrivals, LATEACT strictly Pareto-dominated DIRECT-LATE-BIND in
33.0% of cases under the frozen response-at-least, latency-no-greater,
post-NFE-no-greater definition. Early cases are equal rather than strict, and
some NFE2 arrivals exchange one replayed NFE for response, so the strict
fraction is not expected to be universal. LATEACT was near restart with lower
latency in 100% of late cases.

### NFE accounting

| Policy | Post-arrival NFE mean (median) | Discarded mean (median) | Redone mean (median) |
|---|---:|---:|---:|
| NEXT-BLOCK | 4.010 (4) | 0 (0) | 0 (0) |
| DIRECT-LATE-BIND | 2.000 (2) | 0 (0) | 0 (0) |
| FULL-RESTART | 3.000 (3) | 1.990 (2) | 1.990 (2) |
| **LATEACT** | **2.000 (2)** | **0.990 (1)** | **0.990 (1)** |

The FULL-RESTART and LATEACT discarded/redone ranges were 1-3 and 0-2,
respectively; LATEACT has zero rollback for early arrivals.

### Paired statistics

Using the independent scene/direction as the paired unit (`n=64`), the late
LATEACT-minus-DIRECT response improvement was mean 0.8159, median 0.8389, with
a scene/direction bootstrap mean 95% CI `[0.7690, 0.8585]` and one-sided paired
Wilcoxon `p=1.76e-12`. Direction-clustered latency savings versus restart had
mean 0.3118 s, median 0.3073 s, bootstrap 95% CI `[0.3071, 0.3166]`, and
one-sided paired Wilcoxon `p=1.76e-12`. Arrival-level rows remain available for
all 1,024 matched cases in `mouse_summary.json`.

## Visual and temporal quality

The frozen primary safeguard passes:

- LATEACT-vs-NEW future SSIM: mean 0.9157, median **0.9223**, p05-p95
  0.8397-0.9609;
- temporal-consistency difference from NEW: median +0.00169, range
  -0.00788 to +0.01806;
- temporal deficit within 0.05: **64/64 (100%)**;
- median boundary-jump difference from NEW: +0.00367;
- finite, nonconstant decodes: 64/64.

Thus the precommitted median SSIM >=0.90 and >=90% temporal safeguard both
pass. No additional metric was selected after seeing the videos.

## Runtime and memory

The primary runner measured a mean 14.93 s (median 14.31 s) of synchronized GPU
generation plus VAE decode work per scene/direction audit bundle; the sum was
955.60 GPU-seconds across 64 bundles. With two GPUs, shards ran concurrently.
Mean generation-only work per bundle was 6.21 s and mean decode work was
8.72 s. A representative complete three-NFE block took about 1.17 s before
decode; policy action-to-pixel distributions are reported above rather than
inferred from NFE counts.

- hardware: 2 x NVIDIA GeForce RTX 3090;
- maximum allocated VRAM/GPU: 9,565,199,360 bytes (8.908 GiB);
- maximum reserved VRAM/GPU: 11,943,280,640 bytes (11.123 GiB);
- exact rollback payload overhead: 339,360 GPU bytes (0.324 MiB);
- host checkpoint memory: 0 bytes;
- full live per-fork Matrix-Game cache allocation (not duplicated by LateAct):
  1,670,124,960 bytes.

## Secondary keyboard transfer

The eight-scene, 16-direction secondary is a negative transfer result, not a
newly tuned gate. All directions were evaluator-valid and exact checkpoint
replays passed, but the mouse-derived commitment boundary was not stable across
keyboard directions.

| Quantity | Keyboard result |
|---|---:|
| LATEACT response mean / median | 0.7097 / 0.7635 |
| left-to-right / right-to-left median | 0.5136 / 0.9801 |
| late cases within 0.10 of restart | 26/85 (30.6%) |
| late median latency saved vs restart | 0.3080 s |
| late mean latency-saving fraction | 12.16% |
| LATEACT-vs-NEW future SSIM mean / median | 0.7564 / 0.7378 |
| temporal deficit within 0.05 | 13/16 (81.25%) |

The paired direction-level LATEACT-minus-DIRECT response improvement was mean
0.5927, median 0.6051, bootstrap 95% CI `[0.4509, 0.7305]` (`n=16`, one-sided
Wilcoxon `p=1.53e-5`). This shows that rollback still helps direct binding, but
the fixed after-NFE1 checkpoint is often already too late for keyboard strafe;
it does not recover restart-level response or quality. The correct conclusion
is **action-dependent commitment**, not broad transfer.

## Representative qualitative branches

- Primary montage:
  `/mnt/NAS/data/hl5757/generated_artifacts/lateact/gate2/primary/mouse_qualitative_montage.png`
  (scene `0008_v0`, left-to-right). LATEACT visually tracks FULL/NEW while
  DIRECT remains closer to OLD.
- Secondary montage:
  `/mnt/NAS/data/hl5757/generated_artifacts/lateact/gate2/secondary/keyboard_qualitative_montage.png`
  (scene `0008_v0`, left-to-right). This retained unfavorable case shows
  LATEACT near OLD/DIRECT rather than FULL/NEW.

All 480 condition videos and 480 latent tensors, the 80 scene/direction audit
records, 1,152 paired arrival rows, and both summaries are retained under
`/mnt/NAS/data/hl5757/generated_artifacts/lateact/gate2/` (749 MiB at report
time).

## Final recommendation

Accept Gate 2 as a **STRONG PROJECT RESULT for mouse-yaw asynchronous control**.
The central serving claim is supported: with a measured action-specific
commitment boundary, exact minimal rollback reaches restart-level response at
lower latency and recomputation, while early direct binding avoids unnecessary
rollback. Scope the claim explicitly to measured action families and treat
commitment calibration as action-dependent. Do not begin repair, training, or
a learned allocation/commitment policy without separate review.

Result commit: `c1e6f53` (`Record Gate 2 asynchronous serving result`).
