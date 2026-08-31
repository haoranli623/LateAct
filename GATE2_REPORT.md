# LateAct — canonical technical report

## Executive Summary

LateAct is a commitment-aware serving method for action changes that arrive
during denoising in an interactive video world model. Its primary efficacy
result remains the frozen Matrix-Game mouse-yaw evaluation: the measured
late-action response improves from 0.218 without rollback to 0.997 with the
latest pre-commitment rollback, followed by a 32-context asynchronous test in
which all 676 late arrivals remain near FULL-RESTART while all 348 early
arrivals use exactly zero rollback. The optimized exact rollback state is
339,360 bytes, and mean action-to-pixel latency is 11.93% lower than
FULL-RESTART for late arrivals.

The independent-model evidence now has a deliberately preserved two-stage
chronology. The original minWM lateral `a/d` Phase 4 remains a **FAILED
REPLICATION** because only 6/16 direction-scene pairs passed the frozen
oracle-separation gate, despite descriptive commitment-like response curves
and clean engineering audits. A single predeclared confirmatory follow-up at
commit `f1e2bbc` changed only to the stronger native `j/l` yaw control on eight
new consecutive official prompts. It retained the evaluator, thresholds, and
success gate and passed all criteria: 16/16 oracle-valid pairs, 16/16 monotone
and bounded curves, exact same-action latent equality, and complete
stochastic/noise/cache isolation.

The keyboard result remains an explicit limitation. It exhibited the same
commitment boundary, but the frozen serving confirmation achieved median
future-SSIM 0.6726, below the 0.90 safeguard. LateAct therefore has no
action-general efficacy claim.

## Central claim

> “LateAct’s rollback efficacy is validated on Matrix-Game, while the
> underlying denoising-time commitment phenomenon independently replicates on
> minWM Wan2.1 Action2V under a frozen confirmatory yaw protocol.”

This statement intentionally separates method efficacy from mechanism
replication. minWM was not used to test LateAct rollback, latency savings, or a
cross-model serving policy.

## Result summary

| Evidence | Commitment | Rollback efficacy | Latency benefit | Frozen outcome |
|---|---|---|---|---|
| Matrix-Game mouse-yaw | Confirmed | Confirmed | Confirmed | Primary LateAct efficacy result |
| minWM `a/d` | Descriptive signal | Not tested | Not tested | Confirmatory gate failed, 6/16 oracle-valid |
| minWM `j/l` | Confirmatory replication, 16/16 | Not tested | Not tested | Frozen yaw commitment gate passed |

The Matrix-Game result is the only row supporting the complete LateAct serving
claim. The minWM yaw result supports cross-model replication of the underlying
denoising-time commitment phenomenon only.

**Cross-model claim.** “The underlying denoising-time commitment phenomenon
independently replicates on minWM Wan2.1 Action2V under a frozen confirmatory
yaw protocol.”

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
- base-image-equalized latency saved versus restart: mean **0.311789 s**,
  median **0.313654 s**;
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

The frozen serving hierarchy contains eight independent base images, four
deterministic variants per base, two directions per variant, and 16 arrivals
per direction, for 1,024 matched arrival rows. A late arrival has
`inflight_nfe >= 2`, yielding 676 late arrivals. Because variants and
directions derived from the same base image are not independent, inference
uses the eight base images as clusters (`n=8`). Within each base, the actual
asynchronous late-arrival estimand first averages LATEACT-minus-DIRECT over
the unchanged late rows within each direction and then equally averages the
eight variant/direction cells.

Under this base-cluster analysis, the late-arrival LATEACT-minus-DIRECT
response effect was mean **0.909136**, median **0.930385**, with a
base-cluster bootstrap mean 95% CI `[0.859769, 0.954093]`. The effect was
positive for **8/8 base images**; the exact Wilcoxon signed-rank p-values were
`0.00390625` one-sided and `0.0078125` two-sided.

The corresponding FULL-RESTART-minus-LATEACT late-arrival latency saving was
mean **0.311789 s**, median **0.313654 s**, with a base-cluster bootstrap mean
95% CI `[0.305519, 0.317523]` s. Savings were positive for **8/8 base images**;
the exact Wilcoxon signed-rank p-values were `0.00390625` one-sided and
`0.0078125` two-sided. The mean late-arrival latency reduction remains
**11.93%**. Arrival-level rows remain available for all 1,024 matched cases in
`mouse_summary.json`.

## Visual and temporal quality

The frozen primary safeguard passes:

- LATEACT-vs-NEW future SSIM over 64 scene-variant/direction comparisons:
  mean **0.915660**, median **0.922305**, p05 **0.839679**, and minimum
  **0.718375**;
- future SSIM at least 0.90: **49/64**; at least 0.85: **60/64**; at least
  0.80: **63/64**;
- temporal-consistency difference from NEW: median +0.00169, range
  -0.00788 to +0.01806;
- temporal deficit within 0.05: **64/64 (100%)**;
- median boundary-jump difference from NEW: +0.00367;
- finite, nonconstant decodes: **64/64**.

The frozen SSIM >=0.90 requirement was explicitly a **median-level**
safeguard, not a requirement that every direction reach 0.90. That frozen
median safeguard and the >=90% temporal safeguard both pass. Mouse-yaw
trajectory quality is therefore broadly stable but not uniform across the 64
comparisons. No retrospective per-direction quality threshold is introduced,
and no additional metric was selected after seeing the videos.

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

The eight-scene, 16-direction Gate 2 secondary was the initial keyboard
transfer result. Exact checkpoint replays passed, but its mouse-yaw-oriented
response evaluator reported weak and directionally asymmetric recovery. The
following values are retained as the original chronological result; they are
not evidence that keyboard has a different commitment boundary.

| Quantity | Keyboard result |
|---|---:|
| LATEACT response mean / median | 0.7097 / 0.7635 |
| left-to-right / right-to-left median | 0.5136 / 0.9801 |
| late cases within 0.10 of restart | 26/85 (30.6%) |
| late median latency saved vs restart | 0.3080 s |
| late mean latency-saving fraction | 12.16% |
| LATEACT-vs-NEW future SSIM mean / median | 0.7564 / 0.7378 |
| temporal deficit within 0.05 | 13/16 (81.25%) |

Under that initial evaluator, the paired direction-level
LATEACT-minus-DIRECT response improvement was mean 0.5927, median 0.6051,
bootstrap 95% CI `[0.4509, 0.7305]` (`n=16`, one-sided Wilcoxon
`p=1.53e-5`). The later diagnostic showed that the mouse-yaw score was
mismatched to keyboard A/D locomotion, so these apparent directional-response
shortfalls do not identify a later or action-dependent commitment boundary.

The frozen Phase 3 calibration used an action-appropriate lateral-motion
evaluator and placed keyboard at the same latest-safe boundary as mouse-yaw:
after one OLD NFE the median NEW-action response was 0.9957, falling to 0.1240
after two. In the subsequent 24-context, 48-direction serving confirmation,
the identically bounded rollback policies recovered the intended directional
response (median 1.0188), were bit-exact to each other, and remained within
0.10 of restart on 512/512 late arrivals.

The final frozen gate nevertheless failed on trajectory fidelity. Median
LATEACT-vs-NEW future-SSIM was **0.6726**, below the frozen **0.90** safeguard,
even though the directional response and temporal-consistency checks passed.
The defensible interpretation is therefore a **shared commitment boundary,
but no action-general rollback-efficacy or trajectory-fidelity claim**.

## Independent-model replication: minWM Wan2.1 Action2V

Both minWM experiments use the same pinned official substrate: Wan2.1
Action2V with the released four-step DMD checkpoint, four denoising evaluations
per branch, BF16 precision, and the same frozen affine/Lucas-Kanade horizontal
response evaluator. The intervention changes only the native camera `viewmats`
supplied at each denoising evaluation; prompt, prefix, latent state, cache
history, timesteps, sampler, and materialized stochastic tensors remain
coupled.

### Original `a/d` Phase 4: frozen failed replication

The original Phase 4 fixed eight official prompts, seeds 41000-41007, and both
`a -> d` and `d -> a` directions, yielding 16 direction-scene pairs. Only
**6/16** pairs, three per direction, reached the frozen minimum 8-pixel OLD/NEW
oracle separation. The required coverage was at least 12/16 and at least six
valid scenes per direction. Its formal verdict therefore remains **FAILED
REPLICATION**.

This was not an implementation failure. Affine/LK oracle signs agreed for
16/16 pairs; same-action latents were exact; noise, historical/cache state, and
four-NFE audits passed; and all videos were valid. The valid-pair median
response curve was commitment-like,
`[1.0000, 0.0434, 0.0028, 0.0007, 0.0000]`, but weak absolute lateral-action
separation made the confirmatory effect unidentifiable on ten pairs. No
rollback experiment was authorized or run.

### Frozen `j/l` yaw confirmatory protocol

Commit `f1e2bbc` froze exactly one follow-up on the same pinned model and
checkpoint:

- native `j*3` versus `l*3` yaw, in both switch directions;
- official prompt indices 8-15 with seeds 41008-41015;
- switch positions `s = 0,1,2,3,4` and same-action controls on prompts 8 and 9;
- the unchanged primary evaluator, 8-pixel oracle threshold, and at least
  12/16 overall plus at least 6/8 per-direction success requirements;
- no scene screening or replacement, threshold change, evaluator change,
  alternative action family, or rollback evaluation.

The generated metadata records the frozen config SHA-256
`07e8aa0d65aa62b5b7be4ef8ec7217cf9a35e4913a00a6842a6d26e434bce937`.

### Confirmatory result

**SUCCESSFUL CROSS-MODEL COMMITMENT REPLICATION.**

| Frozen quantity | Confirmatory result |
|---|---:|
| Oracle-valid direction-scene pairs | **16/16** |
| Valid `j -> l` / `l -> j` | **8/8 / 8/8** |
| Minimum oracle gap | **10.238 px** |
| Affine/LK sign agreement | **16/16** |
| Monotone curves | **16/16** |
| Bounded curves | **16/16** |
| Median normalized `R(s)`, `s=0..4` | **1.000000 / 0.145027 / 0.011482 / 0.003144 / 0.000000** |
| Maximum adjacent median response drop | **0.854973** |
| Same-action latent equality | **PASS, exact** |
| Stochastic/noise/cache audits | **PASS** |

The repeatability floor was 0.658011 pixels, so its frozen 10x requirement was
6.580109 pixels; all 16 pairs exceeded both that value and the independent
8-pixel minimum. All 16 direction-scene videos were valid. This result supports
independent replication of denoising-time commitment, not cross-model LateAct
rollback efficacy or systems savings.

## Representative qualitative branches

- Primary montage:
  `artifacts/lateact/gate2/primary/mouse_qualitative_montage.png`
  (scene `0008_v0`, left-to-right). LATEACT visually tracks FULL/NEW while
  DIRECT remains closer to OLD.
- Secondary montage:
  `artifacts/lateact/gate2/secondary/keyboard_qualitative_montage.png`
  (scene `0008_v0`, left-to-right). This retained unfavorable case shows
  LATEACT near OLD/DIRECT rather than FULL/NEW.

All 480 condition videos and 480 latent tensors, the 80 scene/direction audit
records, 1,152 paired arrival rows, and both summaries are retained under
`artifacts/lateact/gate2/` (749 MiB at report
time).

The independent-model yaw montage and machine-readable summary are retained as
`artifacts/lateact/phase4_minwm_yaw_confirmatory/phase4_minwm_yaw_confirmatory_montage.png`
and `phase4_minwm_yaw_confirmatory_summary.json`.

## Claims and evidence

| Claim | Evidence | Status |
|---|---|---|
| Action-conditioned denoising can exhibit a sharp commitment boundary | Matrix-Game mouse-yaw commitment curve; minWM frozen `j/l` replication | Supported on the tested model/action pairs |
| LateAct restores near-restart mouse-yaw response | 32 fresh Matrix-Game contexts; 676/676 late arrivals within 0.10 of restart | Supported |
| LateAct avoids unnecessary work before commitment | 348/348 early arrivals use direct binding with zero rollback | Supported |
| Exact minimal rollback lowers latency versus restart | 11.93% lower mean late-arrival latency with a 339,360-byte exact state | Supported on Matrix-Game mouse-yaw |
| The commitment phenomenon replicates across models | Frozen minWM `j/l` yaw gate passed 16/16 | Supported for Matrix-Game and the pinned minWM yaw protocol |
| LateAct rollback efficacy transfers to minWM | No minWM rollback experiment was run | Not supported |
| A shared rollback boundary guarantees action-general efficacy | Keyboard shared the measured boundary and recovered directional response but failed the trajectory-fidelity gate | Not supported |

## Limitations

- The complete LateAct efficacy and latency result is limited to Matrix-Game
  mouse-yaw. The minWM confirmatory experiment isolates commitment only.
- minWM rollback efficiency, cross-model action-to-pixel latency savings, and
  the full LateAct policy were not evaluated.
- The original minWM `a/d` experiment remains a formal 6/16 failed replication;
  the later yaw success does not retroactively change that verdict.
- Keyboard exhibited the same commitment boundary but failed the frozen visual
  safeguard (`0.6726 < 0.90` median future-SSIM), so there is no
  action-general efficacy claim.
- The evaluators are action-matched optical-flow measurements. The results do
  not establish universal commitment for arbitrary controls, environments,
  architectures, samplers, or world models.
- Latency measurements are specific to the evaluated implementations and RTX
  3090 hardware; no deployment-scale scheduler or multi-user workload was
  tested.
- No training, learned boundary predictor, post-training repair, or lossy
  checkpoint compression was evaluated.

## Contributions

1. A causal denoising-time intervention that measures when a realized action
   branch becomes committed while coupling prefix state, stochastic tensors,
   sampler settings, and all non-action conditioning.
2. LateAct, an asynchronous policy that directly binds early actions and
   restores only the latest pre-commitment state for late actions.
3. An exact 339,360-byte Matrix-Game rollback representation that recovers
   restart-level mouse-yaw response with 11.93% lower mean late-arrival
   latency.
4. A failure-preserving independent-model chronology: the original minWM
   `a/d` gate fails 6/16, while a single frozen native-yaw follow-up passes
   16/16 and independently replicates denoising-time commitment.
5. Explicit negative scope evidence showing that a shared keyboard commitment
   boundary does not imply action-general rollback quality.

## Conclusion

The Matrix-Game mouse-yaw experiments validate the full systems result: a late
action can arrive after direct binding has become ineffective, yet restoring
the exact latest pre-commitment state recovers restart-level control with less
recomputation and lower action-to-pixel latency. The minWM yaw confirmation
strengthens the mechanistic foundation by independently reproducing the sharp
denoising-time commitment curve under a frozen protocol. It does not extend
the rollback or latency result to minWM.

Accordingly, the defensible conclusion is:

> “LateAct’s rollback efficacy is validated on Matrix-Game, while the
> underlying denoising-time commitment phenomenon independently replicates on
> minWM Wan2.1 Action2V under a frozen confirmatory yaw protocol.”

## Recommended paper framing

- Lead with Matrix-Game mouse-yaw as the primary efficacy and systems result:
  `0.218 -> 0.997`, 676/676 late arrivals near restart, 348/348 early arrivals
  with zero rollback, 11.93% lower mean latency, and a 339,360-byte exact
  rollback state.
- Present minWM as independent replication of the underlying commitment
  phenomenon, with the original `a/d` 6/16 failure and the frozen `j/l` 16/16
  success reported in chronological order.
- Present keyboard as a visible negative boundary on action-general efficacy:
  the commitment boundary agrees, but future-SSIM is 0.6726 and fails the 0.90
  safeguard.
- Do not claim minWM rollback efficiency, cross-model latency savings, a
  cross-model-validated LateAct policy, universal action-general behavior, or
  universal commitment across world models.

Matrix-Game Gate 2 result commit: `c1e6f53` (`Record Gate 2 asynchronous
serving result`). minWM yaw confirmatory protocol commit: `f1e2bbc` (`Freeze
minWM yaw confirmatory protocol`).
