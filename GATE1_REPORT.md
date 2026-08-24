# LateAct Gate 1 — Minimal denoising rollback

## Verdict

**A. MINIMAL-ROLLBACK STRONG GO**

For action arrival after two OLD NFEs, restoring the exact state entering NFE2
and recomputing only NFE2-NFE3 under NEW recovers near-oracle signed action
response with one fewer post-arrival NFE than full restart.

- direct late-bind median response: **0.2182**
- minimal-rollback median response: **0.9971**
- full-restart median response: **1.0000**
- rollback improves direct: **16/16** pairs and both directions
- rollback within 0.10 of full restart: **16/16**
- same-action controls: exact BF16 equality on both control scenes
- all state, leakage, RNG, scheduler, and Gate 0 reproduction audits: PASS

The precommitted quality safeguard also passes: median future-frame SSIM to NEW
is 0.9026 and temporal-consistency deficit is within 0.05 on 16/16 pairs. Exact
pixel identity is not universal—the minimum SSIM is 0.5158—because OLD NFE1 is
intentionally retained, but decoded videos remain temporally clean and the
signed action response is restored.

No training, foundation-model edit, or Gate 2 execution was performed.

## Frozen experiment

- Same official universal scenes 0000-0007 as Gate 0
- Same mouse yaw actions: left `-0.1`, right `+0.1`, both directions
- Same prefix seeds `50000+scene` and future seeds `60000+scene`
- Same initial noise and two explicit scheduler re-noising tensors
- Same BF16 model, caches, three-NFE sampler, evaluator, and context write
- Arrival simulated immediately after OLD NFE2

Conditions:

1. **OLD:** OLD NFE1, NFE2, NFE3.
2. **NEW ORACLE:** NEW NFE1, NFE2, NFE3 from the initial noisy latent.
3. **DIRECT:** OLD NFE1-NFE2, then NEW NFE3.
4. **MINIMAL ROLLBACK:** restore the boundary after OLD NFE1, then NEW
   NFE2-NFE3. OLD NFE1 is retained and not recomputed.
5. **FULL RESTART:** discard OLD NFE1-NFE2 and recompute NEW NFE1-NFE3.

Every condition exactly reproduced its corresponding frozen Gate 0 latent hash:
OLD and NEW matched their oracles, DIRECT matched `s=2`, minimal rollback matched
`s=1`, and full restart matched NEW.

## Exact restored state

The rollback boundary is the state *entering NFE2*, after OLD NFE1 prediction
and the first fixed scheduler re-noising operation. The GPU-resident checkpoint
contains:

- BF16 entering-NFE2 latent `[1,16,3,44,80]`;
- visual current-block K/V slices `[2640:5280]` in all 30 layers;
- mouse current-block K/V slices `[3:6]` in active action layers 0-14;
- keyboard current-block K/V slices `[3:6]` in active action layers 0-14;
- `global_end_index` and `local_end_index` for visual, mouse, and keyboard
  caches.

It deliberately does not duplicate:

- historical visual prefix `[0:2640]`;
- historical mouse/keyboard prefix `[0:3]`;
- cross-attention K/V;
- scheduler tensors;
- RNG state.

Those states remain live and were exact-equality or SHA-256 checked before and
after every restore. The FlowMatch scheduler has no step counter; `add_noise`
is a pure lookup/calculation over fixed `sigmas` and `timesteps`. CPU and CUDA
RNG hashes were unchanged because all three stochastic tensors were explicitly
materialized before branching.

Checkpoint storage:

| Component | Bytes |
|---|---:|
| entering NFE2 latent | 337,920 |
| visual current slice | 486,604,800 |
| mouse current slice | 162,201,600 |
| keyboard current slice | 184,320 |
| index metadata | 1,440 |
| **total** | **649,330,080** |

Total is 619.25 MiB (0.649 GB), about 38.9% of the 1.670 GB allocated BF16
runtime-cache buffers measured at Gate 0.

## Leakage and equality audit

After DIRECT consumed OLD-NFE2 state, the rollback path overwrote every saved
current-block slice and restored its cache indices before recomputation. It then
asserted live K/V equality against the checkpoint. This proves no state produced
by OLD NFE2 remained in any readable current-block cache position.

Across all 16 primary pairs:

- entering latent restore: exact
- visual/mouse/keyboard current slices: exact
- historical prefix: exact
- cross-attention K/V and initialization flags: exact
- scheduler hash before/after: exact
- CPU and CUDA RNG hashes before/after: exact
- fixed initial and re-noising tensor hashes: exact
- all five outputs reproduce frozen Gate 0 hashes: exact

Example scene-0000 left-to-right checkpoint hashes are retained in
`gate1_shard_0.json`; all scene-specific hashes are machine-readable in the two
shards.

## Primary signed response

Response uses the frozen Gate 0 evaluator and normalization:

`R = (response - response_old) / (response_new - response_old)`.

| Scene | LR direct | LR rollback | LR full | RL direct | RL rollback | RL full |
|---|---:|---:|---:|---:|---:|---:|
| 0000 | 0.179 | 0.968 | 1.000 | 0.289 | 0.981 | 1.000 |
| 0001 | 0.178 | 0.996 | 1.000 | 0.438 | 1.000 | 1.000 |
| 0002 | 0.248 | 1.007 | 1.000 | 0.136 | 1.010 | 1.000 |
| 0003 | 0.061 | 0.997 | 1.000 | 0.038 | 0.994 | 1.000 |
| 0004 | 0.497 | 0.997 | 1.000 | 0.507 | 1.006 | 1.000 |
| 0005 | 0.245 | 1.003 | 1.000 | 0.191 | 0.997 | 1.000 |
| 0006 | 0.932 | 0.994 | 1.000 | 0.823 | 1.006 | 1.000 |
| 0007 | 0.127 | 0.997 | 1.000 | 0.077 | 1.001 | 1.000 |

Aggregate response:

| Condition | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| direct late bind | 0.3104 | 0.2182 | 0.0381 | 0.9321 |
| minimal rollback | 0.9971 | 0.9971 | 0.9680 | 1.0097 |
| full restart | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Paired rollback-minus-direct effect:

- mean 0.6867; median 0.7740
- bootstrap mean 95% CI `[0.5529, 0.7995]`
- one-sided paired Wilcoxon `p=1.5259e-5`
- median improvement left-to-right 0.7740
- median improvement right-to-left 0.7485

## Visual and temporal quality

The frozen secondary safeguard was median future-frame SSIM to NEW at least
0.90 and temporal-consistency SSIM no more than 0.05 below NEW on at least
14/16 pairs.

- rollback-vs-NEW future SSIM: mean 0.8489, median **0.9026**, range
  0.5158-0.9475; 9/16 individual pairs are at least 0.90
- temporal-consistency difference from NEW: median **+0.00142**
- temporal difference within `-0.05`: **16/16**
- finite, nonconstant valid decodes: **16/16**
- full restart vs NEW SSIM: exactly 1.0

The lower pixel SSIM cases are concentrated in scenes where retaining OLD NFE1
changes fine image trajectory or exact viewpoint, not in temporal collapse.
The scene-0000 qualitative montage visibly shows minimal rollback with the NEW
camera direction and a coherent frame, while DIRECT remains much nearer OLD.

## Same-action negative control

Scenes 0000 and 0001 repeated the full staged procedure with
`OLD == NEW == mouse_left`. Direct, minimal rollback, NEW, OLD, and full restart
all had one identical latent SHA-256 per scene, also matching the frozen Gate 0
mouse-left oracle:

- scene 0000: `b5dd378feab7bf2de64763af2a7ac6872bfaff852c950d694456fada7eaa099b`
- scene 0001: `66534578c6a0ad45affd1f341ce614c88263e07d995a723534fa3bd91190c825`

The rollback operation therefore adds exactly zero BF16 numerical/video error.

## Measured latency and compute

Times are synchronized GPU wall-clock measurements from simulated arrival to a
completed block, including the official context-cache write. Equality/hash
audits are excluded; actual checkpoint restore/reset copies are included.

| Condition | Mean latency | Median | Range |
|---|---:|---:|---:|
| direct continuation | 0.5856 s | 0.5845 s | 0.5745-0.6019 s |
| minimal rollback | 0.8821 s | 0.8800 s | 0.8664-0.9014 s |
| full restart | 1.1835 s | 1.1760 s | 1.1581-1.2356 s |

- GPU checkpoint restore alone: mean 2.57 ms
- minimal added latency over direct: mean 0.2964 s (+50.6%)
- minimal latency saved versus full restart: mean 0.3014 s (25.5%)
- two pre-arrival OLD NFEs: mean 0.5924 s
- peak allocated VRAM with checkpoint resident: 10,733,220,864 bytes
  (9.996 GiB)

| Condition | NFEs before arrival | Discarded | NFEs after arrival | Previously executed NFEs redone | Total NFE compute | Block equivalent |
|---|---:|---:|---:|---:|---:|---:|
| direct | 2 | 0 | 1 | 0 | 3 | 1.000× |
| minimal rollback | 2 | 1 | 2 | 1 | 4 | 1.333× |
| full restart | 2 | 2 | 3 | 2 | 5 | 1.667× |

All final paths execute one additional context write; it is common and excluded
from the denoiser-NFE counts.

## Early-arrival control

For arrival after NFE1, the primary minimal-rollback output is exactly the
zero-rollback DIRECT trajectory: OLD NFE1 has run and NEW is bound for
NFE2-NFE3. Comparing it to full restart shows a median response benefit of only
0.00295, with all 16 pairs already within 0.10 of full restart. Rolling back
before NFE1 therefore provides no meaningful action-response benefit.

This supports the intended boundary-aware rule:

- before commitment boundary: bind directly;
- after commitment boundary: roll back to the latest pre-commitment checkpoint.

## Frozen Gate 2 proposal — not executed

Because Gate 1 is A, the proposed Gate 2 is an asynchronous serving-policy
Pareto evaluation, not training:

1. retain the compact after-NFE1 current-block checkpoint while denoising;
2. if a new action arrives before NFE2 begins, bind it directly with no restore;
3. if it arrives after NFE2, restore the after-NFE1 checkpoint and replay
   NFE2-NFE3 under the new action;
4. retire the checkpoint when the block commits.

Gate 2 should measure action-to-pixel latency, signed response, ordinary video
quality, recomputation, checkpoint memory, and arrival-time distribution against:

- next-block/stale-action control;
- direct late binding;
- full restart.

The proposed headline is the action-to-pixel latency / response / recomputation
Pareto frontier. Gate 2 must not be run without separate review and approval.

## Artifacts

Machine-readable results and all 80 condition videos/latents:

`/mnt/NAS/data/hl5757/generated_artifacts/lateact/gate1`

- `gate1_summary.json`
- `gate1_shard_0.json`
- `gate1_shard_1.json`
- `gate1_qualitative_montage.png`

