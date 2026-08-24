# Phase 4 — minWM independent-model replication

## Frozen verdict

**C. NO COMMITMENT REPLICATION**

This is a conservative gate verdict, not a claim that the raw minWM curve was
flat.  The raw result was sharply commitment-like, but only 6/16
direction-scene pairs passed the evaluator's frozen minimum oracle-separation
gate.  The required 12/16 valid pairs were therefore unavailable.  Phase 4C
rollback was not authorized and was not run.

No Matrix-Game result was modified or reinterpreted.  No training, LoRA,
weight edit, ForgeWM search, Hunyuan run, or additional action-family search
was performed.

## 1. Exact substrate

- Official repository: `https://github.com/shengshu-ai/minWM`
- minWM commit: `df522a26cd4409d3e3e8f269cc98eac069b5df47`
  (`df522a2 Update run_stage3_causal_dmd_camera.sh`)
- Backbone: `Wan-AI/Wan2.1-T2V-1.3B`
- Backbone Hub revision: `37ec512624d61f7aa208f7ea8140a131f93afc9a`
- Released checkpoint: `MIN-Lab/minWM/Wan21/Action2V/dmd/model.pt`
- Checkpoint Hub revision: `21bd74da43b5a061c0b8ff277515088ccd2c798b`
- Checkpoint LFS SHA-256/etag:
  `bdb947d45fb04513305492c2ee393d51d0621ec0e99fd312224f5d61a330aa77`
- Checkpoint path:
  `/mnt/NAS/data/hl5757/models/minwm/checkpoints/Wan21/Action2V/dmd/model.pt`
- Runtime: PyTorch `2.5.1+cu121`, CUDA runtime `12.1`, BF16, one RTX 3090.
  The dedicated NAS wrapper environment pins the two missing official runtime
  dependencies `lmdb==1.7.5` and `av==13.1.0`.

The upstream worktree differs from the pinned commit only by untracked NAS
symlinks under `Wan21/wan_models/` and `ckpts/`; official source files are
unchanged.

## 2. Static and engineering audit

### Four-step loop and exact hook

`Wan21/pipeline/causal_inference.py:28-32` warps the configured
`[1000, 750, 500, 250]` schedule through the shift-5 FlowMatch scheduler.  The
actual four evaluation timesteps measured at runtime were:

`[1000.0, 937.5, 833.3333129882812, 625.0]`.

The exact denoising loop is at
`Wan21/pipeline/causal_inference.py:225-268`.  Each evaluation passes the same
current-block `vm_chunk` as `viewmats` and the same `ks_chunk` as `Ks` into
`pipeline.generator`.  The replication hook changes only which already-built
native `viewmats` tensor is supplied at each of these four calls.  Prompt
embedding, `Ks`, weights, timestep, latent, history, and all stochastic tensors
remain fixed.

The wrapper forwards these values unchanged into `CausalWanModel` at
`Wan21/wan_utils/wan_wrapper.py:253-265`.  PRoPE injects the camera pose into
self-attention.  The ordinary visual cache and the PRoPE cache are both updated
in-place at `Wan21/wan/modules/causal_model.py:337-404`.

### Native control

The official trajectory implementation identifies `a` as left translation and
`d` as right translation, each by 0.08 world units per latent frame
(`Wan21/wan_utils/camera_trajectory.py:6-16,40-46`).  It builds c2w trajectories
and returns inverse w2c matrices (`:82-105`).  Phase 4 uses native `a*3` versus
`d*3` four-pose chunks.  It does not impose Matrix-Game mouse semantics.

### Mutable state and exact coupling

The model mutates:

- the current block's 30-layer visual RoPE K/V slices and index pairs;
- the current block's 30-layer PRoPE K/V slices and index pairs;
- cross-attention K/V on its first prompt-conditioned call;
- the current noisy latent after each of the first three NFEs.

Official inference creates each re-noising tensor implicitly with
`torch.randn_like` at `causal_inference.py:249-255`.  The replication runner
materializes one initial noise tensor and all three re-noising tensors once per
block and reuses their exact BF16 SHA-256 values across branches.  At the fork,
visual/PRoPE indices are restored to 24,960 tokens.  The final block occupies
the remaining 6,240 tokens, so it does not evict prefix state.  Every layer
overwrites its current-block K/V slice before reading it; prefix and cross
state remain shared.

Across all eight scenes:

- exact branch noise hashes: PASS;
- four NFE calls per branch: PASS;
- sampled historical visual/PRoPE bytes and full cross-state hashes after
  prefix-index restoration: PASS;
- intended differing tensor only: `viewmats`: PASS;
- same-action `a -> a` latent equality at every switch position on scenes 0
  and 1: PASS.

The implementation is therefore a clean intervention substrate; verdict D is
not applicable.

## 3. Official feasibility inference

The unmodified official Wan inference entry point generated one `a*19` example
at 832x480, 77 decoded frames, and 16 fps.

- Four NFEs per latent block, five blocks: 20 denoiser evaluations.
- Clean context writes: five additional timestep-zero generator calls.
- Denoising loop: 27.9 s.
- Prompt generation including decode/write: 40.86 s.
- Successful cold wall time including NAS model load: 288.64 s.
- Successful-run max resident host memory: 37,852,388 KiB (36.10 GiB).
- Observed official-run GPU residency during denoising: 15.4 GiB.
- Replication runner peak PyTorch allocation: 19,340,754,432 bytes
  (18.01 GiB); observed process residency was 20,322 MiB (19.85 GiB).
- The model fits one 24 GiB RTX 3090 without sequence parallelism.  A second
  concurrent model was deliberately avoided because duplicated cold host loads
  would reduce the 62 GiB RAM safety margin.

Required large files total about 23.51 GB decimal (21.90 GiB):

- DiT base: 5,676,070,424 bytes;
- UMT5 encoder: 11,361,920,418 bytes;
- VAE: 507,609,880 bytes;
- Action2V DMD checkpoint: 5,959,605,031 bytes.

The dedicated wrapper environment occupies 121 MB.  Phase 4 artifacts occupy
98 MB.  No checkpoint, cache, environment, or artifact was placed under HOME.

## 4. Frozen replication protocol

- Scenes: first eight official `Wan21/prompts/demos.txt` prompts, fixed before
  broad inference.
- Seeds: 41000 through 41007.
- Shared prefix: 16 latent frames (61 decoded frames), identity camera pose,
  four causal blocks, identical per-scene prompt/noise/history.
- Branch: final four-latent block (16 new decoded frames).
- Directions: `a -> d` and `d -> a`.
- Switch positions: old action for 0, 1, 2, 3, or 4 of the four NFEs.
- Same-action negative control: `a -> a` at all five switch positions on the
  first two scenes.
- Evaluator frozen before broad results: accumulated horizontal displacement
  at image center from 416x240 forward/backward-consistent LK tracks and a
  RANSAC partial-affine fit.  Independent stability check: summed median LK x
  translation.
- Frozen validity: at least 8.0 pixels oracle gap, at least 10x same-action
  response floor, at least 20 median inliers, and agreement of the two oracle
  gap signs.
- Frozen Phase 4B pass: at least 12/16 valid, monotone, and bounded pairs; both
  directions represented by at least six valid scenes; maximum adjacent median
  response drop at least 0.15; all videos valid.

The primary normalized statistic is
`R(s)=(response_switch(s)-response_old)/(response_new-response_old)` without
clipping.

## 5. Commitment-curve result

### Oracle separability and repeatability

- Same-action control latent hashes: exact on both control scenes.
- Same-action decoded-metric range: 0.0776 pixels (repeatability floor).
- Oracle gap sign agreement between affine and median-LK metrics: 16/16.
- Oracle affine gap magnitude across all scenes: median 6.851 pixels, range
  1.169–12.767 pixels.
- Median oracle inlier support: 414.75; minimum 221.
- Frozen ≥8-pixel validity: 6/16 direction-scene pairs (three scenes in both
  directions).

### Full response curve

For the six evaluator-valid pairs, median `R(s)` was:

| OLD NFEs `s` | 0 | 1 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|---:|
| median `R(s)` | 1.0000 | 0.0434 | 0.0028 | 0.0007 | 0.0000 |
| mean `R(s)` | 1.0000 | 0.0435 | 0.0015 | 0.0015 | 0.0000 |

For all 16 pairs, including pairs below the frozen absolute gap threshold,
median `R(s)` was:

`1.0000 / 0.0434 / 0.0043 / 0.0007 / 0.0000`.

Raw signed affine response (pixels; positive is rightward image motion) was:

| Scene | all-`a` | all-`d` | `a→d` s1 | s2 | s3 | `d→a` s1 | s2 | s3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 | 5.370 | -3.183 | 4.921 | 5.363 | 5.369 | -2.865 | -3.154 | -3.143 |
| 01 | 8.856 | -3.912 | 8.524 | 8.827 | 8.820 | -3.004 | -3.989 | -3.909 |
| 02 | 4.788 | -1.039 | 4.653 | 4.762 | 4.801 | -0.882 | -1.044 | -1.030 |
| 03 | 0.935 | -0.235 | 0.901 | 0.945 | 0.945 | -0.225 | -0.212 | -0.206 |
| 04 | 8.425 | 3.408 | 8.167 | 8.328 | 8.425 | 5.313 | 3.522 | 3.421 |
| 05 | 7.373 | -0.502 | 7.194 | 7.300 | 7.385 | 0.005 | -0.468 | -0.512 |
| 06 | 7.595 | -2.256 | 7.352 | 7.554 | 7.583 | -1.768 | -2.213 | -2.256 |
| 07 | 0.233 | -1.187 | 0.016 | 0.157 | 0.223 | -0.785 | -1.170 | -1.184 |

The raw columns also make the direction convention clear: an `a→d` switch
after one NFE remains near the all-`a` endpoint, while a `d→a` switch after one
NFE remains near the all-`d` endpoint.

Raw-curve diagnostics over all 16 pairs:

- monotone within tolerance 0.10: 16/16;
- all interior responses within `[-0.10, 1.10]`: 16/16;
- response fell from `s=0` to `s=1`: 16/16;
- `R(1) < 0.5`: 16/16; `R(1) < 0.2`: 14/16;
- paired median drop `R(0)-R(1)`: 0.9566;
- following paired median drops: 0.0355, 0.0037, 0.0007.

This is strong descriptive evidence of a hard first-evaluation commitment in
minWM.  It is not promoted to a successful replication because the absolute
oracle action effect failed the frozen validity count.

### Visual validity

- Finite, nonconstant decoded videos: 16/16 direction-scene curves.
- Temporal-consistency SSIM across the final-block window: median 0.9439,
  range 0.9210–0.9907.
- Prefix-boundary mean absolute jump: median 3.81/255, range 1.84–5.29.
- Manual review of the frozen scene-0 montage found coherent, stable outputs
  without branch-specific corruption.

Representative montage:
`/mnt/NAS/data/hl5757/generated_artifacts/lateact/phase4_minwm/phase4_minwm_curve_montage.png`.

## 6. Runtime

- Eight shared-prefix generation times: mean approximately 19.75 s each.
- Unique branch generation: mean 6.160 s for four NFEs.
- Decode/write: mean 7.417 s per 77-frame video.
- Scene-0 engineering job: 305.48 s including cold load and 13 branches.
- Remaining seven-scene job: 1,167.01 s including one cold load and 61
  branches.
- Total curve execution: 1,472.49 s (24.54 min), excluding the separate
  official feasibility run and CPU analysis.

## 7. Rollback decision

Phase 4B failed its frozen `12/16` evaluator-valid requirement (`6/16`).  The
post-commitment rollback boundary was therefore not selected.  DIRECT,
MINIMAL-ROLLBACK, and FULL-RESTART were not executed.  This obeys the Phase 4C
conditional and avoids turning a suggestive underpowered curve into a tuned
positive result.

## 8. Paper implication

The Matrix-Game mouse-yaw result remains frozen and unchanged.  minWM provides
clean mechanistic evidence that action influence can collapse after the first
of four denoising evaluations, but the native lateral action was not strong
enough on enough frozen prompts to pass the predeclared cross-model replication
gate.  The paper may report this only as a negative/inconclusive independent
replication with a highly suggestive raw curve.  It must not claim cross-model
validation or universal minimal rollback from Phase 4.

## 9. Reproducibility artifacts

- Frozen config: `config/phase4_minwm.yaml`
- Runner: `scripts/run_phase4_minwm.py`
- Analyzer: `scripts/analyze_phase4_minwm.py`
- Raw metadata: `curve_scenes_00_01.json`, `curve_scenes_01_08.json`
- Analysis: `phase4_minwm_curve_summary.json`
- Videos/latents/montage:
  `/mnt/NAS/data/hl5757/generated_artifacts/lateact/phase4_minwm/`

Final LateAct git commit is recorded after this report is committed.
