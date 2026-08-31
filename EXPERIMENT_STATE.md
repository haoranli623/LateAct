# Experiment state

- Project: LateAct — Asynchronous Action Binding in Interactive Video World Models
- Date: 2026-08-24
- Current phase: Phase 4 independent-model commitment replication complete;
  project stopped after one frozen confirmatory yaw follow-up
- Static verdict: PASS
- Foundation model edits by LateAct: none
- Training: forbidden before Gate 0 review
- Upstream repository: `${LATEACT_UPSTREAM}`
- Upstream commit: `71c3cd7f741311f8100f6cf9cde942b6c1378d11`
- Model root: `${LATEACT_MODEL_ROOT}`
- Environment: user-managed Python environment
- Artifact root: `${LATEACT_ARTIFACT_ROOT}`
- Official sampler NFE/block: 3, plus one non-output context-cache write
- Actual denoising timesteps: 1000.0, 908.8427124023438, 713.9794311523438
- Prefix/future design: one 3-latent block each; no cache eviction
- Frozen smoke scenes: official universal 0000, 0001
- Frozen Gate 0 scenes: official universal 0000 through 0007

## Engineering smoke result

- Artifact directory: `artifacts/lateact/smoke-20260823`
- Scenes/runs: 2 scenes, 10 unique runs per scene
- Engineering verdict: PASS
- Exact same-action repeatability: PASS (latent hashes identical; motion range 0)
- Only changed condition tensor: `mouse_cond`
- Historical visual/mouse/keyboard cache slices unchanged: PASS at every NFE and context write
- Cross-attention cache unchanged: PASS
- Coupled stochastic inputs: PASS (one initial + two re-noising SHA-256 values shared per scene)
- Oracle metric validity: 4/4 direction-scene pairs
- Monotone curves: 4/4
- Mean generation time: 1.2194 s per unique branch block
- Peak allocated VRAM: 10,065,501,184 bytes
- Cache bytes at fork: visual 973,210,080; mouse 648,806,880; keyboard 737,760; cross 47,370,240; total 1,670,124,960
- Smoke `p1-p2`: mean 0.7153, median 0.7404 (engineering observation only; thresholds remained frozen)

## Gate 0 result

- Artifact directory: `artifacts/lateact/gate0-20260823`
- Scope complete: 8/8 frozen scenes, 16/16 direction-scene curves
- Evaluator-valid: 16/16
- Monotone: 16/16
- `p1` median: 0.9971
- `p2` median: 0.2182
- paired `p1-p2` median: 0.7740; one-sided Wilcoxon p=1.5259e-5
- Frozen verdict: **B. COMMITMENT-CURVE STRONG GO**
- Stop condition honored: no Gate 1, training, rollback, or repair started

## Gate 1 frozen design

- Primary arrival: after two OLD NFEs
- Minimal checkpoint: state entering NFE2 after OLD NFE1
- Minimal suffix: NEW NFE2 + NEW NFE3; NFE1 is not recomputed
- Direct suffix: NEW NFE3 only
- Full restart: NEW NFE1 + NEW NFE2 + NEW NFE3
- Same scenes/actions/seeds/evaluator as Gate 0
- Same-action controls: scenes 0000 and 0001, mouse-left
- Quality safeguards frozen in `config/gate1.yaml`
- No training or foundation-model edits authorized

## Gate 1 result

- Artifact directory: `artifacts/lateact/gate1`
- Scope: 8/8 scenes, both directions, 16/16 paired conditions
- State/leakage/Gate 0 reproduction audits: PASS 16/16
- Same-action rollback controls: exact on scenes 0000 and 0001
- Direct response median: 0.2182
- Minimal-rollback response median: 0.9971
- Full-restart response median: 1.0000
- Improved vs direct: 16/16
- Within 0.10 of full restart: 16/16
- Quality safeguard: PASS (median SSIM 0.9026; temporal 16/16)
- Mean post-arrival latency: direct 0.5856 s; minimal 0.8821 s; full 1.1835 s
- Checkpoint: 649,330,080 bytes; restore mean 2.57 ms
- Peak allocated VRAM: 10,733,220,864 bytes
- Frozen verdict: **A. MINIMAL-ROLLBACK STRONG GO**
- Gate 2 proposal was subsequently approved and executed as recorded below
- No training or foundation-model edits performed

## Gate 2 frozen design

- Primary: 32 fresh contexts from images 0008-0015 × four prefix variants
- Primary directions: mouse-left/right and reverse
- Arrivals: 16 stratified-uniform wall times per direction over three OLD NFEs
- Policies: NEXT-BLOCK, DIRECT-LATE-BIND, FULL-RESTART, LATEACT
- Boundary: direct after NFE1; rollback to after-NFE1 state after NFE2/NFE3
- Action-to-pixel latency includes context write and VAE decode
- Gate 1 checkpoint reference: 649,330,080 bytes
- Gate 2 exact checkpoint candidate: entering latent plus defensive indices
- Secondary, only after positive primary: 8 contexts, keyboard left/right
- No training, weight changes, lossy state compression, or boundary refitting

## Gate 2 result

- Artifact directory: `artifacts/lateact/gate2`
- Primary scope: 32/32 fresh contexts, 64/64 valid mouse-yaw directions,
  1,024 paired arrivals
- Primary LATEACT response: mean 0.9995, median 0.9999
- Late within 0.10 of FULL-RESTART: 676/676
- Late latency saved: median 0.3069 s; mean fraction 11.93%
- Late fewer redone NFEs: 676/676
- Early direct-bind/zero-rollback exact: 348/348
- Primary quality safeguard: PASS (median future SSIM 0.9223; temporal 64/64)
- Exact optimized checkpoint: 339,360 bytes; logical minimum 337,928 bytes
- Gate 1 checkpoint reference: 649,330,080 bytes
- Peak allocated VRAM: 9,565,199,360 bytes; host checkpoint: 0 bytes
- Primary verdict: **STRONG PROJECT RESULT**
- Secondary keyboard transfer: negative/action-dependent (median response
  0.7635; late within 0.10 of restart 30.6%; quality FAIL)
- Final scope: strong mouse-yaw result; no action-agnostic transfer claim
- Training, weight edits, repair, and additional methods: not started

## Phase 3A frozen design

- Question: does native keyboard A/D have a stable action-specific denoising
  commitment boundary?
- Calibration: 10 fresh rollout contexts from the sole unused official
  universal image `0016`, prefix variants 0-9
- Calibration prefix/future seeds: `120000+variant` / `130000+variant`
- Actions: keyboard-left/right in both switch directions; neutral mouse fixed
- Switch positions: OLD for 0, 1, 2, or 3 of the three NFEs
- Primary evaluator: robust partial-affine RANSAC displacement of image center,
  accumulated from prefix boundary through the generated block
- Independent evaluator check: forward/backward-filtered median LK translation
- Pre-calibration engineering check on existing Gate 2 keyboard oracles: 16/16
  gaps above 8 pixels, sign agreement 16/16, minimum median inlier support 255.5
- Frozen evaluator/quality/curve/boundary criteria: `config/phase3a.yaml`
- Frozen mouse latest safe switch: after NFE1
- Phase 3B permitted only for Phase 3A verdict A or B
- No training, weight edit, learned boundary, adaptive per-scene rule, or action
  expansion authorized

## Phase 3A frozen result

- Artifact directory:
  `artifacts/lateact/phase3/calibration`
- Scope: 10/10 fresh stochastic contexts, 20/20 evaluator-valid directions
- State/noise/condition audits: PASS 10/10 contexts and 60/60 runs
- Median keyboard response after 0/1/2/3 OLD NFEs:
  `1.0000 / 0.9957 / 0.1240 / 0.0000`
- Bootstrap median 95% CI after one OLD NFE: `[0.9827, 1.0203]`
- Bootstrap median 95% CI after two OLD NFEs: `[0.0401, 0.1711]`
- Monotone fraction: 19/20; bounded fraction: 19/20; quality: 40/40
  switch-direction cases
- Frozen boundary audit: latest safe switch after NFE1; next median 0.1240;
  median drop 0.8717; individual bracketing 20/20
- Mouse latest safe switch: after NFE1
- Phase 3A verdict: **B. SAME BOUNDARY**

## Phase 3B frozen design

- Confirmation: 24 fresh stochastic contexts from official image `0016`,
  variants 10-33, disjoint from calibration
- Prefix/current/next seed bases: 140000 / 150000 / 160000
- Both keyboard A/D directions; 16 frozen continuous arrivals per direction
- Policies: NEXT-BLOCK, DIRECT-LATE-BIND, FULL-RESTART,
  MOUSE-BOUNDARY LATEACT, ACTION-SPECIFIC LATEACT
- Mouse and keyboard boundaries are both after NFE1, so the last two policies
  are required to be bit-exact
- Primary evaluator: frozen robust affine lateral translation
- Exact criteria: `config/phase3b.yaml`
- Strong action-dependent-boundary support is impossible under frozen Phase 3A
  B; Phase 3B tests same-boundary reproduction and the Gate 2 evaluator diagnosis
- No training or new method authorized

## Phase 3B frozen result and final Phase 3 decision

- Artifact directory:
  `artifacts/lateact/phase3/confirmation`
- Scope: 24/24 fresh stochastic contexts, 48/48 valid directions, 768 arrivals
- Mouse-boundary and action-specific LateAct: bit-exact 768/768; response
  difference exactly 0 because both frozen boundaries are after NFE1
- LateAct response: mean 1.0207, median 1.0188
- Late within 0.10 of restart: 512/512; fewer redone NFEs: 512/512
- Late latency saved: median 0.3105 s; mean fraction 12.00%
- Early zero rollback: 256/256
- Temporal quality: PASS 48/48
- Pixel-fidelity quality: FAIL (median future SSIM 0.6726 < 0.90)
- Phase 3B verdict: **CONFIRMATION FAILED**
- Action-dependent commitment supported: NO; keyboard and mouse boundaries same
- Gate 2 keyboard response failure: evaluator mismatch; Gate 2 quality failure
  reproduces
- Paper recommendation: retain mouse-only main claim
- Training, second-model replication, and new methods: not started

## Phase 4 minWM frozen design

- Independent substrate: official minWM Wan2.1 Action2V four-step DMD
- Upstream commit: `df522a26cd4409d3e3e8f269cc98eac069b5df47`
- Backbone revision: `37ec512624d61f7aa208f7ea8140a131f93afc9a`
- DMD checkpoint revision: `21bd74da43b5a061c0b8ff277515088ccd2c798b`
- Native action pair: lateral camera `a` / `d`
- Eight frozen official prompts, seeds 41000-41007
- Shared 16-latent identity-camera prefix; final four-latent block branched
- Switch positions: old action for 0/1/2/3/4 of four NFEs
- Exactly materialized initial noise and three re-noising tensors per block
- Primary metric: frozen 416x240 robust partial-affine center displacement
- Rollback authorized only after a positive Phase 4B curve gate

## Phase 4 minWM frozen result

- Static intervention substrate: PASS
- Official feasibility inference: PASS on one RTX 3090
- Effective NFE timesteps: `1000.0 / 937.5 / 833.3333 / 625.0`
- Exact noise/action/state audits: PASS on 8/8 scenes
- Same-action latent repeatability: bit-exact on both control scenes
- Visual validity: 16/16 directional curves
- Oracle sign agreement: 16/16
- Frozen >=8-pixel oracle-separation validity: 6/16 pairs
- Valid-pair median response for switch positions 0/1/2/3/4:
  `1.0000 / 0.0434 / 0.0028 / 0.0007 / 0.0000`
- All-pair descriptive median response:
  `1.0000 / 0.0434 / 0.0043 / 0.0007 / 0.0000`
- Raw monotonic curves: 16/16; paired median first-step drop: 0.9566
- Frozen Phase 4B requirement: FAIL (needs at least 12/16 valid)
- Phase 4C rollback: not authorized and not run
- Peak PyTorch GPU allocation: 19,340,754,432 bytes (18.01 GiB)
- Mean four-NFE branch generation: 6.160 s; mean decode: 7.417 s
- Frozen verdict: **C. NO COMMITMENT REPLICATION**
- Interpretation: highly suggestive hard first-NFE raw curve, but insufficient
  frozen oracle-valid coverage for a cross-model claim
- Training, rollback, weight edits, Hunyuan, ForgeWM, and additional substrates:
  not started

## Phase 4 minWM frozen yaw confirmation

- The original native `a/d` result above remains a formal failed replication:
  only 6/16 direction-scene pairs passed the frozen oracle-separation gate.
- Exactly one confirmatory follow-up was frozen at commit `f1e2bbc` before its
  outputs were generated.
- Native action pair: yaw `j` / `l`; official prompt indices 8-15; seeds
  41008-41015; both directions; switch positions 0/1/2/3/4.
- Upstream source, Wan2.1 backbone, Action2V checkpoint, evaluator, thresholds,
  and success gate were unchanged from the original Phase 4 protocol.
- Oracle-valid, monotone, and bounded direction-scene pairs: 16/16.
- Median response for switch positions 0/1/2/3/4:
  `1.000000 / 0.145027 / 0.011482 / 0.003144 / 0.000000`.
- Same-action latent equality and stochastic/noise/cache isolation: PASS.
- Frozen interpretation: independent cross-model replication of the
  denoising-time commitment mechanism only.
- minWM rollback efficacy, checkpoint savings, and latency were not tested.
- Full LateAct rollback efficacy remains validated only on Matrix-Game
  mouse-yaw.
