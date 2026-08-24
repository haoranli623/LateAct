# Experiment state

- Project: LateAct — Asynchronous Action Binding in Interactive Video World Models
- Date: 2026-08-23
- Current phase: Phase 3A protocol frozen; calibration pending
- Static verdict: PASS
- Foundation model edits by LateAct: none
- Training: forbidden before Gate 0 review
- Upstream repository: `/mnt/NAS/data/hl5757/third_party/Matrix-Game`
- Upstream commit: `71c3cd7f741311f8100f6cf9cde942b6c1378d11`
- Model root: `/mnt/NAS/data/hl5757/models/matrix-game-2`
- Environment: `/mnt/NAS/data/hl5757/conda_envs/branch-safe-kv`
- Artifact root: `/mnt/NAS/data/hl5757/generated_artifacts/lateact`
- Official sampler NFE/block: 3, plus one non-output context-cache write
- Actual denoising timesteps: 1000.0, 908.8427124023438, 713.9794311523438
- Prefix/future design: one 3-latent block each; no cache eviction
- Frozen smoke scenes: official universal 0000, 0001
- Frozen Gate 0 scenes: official universal 0000 through 0007

## Engineering smoke result

- Artifact directory: `/mnt/NAS/data/hl5757/generated_artifacts/lateact/smoke-20260823`
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

- Artifact directory: `/mnt/NAS/data/hl5757/generated_artifacts/lateact/gate0-20260823`
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

- Artifact directory: `/mnt/NAS/data/hl5757/generated_artifacts/lateact/gate1`
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

- Artifact directory: `/mnt/NAS/data/hl5757/generated_artifacts/lateact/gate2`
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
