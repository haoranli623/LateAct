# Experiment state

- Project: LateAct — Asynchronous Action Binding in Interactive Video World Models
- Date: 2026-08-23
- Current phase: Gate 1 protocol frozen; implementation pending
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
