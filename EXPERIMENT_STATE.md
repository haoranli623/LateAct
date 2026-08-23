# Experiment state

- Project: LateAct — Asynchronous Action Binding in Interactive Video World Models
- Date: 2026-08-23
- Current phase: engineering smoke passed; frozen Gate 0 approved by protocol
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
