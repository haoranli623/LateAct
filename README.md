# LateAct

LateAct tests asynchronous action binding in an autoregressive interactive
video diffusion model: after denoising of the current output block has begun,
how much can a newly arrived action still causally redirect that same block
without restarting its latent trajectory?

The frozen novelty boundary is **late action arrival during a block's denoising
trajectory**. This is not action representation, generic controllability, or a
model-editing project.

The project uses the existing Matrix-Game 2.0 installation read-only:

- upstream: `/mnt/NAS/data/hl5757/third_party/Matrix-Game`
- weights: `/mnt/NAS/data/hl5757/models/matrix-game-2`
- environment: `/mnt/NAS/data/hl5757/conda_envs/branch-safe-kv`
- generated artifacts: `/mnt/NAS/data/hl5757/generated_artifacts/lateact`

See `STATIC_AUDIT.md`, `GATE0_REPORT.md`, and `GATE1_REPORT.md` for the frozen
gates. Gate 1 concludes **A. MINIMAL-ROLLBACK STRONG GO** and stops before the
proposed asynchronous serving-policy Gate 2.
