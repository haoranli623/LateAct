# LateAct disaster recovery

This private branch is a portable scientific recovery snapshot for the final
NeurIPS 2026 World Models in Physical AI submission. It preserves source,
protocols, reports, the submitted paper, frozen numerical evidence, and the
otherwise-local Matrix-Game cache dependency. It intentionally excludes
licensed model weights, caches, Conda binaries, and the bulk generated
video/latent tree.

## 1. Repository layout

- `src/lateact/`: runtime, intervention, and rollback implementation.
- `scripts/`: frozen generation and analysis programs.
- `config/`: frozen experimental protocols; `storage.env.example` is portable,
  while machine-specific `storage.env` is intentionally absent.
- top-level reports: chronological audits, protocols, outcomes, and the
  canonical `GATE2_REPORT.md`.
- `results/frozen/`: 27 JSON records and nine qualitative montages.
- `robustness/gate2_mouse_clustered/`: corrected n=8 clustered inference.
- `paper/`: exact submitted source, figures, style, and final PDF.
- `recovery/vendor/branchsafe/`: local cache dependency with no upstream remote.
- `recovery/patches/`: exact Matrix-Game source patch.
- `recovery/environment/`: observed package inventories.
- `recovery/artifact_manifest.tsv` and `recovery/SHA256SUMS`: recovery inventory
  and integrity checks.

## 2. Frozen scientific state

Matrix-Game 2.0 mouse-yaw is the full LateAct efficacy setting. Gate 2 contains
1,024 paired arrivals: 676 late and 348 early. LateAct has median response
0.9999, all late arrivals are within 0.10 of restart, every early arrival uses
zero rollback, mean late-arrival latency is 11.93% below restart, and the exact
checkpoint is 339,360 bytes. Serving inference uses eight independent base
images; mouse-yaw future-SSIM has median 0.9223.

Keyboard shares the after-NFE1 commitment boundary and recovers directional
response, but future-SSIM is 0.6726, below the frozen 0.90 median safeguard.
There is no action-general trajectory-fidelity claim.

The original minWM native `a/d` study failed its frozen oracle-separation gate
at 6/16. Exactly one frozen native `j/l` yaw confirmation then passed 16/16.
This independently replicates the commitment mechanism only; minWM did not test
LateAct rollback or latency.

## 3. Obtain upstream source

Exact provenance is machine-readable in `UPSTREAMS.yaml`.

```bash
git clone https://github.com/SkyworkAI/Matrix-Game.git third_party/Matrix-Game
git -C third_party/Matrix-Game checkout 71c3cd7f741311f8100f6cf9cde942b6c1378d11

git clone https://github.com/shengshu-ai/minWM.git third_party/minWM
git -C third_party/minWM checkout df522a26cd4409d3e3e8f269cc98eac069b5df47
```

## 4. Restore the Matrix-Game cache hook

Apply the archived patch to the clean pinned checkout and install the vendored
package into the environment used for Matrix-Game:

```bash
git -C third_party/Matrix-Game apply \
  "$(pwd)/recovery/patches/matrix_game_cache_hook.patch"
python -m pip install -e recovery/vendor/branchsafe
python -m pytest recovery/vendor/branchsafe/tests
```

The patch changes only
`Matrix-Game-2/wan/modules/causal_model.py` and routes visual causal-cache
insertion through `branchsafe.cache_ops.append_and_read`. The archived patch was
verified by applying it to the clean pinned file and comparing the result
byte-for-byte with the experiment checkout.

## 5. Obtain model weights

Do not retrieve weights from an untrusted mirror and do not commit them here.

Matrix-Game documents:

```bash
huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir models/matrix-game-2
```

The historical Matrix-Game Hub revision and local model-file hashes were not
recorded; exact weight identity therefore remains the principal recovery
uncertainty.

For minWM, use the exact revisions in `UPSTREAMS.yaml`:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
  --local-dir models/minwm/Wan2.1-T2V-1.3B
huggingface-cli download MIN-Lab/minWM Wan21/Action2V/dmd/model.pt \
  --revision 21bd74da43b5a061c0b8ff277515088ccd2c798b \
  --local-dir models/minwm/checkpoints
```

Verify the Action2V checkpoint against the recorded LFS etag/SHA in
`UPSTREAMS.yaml`.

## 6. Reconstruct environments

Read `ENVIRONMENT.md` before installing. It distinguishes historically
observed runtime versions from the later server package state. Start with a
clean Python 3.10 environment and PyTorch 2.5.1+cu121, install the pinned
upstream dependencies, then add the vendored BranchSafe package. The files in
`recovery/environment/` are evidence for troubleshooting, not guaranteed clean
lockfiles.

Copy `config/storage.env.example` to `config/storage.env`, update paths for the
new machine, and source it. Never commit the populated file.

## 7. Obtain evaluation inputs

No separate training dataset is needed. Matrix-Game evaluation uses bundled
universal images `0000` through `0016` from the pinned checkout. minWM uses
`Wan21/prompts/demos.txt`; exact prompt indices, seeds, and prompt hashes are in
`UPSTREAMS.yaml` and the frozen YAML protocols.

## 8. Reproduce key results without model inference

Install NumPy and SciPy, then run:

```bash
PYTHON_BIN=python recovery/reproduce_key_results.sh
```

This fails if required files are missing, recomputes the Gate 2 clustered
robustness analysis from the two raw frozen mouse shards, and verifies the
central Gate 0–2, keyboard, minWM, and PDF facts. It never loads a world model.

To also regenerate the cross-model response-curve figure in an isolated output
directory:

```bash
LATEACT_RECOVERY_OUTPUT=/tmp/lateact-recovery-check \
  PYTHON_BIN=python recovery/reproduce_key_results.sh --figures
```

The complete paper can be rebuilt with Tectonic from `paper/main.tex`; the
authoritative submitted PDF is `paper/LateAct_NeurIPS2026_Workshop.pdf` with
SHA256 `3b6eb37388b6bc62120a8f11e910b1d384ff1ec3abde429ef3ccabef73cae5b7`.

## 9. Full-generation experiments

The runners under `scripts/` require the corresponding upstream checkout,
weights, CUDA-capable PyTorch environment, and RTX-3090-class GPU memory.
Frozen configurations specify scenes, seeds, actions, timesteps, arrival
schedules, and gates. Running them is not required to recover the reported
scientific conclusions because their numerical records are stored in Git.

## 10. Intentionally omitted large artifacts

The original generated artifact tree was approximately 1.9 GB and contained
1,147 MP4 videos and 1,166 latent `.pt` files. These are not normal Git objects.
Every omitted file's original path, byte size, scientific role, regeneration
status, and SHA256 is recorded in `artifact_manifest.tsv`. Licensed model
weights, Hugging Face/Torch caches, environment binaries, and upstream Git
packs are also excluded and documented by category.

Use `sha256sum -c recovery/SHA256SUMS` after cloning to verify the portable
snapshot before relying on it.
