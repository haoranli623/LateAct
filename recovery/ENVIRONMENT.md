# LateAct environment record

This file separates runtime facts recorded by the experiments from the package
state that happened to remain on the original server. The latter is evidence,
not a claim of an exact or minimal reconstruction.

## A. Experimentally observed historical runtime

Frozen shard metadata and reports record:

- GPU: NVIDIA GeForce RTX 3090, 24 GiB each; the Matrix-Game jobs used a
  two-GPU server and minWM inference used one GPU at a time.
- PyTorch: `2.5.1+cu121`.
- PyTorch CUDA runtime: `12.1`.
- Precision: BF16 for the evaluated model paths.
- minWM-specific runtime additions: `lmdb==1.7.5` and `av==13.1.0`.
- minWM peak PyTorch allocation: 19,340,754,432 bytes.

The original NVIDIA driver version was not saved in the experiment metadata.
At recovery-archive time, `nvidia-smi` could no longer communicate with the
driver, so the historical driver remains unknown rather than inferred.

## B. Current server package state at archive time

Archive date: 2026-08-30.

The directory named `branch-safe-kv` is a Conda environment with:

- Python 3.10.20
- PyTorch 2.5.1+cu121
- CUDA runtime reported by PyTorch: 12.1

Its recorded package state is in:

- `environment/branch-safe-kv.from-history.yml`
- `environment/branch-safe-kv.conda-explicit.txt`
- `environment/branch-safe-kv.pip-freeze.txt`

The directory named `minwm-wan21` is not an independent Conda environment. It
is a Python venv whose interpreter points into `branch-safe-kv`, enables system
site packages, and adds minWM-specific packages. Its combined visible package
state is archived as `environment/minwm-wan21.pip-freeze.txt`.

The current environments contain both CUDA 12 and CUDA 13 wheel families.
They ran the frozen experiments through PyTorch 2.5.1+cu121, but the mixed
package inventory should not be treated as a clean lockfile. Prefer rebuilding
from Python 3.10, PyTorch 2.5.1+cu121, the pinned upstream requirements, and the
specific packages needed by each runner. Use the freeze/explicit files to
resolve compatibility questions, not as an unconditional one-command install.

## Suggested reconstruction order

1. Create a clean Python 3.10 environment.
2. Install PyTorch 2.5.1 and torchvision 0.20.1 for CUDA 12.1.
3. Install the dependencies required by the pinned Matrix-Game repository.
4. Install the vendored BranchSafe package from `recovery/vendor/branchsafe`.
5. For minWM, install the pinned upstream requirements plus `lmdb==1.7.5` and
   `av==13.1.0`.
6. Install analysis dependencies including NumPy, SciPy, pandas, OpenCV,
   scikit-image, PyYAML, and matplotlib.
7. Run `recovery/reproduce_key_results.sh` before attempting model inference.

Exact GPU kernels, driver behavior, and package solver output may differ on a
future machine. The frozen JSON results are the authoritative numerical record.
