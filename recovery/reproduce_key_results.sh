#!/usr/bin/env bash
set -euo pipefail

recovery_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(CDPATH= cd -- "${recovery_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${python_bin}" >&2
  echo "Set PYTHON_BIN to a Python with NumPy and SciPy installed." >&2
  exit 2
fi

required=(
  "results/frozen/gate0/gate0_summary.json"
  "results/frozen/gate1/gate1_summary.json"
  "results/frozen/gate2/primary/mouse_shard_0.json"
  "results/frozen/gate2/primary/mouse_shard_1.json"
  "results/frozen/gate2/primary/mouse_summary.json"
  "results/frozen/phase3/calibration/phase3a_summary.json"
  "results/frozen/phase3/confirmation/phase3b_summary.json"
  "results/frozen/phase4_minwm_ad/phase4_minwm_curve_summary.json"
  "results/frozen/phase4_minwm_yaw/phase4_minwm_yaw_confirmatory_summary.json"
  "robustness/gate2_mouse_clustered/robustness_summary.json"
  "paper/LateAct_NeurIPS2026_Workshop.pdf"
)

for relative in "${required[@]}"; do
  if [[ ! -f "${repo_root}/${relative}" ]]; then
    echo "ERROR: missing frozen recovery artifact: ${relative}" >&2
    exit 3
  fi
done

if [[ -n "${LATEACT_RECOVERY_OUTPUT:-}" ]]; then
  output_dir="${LATEACT_RECOVERY_OUTPUT}"
  mkdir -p "${output_dir}"
  cleanup=0
else
  output_dir="$(mktemp -d -t lateact-analysis-only.XXXXXX)"
  cleanup=1
fi

cleanup_output() {
  if [[ "${cleanup}" -eq 1 ]]; then
    rm -rf -- "${output_dir}"
  fi
}
trap cleanup_output EXIT

mkdir -p "${output_dir}/robustness"
"${python_bin}" \
  "${repo_root}/robustness/gate2_mouse_clustered/analyze_gate2_mouse_robustness.py" \
  --input "${repo_root}/results/frozen/gate2/primary" \
  --output "${output_dir}/robustness"

"${python_bin}" "${repo_root}/recovery/verify_frozen_results.py" \
  --repo "${repo_root}" \
  --generated-robustness "${output_dir}/robustness/robustness_summary.json"

if [[ "${1:-}" == "--figures" ]]; then
  legacy="${output_dir}/figure_inputs"
  mkdir -p \
    "${legacy}/gate0-20260823" \
    "${legacy}/phase4_minwm" \
    "${legacy}/phase4_minwm_yaw_confirmatory" \
    "${output_dir}/paper/figures"
  cp "${repo_root}/results/frozen/gate0/gate0_summary.json" \
    "${legacy}/gate0-20260823/gate0_summary.json"
  cp "${repo_root}/results/frozen/phase4_minwm_ad/phase4_minwm_curve_summary.json" \
    "${legacy}/phase4_minwm/phase4_minwm_curve_summary.json"
  cp "${repo_root}/results/frozen/phase4_minwm_yaw/phase4_minwm_yaw_confirmatory_summary.json" \
    "${legacy}/phase4_minwm_yaw_confirmatory/phase4_minwm_yaw_confirmatory_summary.json"
  cp "${repo_root}/paper/make_figures.py" "${output_dir}/paper/make_figures.py"
  LATEACT_ARTIFACT_ROOT="${legacy}" "${python_bin}" "${output_dir}/paper/make_figures.py"
  test -s "${output_dir}/paper/figures/cross_model_commitment.pdf"
  test -s "${output_dir}/paper/figures/cross_model_commitment.png"
  echo "PASS: paper response-curve figure regenerated without model inference"
fi

echo "PASS: analysis-only LateAct recovery checks completed"
if [[ "${cleanup}" -eq 0 ]]; then
  echo "Outputs retained at: ${output_dir}"
fi
