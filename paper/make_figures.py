#!/usr/bin/env python3
"""Build paper-only figures from frozen LateAct JSON summaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


artifact_root = Path(os.environ["LATEACT_ARTIFACT_ROOT"])
output_dir = Path(__file__).resolve().parent / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

gate0 = load(artifact_root / "gate0-20260823" / "gate0_summary.json")
minwm_ad = load(
    artifact_root / "phase4_minwm" / "phase4_minwm_curve_summary.json"
)
minwm_yaw = load(
    artifact_root
    / "phase4_minwm_yaw_confirmatory"
    / "phase4_minwm_yaw_confirmatory_summary.json"
)

matrix_curves = [
    direction["new_action_fraction_s0_to_s3"]
    for scene in gate0["curves"].values()
    for direction in scene.values()
]
matrix_median = [median(values) for values in zip(*matrix_curves)]

yaw_curves = [
    [entry["response_retention"][str(step)] for step in range(5)]
    for entry in minwm_yaw["per_direction"]
]
yaw_median = [
    minwm_yaw["response_by_switch_after_nfe"][str(step)]["median"]
    for step in range(5)
]
ad_median = [
    minwm_ad["response_by_switch_after_nfe"][str(step)]["median"]
    for step in range(5)
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), constrained_layout=True)

blue = "#2468C9"
teal = "#168C79"
orange = "#D98216"
gray = "#8A96A3"

ax = axes[0]
x_matrix = list(range(4))
for curve in matrix_curves:
    ax.plot(x_matrix, curve, color=blue, alpha=0.18, linewidth=0.7)
ax.plot(
    x_matrix,
    matrix_median,
    color=blue,
    marker="o",
    linewidth=2.3,
    markersize=4.5,
    label="median (16/16 valid)",
)
ax.axvspan(1, 2, color=orange, alpha=0.09, linewidth=0)
ax.text(1.5, 0.61, "commitment\nboundary", ha="center", va="center", color=orange)
ax.annotate("0.997", (1, matrix_median[1]), xytext=(1.06, 0.90), color=blue)
ax.annotate("0.218", (2, matrix_median[2]), xytext=(2.06, 0.29), color=blue)
ax.set_title("(a) Matrix-Game mouse-yaw")
ax.set_xlabel("OLD-action NFEs before switch")
ax.set_ylabel("Normalized NEW-action response $R(s)$")
ax.set_xticks(x_matrix)
ax.set_ylim(-0.08, 1.08)
ax.grid(axis="y", color="#D9DEE5", linewidth=0.6)
ax.legend(loc="lower left", frameon=False)

ax = axes[1]
x_minwm = list(range(5))
for curve in yaw_curves:
    ax.plot(x_minwm, curve, color=teal, alpha=0.18, linewidth=0.7)
ax.plot(
    x_minwm,
    yaw_median,
    color=teal,
    marker="o",
    linewidth=2.3,
    markersize=4.5,
    label="j/l yaw (16/16 valid)",
)
ax.plot(
    x_minwm,
    ad_median,
    color=orange,
    marker="s",
    linewidth=1.5,
    markersize=3.7,
    linestyle="--",
    label="a/d lateral (6/16 valid)",
)
ax.axvspan(0, 1, color=orange, alpha=0.09, linewidth=0)
ax.annotate("0.145", (1, yaw_median[1]), xytext=(1.08, 0.22), color=teal)
ax.set_title("(b) minWM Wan2.1 Action2V")
ax.set_xlabel("OLD-action NFEs before switch")
ax.set_xticks(x_minwm)
ax.set_ylim(-0.08, 1.08)
ax.grid(axis="y", color="#D9DEE5", linewidth=0.6)
ax.legend(loc="upper right", frameon=False)

for ax in axes:
    ax.axhline(0, color=gray, linewidth=0.7)
    ax.axhline(1, color=gray, linewidth=0.7, linestyle=":")

fig.savefig(output_dir / "cross_model_commitment.pdf", bbox_inches="tight")
fig.savefig(output_dir / "cross_model_commitment.png", dpi=300, bbox_inches="tight")
plt.close(fig)
