#!/usr/bin/env python3
"""Aggregate Gate 1 shards and apply the frozen rollback verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def load_scenes(root: Path) -> tuple[dict, list[dict]]:
    reports, scenes = [], {}
    for path in sorted(root.glob("gate1_shard_*.json")):
        report = json.loads(path.read_text())
        if report["status"] != "complete":
            raise RuntimeError(f"incomplete shard: {path}")
        reports.append(report)
        overlap = scenes.keys() & report["scenes"].keys()
        if overlap:
            raise RuntimeError(f"duplicate scenes: {sorted(overlap)}")
        scenes.update(report["scenes"])
    if len(scenes) != 8:
        raise RuntimeError(f"expected 8 scenes, found {len(scenes)}")
    return scenes, reports


def bootstrap_ci(values: np.ndarray, seed: int = 1) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def response(direction: dict, condition: str) -> float:
    metrics = direction["metrics"]
    old = float(metrics["old"]["motion"]["lk_signed_sum"])
    new = float(metrics["new_oracle"]["motion"]["lk_signed_sum"])
    value = float(metrics[condition]["motion"]["lk_signed_sum"])
    return (value - old) / (new - old)


def montage(root: Path, destination: Path) -> None:
    scene, direction = "0000", "left_to_right"
    conditions = ["old", "direct_late_bind", "minimal_rollback", "new_oracle", "full_restart"]
    labels = ["OLD", "DIRECT", "MIN ROLLBACK", "NEW", "FULL RESTART"]
    cells = []
    for condition, label in zip(conditions, labels):
        path = root / scene / f"{direction}_{condition}.mp4"
        capture = cv2.VideoCapture(str(path))
        frame = None
        while True:
            ok, candidate = capture.read()
            if not ok:
                break
            frame = candidate
        capture.release()
        if frame is None:
            raise RuntimeError(f"could not read {path}")
        frame = cv2.resize(frame, (320, 176))
        cv2.rectangle(frame, (0, 0), (180, 25), (0, 0, 0), -1)
        cv2.putText(frame, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(frame)
    cv2.imwrite(str(destination), np.hstack(cells))


def main() -> None:
    args = parse_args()
    scenes, reports = load_scenes(args.input)
    rows = []
    for scene_name, scene in sorted(scenes.items()):
        for direction_name, direction in scene["directions"].items():
            values = {
                condition: response(direction, condition)
                for condition in ("old", "new_oracle", "direct_late_bind", "minimal_rollback", "full_restart")
            }
            latency = direction["audit"]["latency_seconds"]
            row = {
                "scene": scene_name,
                "direction": direction_name,
                "response": values,
                "rollback_minus_direct": values["minimal_rollback"] - values["direct_late_bind"],
                "rollback_abs_gap_full": abs(values["minimal_rollback"] - values["full_restart"]),
                "minimal_vs_new_future_ssim": direction["quality"]["minimal_vs_new_future_ssim"],
                "direct_vs_new_future_ssim": direction["quality"]["direct_vs_new_future_ssim"],
                "full_vs_new_future_ssim": direction["quality"]["full_vs_new_future_ssim"],
                "minimal_temporal_minus_new": direction["quality"]["minimal_temporal_minus_new"],
                "decodes_valid": direction["quality"]["all_decodes_finite_and_nonconstant"],
                "latency_seconds": latency,
                "minimal_added_vs_direct": latency["minimal_from_arrival"] - latency["direct_from_arrival"],
                "minimal_saved_vs_full": latency["full_from_arrival"] - latency["minimal_from_arrival"],
                "peak_allocated_bytes": direction["audit"]["peak_allocated_bytes"],
                "checkpoint_bytes": direction["audit"]["checkpoint_bytes"],
                "state_audit_pass": all(
                    direction["audit"][key]
                    for key in (
                        "latent_restore_exact",
                        "checkpoint_live_exact_before_recompute",
                        "historical_prefix_exact",
                        "cross_attention_exact",
                        "no_old_nfe2_state_leak",
                        "all_outputs_reproduce_gate0",
                    )
                )
                and direction["audit"]["scheduler_hash_before_after"][0]
                == direction["audit"]["scheduler_hash_before_after"][1]
                and direction["audit"]["rng_hashes_before_after"][0]
                == direction["audit"]["rng_hashes_before_after"][1],
            }
            rows.append(row)

    direct = np.array([row["response"]["direct_late_bind"] for row in rows])
    minimal = np.array([row["response"]["minimal_rollback"] for row in rows])
    full = np.array([row["response"]["full_restart"] for row in rows])
    improvement = minimal - direct
    gap = np.abs(minimal - full)
    ssim = np.array([row["minimal_vs_new_future_ssim"] for row in rows])
    temporal = np.array([row["minimal_temporal_minus_new"] for row in rows])
    try:
        improvement_p = float(wilcoxon(minimal, direct, alternative="greater").pvalue)
    except ValueError:
        improvement_p = None

    same_controls = [
        scene["same_action_control"]
        for scene in scenes.values()
        if scene["same_action_control"] is not None
    ]
    engineering = {
        "all_state_audits_pass": all(row["state_audit_pass"] for row in rows),
        "all_decodes_valid": all(row["decodes_valid"] for row in rows),
        "same_action_control_count": len(same_controls),
        "all_same_action_controls_exact": len(same_controls) == 2 and all(value["all_exact"] for value in same_controls),
        "all_gate0_reproduction_exact": all(row["state_audit_pass"] for row in rows),
    }
    direction_improvement = {
        direction: float(np.median([row["rollback_minus_direct"] for row in rows if row["direction"] == direction]))
        for direction in ("left_to_right", "right_to_left")
    }
    improved_count = int((improvement > 0).sum())
    near_restart_count = int((gap <= 0.10).sum())
    temporal_ok_count = int((temporal >= -0.05).sum())
    quality_pass = bool(float(np.median(ssim)) >= 0.90 and temporal_ok_count >= 14)

    strong = bool(
        all(value if isinstance(value, bool) else value == 2 for value in engineering.values())
        and float(np.median(minimal)) >= 0.90
        and improved_count >= 14
        and all(value > 0 for value in direction_improvement.values())
        and near_restart_count >= 14
        and quality_pass
    )
    if strong:
        verdict = "A. MINIMAL-ROLLBACK STRONG GO"
    elif engineering["all_state_audits_pass"] and engineering["all_same_action_controls_exact"] and float(np.median(minimal)) >= 0.70 and improved_count >= 12:
        verdict = "B. PARTIAL GO"
    elif engineering["all_state_audits_pass"] and int((np.abs(full - 1.0) <= 1e-6).sum()) >= 14:
        verdict = "C. FULL-RESTART REQUIRED"
    else:
        verdict = "D. NO-GO"

    latency_keys = ("direct_from_arrival", "minimal_from_arrival", "full_from_arrival")
    latency_summary = {
        key: {
            "mean": float(np.mean([row["latency_seconds"][key] for row in rows])),
            "median": float(np.median([row["latency_seconds"][key] for row in rows])),
        }
        for key in latency_keys
    }
    latency_summary["minimal_added_vs_direct_mean"] = float(np.mean([row["minimal_added_vs_direct"] for row in rows]))
    latency_summary["minimal_saved_vs_full_mean"] = float(np.mean([row["minimal_saved_vs_full"] for row in rows]))
    latency_summary["prearrival_two_nfes_mean"] = float(np.mean([row["latency_seconds"]["prearrival_total"] for row in rows]))

    checkpoint = rows[0]["checkpoint_bytes"]
    if any(row["checkpoint_bytes"] != checkpoint for row in rows):
        raise RuntimeError("checkpoint byte cost changed across pairs")
    summary = {
        "verdict": verdict,
        "scene_count": 8,
        "paired_condition_count": 16,
        "engineering": engineering,
        "response": {
            "direct": {"mean": float(direct.mean()), "median": float(np.median(direct)), "min": float(direct.min()), "max": float(direct.max())},
            "minimal_rollback": {"mean": float(minimal.mean()), "median": float(np.median(minimal)), "min": float(minimal.min()), "max": float(minimal.max())},
            "full_restart": {"mean": float(full.mean()), "median": float(np.median(full)), "min": float(full.min()), "max": float(full.max())},
            "improved_pair_count": improved_count,
            "near_full_restart_count": near_restart_count,
            "direction_median_improvement": direction_improvement,
        },
        "paired_effect": {
            "rollback_minus_direct_mean": float(improvement.mean()),
            "rollback_minus_direct_median": float(np.median(improvement)),
            "bootstrap_mean_95_percent_ci": bootstrap_ci(improvement),
            "wilcoxon_greater_p": improvement_p,
        },
        "quality": {
            "minimal_vs_new_future_ssim_mean": float(ssim.mean()),
            "minimal_vs_new_future_ssim_median": float(np.median(ssim)),
            "minimal_vs_new_future_ssim_min": float(ssim.min()),
            "temporal_minus_new_median": float(np.median(temporal)),
            "temporal_within_0_05_count": temporal_ok_count,
            "quality_pass": quality_pass,
        },
        "early_arrival_control": {
            "interpretation": "minimal rollback trajectory is zero-rollback direct binding for arrival after NFE1; full restart is the rollback comparator",
            "median_full_restart_benefit": float(np.median(full - minimal)),
            "within_0_10_count": near_restart_count,
            "no_meaningful_rollback_benefit": near_restart_count >= 14,
        },
        "latency_seconds": latency_summary,
        "compute": rows[0]["latency_seconds"] and next(iter(scenes.values()))["directions"]["left_to_right"]["audit"]["compute"],
        "checkpoint_bytes": checkpoint,
        "peak_allocated_bytes": int(max(row["peak_allocated_bytes"] for row in rows)),
        "per_pair": rows,
    }
    (args.input / "gate1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    montage(args.input, args.input / "gate1_qualitative_montage.png")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

