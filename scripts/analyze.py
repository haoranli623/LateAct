#!/usr/bin/env python3
"""Analyze LateAct smoke/Gate 0 shards and apply frozen verdict rules."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "gate0"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--smoke-input", type=Path)
    parser.add_argument("--project", type=Path, required=True)
    return parser.parse_args()


def load_scenes(root: Path, phase: str) -> tuple[dict, list[dict]]:
    reports = []
    for path in sorted(root.glob(f"{phase}_shard_*.json")):
        value = json.loads(path.read_text())
        if value["status"] != "complete":
            raise RuntimeError(f"incomplete shard: {path}")
        reports.append(value)
    if not reports:
        raise RuntimeError(f"no {phase} reports in {root}")
    scenes = {}
    for report in reports:
        overlap = scenes.keys() & report["scenes"].keys()
        if overlap:
            raise RuntimeError(f"duplicate scenes: {sorted(overlap)}")
        scenes.update(report["scenes"])
    return scenes, reports


def score(scene: dict, run: str, metric: str = "lk_signed_sum") -> float:
    return float(scene["runs"][run]["motion"][metric])


def direction_curve(scene: dict, direction: str) -> dict:
    if direction == "left_to_right":
        old, new = "oracle_mouse_left", "oracle_mouse_right"
        names = [new, "lr_s1", "lr_s2", old]
    else:
        old, new = "oracle_mouse_right", "oracle_mouse_left"
        names = [new, "rl_s1", "rl_s2", old]
    old_score, new_score = score(scene, old), score(scene, new)
    gap = new_score - old_score
    fractions = [
        (score(scene, name) - old_score) / gap if gap != 0 else float("nan")
        for name in names
    ]
    dense_gap = score(scene, new, "dense_signed_sum") - score(scene, old, "dense_signed_sum")
    median_tracks = min(
        scene["runs"][old]["motion"]["median_track_count"],
        scene["runs"][new]["motion"]["median_track_count"],
    )
    return {
        "old_run": old,
        "new_run": new,
        "run_names_s0_to_s3": names,
        "scores_s0_to_s3": [score(scene, name) for name in names],
        "new_action_fraction_s0_to_s3": fractions,
        "oracle_gap": gap,
        "dense_oracle_gap": dense_gap,
        "median_oracle_track_count": median_tracks,
        "oracle_latents_distinct": scene["runs"][old]["latent_sha256"] != scene["runs"][new]["latent_sha256"],
    }


def bootstrap_ci(values: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def make_montage(root: Path, scene: str, destination: Path) -> None:
    labels = ["LR s=0", "LR s=1", "LR s=2", "LR s=3", "RL s=0", "RL s=1", "RL s=2", "RL s=3"]
    files = [
        "oracle_mouse_right.mp4", "lr_s1.mp4", "lr_s2.mp4", "oracle_mouse_left.mp4",
        "oracle_mouse_left.mp4", "rl_s1.mp4", "rl_s2.mp4", "oracle_mouse_right.mp4",
    ]
    cells = []
    for label, filename in zip(labels, files):
        capture = cv2.VideoCapture(str(root / scene / filename))
        frame = None
        while True:
            ok, candidate = capture.read()
            if not ok:
                break
            frame = candidate
        capture.release()
        if frame is None:
            raise RuntimeError(f"could not read {filename}")
        frame = cv2.resize(frame, (320, 176))
        cv2.rectangle(frame, (0, 0), (130, 25), (0, 0, 0), -1)
        cv2.putText(frame, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(frame)
    montage = np.vstack([np.hstack(cells[:4]), np.hstack(cells[4:])])
    cv2.imwrite(str(destination), montage)


def main() -> None:
    args = parse_args()
    scenes, reports = load_scenes(args.input, args.phase)
    expected = 2 if args.phase == "smoke" else 8
    if len(scenes) != expected:
        raise RuntimeError(f"expected {expected} scenes, found {len(scenes)}")

    if args.phase == "smoke":
        floor = max(
            float(scene["engineering_checks"]["same_action_motion_range"])
            for scene in scenes.values()
        )
    else:
        if args.smoke_input is None:
            raise ValueError("--smoke-input is required for Gate 0")
        smoke_scenes, _ = load_scenes(args.smoke_input, "smoke")
        floor = max(
            float(scene["engineering_checks"]["same_action_motion_range"])
            for scene in smoke_scenes.values()
        )

    curves = {}
    flat = []
    for scene_name, scene in sorted(scenes.items()):
        curves[scene_name] = {}
        for direction in ("left_to_right", "right_to_left"):
            curve = direction_curve(scene, direction)
            gap_floor = max(10.0 * floor, 1e-6)
            curve["evaluator_valid"] = bool(
                curve["oracle_latents_distinct"]
                and abs(curve["oracle_gap"]) > gap_floor
                and curve["median_oracle_track_count"] >= 20
                and curve["oracle_gap"] * curve["dense_oracle_gap"] > 0
            )
            values = curve["new_action_fraction_s0_to_s3"]
            curve["monotone_tolerance_0_10"] = bool(
                all(values[index] + 0.10 >= values[index + 1] for index in range(3))
            )
            curves[scene_name][direction] = curve
            flat.append(curve)

    engineering = {
        "all_noise_hashes_match": all(
            scene["engineering_checks"]["all_noise_hashes_match"] for scene in scenes.values()
        ),
        "only_mouse_condition_differs": all(
            scene["engineering_checks"]["only_mouse_condition_differs"] for scene in scenes.values()
        ),
        "all_historical_cache_checks_passed": all(
            scene["engineering_checks"]["all_historical_cache_checks_passed"] for scene in scenes.values()
        ),
        "all_oracle_latents_distinct": all(
            scene["engineering_checks"]["oracle_latents_distinct"] for scene in scenes.values()
        ),
    }
    if args.phase == "smoke":
        engineering["same_action_controls_exact"] = all(
            scene["engineering_checks"]["same_action_controls_exact"] for scene in scenes.values()
        )

    valid_count = sum(curve["evaluator_valid"] for curve in flat)
    monotone_count = sum(
        curve["evaluator_valid"] and curve["monotone_tolerance_0_10"] for curve in flat
    )
    p1 = np.array([curve["new_action_fraction_s0_to_s3"][1] for curve in flat], dtype=float)
    p2 = np.array([curve["new_action_fraction_s0_to_s3"][2] for curve in flat], dtype=float)
    delta = p1 - p2
    try:
        paired_p = float(wilcoxon(p1, p2, alternative="greater").pvalue)
    except ValueError:
        paired_p = None

    runtime_values = [
        run["generation_seconds"] for scene in scenes.values() for run in scene["runs"].values()
    ]
    peak_allocated = max(
        run["peak_allocated_bytes"] for scene in scenes.values() for run in scene["runs"].values()
    )
    cache_example = next(iter(next(iter(scenes.values()))["runs"].values()))["cache_bytes_at_fork"]
    summary = {
        "phase": args.phase,
        "scene_count": len(scenes),
        "direction_scene_count": len(flat),
        "engineering": engineering,
        "same_action_lk_motion_floor": floor,
        "valid_direction_count": valid_count,
        "monotone_direction_count": monotone_count,
        "curves": curves,
        "paired_effect": {
            "p1_minus_p2_mean": float(np.nanmean(delta)),
            "p1_minus_p2_median": float(np.nanmedian(delta)),
            "bootstrap_mean_95_percent_ci": bootstrap_ci(delta),
            "wilcoxon_greater_p": paired_p,
        },
        "runtime": {
            "generation_seconds_total": float(np.sum(runtime_values)),
            "generation_seconds_mean_per_unique_run": float(np.mean(runtime_values)),
            "peak_allocated_bytes": int(peak_allocated),
            "cache_bytes_at_fork": cache_example,
            "gpu_devices": sorted({report["gpu"] for report in reports}),
        },
    }

    if args.phase == "smoke":
        evaluator_pass = valid_count == 4
        summary["engineering_pass"] = bool(all(engineering.values()) and evaluator_pass)
        summary["recommendation"] = "PROCEED_TO_FROZEN_GATE0" if summary["engineering_pass"] else "STOP_SMOKE_FAILURE"
        output_path = args.input / "smoke_summary.json"
    else:
        total = 16
        p2_late_count = sum(
            curve["evaluator_valid"] and curve["new_action_fraction_s0_to_s3"][2] >= 0.8
            for curve in flat
        )
        bounded_count = sum(
            curve["evaluator_valid"]
            and -0.10 <= curve["new_action_fraction_s0_to_s3"][1] <= 1.10
            and -0.10 <= curve["new_action_fraction_s0_to_s3"][2] <= 1.10
            for curve in flat
        )
        hard_count = sum(
            curve["evaluator_valid"] and curve["new_action_fraction_s0_to_s3"][1] <= 0.2
            for curve in flat
        )
        base_ok = all(engineering.values()) and valid_count >= 12
        if base_ok and p2_late_count >= 12 and monotone_count >= 12:
            verdict = "A. FREE-LATE-BINDING"
        elif base_ok and monotone_count >= 12 and float(np.nanmedian(delta)) >= 0.15 and bounded_count >= 12:
            verdict = "B. COMMITMENT-CURVE STRONG GO"
        elif base_ok and hard_count >= 12:
            verdict = "C. HARD-COMMITMENT CONDITIONAL GO"
        else:
            verdict = "D. NO-GO"
        summary["criterion_counts"] = {
            "total": total,
            "p2_at_least_0_8": p2_late_count,
            "monotone": monotone_count,
            "bounded_intermediates": bounded_count,
            "p1_at_most_0_2": hard_count,
        }
        summary["verdict"] = verdict
        output_path = args.input / "gate0_summary.json"

    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    representative = sorted(scenes)[0]
    make_montage(args.input, representative, args.input / f"{args.phase}_qualitative_montage.png")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

