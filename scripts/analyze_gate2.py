#!/usr/bin/env python3
"""Aggregate one Gate 2 action family and evaluate frozen serving criteria."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


POLICIES = ("next_block", "direct_late_bind", "full_restart", "lateact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("mouse", "keyboard"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def load(root: Path, family: str) -> tuple[dict, list[dict]]:
    scenes, reports = {}, []
    for path in sorted(root.glob(f"{family}_shard_*.json")):
        value = json.loads(path.read_text())
        if value["status"] != "complete":
            raise RuntimeError(f"incomplete shard: {path}")
        reports.append(value)
        overlap = scenes.keys() & value["scenes"].keys()
        if overlap:
            raise RuntimeError(f"duplicate scenes: {sorted(overlap)}")
        scenes.update(value["scenes"])
    expected = 32 if family == "mouse" else 8
    if len(scenes) != expected:
        raise RuntimeError(f"expected {expected} scenes, found {len(scenes)}")
    return scenes, reports


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int = 20260824) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def make_montage(root: Path, family: str, scene: str) -> None:
    direction = "left_to_right"
    names = ["old", "direct_late", "lateact", "new_oracle", "next_new"]
    labels = ["OLD/NEXT", "DIRECT LATE", "LATEACT", "FULL/NEW", "NEXT BLOCK"]
    cells = []
    for name, label in zip(names, labels):
        path = root / scene / f"{direction}_{name}.mp4"
        capture = cv2.VideoCapture(str(path))
        frame = None
        while True:
            ok, candidate = capture.read()
            if not ok:
                break
            frame = candidate
        capture.release()
        if frame is None:
            raise RuntimeError(f"cannot read {path}")
        frame = cv2.resize(frame, (320, 176))
        cv2.rectangle(frame, (0, 0), (175, 25), (0, 0, 0), -1)
        cv2.putText(frame, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(frame)
    cv2.imwrite(str(root / f"{family}_qualitative_montage.png"), np.hstack(cells))


def main() -> None:
    args = parse_args()
    scenes, reports = load(args.input, args.family)
    directions = []
    cases = []
    for scene_name, scene in sorted(scenes.items()):
        for direction_name, direction in sorted(scene["directions"].items()):
            direction_row = {
                "scene": scene_name,
                "direction": direction_name,
                "valid": direction["evaluator_valid"],
                "response_early": direction["response"]["lateact"],
                "response_late_direct": direction["response"]["direct_late"],
                "response_lateact": direction["response"]["lateact"],
                "quality": direction["quality"],
                "audit": direction["audit"],
            }
            directions.append(direction_row)
            for arrival in direction["arrivals"]:
                row = {
                    "scene": scene_name,
                    "direction": direction_name,
                    "valid": direction["evaluator_valid"],
                    "arrival_index": arrival["arrival_index"],
                    "normalized_time": arrival["normalized_time"],
                    "arrival_seconds": arrival["arrival_seconds"],
                    "inflight_nfe": arrival["inflight_nfe"],
                    "fully_completed_nfe_at_arrival": arrival["fully_completed_nfe_at_arrival"],
                    "inflight_fraction": arrival["inflight_fraction"],
                    "wait_to_boundary_seconds": arrival["wait_to_boundary_seconds"],
                    "policies": arrival["policies"],
                }
                cases.append(row)

    valid_directions = [row for row in directions if row["valid"]]
    valid_cases = [row for row in cases if row["valid"]]
    late_cases = [row for row in valid_cases if row["inflight_nfe"] >= 2]
    early_cases = [row for row in valid_cases if row["inflight_nfe"] == 1]
    if not valid_cases:
        raise RuntimeError("no evaluator-valid Gate 2 cases")

    policy_summary = {}
    for policy in POLICIES:
        response = [row["policies"][policy]["response_current"] for row in valid_cases]
        latency = [row["policies"][policy]["action_to_pixel_seconds"] for row in valid_cases]
        post = [row["policies"][policy]["post_boundary_nfe"] for row in valid_cases]
        discarded = [row["policies"][policy]["discarded_nfe"] for row in valid_cases]
        redone = [row["policies"][policy]["redone_nfe"] for row in valid_cases]
        policy_summary[policy] = {
            "response": stats(response),
            "action_to_pixel_seconds": stats(latency),
            "post_boundary_nfe": stats(post),
            "discarded_nfe": stats(discarded),
            "redone_nfe": stats(redone),
        }

    lateact_response = np.array(
        [row["policies"]["lateact"]["response_current"] for row in valid_cases]
    )
    direct_response = np.array(
        [row["policies"]["direct_late_bind"]["response_current"] for row in valid_cases]
    )
    improvement = lateact_response - direct_response
    late_gap = np.array(
        [
            abs(
                row["policies"]["lateact"]["response_current"]
                - row["policies"]["full_restart"]["response_current"]
            )
            for row in late_cases
        ]
    )
    late_latency = np.array(
        [row["policies"]["lateact"]["action_to_pixel_seconds"] for row in late_cases]
    )
    restart_latency = np.array(
        [row["policies"]["full_restart"]["action_to_pixel_seconds"] for row in late_cases]
    )
    latency_saved = restart_latency - late_latency
    strict_pareto = []
    near_restart_lower_latency = []
    for row in valid_cases:
        lateact = row["policies"]["lateact"]
        direct = row["policies"]["direct_late_bind"]
        axes = (
            lateact["response_current"] >= direct["response_current"],
            lateact["action_to_pixel_seconds"] <= direct["action_to_pixel_seconds"],
            lateact["post_boundary_nfe"] <= direct["post_boundary_nfe"],
        )
        strict_pareto.append(
            all(axes)
            and (
                lateact["response_current"] > direct["response_current"]
                or lateact["action_to_pixel_seconds"] < direct["action_to_pixel_seconds"]
                or lateact["post_boundary_nfe"] < direct["post_boundary_nfe"]
            )
        )
        full = row["policies"]["full_restart"]
        near_restart_lower_latency.append(
            abs(lateact["response_current"] - full["response_current"]) <= 0.10
            and lateact["action_to_pixel_seconds"] < full["action_to_pixel_seconds"]
        )

    early_exact = all(
        row["policies"]["lateact"]["response_current"]
        == row["policies"]["direct_late_bind"]["response_current"]
        and abs(
            row["policies"]["lateact"]["action_to_pixel_seconds"]
            - row["policies"]["direct_late_bind"]["action_to_pixel_seconds"]
        )
        < 1e-9
        and not row["policies"]["lateact"]["checkpoint_used"]
        for row in early_cases
    )
    fewer_redone = all(
        row["policies"]["lateact"]["redone_nfe"]
        < row["policies"]["full_restart"]["redone_nfe"]
        for row in late_cases
    )
    ssim = np.array([row["quality"]["lateact_vs_new_future_ssim"] for row in valid_directions])
    temporal = np.array([row["quality"]["lateact_temporal_minus_new"] for row in valid_directions])
    boundary = np.array([row["quality"]["lateact_boundary_jump_minus_new"] for row in valid_directions])
    quality_pass = bool(
        np.median(ssim) >= 0.90
        and np.mean(temporal >= -0.05) >= 0.90
        and all(row["quality"]["all_decodes_valid"] for row in valid_directions)
    )
    valid_fraction = len(valid_directions) / len(directions)
    direction_medians = {
        name: float(
            np.median([row["response_lateact"] for row in valid_directions if row["direction"] == name])
        )
        for name in ("left_to_right", "right_to_left")
    }
    late_near_fraction = float(np.mean(late_gap <= 0.10))
    late_abs_saved = float(latency_saved.mean())
    late_fraction_saved = float(latency_saved.mean() / restart_latency.mean())
    checkpoint_audits = all(
        row["audit"]["independent_early_bind_exact"]
        and row["audit"]["historical_prefix_exact"]
        and row["audit"]["rng_unchanged"]
        and row["audit"]["scheduler_unchanged"]
        for row in directions
    )

    criteria = {
        "overall_median_response_at_least_0_90": float(np.median(lateact_response)) >= 0.90,
        "late_within_0_10_restart_at_least_90_percent": late_near_fraction >= 0.90,
        "lateact_fewer_redone_nfe_than_restart": fewer_redone,
        "late_latency_saved_at_least_0_20_seconds": late_abs_saved >= 0.20,
        "late_latency_saved_at_least_10_percent": late_fraction_saved >= 0.10,
        "early_zero_rollback_exact": early_exact,
        "quality_pass": quality_pass,
        "valid_direction_fraction_at_least_0_90": valid_fraction >= 0.90,
        "both_direction_median_response_at_least_0_90": all(value >= 0.90 for value in direction_medians.values()),
        "checkpoint_audits_pass": checkpoint_audits,
    }
    positive = all(criteria.values())
    checkpoint_bytes = directions[0]["audit"]["optimized_checkpoint_bytes"]
    if any(row["audit"]["optimized_checkpoint_bytes"] != checkpoint_bytes for row in directions):
        raise RuntimeError("checkpoint size changed")
    summary = {
        "family": args.family,
        "verdict": "STRONG PROJECT RESULT" if positive else "GATE 2 NOT STRONG",
        "positive": positive,
        "scene_count": len(scenes),
        "direction_count": len(directions),
        "valid_direction_count": len(valid_directions),
        "valid_direction_fraction": valid_fraction,
        "arrival_case_count": len(cases),
        "valid_arrival_case_count": len(valid_cases),
        "arrival_class_counts": dict(Counter(row["inflight_nfe"] for row in valid_cases)),
        "criteria": criteria,
        "policies": policy_summary,
        "paired_effect": {
            "lateact_minus_direct_response_mean": float(improvement.mean()),
            "lateact_minus_direct_response_median": float(np.median(improvement)),
            "bootstrap_mean_95_percent_ci": bootstrap_mean_ci(improvement),
        },
        "late_arrivals": {
            "case_count": len(late_cases),
            "within_0_10_restart_fraction": late_near_fraction,
            "latency_saved_vs_restart_seconds": stats(latency_saved.tolist()),
            "mean_latency_saved_fraction": late_fraction_saved,
            "fewer_redone_nfe_fraction": float(
                np.mean(
                    [
                        row["policies"]["lateact"]["redone_nfe"]
                        < row["policies"]["full_restart"]["redone_nfe"]
                        for row in late_cases
                    ]
                )
            ),
        },
        "early_arrivals": {
            "case_count": len(early_cases),
            "zero_rollback_exact": early_exact,
        },
        "pareto": {
            "lateact_strictly_dominates_direct_fraction": float(np.mean(strict_pareto)),
            "lateact_near_restart_with_lower_latency_fraction": float(np.mean(near_restart_lower_latency)),
        },
        "quality": {
            "lateact_vs_new_future_ssim": stats(ssim.tolist()),
            "temporal_minus_new": stats(temporal.tolist()),
            "boundary_jump_minus_new": stats(boundary.tolist()),
            "temporal_within_0_05_fraction": float(np.mean(temporal >= -0.05)),
            "pass": quality_pass,
        },
        "direction_median_response": direction_medians,
        "checkpoint": {
            "gate1_current_implementation_bytes": 649330080,
            "gate2_optimized_exact_bytes": checkpoint_bytes,
            "save_seconds": stats([row["audit"]["checkpoint_save_seconds"] for row in directions]),
            "restore_seconds": stats([row["audit"]["checkpoint_restore_seconds"] for row in directions]),
            "host_bytes": 0,
            "peak_allocated_bytes": max(row["audit"]["peak_allocated_bytes"] for row in directions),
            "audits_pass": checkpoint_audits,
        },
        "per_direction": directions,
        "per_arrival": cases,
        "gpu_devices": sorted({report["gpu"] for report in reports}),
    }
    path = args.input / f"{args.family}_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    make_montage(args.input, args.family, sorted(scenes)[0])
    print(json.dumps({key: value for key, value in summary.items() if key not in ("per_direction", "per_arrival")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

