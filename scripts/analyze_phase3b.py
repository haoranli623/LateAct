#!/usr/bin/env python3
"""Analyze the frozen Phase 3B keyboard serving confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


POLICIES = (
    "next_block",
    "direct_late_bind",
    "full_restart",
    "mouse_boundary_lateact",
    "action_specific_lateact",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p05": float(np.quantile(data, 0.05)),
        "p95": float(np.quantile(data, 0.95)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def bootstrap_mean_ci(values: list[float], seed: int = 20260825) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(10000, len(data)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def load(root: Path) -> tuple[dict, list[dict]]:
    reports = [json.loads(path.read_text()) for path in sorted(root.glob("phase3b_shard_*.json"))]
    if len(reports) != 2 or any(report["status"] != "complete" for report in reports):
        raise RuntimeError("Phase 3B requires two complete shards")
    contexts = {}
    for report in reports:
        overlap = set(contexts) & set(report["contexts"])
        if overlap:
            raise RuntimeError(f"duplicate Phase 3B contexts: {sorted(overlap)}")
        contexts.update(report["contexts"])
    if len(contexts) != 24:
        raise RuntimeError(f"expected 24 confirmation contexts, found {len(contexts)}")
    return contexts, reports


def make_montage(root: Path, context: str) -> None:
    direction = "left_to_right"
    names = ["old", "direct_late", "lateact", "new_oracle", "next_new"]
    labels = ["OLD", "DIRECT LATE", "BOTH LATEACT", "FULL/NEW", "NEXT BLOCK"]
    cells = []
    for name, label in zip(names, labels):
        capture = cv2.VideoCapture(str(root / context / f"{direction}_{name}.mp4"))
        frame = None
        while True:
            ok, candidate = capture.read()
            if not ok:
                break
            frame = candidate
        capture.release()
        if frame is None:
            raise RuntimeError(f"cannot read montage input {name}")
        frame = cv2.resize(frame, (320, 176))
        cv2.rectangle(frame, (0, 0), (185, 25), (0, 0, 0), -1)
        cv2.putText(frame, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(frame)
    cv2.imwrite(str(root / "phase3b_keyboard_serving_montage.png"), np.hstack(cells))


def main() -> None:
    args = parse_args()
    contexts, reports = load(args.input)
    directions = []
    arrivals = []
    for context_name, context in sorted(contexts.items()):
        for direction_name, direction in sorted(context["directions"].items()):
            directions.append({
                "context": context_name,
                "direction": direction_name,
                "valid": direction["evaluator_valid"],
                "direct_response": direction["response"]["direct_late"],
                "lateact_response": direction["response"]["lateact"],
                "quality": direction["quality"],
                "audit": direction["audit"],
            })
            for arrival in direction["arrivals"]:
                arrivals.append({
                    "context": context_name,
                    "direction": direction_name,
                    "valid": direction["evaluator_valid"],
                    **arrival,
                })
    valid_directions = [row for row in directions if row["valid"]]
    valid_arrivals = [row for row in arrivals if row["valid"]]
    late = [row for row in valid_arrivals if row["inflight_nfe"] >= 2]
    early = [row for row in valid_arrivals if row["inflight_nfe"] == 1]
    if not valid_arrivals:
        raise RuntimeError("no valid Phase 3B arrivals")

    policy_summary = {}
    for policy in POLICIES:
        policy_summary[policy] = {
            "response": stats([row["policies"][policy]["response_current"] for row in valid_arrivals]),
            "action_to_pixel_seconds": stats([
                row["policies"][policy]["action_to_pixel_seconds"] for row in valid_arrivals
            ]),
            "post_boundary_nfe": stats([
                row["policies"][policy]["post_boundary_nfe"] for row in valid_arrivals
            ]),
            "discarded_nfe": stats([
                row["policies"][policy]["discarded_nfe"] for row in valid_arrivals
            ]),
            "redone_nfe": stats([
                row["policies"][policy]["redone_nfe"] for row in valid_arrivals
            ]),
        }

    policy_bit_exact = all(
        row["policies"]["mouse_boundary_lateact"]
        == row["policies"]["action_specific_lateact"]
        for row in arrivals
    ) and all(row["audit"]["mouse_action_specific_policy_bit_exact"] for row in directions)
    late_gap = np.asarray([
        abs(row["policies"]["action_specific_lateact"]["response_current"] - 1.0)
        for row in late
    ])
    latency_saved = np.asarray([
        row["policies"]["full_restart"]["action_to_pixel_seconds"]
        - row["policies"]["action_specific_lateact"]["action_to_pixel_seconds"]
        for row in late
    ])
    latency_fraction = np.asarray([
        saved / row["policies"]["full_restart"]["action_to_pixel_seconds"]
        for saved, row in zip(latency_saved, late)
    ])
    fewer_redone = np.asarray([
        row["policies"]["action_specific_lateact"]["redone_nfe"]
        < row["policies"]["full_restart"]["redone_nfe"]
        for row in late
    ])
    early_zero = all(
        row["policies"]["action_specific_lateact"]["redone_nfe"] == 0
        and not row["policies"]["action_specific_lateact"]["checkpoint_used"]
        for row in early
    )
    quality_ssim = [row["quality"]["lateact_vs_new_future_ssim"] for row in valid_directions]
    quality_temporal = [row["quality"]["lateact_temporal_minus_new"] for row in valid_directions]
    quality_pass = bool(
        np.median(quality_ssim) >= 0.90
        and np.mean(np.asarray(quality_temporal) >= -0.05) >= 0.90
        and all(row["quality"]["all_decodes_valid"] for row in valid_directions)
    )
    valid_fraction = len(valid_directions) / len(directions)
    criteria = {
        "valid_direction_fraction_at_least_0_90": valid_fraction >= 0.90,
        "median_response_at_least_0_90": policy_summary["action_specific_lateact"]["response"]["median"] >= 0.90,
        "late_within_0_10_restart_at_least_0_90": float(np.mean(late_gap <= 0.10)) >= 0.90,
        "late_fewer_redone_nfe": bool(np.all(fewer_redone)),
        "late_latency_saved_at_least_0_20_seconds": float(np.median(latency_saved)) >= 0.20,
        "late_latency_saved_at_least_10_percent": float(np.mean(latency_fraction)) >= 0.10,
        "early_zero_rollback": early_zero,
        "quality_pass": quality_pass,
        "mouse_action_specific_policy_bit_exact": policy_bit_exact,
    }
    serving_success = all(criteria.values())
    improvement_by_direction = [
        row["lateact_response"] - row["direct_response"] for row in valid_directions
    ]
    action_vs_mouse = [
        row["policies"]["action_specific_lateact"]["response_current"]
        - row["policies"]["mouse_boundary_lateact"]["response_current"]
        for row in valid_arrivals
    ]
    action_dependent_support = bool(
        False  # Phase 3A froze identical mouse and keyboard boundaries.
        and np.median(action_vs_mouse) >= 0.15
        and np.mean(np.asarray(action_vs_mouse) >= 0.10) >= 0.75
    )
    if serving_success and policy_bit_exact:
        verdict = "SAME-BOUNDARY KEYBOARD COMMITMENT CONFIRMED"
    else:
        verdict = "PHASE 3B CONFIRMATION FAILED"

    summary = {
        "phase": "phase3b_same_boundary_confirmation",
        "context_count": len(contexts),
        "direction_count": len(directions),
        "valid_direction_count": len(valid_directions),
        "valid_direction_fraction": valid_fraction,
        "arrival_count": len(arrivals),
        "valid_arrival_count": len(valid_arrivals),
        "early_arrival_count": len(early),
        "late_arrival_count": len(late),
        "policies": policy_summary,
        "late": {
            "within_0_10_restart_fraction": float(np.mean(late_gap <= 0.10)),
            "fewer_redone_nfe_fraction": float(np.mean(fewer_redone)),
            "latency_saved_seconds": stats(latency_saved.tolist()),
            "mean_latency_saved_fraction": float(np.mean(latency_fraction)),
        },
        "early_zero_rollback_exact": early_zero,
        "quality": {
            "future_ssim_vs_new": stats(quality_ssim),
            "temporal_minus_new": stats(quality_temporal),
            "temporal_within_0_05_fraction": float(np.mean(np.asarray(quality_temporal) >= -0.05)),
            "pass": quality_pass,
        },
        "paired_direction_effect": {
            "lateact_minus_direct_mean": float(np.mean(improvement_by_direction)),
            "lateact_minus_direct_median": float(np.median(improvement_by_direction)),
            "bootstrap_mean_95_percent_ci": bootstrap_mean_ci(improvement_by_direction),
        },
        "action_specific_vs_mouse_boundary": {
            "mean_response_difference": float(np.mean(action_vs_mouse)),
            "median_response_difference": float(np.median(action_vs_mouse)),
            "fraction_improving_by_at_least_0_10": float(np.mean(np.asarray(action_vs_mouse) >= 0.10)),
            "policy_bit_exact": policy_bit_exact,
        },
        "criteria": criteria,
        "serving_success": serving_success,
        "action_dependent_commitment_supported": action_dependent_support,
        "verdict": verdict,
        "per_direction": directions,
        "per_arrival": arrivals,
        "gpu_devices": sorted({report["gpu"] for report in reports}),
        "peak_allocated_bytes": max(row["audit"]["peak_allocated_bytes"] for row in directions),
    }
    (args.input / "phase3b_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    make_montage(args.input, sorted(contexts)[0])
    print(json.dumps({key: value for key, value in summary.items() if key not in ("per_direction", "per_arrival")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
