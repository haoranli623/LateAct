#!/usr/bin/env python3
"""Analyze the frozen Phase 3A keyboard commitment calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DIRECTION_RUNS = {
    "left_to_right": {
        0: "oracle_keyboard_right",
        1: "lr_s1",
        2: "lr_s2",
        3: "oracle_keyboard_left",
    },
    "right_to_left": {
        0: "oracle_keyboard_left",
        1: "rl_s1",
        2: "rl_s2",
        3: "oracle_keyboard_right",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p05": float(np.quantile(data, 0.05)),
        "p95": float(np.quantile(data, 0.95)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def bootstrap_median_ci(values: list[float], seed: int) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.median(rng.choice(data, size=(10000, len(data)), replace=True), axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def load(root: Path) -> tuple[dict, list[dict]]:
    reports = [json.loads(path.read_text()) for path in sorted(root.glob("phase3a_shard_*.json"))]
    if len(reports) != 2 or any(report["status"] != "complete" for report in reports):
        raise RuntimeError("Phase 3A requires two complete shards")
    contexts = {}
    for report in reports:
        overlap = set(contexts) & set(report["contexts"])
        if overlap:
            raise RuntimeError(f"duplicate contexts: {sorted(overlap)}")
        contexts.update(report["contexts"])
    if len(contexts) != 10:
        raise RuntimeError(f"expected 10 calibration contexts, found {len(contexts)}")
    return contexts, reports


def make_montage(root: Path, context: str) -> None:
    names = ["oracle_keyboard_left", "lr_s1", "lr_s2", "oracle_keyboard_right"]
    labels = ["LEFT ORACLE", "L->R AFTER 1", "L->R AFTER 2", "RIGHT ORACLE"]
    cells = []
    for name, label in zip(names, labels):
        capture = cv2.VideoCapture(str(root / context / f"{name}.mp4"))
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
        cv2.rectangle(frame, (0, 0), (190, 25), (0, 0, 0), -1)
        cv2.putText(frame, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(frame)
    cv2.imwrite(str(root / "phase3a_keyboard_curve_montage.png"), np.hstack(cells))


def main() -> None:
    args = parse_args()
    contexts, reports = load(args.input)
    rows = []
    context_audits = []
    for context_name, context in sorted(contexts.items()):
        runs = context["runs"]
        checks = context["engineering_checks"]
        audit_pass = all(bool(value) for value in checks.values())
        context_audits.append(audit_pass)
        left_score = runs["oracle_keyboard_left"]["lateral_translation"]["affine_center_signed_sum"]
        right_score = runs["oracle_keyboard_right"]["lateral_translation"]["affine_center_signed_sum"]
        semantic_direction = left_score > right_score
        for direction, mapping in DIRECTION_RUNS.items():
            selected = {position: runs[name] for position, name in mapping.items()}
            raw = {
                position: value["lateral_translation"]["affine_center_signed_sum"]
                for position, value in selected.items()
            }
            raw_stability = {
                position: value["lateral_translation"]["median_lk_signed_sum"]
                for position, value in selected.items()
            }
            denominator = raw[0] - raw[3]
            response = {
                position: (raw[position] - raw[3]) / denominator if denominator != 0 else None
                for position in range(4)
            }
            oracle_sign_agreement = (
                np.sign(raw[0] - raw[3]) == np.sign(raw_stability[0] - raw_stability[3])
            )
            inlier_support = min(
                selected[0]["lateral_translation"]["median_inlier_count"],
                selected[3]["lateral_translation"]["median_inlier_count"],
            )
            valid = bool(
                audit_pass
                and abs(denominator) >= 2.0
                and inlier_support >= 20
                and oracle_sign_agreement
                and semantic_direction
            )
            values = [response[position] for position in range(4)]
            monotone = bool(valid and all(values[index + 1] <= values[index] + 0.10 for index in range(3)))
            bounded = bool(valid and all(-0.10 <= response[position] <= 1.10 for position in (1, 2)))
            oracle_temporal_floor = min(
                selected[0]["temporal_consistency_ssim"],
                selected[3]["temporal_consistency_ssim"],
            )
            oracle_jump_ceiling = max(selected[0]["boundary_jump"], selected[3]["boundary_jump"])
            switch_quality = {
                position: {
                    "temporal_deficit": selected[position]["temporal_consistency_ssim"] - oracle_temporal_floor,
                    "boundary_jump_excess": selected[position]["boundary_jump"] - oracle_jump_ceiling,
                    "pass": bool(
                        selected[position]["decode_valid"]
                        and selected[position]["temporal_consistency_ssim"] >= oracle_temporal_floor - 0.05
                        and selected[position]["boundary_jump"] <= oracle_jump_ceiling + 10.0
                    ),
                }
                for position in (1, 2)
            }
            rows.append({
                "context": context_name,
                "direction": direction,
                "valid": valid,
                "semantic_direction_consistent": semantic_direction,
                "oracle_sign_agreement": bool(oracle_sign_agreement),
                "oracle_gap": denominator,
                "median_oracle_inlier_support": inlier_support,
                "raw_affine_response": raw,
                "raw_median_lk_response": raw_stability,
                "response_retention": response,
                "monotone": monotone,
                "bounded": bounded,
                "switch_quality": switch_quality,
            })

    valid_rows = [row for row in rows if row["valid"]]
    valid_fraction = len(valid_rows) / len(rows)
    if not valid_rows:
        raise RuntimeError("no evaluator-valid Phase 3A directions")
    by_position = {}
    for position in range(4):
        values = [row["response_retention"][position] for row in valid_rows]
        by_position[str(position)] = {
            **stats(values),
            "bootstrap_median_95_percent_ci": bootstrap_median_ci(values, 20260824 + position),
        }
    monotone_fraction = float(np.mean([row["monotone"] for row in valid_rows]))
    bounded_fraction = float(np.mean([row["bounded"] for row in valid_rows]))
    quality_values = [
        row["switch_quality"][position]["pass"]
        for row in valid_rows for position in (1, 2)
    ]
    quality_fraction = float(np.mean(quality_values))
    structured = bool(
        valid_fraction >= 0.80
        and monotone_fraction >= 0.80
        and bounded_fraction >= 0.80
        and quality_fraction >= 0.90
    )

    safe_positions = []
    safe_audits = {}
    for position in (0, 1, 2):
        values = np.asarray([row["response_retention"][position] for row in valid_rows])
        median = float(np.median(values))
        ci = by_position[str(position)]["bootstrap_median_95_percent_ci"]
        near_fraction = float(np.mean(values >= 0.80))
        safe = bool(median >= 0.90 and ci[0] >= 0.80 and near_fraction >= 0.80)
        safe_audits[str(position)] = {
            "median": median,
            "bootstrap_lower": ci[0],
            "individual_near_oracle_fraction": near_fraction,
            "safe": safe,
        }
        if safe:
            safe_positions.append(position)
    candidate = max(safe_positions) if safe_positions else None
    boundary_audit = None
    stable_boundary = False
    if candidate is not None and candidate < 3:
        current = np.asarray([row["response_retention"][candidate] for row in valid_rows])
        following = np.asarray([row["response_retention"][candidate + 1] for row in valid_rows])
        drop = float(np.median(current) - np.median(following))
        bracket = float(np.mean((current >= 0.80) & (following <= 0.80)))
        boundary_audit = {
            "candidate_latest_safe_switch_after_nfe": candidate,
            "next_median": float(np.median(following)),
            "median_drop": drop,
            "individual_bracketing_fraction": bracket,
        }
        stable_boundary = bool(
            structured
            and np.median(following) <= 0.75
            and drop >= 0.15
            and bracket >= 0.75
        )

    intermediate_progress = max(
        by_position[str(position)]["median"] - by_position[str(position + 1)]["median"]
        for position in (0, 1)
    ) >= 0.15
    if stable_boundary and candidate != 1:
        verdict_code = "A"
        verdict = "STRUCTURED ACTION-DEPENDENT COMMITMENT"
    elif stable_boundary and candidate == 1:
        verdict_code = "B"
        verdict = "SAME BOUNDARY"
    elif valid_fraction >= 0.80 and quality_fraction >= 0.90 and intermediate_progress:
        verdict_code = "C"
        verdict = "WEAK / UNSTABLE COMMITMENT"
    else:
        verdict_code = "D"
        verdict = "NO-GO FOR KEYBOARD"

    direction_medians = {
        direction: {
            str(position): float(np.median([
                row["response_retention"][position]
                for row in valid_rows if row["direction"] == direction
            ]))
            for position in range(4)
        }
        for direction in DIRECTION_RUNS
    }
    summary = {
        "phase": "phase3a_keyboard_commitment_calibration",
        "context_count": len(contexts),
        "direction_count": len(rows),
        "valid_direction_count": len(valid_rows),
        "valid_direction_fraction": valid_fraction,
        "context_state_audits_pass": all(context_audits),
        "metric": "robust_affine_center_lateral_translation",
        "response_by_switch_after_nfe": by_position,
        "direction_median_response": direction_medians,
        "monotone_fraction": monotone_fraction,
        "bounded_fraction": bounded_fraction,
        "quality_pass_fraction": quality_fraction,
        "structured_curve": structured,
        "safe_position_audits": safe_audits,
        "boundary_audit": boundary_audit,
        "stable_boundary": stable_boundary,
        "keyboard_latest_safe_switch_after_nfe": candidate if stable_boundary else None,
        "mouse_latest_safe_switch_after_nfe": 1,
        "verdict_code": verdict_code,
        "verdict": verdict,
        "phase3b_authorized": verdict_code in ("A", "B"),
        "per_direction": rows,
        "gpu_devices": sorted({report["gpu"] for report in reports}),
        "peak_allocated_bytes": max(
            run["peak_allocated_bytes"]
            for context in contexts.values() for run in context["runs"].values()
        ),
    }
    (args.input / "phase3a_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    make_montage(args.input, sorted(contexts)[0])
    print(json.dumps({key: value for key, value in summary.items() if key != "per_direction"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
