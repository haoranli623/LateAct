#!/usr/bin/env python3
"""Analyze the frozen minWM commitment-curve replication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from phase4_minwm_protocol import (
    PROJECT,
    actions,
    control_prompt_indices,
    direction_mappings,
    load_and_validate_prompts,
    load_protocol,
    prompt_indices,
    seed_by_prompt,
    switch_positions,
    unique_run_specs,
    validate_confirmatory_output,
)


PREFIX_PIXEL_FRAMES = 61
DEFAULT_CONFIG = PROJECT / "config" / "phase4_minwm.yaml"
UPSTREAM = Path(os.environ.get("LATEACT_MINWM_UPSTREAM", PROJECT / "third_party" / "minWM"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config, prompt identity, expected runs, and input isolation only",
    )
    return parser.parse_args()


def load_frames(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != 77:
        raise RuntimeError(f"expected 77 frames in {path}, found {len(frames)}")
    return np.asarray(frames)


def lateral_translation(frames: np.ndarray) -> dict:
    gray = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (416, 240))
        for frame in frames[PREFIX_PIXEL_FRAMES - 1:]
    ]
    affine_dx = []
    median_dx = []
    tracked_counts = []
    inlier_counts = []
    center = np.array([208.0, 120.0, 1.0], dtype=np.float64)
    for previous, current in zip(gray, gray[1:]):
        points = cv2.goodFeaturesToTrack(
            previous, maxCorners=900, qualityLevel=0.01, minDistance=5, blockSize=7
        )
        if points is None:
            affine_dx.append(float("nan")); median_dx.append(float("nan"))
            tracked_counts.append(0); inlier_counts.append(0)
            continue
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, points, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if moved is None or status is None:
            affine_dx.append(float("nan")); median_dx.append(float("nan"))
            tracked_counts.append(0); inlier_counts.append(0)
            continue
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, moved, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if backward is None or backward_status is None:
            affine_dx.append(float("nan")); median_dx.append(float("nan"))
            tracked_counts.append(0); inlier_counts.append(0)
            continue
        round_trip = np.linalg.norm((backward - points).reshape(-1, 2), axis=1)
        valid = (
            status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.isfinite(round_trip)
            & (round_trip < 1.5)
        )
        first = points.reshape(-1, 2)[valid]
        second = moved.reshape(-1, 2)[valid]
        tracked_counts.append(int(len(first)))
        median_dx.append(float(np.median(second[:, 0] - first[:, 0])) if len(first) else float("nan"))
        matrix, inliers = (None, None)
        if len(first) >= 3:
            matrix, inliers = cv2.estimateAffinePartial2D(
                first, second, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                maxIters=2000, confidence=0.99, refineIters=10,
            )
        if matrix is None or inliers is None:
            affine_dx.append(float("nan")); inlier_counts.append(0)
        else:
            affine_dx.append(float((matrix @ center)[0] - center[0]))
            inlier_counts.append(int(inliers.sum()))
    return {
        "affine_center_signed_sum": float(np.nansum(affine_dx)),
        "median_lk_signed_sum": float(np.nansum(median_dx)),
        "affine_per_transition": affine_dx,
        "median_lk_per_transition": median_dx,
        "tracked_counts": tracked_counts,
        "inlier_counts": inlier_counts,
        "median_inlier_count": float(np.median(inlier_counts)),
    }


def temporal_quality(frames: np.ndarray) -> dict:
    segment = frames[PREFIX_PIXEL_FRAMES - 1:]
    temporal = float(np.mean([
        structural_similarity(a, b, data_range=255, channel_axis=2)
        for a, b in zip(segment[:-1], segment[1:])
    ]))
    jump = float(np.mean(np.abs(
        frames[PREFIX_PIXEL_FRAMES].astype(np.float32)
        - frames[PREFIX_PIXEL_FRAMES - 1].astype(np.float32)
    )))
    return {
        "temporal_consistency_ssim": temporal,
        "boundary_jump": jump,
        "decode_valid": bool(np.isfinite(frames).all() and np.std(frames) > 0),
    }


def stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)), "median": float(np.median(data)),
        "min": float(np.min(data)), "max": float(np.max(data)),
        "p05": float(np.quantile(data, 0.05)), "p95": float(np.quantile(data, 0.95)),
    }


def make_montage(
    root: Path,
    scene: str,
    protocol: dict,
    filename: str,
) -> None:
    negative, positive = actions(protocol)
    names = [
        f"oracle_{positive}",
        f"{negative}_to_{positive}_s1",
        f"{negative}_to_{positive}_s2",
        f"{negative}_to_{positive}_s3",
        f"oracle_{negative}",
    ]
    labels = [
        f"NEW {positive}",
        f"{negative}->{positive} s1",
        f"{negative}->{positive} s2",
        f"{negative}->{positive} s3",
        f"OLD {negative}",
    ]
    cells = []
    for name, label in zip(names, labels):
        frames = load_frames(root / scene / f"{name}.mp4")
        frame = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (416, 240))
        cv2.rectangle(frame, (0, 0), (150, 28), (0, 0, 0), -1)
        cv2.putText(frame, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(frame)
    cv2.imwrite(str(root / filename), np.hstack(cells))


def validate_reports(
    reports: list[dict],
    scenes: dict,
    protocol: dict,
    prompts: list[str],
) -> None:
    expected_indices = prompt_indices(protocol)
    expected_scenes = {f"scene_{index:02d}" for index in expected_indices}
    if set(scenes) != expected_scenes:
        raise RuntimeError(
            f"expected frozen scenes {sorted(expected_scenes)}, found {sorted(scenes)}"
        )
    negative, positive = actions(protocol)
    seed_map = seed_by_prompt(protocol)
    control_indices = set(control_prompt_indices(protocol))
    base_specs = unique_run_specs(protocol)
    base_runs = {name for name, _, _, _ in base_specs}
    expected_metadata = {name: (old, new, switch) for name, old, new, switch in base_specs}
    for report in reports:
        if protocol.get("phase") == "phase4_minwm_yaw_confirmatory":
            expected_top = {
                "phase": protocol["phase"],
                "protocol_config_sha256": protocol["_config_sha256"],
                "negative_action": negative,
                "positive_action": positive,
                "trajectory_steps_per_branch": int(protocol["control"]["trajectory_steps_per_branch"]),
                "switch_positions": switch_positions(protocol),
                "same_action_control_prompt_indices": sorted(control_indices),
            }
            for key, expected in expected_top.items():
                if report.get(key) != expected:
                    raise RuntimeError(
                        f"report/config mismatch for {key}: expected {expected!r}, got {report.get(key)!r}"
                    )
            if set(report.get("camera_viewmat_hashes", {})) != {negative, positive}:
                raise RuntimeError("report camera tensors do not match the frozen action pair")
    for index in expected_indices:
        scene_name = f"scene_{index:02d}"
        scene = scenes[scene_name]
        if scene.get("prompt") != prompts[index] or int(scene.get("seed")) != seed_map[index]:
            raise RuntimeError(f"prompt or seed substitution detected in {scene_name}")
        expected_runs = set(base_runs)
        if index in control_indices:
            for switch in switch_positions(protocol):
                name = f"{negative}_to_{negative}_s{switch}"
                expected_runs.add(name)
                expected_metadata[name] = (negative, negative, switch)
        if set(scene["runs"]) != expected_runs:
            raise RuntimeError(
                f"action/run mismatch in {scene_name}: expected {sorted(expected_runs)}, "
                f"found {sorted(scene['runs'])}"
            )
        for run_name, run in scene["runs"].items():
            expected = expected_metadata[run_name]
            actual = (run.get("old_action"), run.get("new_action"), int(run.get("switch_after")))
            if actual != expected:
                raise RuntimeError(
                    f"action metadata mismatch for {scene_name}/{run_name}: {actual} != {expected}"
                )


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.config)
    prompts = load_and_validate_prompts(UPSTREAM, protocol)
    validate_confirmatory_output(args.input, protocol, require_empty=False)
    directions = direction_mappings(protocol)
    expected_scene_names = [f"scene_{index:02d}" for index in prompt_indices(protocol)]
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_only_no_artifact_analysis",
            "config": protocol["_config_path"],
            "config_sha256": protocol["_config_sha256"],
            "input": str(args.input.resolve()),
            "actions": list(actions(protocol)),
            "expected_scenes": expected_scene_names,
            "expected_directions": list(directions),
            "control_prompt_indices": control_prompt_indices(protocol),
            "thresholds": {
                "min_oracle_gap_pixels": protocol["validity"]["min_oracle_gap_pixels"],
                "same_action_floor_multiplier": protocol["validity"]["same_action_floor_multiplier"],
                "min_tracks_per_transition": protocol["evaluator"]["min_tracks_per_transition"],
                "min_valid_direction_scene_pairs": protocol["phase4b_pass"]["min_valid_direction_scene_pairs"],
                "min_monotone_pairs": protocol["phase4b_pass"]["min_monotone_pairs"],
                "min_bounded_pairs": protocol["phase4b_pass"]["min_bounded_pairs"],
                "min_median_adjacent_drop": protocol["phase4b_pass"]["min_median_adjacent_drop"],
                "min_directions_each_orientation": protocol["phase4b_pass"]["min_directions_each_orientation"],
            },
        }, indent=2, sort_keys=True))
        return
    reports = [json.loads(path.read_text()) for path in sorted(args.input.glob("curve_scenes_*.json"))]
    reports = [report for report in reports if report.get("status") == "complete"]
    scenes = {}
    for report in reports:
        overlap = set(scenes) & set(report["scenes"])
        if overlap:
            raise RuntimeError(f"duplicate completed scenes: {sorted(overlap)}")
        scenes.update(report["scenes"])
    validate_reports(reports, scenes, protocol, prompts)

    metrics = {}
    for scene, value in sorted(scenes.items()):
        metrics[scene] = {}
        for run_name in value["runs"]:
            frames = load_frames(args.input / scene / f"{run_name}.mp4")
            metrics[scene][run_name] = {
                "motion": lateral_translation(frames),
                "quality": temporal_quality(frames),
            }

    controls = []
    negative, _ = actions(protocol)
    controls_scenes = [f"scene_{index:02d}" for index in control_prompt_indices(protocol)]
    for scene in controls_scenes:
        hashes = [
            scenes[scene]["runs"][f"{negative}_to_{negative}_s{s}"]["latent_sha256"]
            for s in switch_positions(protocol)
        ]
        scores = [
            metrics[scene][f"{negative}_to_{negative}_s{s}"]["motion"]["affine_center_signed_sum"]
            for s in switch_positions(protocol)
        ]
        controls.append({
            "scene": scene,
            "latent_hashes_exact": len(set(hashes)) == 1,
            "score_range": float(max(scores) - min(scores)),
            "scores": scores,
        })
    repeatability_floor = max(control["score_range"] for control in controls)

    rows = []
    for scene, value in sorted(scenes.items()):
        audit = value["engineering_checks"]
        audit_pass = all(item is True for item in audit.values() if item is not None)
        for direction, mapping in directions.items():
            primary = {
                position: metrics[scene][run]["motion"]["affine_center_signed_sum"]
                for position, run in mapping.items()
            }
            stability = {
                position: metrics[scene][run]["motion"]["median_lk_signed_sum"]
                for position, run in mapping.items()
            }
            denominator = primary[0] - primary[4]
            response = {
                position: ((primary[position] - primary[4]) / denominator if denominator else None)
                for position in range(5)
            }
            support = min(
                metrics[scene][mapping[0]]["motion"]["median_inlier_count"],
                metrics[scene][mapping[4]]["motion"]["median_inlier_count"],
            )
            sign_agreement = np.sign(denominator) == np.sign(stability[0] - stability[4])
            valid = bool(
                audit_pass
                and abs(denominator) >= float(protocol["validity"]["min_oracle_gap_pixels"])
                and abs(denominator) >= float(protocol["validity"]["same_action_floor_multiplier"]) * max(repeatability_floor, 1e-6)
                and support >= int(protocol["evaluator"]["min_tracks_per_transition"])
                and sign_agreement
            )
            values = [response[position] for position in range(5)]
            tolerance = float(protocol["phase4b_pass"]["monotonic_tolerance"])
            bounded_min, bounded_max = [float(value) for value in protocol["phase4b_pass"]["bounded_range"]]
            monotone = bool(valid and all(values[index + 1] <= values[index] + tolerance for index in range(4)))
            bounded = bool(valid and all(bounded_min <= response[position] <= bounded_max for position in (1, 2, 3)))
            quality = all(metrics[scene][mapping[position]]["quality"]["decode_valid"] for position in range(5))
            rows.append({
                "scene": scene, "direction": direction, "valid": valid,
                "oracle_gap": denominator, "oracle_sign_agreement": bool(sign_agreement),
                "median_oracle_inlier_support": support, "raw_response": primary,
                "raw_stability_response": stability, "response_retention": response,
                "monotone": monotone, "bounded": bounded, "visual_valid": quality,
            })

    valid_rows = [row for row in rows if row["valid"]]
    by_position = {
        str(position): stats([row["response_retention"][position] for row in valid_rows])
        for position in range(5)
    } if valid_rows else {}
    adjacent_drops = {
        f"{position}_to_{position + 1}": (
            by_position[str(position)]["median"] - by_position[str(position + 1)]["median"]
        ) for position in range(4)
    } if valid_rows else {}
    direction_valid = {
        direction: sum(row["valid"] for row in rows if row["direction"] == direction)
        for direction in directions
    }
    gate = protocol["phase4b_pass"]
    all_audits_pass = all(
        all(item is True for item in scene["engineering_checks"].values() if item is not None)
        for scene in scenes.values()
    )
    phase4b_pass = bool(
        len(valid_rows) >= int(gate["min_valid_direction_scene_pairs"])
        and sum(row["monotone"] for row in rows) >= int(gate["min_monotone_pairs"])
        and sum(row["bounded"] for row in rows) >= int(gate["min_bounded_pairs"])
        and max(adjacent_drops.values(), default=-np.inf) >= float(gate["min_median_adjacent_drop"])
        and all(count >= int(gate["min_directions_each_orientation"]) for count in direction_valid.values())
        and sum(row["visual_valid"] for row in rows) >= int(gate.get("required_visual_valid_pairs", len(rows)))
        and all(row["visual_valid"] for row in rows)
        and all(control["latent_hashes_exact"] for control in controls)
        and (
            all_audits_pass
            if bool(gate.get("require_stochastic_noise_cache_audits", True))
            else True
        )
    )

    rollback_boundary = None
    if phase4b_pass:
        candidates = []
        for position in range(3):
            current = by_position[str(position)]["median"]
            following = by_position[str(position + 1)]["median"]
            rollback = protocol.get("rollback", {})
            if (
                current >= float(rollback.get("boundary_min_current_response", 0.80))
                and following <= float(rollback.get("boundary_max_following_response", 0.65))
                and current - following >= float(rollback.get("boundary_min_adjacent_drop", 0.15))
            ):
                candidates.append(position)
        rollback_boundary = max(candidates) if candidates else None
        if rollback_boundary is None:
            phase4b_pass = False

    summary = {
        "phase": protocol["phase"],
        "protocol_config": protocol["_config_path"],
        "protocol_config_sha256": protocol["_config_sha256"],
        "scene_count": len(scenes), "direction_count": len(rows),
        "valid_direction_count": len(valid_rows), "direction_valid_counts": direction_valid,
        "repeatability_floor": repeatability_floor, "same_action_controls": controls,
        "response_by_switch_after_nfe": by_position, "adjacent_median_drops": adjacent_drops,
        "monotone_count": sum(row["monotone"] for row in rows),
        "bounded_count": sum(row["bounded"] for row in rows),
        "visual_valid_count": sum(row["visual_valid"] for row in rows),
        "all_engineering_audits_pass": all_audits_pass,
        "phase4b_pass": phase4b_pass, "rollback_boundary_after_nfe": rollback_boundary,
        "per_direction": rows, "per_run_metrics": metrics,
        "peak_allocated_bytes": max(
            run["peak_allocated_bytes"]
            for scene in scenes.values() for run in scene["runs"].values()
        ),
        "mean_branch_generation_seconds": float(np.mean([
            run["generation_seconds"] for scene in scenes.values() for run in scene["runs"].values()
        ])),
        "mean_decode_seconds": float(np.mean([
            run["decode_seconds"] for scene in scenes.values() for run in scene["runs"].values()
        ])),
    }
    summary_filename = protocol.get("analysis", {}).get(
        "summary_filename", "phase4_minwm_curve_summary.json"
    )
    montage_filename = protocol.get("analysis", {}).get(
        "montage_filename", "phase4_minwm_curve_montage.png"
    )
    (args.input / summary_filename).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    make_montage(args.input, controls_scenes[0], protocol, montage_filename)
    compact = {key: value for key, value in summary.items() if key not in ("per_direction", "per_run_metrics")}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
