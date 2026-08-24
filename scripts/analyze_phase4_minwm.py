#!/usr/bin/env python3
"""Analyze the frozen minWM commitment-curve replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


PREFIX_PIXEL_FRAMES = 61
DIRECTIONS = {
    "a_to_d": {0: "oracle_d", 1: "a_to_d_s1", 2: "a_to_d_s2", 3: "a_to_d_s3", 4: "oracle_a"},
    "d_to_a": {0: "oracle_a", 1: "d_to_a_s1", 2: "d_to_a_s2", 3: "d_to_a_s3", 4: "oracle_d"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
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


def make_montage(root: Path, scene: str) -> None:
    names = ["oracle_d", "a_to_d_s1", "a_to_d_s2", "a_to_d_s3", "oracle_a"]
    labels = ["NEW d", "a->d s1", "a->d s2", "a->d s3", "OLD a"]
    cells = []
    for name, label in zip(names, labels):
        frames = load_frames(root / scene / f"{name}.mp4")
        frame = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (416, 240))
        cv2.rectangle(frame, (0, 0), (150, 28), (0, 0, 0), -1)
        cv2.putText(frame, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(frame)
    cv2.imwrite(str(root / "phase4_minwm_curve_montage.png"), np.hstack(cells))


def main() -> None:
    args = parse_args()
    reports = [json.loads(path.read_text()) for path in sorted(args.input.glob("curve_scenes_*.json"))]
    reports = [report for report in reports if report.get("status") == "complete"]
    scenes = {}
    for report in reports:
        overlap = set(scenes) & set(report["scenes"])
        if overlap:
            raise RuntimeError(f"duplicate completed scenes: {sorted(overlap)}")
        scenes.update(report["scenes"])
    if len(scenes) != 8:
        raise RuntimeError(f"frozen replication requires eight complete scenes, found {len(scenes)}")

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
    for scene in ("scene_00", "scene_01"):
        hashes = [scenes[scene]["runs"][f"a_to_a_s{s}"]["latent_sha256"] for s in range(5)]
        scores = [metrics[scene][f"a_to_a_s{s}"]["motion"]["affine_center_signed_sum"] for s in range(5)]
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
        for direction, mapping in DIRECTIONS.items():
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
                audit_pass and abs(denominator) >= 8.0
                and abs(denominator) >= 10.0 * max(repeatability_floor, 1e-6)
                and support >= 20 and sign_agreement
            )
            values = [response[position] for position in range(5)]
            monotone = bool(valid and all(values[index + 1] <= values[index] + 0.10 for index in range(4)))
            bounded = bool(valid and all(-0.10 <= response[position] <= 1.10 for position in (1, 2, 3)))
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
        for direction in DIRECTIONS
    }
    phase4b_pass = bool(
        len(valid_rows) >= 12
        and sum(row["monotone"] for row in rows) >= 12
        and sum(row["bounded"] for row in rows) >= 12
        and max(adjacent_drops.values(), default=-np.inf) >= 0.15
        and all(count >= 6 for count in direction_valid.values())
        and all(row["visual_valid"] for row in rows)
        and all(control["latent_hashes_exact"] for control in controls)
    )

    rollback_boundary = None
    if phase4b_pass:
        candidates = []
        for position in range(3):
            current = by_position[str(position)]["median"]
            following = by_position[str(position + 1)]["median"]
            if current >= 0.80 and following <= 0.65 and current - following >= 0.15:
                candidates.append(position)
        rollback_boundary = max(candidates) if candidates else None
        if rollback_boundary is None:
            phase4b_pass = False

    summary = {
        "phase": "phase4_minwm_commitment_curve",
        "scene_count": len(scenes), "direction_count": len(rows),
        "valid_direction_count": len(valid_rows), "direction_valid_counts": direction_valid,
        "repeatability_floor": repeatability_floor, "same_action_controls": controls,
        "response_by_switch_after_nfe": by_position, "adjacent_median_drops": adjacent_drops,
        "monotone_count": sum(row["monotone"] for row in rows),
        "bounded_count": sum(row["bounded"] for row in rows),
        "visual_valid_count": sum(row["visual_valid"] for row in rows),
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
    (args.input / "phase4_minwm_curve_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    make_montage(args.input, "scene_00")
    compact = {key: value for key, value in summary.items() if key not in ("per_direction", "per_run_metrics")}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
