#!/usr/bin/env python3
"""Verify the central LateAct conclusions from archived JSON only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, *, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"expected {expected}, found {actual}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--generated-robustness", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo.resolve()
    frozen = root / "results" / "frozen"

    json_files = sorted(frozen.rglob("*.json"))
    if len(json_files) != 27:
        raise AssertionError(f"expected 27 frozen JSON files, found {len(json_files)}")
    montages = sorted(frozen.rglob("*montage.png"))
    if len(montages) != 9:
        raise AssertionError(f"expected 9 frozen montages, found {len(montages)}")

    gate0 = load(frozen / "gate0" / "gate0_summary.json")
    assert gate0["verdict"] == "B. COMMITMENT-CURVE STRONG GO"
    assert gate0["valid_direction_count"] == 16
    curves = [item for scene in gate0["curves"].values() for item in scene.values()]
    close(median(item["new_action_fraction_s0_to_s3"][1] for item in curves), 0.997053723726691)
    close(median(item["new_action_fraction_s0_to_s3"][2] for item in curves), 0.21823602063469516)

    gate1 = load(frozen / "gate1" / "gate1_summary.json")
    assert gate1["verdict"] == "A. MINIMAL-ROLLBACK STRONG GO"
    close(gate1["response"]["direct"]["median"], 0.21823602063469516)
    close(gate1["response"]["minimal_rollback"]["median"], 0.997053723726691)
    assert gate1["response"]["near_full_restart_count"] == 16

    gate2 = load(frozen / "gate2" / "primary" / "mouse_summary.json")
    assert gate2["verdict"] == "STRONG PROJECT RESULT"
    assert gate2["arrival_case_count"] == 1024
    assert gate2["late_arrivals"]["case_count"] == 676
    assert gate2["early_arrivals"]["case_count"] == 348
    assert gate2["early_arrivals"]["zero_rollback_exact"] is True
    close(gate2["policies"]["lateact"]["response"]["median"], 0.999913300464828)
    close(gate2["late_arrivals"]["mean_latency_saved_fraction"], 0.11928583948989827)
    assert gate2["checkpoint"]["gate2_optimized_exact_bytes"]["total"] == 339360
    close(gate2["quality"]["lateact_vs_new_future_ssim"]["median"], 0.9223045246171051)

    phase3a = load(frozen / "phase3" / "calibration" / "phase3a_summary.json")
    assert phase3a["verdict"] == "SAME BOUNDARY"
    assert phase3a["keyboard_latest_safe_switch_after_nfe"] == 1
    close(phase3a["response_by_switch_after_nfe"]["1"]["median"], 0.9956907883528071)
    close(phase3a["response_by_switch_after_nfe"]["2"]["median"], 0.1240237601161071)

    phase3b = load(frozen / "phase3" / "confirmation" / "phase3b_summary.json")
    assert phase3b["verdict"] == "PHASE 3B CONFIRMATION FAILED"
    assert phase3b["late_arrival_count"] == 512
    assert phase3b["early_arrival_count"] == 256
    close(phase3b["policies"]["action_specific_lateact"]["response"]["median"], 1.0187847734762792)
    close(phase3b["quality"]["future_ssim_vs_new"]["median"], 0.672600241089411)
    assert phase3b["quality"]["pass"] is False

    minwm_ad = load(frozen / "phase4_minwm_ad" / "phase4_minwm_curve_summary.json")
    assert minwm_ad["direction_count"] == 16
    assert minwm_ad["valid_direction_count"] == 6
    assert minwm_ad["phase4b_pass"] is False

    minwm_yaw = load(
        frozen / "phase4_minwm_yaw" / "phase4_minwm_yaw_confirmatory_summary.json"
    )
    assert minwm_yaw["direction_count"] == 16
    assert minwm_yaw["valid_direction_count"] == 16
    assert minwm_yaw["phase4b_pass"] is True
    close(minwm_yaw["response_by_switch_after_nfe"]["1"]["median"], 0.14502679509494082)

    robustness_path = args.generated_robustness or (
        root / "robustness" / "gate2_mouse_clustered" / "robustness_summary.json"
    )
    robustness = load(robustness_path)
    response = robustness["response"]["base_image_cluster_inference_primary"]
    latency = robustness["latency"]["base_image_cluster_inference_primary"]
    quality = robustness["quality"]["future_ssim"]
    assert response["n"] == latency["n"] == 8
    close(response["mean"], 0.9091360548028863)
    close(response["mean_percentile_95_ci"][0], 0.85976894837631)
    close(response["mean_percentile_95_ci"][1], 0.9540934386765971)
    close(latency["mean"], 0.3117885655956343)
    close(latency["mean_percentile_95_ci"][0], 0.3055191419989569)
    close(latency["mean_percentile_95_ci"][1], 0.31752334906195756)
    close(quality["median"], 0.9223045246171051)

    pdf = root / "paper" / "LateAct_NeurIPS2026_Workshop.pdf"
    expected_pdf = "3b6eb37388b6bc62120a8f11e910b1d384ff1ec3abde429ef3ccabef73cae5b7"
    if sha256(pdf) != expected_pdf:
        raise AssertionError("submitted PDF SHA256 mismatch")

    print("PASS: frozen LateAct cardinalities, effects, scope, and PDF checksum verified")
    print("Matrix-Game: 1024 arrivals; 676 late; 348 early; checkpoint 339360 bytes")
    print("Clustered inference: response n=8 mean 0.909136; latency saving 0.311789 s")
    print("Keyboard: shared boundary, response recovered, future SSIM 0.672600 < 0.90")
    print("minWM: a/d 6/16 failed; frozen j/l yaw 16/16 mechanism replication")


if __name__ == "__main__":
    main()
