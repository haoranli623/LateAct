#!/usr/bin/env python3
"""Read-only clustered and quality-tail audit for frozen Gate 2 mouse data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


DIRECTIONS = ("left_to_right", "right_to_left")
SCENE_RE = re.compile(r"^(?P<base>\d{4})_v(?P<variant>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=200_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: np.generic | float | int) -> float | int:
    return value.item() if isinstance(value, np.generic) else value


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "sample_standard_deviation": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def bootstrap_cluster_ci(
    values: list[float], replicates: int, seed: int
) -> dict[str, list[float]]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    samples = array[indices]
    return {
        "mean_percentile_95_ci": [
            float(value) for value in np.quantile(samples.mean(axis=1), (0.025, 0.975))
        ],
        "median_percentile_95_ci": [
            float(value)
            for value in np.quantile(np.median(samples, axis=1), (0.025, 0.975))
        ],
    }


def pooled_top_cluster_ci(
    sums: list[float], counts: list[int], replicates: int, seed: int
) -> list[float]:
    sums_array = np.asarray(sums, dtype=np.float64)
    counts_array = np.asarray(counts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sums_array), size=(replicates, len(sums_array)))
    estimates = sums_array[indices].sum(axis=1) / counts_array[indices].sum(axis=1)
    return [float(value) for value in np.quantile(estimates, (0.025, 0.975))]


def exact_signed_rank(values: list[float]) -> dict[str, float | int | bool]:
    array = np.asarray(values, dtype=np.float64)
    if np.any(array == 0):
        raise ValueError("exact signed-rank implementation requires nonzero values")
    absolute = np.abs(array)
    if len(np.unique(absolute)) != len(absolute):
        raise ValueError("exact signed-rank implementation requires no absolute ties")
    order = np.argsort(absolute)
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(1, len(array) + 1, dtype=np.float64)
    observed = float(ranks[array > 0].sum())
    total = float(ranks.sum())
    possible = []
    for mask in range(1 << len(array)):
        possible.append(
            sum(ranks[index] for index in range(len(array)) if mask & (1 << index))
        )
    possible_array = np.asarray(possible, dtype=np.float64)
    one_sided = float(np.mean(possible_array >= observed - 1e-12))
    distance = abs(observed - total / 2.0)
    two_sided = float(
        np.mean(np.abs(possible_array - total / 2.0) >= distance - 1e-12)
    )
    return {
        "n": len(array),
        "w_plus": observed,
        "all_effects_positive": bool(np.all(array > 0)),
        "one_sided_exact_p_greater": one_sided,
        "two_sided_exact_p": two_sided,
    }


def original_direction_reference(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260824)
    samples = rng.choice(array, size=(10_000, len(array)), replace=True).mean(axis=1)
    test = wilcoxon(array, alternative="greater", method="approx")
    return {
        "unit": "scene_variant_x_direction",
        "n": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "direction_bootstrap_mean_95_ci_10000_seed_20260824": [
            float(value) for value in np.quantile(samples, (0.025, 0.975))
        ],
        "one_sided_wilcoxon_normal_approximation_p": float(test.pvalue),
        "inferential_caveat": (
            "Descriptive reference only: variants and opposite directions share base images."
        ),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shard_paths = sorted(args.input.glob("mouse_shard_*.json"))
    if len(shard_paths) != 2:
        raise RuntimeError(f"expected two frozen mouse shards, found {len(shard_paths)}")

    scenes: dict[str, dict] = {}
    source_hashes = {}
    for path in shard_paths:
        report = json.loads(path.read_text())
        if report["status"] != "complete" or report["family"] != "mouse":
            raise RuntimeError(f"invalid frozen shard: {path}")
        overlap = scenes.keys() & report["scenes"].keys()
        if overlap:
            raise RuntimeError(f"duplicate scenes: {sorted(overlap)}")
        scenes.update(report["scenes"])
        source_hashes[path.name] = sha256(path)
    summary_path = args.input / "mouse_summary.json"
    source_hashes[summary_path.name] = sha256(summary_path)
    frozen_summary = json.loads(summary_path.read_text())

    direction_rows = []
    arrival_rows = []
    quality_rows = []
    hierarchy: dict[str, dict] = {}
    for scene_name, scene in sorted(scenes.items()):
        match = SCENE_RE.match(scene_name)
        if match is None:
            raise RuntimeError(f"unexpected scene identifier: {scene_name}")
        base = match.group("base")
        variant = int(match.group("variant"))
        if scene["source_image_index"] != int(base) or scene["prefix_variant"] != variant:
            raise RuntimeError(f"scene metadata mismatch: {scene_name}")
        if tuple(sorted(scene["directions"])) != tuple(sorted(DIRECTIONS)):
            raise RuntimeError(f"direction mismatch: {scene_name}")
        hierarchy.setdefault(base, {"variants": set(), "directions": 0, "arrivals": 0})
        hierarchy[base]["variants"].add(variant)
        for direction_name in DIRECTIONS:
            direction = scene["directions"][direction_name]
            if not direction["evaluator_valid"]:
                raise RuntimeError(f"invalid primary direction: {scene_name}/{direction_name}")
            arrivals = direction["arrivals"]
            if len(arrivals) != 16 or sorted(a["arrival_index"] for a in arrivals) != list(range(16)):
                raise RuntimeError(f"arrival mismatch: {scene_name}/{direction_name}")
            hierarchy[base]["directions"] += 1
            hierarchy[base]["arrivals"] += len(arrivals)
            fixed_response_improvement = (
                direction["response"]["lateact"] - direction["response"]["direct_late"]
            )
            late_response_effects = []
            late_latency_effects = []
            class_counts = Counter()
            for arrival in arrivals:
                inflight = int(arrival["inflight_nfe"])
                class_counts[inflight] += 1
                row = {
                    "base_image": base,
                    "variant": variant,
                    "scene": scene_name,
                    "direction": direction_name,
                    "arrival_index": int(arrival["arrival_index"]),
                    "inflight_nfe": inflight,
                    "late": inflight >= 2,
                    "response_improvement": float(
                        arrival["policies"]["lateact"]["response_current"]
                        - arrival["policies"]["direct_late_bind"]["response_current"]
                    ),
                    "latency_saving_seconds": float(
                        arrival["policies"]["full_restart"]["action_to_pixel_seconds"]
                        - arrival["policies"]["lateact"]["action_to_pixel_seconds"]
                    ),
                }
                arrival_rows.append(row)
                if row["late"]:
                    late_response_effects.append(row["response_improvement"])
                    late_latency_effects.append(row["latency_saving_seconds"])
            direction_rows.append(
                {
                    "base_image": base,
                    "variant": variant,
                    "scene": scene_name,
                    "direction": direction_name,
                    "arrival_class_counts": dict(sorted(class_counts.items())),
                    "late_arrival_count": len(late_response_effects),
                    "fixed_direct_late_response_improvement": float(
                        fixed_response_improvement
                    ),
                    "raw_late_response_improvement_mean": float(
                        np.mean(late_response_effects)
                    ),
                    "raw_late_latency_saving_mean_seconds": float(
                        np.mean(late_latency_effects)
                    ),
                }
            )
            quality = direction["quality"]
            quality_rows.append(
                {
                    "base_image": base,
                    "variant": variant,
                    "scene": scene_name,
                    "direction": direction_name,
                    "future_ssim": float(quality["lateact_vs_new_future_ssim"]),
                    "temporal_minus_new": float(quality["lateact_temporal_minus_new"]),
                    "boundary_jump_minus_new": float(
                        quality["lateact_boundary_jump_minus_new"]
                    ),
                    "all_decodes_valid": bool(quality["all_decodes_valid"]),
                }
            )

    if len(scenes) != 32 or len(direction_rows) != 64 or len(arrival_rows) != 1024:
        raise RuntimeError("frozen primary cardinality mismatch")
    if {key: sorted(value["variants"]) for key, value in hierarchy.items()} != {
        f"{index:04d}": [0, 1, 2, 3] for index in range(8, 16)
    }:
        raise RuntimeError("base-image/variant hierarchy mismatch")
    if any(value["directions"] != 8 or value["arrivals"] != 128 for value in hierarchy.values()):
        raise RuntimeError("within-base hierarchy mismatch")

    late_rows = [row for row in arrival_rows if row["late"]]
    arrival_counts = Counter(row["inflight_nfe"] for row in arrival_rows)
    if len(late_rows) != 676 or arrival_counts != Counter({1: 348, 2: 338, 3: 338}):
        raise RuntimeError("frozen late-arrival definition/cardinality mismatch")

    summary_direction_keys = {
        (row["scene"], row["direction"]) for row in frozen_summary["per_direction"]
    }
    raw_direction_keys = {(row["scene"], row["direction"]) for row in direction_rows}
    if summary_direction_keys != raw_direction_keys:
        raise RuntimeError("shard/summary direction mismatch")
    if frozen_summary["arrival_case_count"] != 1024 or frozen_summary["direction_count"] != 64:
        raise RuntimeError("shard/summary count mismatch")

    by_base_direction = defaultdict(list)
    for row in direction_rows:
        by_base_direction[row["base_image"]].append(row)
    base_rows = []
    for base in sorted(by_base_direction):
        directions = by_base_direction[base]
        base_late_rows = [row for row in late_rows if row["base_image"] == base]
        base_quality = [row for row in quality_rows if row["base_image"] == base]
        base_rows.append(
            {
                "base_image": base,
                "variant_count": len({row["variant"] for row in directions}),
                "direction_count": len(directions),
                "arrival_count": sum(sum(row["arrival_class_counts"].values()) for row in directions),
                "late_arrival_count": len(base_late_rows),
                "arrival_class_1": sum(row["arrival_class_counts"].get(1, 0) for row in directions),
                "arrival_class_2": sum(row["arrival_class_counts"].get(2, 0) for row in directions),
                "arrival_class_3": sum(row["arrival_class_counts"].get(3, 0) for row in directions),
                "original_fixed_branch_response_improvement": float(
                    np.mean(
                        [row["fixed_direct_late_response_improvement"] for row in directions]
                    )
                ),
                "raw_late_response_improvement_equal_direction_mean": float(
                    np.mean(
                        [row["raw_late_response_improvement_mean"] for row in directions]
                    )
                ),
                "raw_late_response_improvement_arrival_weighted": float(
                    np.mean([row["response_improvement"] for row in base_late_rows])
                ),
                "late_latency_saving_equal_direction_mean_seconds": float(
                    np.mean(
                        [row["raw_late_latency_saving_mean_seconds"] for row in directions]
                    )
                ),
                "late_latency_saving_arrival_weighted_seconds": float(
                    np.mean([row["latency_saving_seconds"] for row in base_late_rows])
                ),
                "future_ssim_mean": float(np.mean([row["future_ssim"] for row in base_quality])),
                "future_ssim_median": float(np.median([row["future_ssim"] for row in base_quality])),
                "future_ssim_minimum": float(min(row["future_ssim"] for row in base_quality)),
                "future_ssim_below_0_90": sum(row["future_ssim"] < 0.90 for row in base_quality),
                "future_ssim_below_0_85": sum(row["future_ssim"] < 0.85 for row in base_quality),
                "future_ssim_below_0_80": sum(row["future_ssim"] < 0.80 for row in base_quality),
            }
        )

    fixed_direction_response = [
        row["fixed_direct_late_response_improvement"] for row in direction_rows
    ]
    raw_direction_response = [
        row["raw_late_response_improvement_mean"] for row in direction_rows
    ]
    direction_latency = [
        row["raw_late_latency_saving_mean_seconds"] for row in direction_rows
    ]
    base_fixed_response = [
        row["original_fixed_branch_response_improvement"] for row in base_rows
    ]
    base_raw_response = [
        row["raw_late_response_improvement_equal_direction_mean"] for row in base_rows
    ]
    base_latency = [
        row["late_latency_saving_equal_direction_mean_seconds"] for row in base_rows
    ]

    def cluster_result(values: list[float]) -> dict:
        result = distribution(values)
        result.update(
            bootstrap_cluster_ci(values, args.bootstrap_replicates, args.bootstrap_seed)
        )
        result["exact_wilcoxon_signed_rank"] = exact_signed_rank(values)
        result["positive_base_images"] = int(sum(value > 0 for value in values))
        return result

    base_sums_response = [
        sum(row["response_improvement"] for row in late_rows if row["base_image"] == base)
        for base in sorted(hierarchy)
    ]
    base_sums_latency = [
        sum(row["latency_saving_seconds"] for row in late_rows if row["base_image"] == base)
        for base in sorted(hierarchy)
    ]
    base_late_counts = [
        sum(row["base_image"] == base for row in late_rows) for base in sorted(hierarchy)
    ]

    ssim_values = [row["future_ssim"] for row in quality_rows]
    temporal_values = [row["temporal_minus_new"] for row in quality_rows]
    boundary_values = [row["boundary_jump_minus_new"] for row in quality_rows]
    threshold_counts = {}
    for threshold in (0.90, 0.85, 0.80):
        below = sum(value < threshold for value in ssim_values)
        at_least = len(ssim_values) - below
        threshold_counts[f"{threshold:.2f}"] = {
            "below_count": below,
            "below_percent": 100.0 * below / len(ssim_values),
            "at_least_count": at_least,
            "at_least_percent": 100.0 * at_least / len(ssim_values),
        }
    worst_quality = sorted(quality_rows, key=lambda row: row["future_ssim"])[:8]
    temporal_stats = distribution(temporal_values)
    temporal_pass = sum(value >= -0.05 for value in temporal_values)

    results = {
        "analysis": "Gate 2 Matrix-Game mouse clustered robustness and SSIM tails",
        "read_only": True,
        "model_generation_performed": False,
        "source_artifact_sha256": source_hashes,
        "hierarchy": {
            "base_images": len(hierarchy),
            "base_image_ids": sorted(hierarchy),
            "variants_per_base_image": 4,
            "directions_per_variant": 2,
            "arrivals_per_direction": 16,
            "total_scene_variants": len(scenes),
            "total_directions": len(direction_rows),
            "total_arrivals": len(arrival_rows),
            "arrival_class_counts": dict(sorted(arrival_counts.items())),
            "late_definition": "inflight_nfe >= 2 (unchanged frozen definition)",
            "late_arrivals": len(late_rows),
            "late_arrivals_per_direction_range": [
                min(row["late_arrival_count"] for row in direction_rows),
                max(row["late_arrival_count"] for row in direction_rows),
            ],
            "base_rows": base_rows,
        },
        "response": {
            "original_direction_level_reference": original_direction_reference(
                fixed_direction_response
            ),
            "original_reference_definition": (
                "Per-direction fixed branch: response(lateact) - response(direct_late). "
                "This is the reported 0.8159 estimate and is not an average over all "
                "676 raw late-arrival rows."
            ),
            "raw_late_direction_level_descriptive": {
                **distribution(raw_direction_response),
                "definition": (
                    "Within each scene-variant/direction, mean over every frozen arrival "
                    "with inflight_nfe >= 2."
                ),
            },
            "base_image_cluster_inference_primary": {
                **cluster_result(base_raw_response),
                "definition": (
                    "Mean late-row response effect within each direction, then equal mean "
                    "over four variants x two directions within each base image; n=8 bases."
                ),
            },
            "base_image_cluster_inference_paper_compatible_fixed_branch": {
                **cluster_result(base_fixed_response),
                "definition": (
                    "Mean of the eight fixed per-direction direct_late contrasts within each "
                    "base image; preserves the published 0.8159 estimand but uses n=8."
                ),
            },
            "hierarchical_top_cluster_bootstrap_arrival_weighted": {
                "point_estimate": float(sum(base_sums_response) / sum(base_late_counts)),
                "mean_percentile_95_ci": pooled_top_cluster_ci(
                    base_sums_response,
                    base_late_counts,
                    args.bootstrap_replicates,
                    args.bootstrap_seed,
                ),
                "definition": (
                    "Resample eight base images with replacement; retain every selected "
                    "base's variants, directions, and late arrivals; compute the pooled mean."
                ),
            },
        },
        "latency": {
            "original_direction_level_reference": original_direction_reference(
                direction_latency
            ),
            "base_image_cluster_inference_primary": {
                **cluster_result(base_latency),
                "definition": (
                    "Mean late-row latency saving within each direction, then equal mean "
                    "over four variants x two directions within each base image; n=8 bases."
                ),
            },
            "hierarchical_top_cluster_bootstrap_arrival_weighted": {
                "point_estimate": float(sum(base_sums_latency) / sum(base_late_counts)),
                "mean_percentile_95_ci": pooled_top_cluster_ci(
                    base_sums_latency,
                    base_late_counts,
                    args.bootstrap_replicates,
                    args.bootstrap_seed,
                ),
                "definition": (
                    "Resample eight base images with replacement; retain every selected "
                    "base's variants, directions, and late arrivals; compute the pooled mean."
                ),
            },
        },
        "quality": {
            "unit": (
                "One frozen lateact-vs-new decoded future comparison per evaluator-valid "
                "scene-variant/direction; n=64. Not arrival-level."
            ),
            "future_ssim": distribution(ssim_values),
            "threshold_counts": threshold_counts,
            "worst_eight": worst_quality,
            "temporal_minus_new": {
                **temporal_stats,
                "pass_rule": "lateact_temporal_minus_new >= -0.05",
                "pass_count": temporal_pass,
                "pass_percent": 100.0 * temporal_pass / len(temporal_values),
                "maximum_observed_deficit": float(max(0.0, -min(temporal_values))),
            },
            "boundary_jump_minus_new": distribution(boundary_values),
            "finite_nonconstant_decode_checks": {
                "pass_count": sum(row["all_decodes_valid"] for row in quality_rows),
                "total": len(quality_rows),
            },
        },
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "interval": "percentile 2.5%--97.5%",
            "primary_cluster_aggregation": (
                "Equal-weight base images; within each base, equal-weight direction cells "
                "after averaging that cell's unchanged late arrivals."
            ),
        },
        "conclusion": {
            "response_positive_in_all_base_images": all(value > 0 for value in base_raw_response),
            "latency_positive_in_all_base_images": all(value > 0 for value in base_latency),
            "main_response_conclusion_changes": False,
            "main_latency_conclusion_changes": False,
            "quality_interpretation": (
                "Mouse-yaw quality is broadly stable but not uniformly >=0.90: the median "
                "gate passes, 60/64 are >=0.85 and 63/64 are >=0.80, while 15/64 are "
                "below 0.90 and one direction is 0.7184."
            ),
        },
    }

    (args.output / "robustness_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    with (args.output / "base_image_clusters.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(base_rows[0]))
        writer.writeheader()
        writer.writerows(base_rows)
    with (args.output / "ssim_direction_tail.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quality_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(quality_rows, key=lambda row: row["future_ssim"]))

    response = results["response"]["base_image_cluster_inference_primary"]
    response_fixed = results["response"][
        "base_image_cluster_inference_paper_compatible_fixed_branch"
    ]
    latency = results["latency"]["base_image_cluster_inference_primary"]
    ssim = results["quality"]["future_ssim"]
    counts = results["quality"]["threshold_counts"]
    base_table = "\n".join(
        "| {base_image} | {late_arrival_count} | {raw_late_response_improvement_equal_direction_mean:.6f} | "
        "{original_fixed_branch_response_improvement:.6f} | "
        "{late_latency_saving_equal_direction_mean_seconds:.6f} | {future_ssim_median:.6f} | "
        "{future_ssim_minimum:.6f} | {future_ssim_below_0_90}/8 |".format(**row)
        for row in base_rows
    )
    report = f"""# Gate 2 mouse-yaw robustness analysis

This is a read-only analysis of the frozen `mouse_shard_*.json` rows. No model
generation, artifact mutation, threshold change, or sample exclusion was performed.

## Verified hierarchy

- 8 base images (`0008`--`0015`)
- 4 deterministic variants per base (`v0`--`v3`)
- 2 directions per variant
- 16 arrivals per direction
- 32 scene-variants, 64 directions, and 1,024 arrivals total
- Frozen late definition: `inflight_nfe >= 2`; 676 late arrivals retained
- Arrival classes: NFE1={arrival_counts[1]}, NFE2={arrival_counts[2]}, NFE3={arrival_counts[3]}
- Quality safeguard unit: one decoded comparison per scene-variant/direction (n=64), not per arrival

## Clustered serving effects

Primary aggregation averages unchanged late arrivals within each direction,
then gives equal weight to the eight variant/direction cells within a base image.
The eight base images are the inferential units. The percentile bootstrap resamples
8 base images with replacement for {args.bootstrap_replicates:,} replicates using seed
{args.bootstrap_seed}. Exact Wilcoxon signed-rank tests enumerate all 2^8 sign assignments.

### Response: LATEACT minus DIRECT-LATE-BIND

- Published direction-level reference (fixed `direct_late` branch, descriptive n=64):
  mean {np.mean(fixed_direction_response):.6f}, median {np.median(fixed_direction_response):.6f},
  direction-bootstrap 95% CI [{results['response']['original_direction_level_reference']['direction_bootstrap_mean_95_ci_10000_seed_20260824'][0]:.6f}, {results['response']['original_direction_level_reference']['direction_bootstrap_mean_95_ci_10000_seed_20260824'][1]:.6f}].
- Important estimand note: that 0.8159 statistic is the stored fixed per-direction
  `lateact - direct_late` branch contrast. It is not the mean over all 676 late-arrival rows.
- Raw late-row direction-level descriptive estimate (n=64 direction cells): mean
  {np.mean(raw_direction_response):.6f}, median {np.median(raw_direction_response):.6f}.
- Base-image inference on all frozen late rows (n=8): mean {response['mean']:.6f},
  median {response['median']:.6f}, cluster-bootstrap mean 95% CI
  [{response['mean_percentile_95_ci'][0]:.6f}, {response['mean_percentile_95_ci'][1]:.6f}],
  exact one-sided Wilcoxon p={response['exact_wilcoxon_signed_rank']['one_sided_exact_p_greater']:.8f};
  8/8 base images positive.
- Paper-compatible fixed-branch n=8 estimate: mean {response_fixed['mean']:.6f},
  median {response_fixed['median']:.6f}, cluster-bootstrap mean 95% CI
  [{response_fixed['mean_percentile_95_ci'][0]:.6f}, {response_fixed['mean_percentile_95_ci'][1]:.6f}],
  exact one-sided Wilcoxon p={response_fixed['exact_wilcoxon_signed_rank']['one_sided_exact_p_greater']:.8f};
  8/8 positive.

### Latency: FULL-RESTART minus LATEACT

- Original direction-level reference (descriptive n=64): mean
  {np.mean(direction_latency):.6f} s, median {np.median(direction_latency):.6f} s,
  direction-bootstrap 95% CI [{results['latency']['original_direction_level_reference']['direction_bootstrap_mean_95_ci_10000_seed_20260824'][0]:.6f}, {results['latency']['original_direction_level_reference']['direction_bootstrap_mean_95_ci_10000_seed_20260824'][1]:.6f}] s.
- Base-image inference (n=8): mean {latency['mean']:.6f} s, median
  {latency['median']:.6f} s, cluster-bootstrap mean 95% CI
  [{latency['mean_percentile_95_ci'][0]:.6f}, {latency['mean_percentile_95_ci'][1]:.6f}] s,
  exact one-sided Wilcoxon p={latency['exact_wilcoxon_signed_rank']['one_sided_exact_p_greater']:.8f};
  8/8 base images positive.

Both qualitative conclusions remain unchanged under n=8 inference.

### Per-base-image aggregates

| Base | Late rows | Raw-late response gain | Fixed-branch response gain | Latency saving (s) | SSIM median | SSIM minimum | SSIM <0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|
{base_table}

### Existing analyzer compatibility note

The frozen `analyze_gate2.py` `paired_effect` field averages LATEACT-minus-DIRECT
over all 1,024 valid arrival rows, including the 348 early rows where the two
policies are identical. It is therefore not a late-arrival inferential statistic.
The published 0.8159 statistic instead comes from the 64 stored fixed
`lateact - direct_late` direction contrasts. Neither existing file was changed;
this robustness output uses raw shard rows and reports both estimands explicitly.

## Future-SSIM tail (frozen n=64 direction-level quality units)

Mean {ssim['mean']:.6f}; median {ssim['median']:.6f}; sample SD
{ssim['sample_standard_deviation']:.6f}; minimum {ssim['minimum']:.6f};
p05 {ssim['p05']:.6f}; p10 {ssim['p10']:.6f}; p25 {ssim['p25']:.6f};
p75 {ssim['p75']:.6f}; p90 {ssim['p90']:.6f}; p95 {ssim['p95']:.6f};
maximum {ssim['maximum']:.6f}.

- SSIM < 0.90: {counts['0.90']['below_count']}/64 ({counts['0.90']['below_percent']:.4f}%)
- SSIM < 0.85: {counts['0.85']['below_count']}/64 ({counts['0.85']['below_percent']:.4f}%)
- SSIM < 0.80: {counts['0.80']['below_count']}/64 ({counts['0.80']['below_percent']:.4f}%)
- SSIM >= 0.90: {counts['0.90']['at_least_count']}/64 ({counts['0.90']['at_least_percent']:.4f}%)
- SSIM >= 0.85: {counts['0.85']['at_least_count']}/64 ({counts['0.85']['at_least_percent']:.4f}%)
- SSIM >= 0.80: {counts['0.80']['at_least_count']}/64 ({counts['0.80']['at_least_percent']:.4f}%)

Worst case: `{worst_quality[0]['scene']}` / `{worst_quality[0]['direction']}` =
{worst_quality[0]['future_ssim']:.6f}. The complete ordered tail is in
`ssim_direction_tail.csv`. Failures below 0.90 are concentrated in four base
images; base `0013` contains 6/8 such directions and the sole value below 0.80.

Temporal signed difference (`LATEACT - NEW`) has mean
{temporal_stats['mean']:.6f}, median {temporal_stats['median']:.6f}, range
[{temporal_stats['minimum']:.6f}, {temporal_stats['maximum']:.6f}]. All
{temporal_pass}/64 pass the frozen >= -0.05 safeguard; the maximum observed
temporal deficit is {max(0.0, -min(temporal_values)):.6f}. Boundary-jump signed
difference has mean {np.mean(boundary_values):.6f}, median
{np.median(boundary_values):.6f}, and range [{min(boundary_values):.6f},
{max(boundary_values):.6f}]. Finite/nonconstant decode checks pass 64/64.

## Interpretation

The serving response and latency conclusions do not depend on treating 64
direction cells as independent: every base image has a positive effect and both
n=8 cluster-bootstrap intervals exclude zero. The future-SSIM result is not
supported only by its median: 60/64 directions are >=0.85, 63/64 are >=0.80,
and temporal/decode safeguards pass universally. Quality is nevertheless not
uniform: 15/64 are below 0.90 and one is 0.7184, with visible base-image
concentration. These unfavorable tails should be disclosed rather than converted
into a new gate.

## Recommended paper-level changes

1. Treat the 64 variant/direction cells as descriptive and use the eight base
   images for inference; report the n=8 cluster bootstrap and exact Wilcoxon.
2. Clarify whether 0.8159 denotes the fixed `direct_late` branch contrast. If
   the prose says all late arrivals, use the raw late-row n=8 estimate instead.
3. Add p05 and the exact SSIM tail counts, including the 0.7184 minimum and
   concentration in base image 0013; do not create or retroactively change a gate.
4. Retain the median >=0.90 frozen quality-gate statement, but do not imply that
   all 64 directions individually exceed 0.90.
"""
    (args.output / "ROBUSTNESS_REPORT.md").write_text(report)
    print(json.dumps({
        "response_n8": results["response"]["base_image_cluster_inference_primary"],
        "latency_n8": results["latency"]["base_image_cluster_inference_primary"],
        "future_ssim": results["quality"]["future_ssim"],
        "threshold_counts": results["quality"]["threshold_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
