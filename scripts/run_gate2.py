#!/usr/bin/env python3
"""Run one primary or secondary LateAct Gate 2 serving-evaluation shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lateact.experiment import (
    atomic_json,
    decoded_uint8,
    load_models,
    make_condition,
    make_transition_condition,
    prepare_scene,
    save_video,
    scene_path,
    signed_camera_motion,
)
from lateact.rollback import (
    capture_logical_boundary,
    commit_context,
    logical_checkpoint_hashes,
    logical_checkpoint_nbytes,
    reset_to_prefix,
    restore_logical_boundary,
    rng_hashes,
    run_nfe,
    run_steps,
    scheduler_hash,
)
from lateact.runtime import (
    assert_history_guard,
    generate_prefix,
    make_history_guard,
    make_noise_bundle,
    noise_hashes,
    restore_pipeline,
    snapshot_pipeline,
    tensor_sha256,
    tree_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("mouse", "keyboard"), required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    return parser.parse_args()


def family_spec(family: str) -> tuple[int, int, tuple[tuple[str, str, str], ...]]:
    if family == "mouse":
        return 4, 16, (
            ("left_to_right", "mouse_left", "mouse_right"),
            ("right_to_left", "mouse_right", "mouse_left"),
        )
    return 1, 8, (
        ("left_to_right", "keyboard_left", "keyboard_right"),
        ("right_to_left", "keyboard_right", "keyboard_left"),
    )


def scene_specs(variants: int) -> list[tuple[str, int, int, int]]:
    result = []
    ordinal = 0
    for image_index in range(8, 16):
        for variant in range(variants):
            result.append((f"{image_index:04d}_v{variant}", image_index, variant, ordinal))
            ordinal += 1
    return result


def sync_elapsed(device, started: float) -> float:
    torch.cuda.synchronize(device)
    return time.perf_counter() - started


def timed_nfe(pipeline, *, noisy, condition, current_start, step, renoise, device):
    started = time.perf_counter()
    denoised, next_input = run_nfe(
        pipeline,
        noisy_input=noisy,
        condition=condition,
        current_start=current_start,
        step_index=step,
        renoise=renoise,
    )
    return denoised, next_input, sync_elapsed(device, started)


def timed_context(pipeline, output, condition, current_start, device) -> float:
    started = time.perf_counter()
    commit_context(pipeline, output, condition, current_start)
    return sync_elapsed(device, started)


def decode(vae, latent: torch.Tensor, device) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    value = vae.decode(
        latent.to(torch.bfloat16),
        device=device,
        tiled=True,
        tile_size=[44, 80],
        tile_stride=[23, 38],
    )
    seconds = sync_elapsed(device, started)
    frames = decoded_uint8(value)
    del value
    return frames, seconds


def frame_ssim(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.mean(
            [structural_similarity(a, b, data_range=255, channel_axis=2) for a, b in zip(first, second)]
        )
    )


def temporal_consistency(frames: np.ndarray, prefix_frames: int) -> float:
    segment = frames[prefix_frames - 1 :]
    return frame_ssim(segment[:-1], segment[1:])


def boundary_jump(frames: np.ndarray, prefix_frames: int) -> float:
    previous = frames[prefix_frames - 1].astype(np.float32)
    current = frames[prefix_frames].astype(np.float32)
    return float(np.mean(np.abs(current - previous)))


def stratified_arrivals(count: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(values)
    return [float(value) for value in values]


def reset(pipeline, initial, guard, globals_):
    return reset_to_prefix(
        pipeline,
        initial_latent=initial,
        history_guard=guard,
        prefix_global_ends=globals_,
        verify=False,
    )


@torch.no_grad()
def run_current_full(
    pipeline, *, initial, condition, current_start, renoise, guard, globals_, device
) -> tuple[torch.Tensor, dict]:
    reset_started = time.perf_counter()
    current = reset(pipeline, initial, guard, globals_)
    reset_seconds = sync_elapsed(device, reset_started)
    nfe_times = []
    output = None
    for step in range(3):
        output, next_input, seconds = timed_nfe(
            pipeline,
            noisy=current,
            condition=condition,
            current_start=current_start,
            step=step,
            renoise=renoise,
            device=device,
        )
        nfe_times.append(seconds)
        if next_input is not None:
            current = next_input
    assert output is not None
    context_seconds = timed_context(pipeline, output, condition, current_start, device)
    assert_history_guard(pipeline, guard)
    return output.cpu(), {
        "reset": reset_seconds,
        "nfes": nfe_times,
        "context": context_seconds,
        "generation_total": reset_seconds + sum(nfe_times) + context_seconds,
    }


@torch.no_grad()
def run_direction(
    pipeline,
    vae,
    *,
    prefix: torch.Tensor,
    prefix_snapshot,
    current_noise,
    next_noise,
    old_condition: dict,
    new_condition: dict,
    transition_condition: dict,
    current_start: int,
    guard: dict,
    globals_: dict,
    arrivals: list[float],
    device,
    motion_fn=signed_camera_motion,
    motion_value_key: str = "lk_signed_sum",
    stability_value_key: str = "dense_signed_sum",
    support_key: str = "median_track_count",
    minimum_oracle_gap: float = 1e-6,
) -> tuple[dict, dict[str, torch.Tensor], dict[str, np.ndarray]]:
    initial = current_noise.initial.to(device=device, dtype=torch.bfloat16)
    scheduler_before = scheduler_hash(pipeline)
    rng_before = rng_hashes(device)
    torch.cuda.reset_peak_memory_stats(device)

    # Produce the direct late (s=2) trajectory and capture the optimized boundary.
    reset(pipeline, initial, guard, globals_)
    old0, entering2, _ = timed_nfe(
        pipeline, noisy=initial, condition=old_condition, current_start=current_start,
        step=0, renoise=current_noise.renoise, device=device
    )
    assert entering2 is not None

    # Operational checkpoint-save latency excludes heavy audit hashes.
    save_started = time.perf_counter()
    save_probe = entering2.clone()
    index_probe = [
        (int(cache["global_end_index"].item()), int(cache["local_end_index"].item()))
        for caches in (pipeline.kv_cache1, pipeline.kv_cache_mouse, pipeline.kv_cache_keyboard)
        for cache in caches
    ]
    checkpoint_save_seconds = sync_elapsed(device, save_started)
    del save_probe, index_probe
    checkpoint = capture_logical_boundary(
        pipeline, entering_latent=entering2, history_guard=guard, step_entering=1
    )
    _, entering3, _ = timed_nfe(
        pipeline, noisy=entering2, condition=old_condition, current_start=current_start,
        step=1, renoise=current_noise.renoise, device=device
    )
    assert entering3 is not None
    direct_started = time.perf_counter()
    direct_late = run_steps(
        pipeline, noisy_input=entering3, condition=new_condition,
        current_start=current_start, start_step=2, renoise=current_noise.renoise
    )
    direct_suffix_seconds = sync_elapsed(device, direct_started)

    # Optimized rollback restores latent and indices only; stale current K/V must
    # be overwritten before read by NFE2.
    restore_started = time.perf_counter()
    restored = restore_logical_boundary(pipeline, checkpoint, guard, verify=False)
    restore_seconds = sync_elapsed(device, restore_started)
    rollback_started = time.perf_counter()
    lateact = run_steps(
        pipeline, noisy_input=restored, condition=new_condition,
        current_start=current_start, start_step=1, renoise=current_noise.renoise
    )
    rollback_suffix_seconds = sync_elapsed(device, rollback_started)
    assert_history_guard(pipeline, guard)

    # Full restart / NEW oracle.
    new_oracle, new_timing = run_current_full(
        pipeline, initial=initial, condition=new_condition, current_start=current_start,
        renoise=current_noise.renoise, guard=guard, globals_=globals_, device=device
    )

    # Independent early direct bind must be bit-identical to optimized rollback.
    reset(pipeline, initial, guard, globals_)
    _, independent_entering2, _ = timed_nfe(
        pipeline, noisy=initial, condition=old_condition, current_start=current_start,
        step=0, renoise=current_noise.renoise, device=device
    )
    assert independent_entering2 is not None
    early_started = time.perf_counter()
    early_direct = run_steps(
        pipeline, noisy_input=independent_entering2, condition=new_condition,
        current_start=current_start, start_step=1, renoise=current_noise.renoise
    )
    early_suffix_seconds = sync_elapsed(device, early_started)
    if not torch.equal(early_direct, lateact):
        raise RuntimeError("latent-plus-index rollback does not reproduce early direct binding")

    # OLD current timeline, then NEXT-BLOCK under NEW.
    old, old_timing = run_current_full(
        pipeline, initial=initial, condition=old_condition, current_start=current_start,
        renoise=current_noise.renoise, guard=guard, globals_=globals_, device=device
    )
    next_initial = next_noise.initial.to(device=device, dtype=torch.bfloat16)
    next_started = time.perf_counter()
    next_new = run_steps(
        pipeline, noisy_input=next_initial, condition=transition_condition,
        current_start=current_start + 3, start_step=0, renoise=next_noise.renoise
    )
    next_generation_seconds = sync_elapsed(device, next_started)
    serving_peak_allocated = torch.cuda.max_memory_allocated(device)
    serving_peak_reserved = torch.cuda.max_memory_reserved(device)

    # Reconstruct OLD current and generate the next-block OLD endpoint.
    restore_pipeline(pipeline, prefix_snapshot, device)
    old_repeat, _ = run_current_full(
        pipeline, initial=initial, condition=old_condition, current_start=current_start,
        renoise=current_noise.renoise, guard=guard, globals_=globals_, device=device
    )
    next_old = run_steps(
        pipeline, noisy_input=next_initial, condition=old_condition,
        current_start=current_start + 3, start_step=0, renoise=next_noise.renoise
    ).cpu()
    if not torch.equal(old, old_repeat):
        raise RuntimeError("OLD current reconstruction changed")

    outputs = {
        "old": old,
        "new_oracle": new_oracle,
        "direct_late": direct_late.cpu(),
        "lateact": lateact.cpu(),
        "next_new": next_new.cpu(),
        "next_old": next_old,
    }

    frames = {}
    decode_times = {}
    for name in ("old", "new_oracle", "direct_late", "lateact"):
        frames[name], decode_times[name] = decode(
            vae, torch.cat([prefix.cpu(), outputs[name]], dim=2), device
        )
    frames["next_new"], decode_times["next_new"] = decode(
        vae, torch.cat([prefix.cpu(), old, outputs["next_new"]], dim=2), device
    )
    frames["next_old"], decode_times["next_old"] = decode(
        vae, torch.cat([prefix.cpu(), old, outputs["next_old"]], dim=2), device
    )

    prefix_frames, next_prefix_frames = 9, 21
    current_metrics = {
        name: {
            "motion": motion_fn(frames[name], prefix_frames),
            "temporal_consistency_ssim": temporal_consistency(frames[name], prefix_frames),
            "boundary_jump": boundary_jump(frames[name], prefix_frames),
        }
        for name in ("old", "new_oracle", "direct_late", "lateact")
    }
    next_metrics = {
        name: motion_fn(frames[name], next_prefix_frames)
        for name in ("next_old", "next_new")
    }
    old_score = current_metrics["old"]["motion"][motion_value_key]
    new_score = current_metrics["new_oracle"]["motion"][motion_value_key]
    denominator = new_score - old_score
    response = {
        name: (current_metrics[name]["motion"][motion_value_key] - old_score) / denominator
        if denominator != 0 else None
        for name in ("old", "new_oracle", "direct_late", "lateact")
    }
    next_denominator = next_metrics["next_new"][motion_value_key] - next_metrics["next_old"][motion_value_key]
    next_response = 1.0 if next_denominator != 0 else None

    old_nfes = old_timing["nfes"]
    denoise_total = sum(old_nfes)
    arrival_rows = []
    cumulative = np.cumsum(old_nfes)
    for arrival_index, normalized in enumerate(arrivals):
        timestamp = normalized * denoise_total
        nfe_index = int(np.searchsorted(cumulative, timestamp, side="right"))
        nfe_index = min(nfe_index, 2)
        completed_boundary = nfe_index + 1
        start_time = 0.0 if nfe_index == 0 else float(cumulative[nfe_index - 1])
        residual = float(cumulative[nfe_index] - timestamp)
        inflight_fraction = (timestamp - start_time) / old_nfes[nfe_index]
        remaining_old = sum(old_nfes[completed_boundary:]) + old_timing["context"] + decode_times["old"]
        next_total = next_generation_seconds + decode_times["next_new"]
        next_latency = residual + remaining_old + next_total
        full_latency = residual + new_timing["generation_total"] + decode_times["new_oracle"]

        if completed_boundary == 1:
            direct_response = response["lateact"]
            direct_latency = residual + early_suffix_seconds + decode_times["lateact"]
            lateact_latency = direct_latency
            direct_post = 2
            late_discarded = 0
            checkpoint_used = False
            direct_output = "early_direct"
        elif completed_boundary == 2:
            direct_response = response["direct_late"]
            direct_latency = residual + direct_suffix_seconds + decode_times["direct_late"]
            lateact_latency = residual + restore_seconds + rollback_suffix_seconds + decode_times["lateact"]
            direct_post = 1
            late_discarded = 1
            checkpoint_used = True
            direct_output = "direct_late"
        else:
            direct_response = 0.0
            direct_latency = next_latency
            lateact_latency = residual + restore_seconds + rollback_suffix_seconds + decode_times["lateact"]
            direct_post = 3
            late_discarded = 2
            checkpoint_used = True
            direct_output = "old_then_next"

        policy = {
            "next_block": {
                "response_current": 0.0,
                "eventual_next_response": next_response,
                "action_to_pixel_seconds": next_latency,
                "discarded_nfe": 0,
                "redone_nfe": 0,
                "post_boundary_nfe": (3 - completed_boundary) + 3,
            },
            "direct_late_bind": {
                "response_current": direct_response,
                "eventual_next_response": 1.0 if completed_boundary == 3 else None,
                "action_to_pixel_seconds": direct_latency,
                "discarded_nfe": 0,
                "redone_nfe": 0,
                "post_boundary_nfe": direct_post,
            },
            "full_restart": {
                "response_current": 1.0,
                "eventual_next_response": None,
                "action_to_pixel_seconds": full_latency,
                "discarded_nfe": completed_boundary,
                "redone_nfe": completed_boundary,
                "post_boundary_nfe": 3,
            },
            "lateact": {
                "response_current": response["lateact"],
                "eventual_next_response": None,
                "action_to_pixel_seconds": lateact_latency,
                "discarded_nfe": late_discarded,
                "redone_nfe": late_discarded,
                "post_boundary_nfe": 2,
                "checkpoint_used": checkpoint_used,
            },
        }
        arrival_rows.append(
            {
                "arrival_index": arrival_index,
                "normalized_time": normalized,
                "arrival_seconds": timestamp,
                "inflight_nfe": completed_boundary,
                "fully_completed_nfe_at_arrival": completed_boundary - 1,
                "inflight_fraction": inflight_fraction,
                "wait_to_boundary_seconds": residual,
                "direct_output": direct_output,
                "policies": policy,
            }
        )

    dense_gap = (
        current_metrics["new_oracle"]["motion"][stability_value_key]
        - current_metrics["old"]["motion"][stability_value_key]
    )
    track_support = min(
        current_metrics["old"]["motion"][support_key],
        current_metrics["new_oracle"]["motion"][support_key],
    )
    evaluator_valid = bool(
        tensor_sha256(old) != tensor_sha256(new_oracle)
        and abs(denominator) > minimum_oracle_gap
        and denominator * dense_gap > 0
        and track_support >= 20
    )
    future = slice(prefix_frames, None)
    quality = {
        "lateact_vs_new_future_ssim": frame_ssim(frames["lateact"][future], frames["new_oracle"][future]),
        "direct_late_vs_new_future_ssim": frame_ssim(frames["direct_late"][future], frames["new_oracle"][future]),
        "lateact_temporal_minus_new": current_metrics["lateact"]["temporal_consistency_ssim"] - current_metrics["new_oracle"]["temporal_consistency_ssim"],
        "lateact_boundary_jump_minus_new": current_metrics["lateact"]["boundary_jump"] - current_metrics["new_oracle"]["boundary_jump"],
        "all_decodes_valid": all(np.isfinite(value).all() and np.ptp(value) > 0 for value in frames.values()),
    }
    if rng_hashes(device) != rng_before:
        raise RuntimeError("implicit RNG state changed in Gate 2")
    if scheduler_hash(pipeline) != scheduler_before:
        raise RuntimeError("scheduler changed in Gate 2")
    audit = {
        "optimized_checkpoint_hashes": logical_checkpoint_hashes(checkpoint),
        "optimized_checkpoint_bytes": logical_checkpoint_nbytes(checkpoint),
        "gate1_checkpoint_bytes": 649330080,
        "checkpoint_save_seconds": checkpoint_save_seconds,
        "checkpoint_restore_seconds": restore_seconds,
        "host_checkpoint_bytes": 0,
        "independent_early_bind_exact": tensor_sha256(early_direct) == tensor_sha256(lateact.cpu()),
        "historical_prefix_exact": True,
        "cross_attention_hash": tree_sha256(pipeline.crossattn_cache),
        "rng_unchanged": True,
        "scheduler_unchanged": True,
        "peak_allocated_bytes": serving_peak_allocated,
        "peak_reserved_bytes": serving_peak_reserved,
    }
    result = {
        "evaluator_valid": evaluator_valid,
        "oracle_gap": denominator,
        "dense_oracle_gap": dense_gap,
        "median_track_support": track_support,
        "response": response,
        "next_block_response": next_response,
        "quality": quality,
        "timing_components": {
            "old": old_timing,
            "new_full_restart": new_timing,
            "early_direct_suffix": early_suffix_seconds,
            "direct_late_suffix": direct_suffix_seconds,
            "lateact_restore": restore_seconds,
            "lateact_suffix": rollback_suffix_seconds,
            "next_generation": next_generation_seconds,
            "decode": decode_times,
        },
        "arrivals": arrival_rows,
        "audit": audit,
    }
    # The next-block endpoint fork rolls the local cache; restore the immutable
    # prefix snapshot for the next action direction. This experimental fork
    # snapshot is host-side and is not part of the serving policy checkpoint.
    restore_pipeline(pipeline, prefix_snapshot, device)
    return result, outputs, frames


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    variants, arrival_count, directions = family_spec(args.family)
    specs = scene_specs(variants)
    selected = [spec for index, spec in enumerate(specs) if index % args.shard_count == args.shard_index]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    code_root, pipeline, vae = load_models(args.upstream, args.model_root, device)
    report_path = output / f"{args.family}_shard_{args.shard_index}.json"
    report = {
        "status": "running",
        "family": args.family,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "arrival_protocol": {
            "distribution": "stratified_uniform_over_three_nfe_wall_interval",
            "count_per_direction": arrival_count,
            "seed": 20260824,
            "intervention": "next_completed_nfe_boundary",
        },
        "scenes": {},
    }
    atomic_json(report_path, report)

    prefix_latents, total_latents = 3, 9
    raw_frames = (total_latents - 1) * 4 + 1
    current_start_raw, next_start_raw = 9, 21
    for scene_name, image_index, variant, ordinal in selected:
        scene_dir = output / scene_name
        scene_dir.mkdir(exist_ok=True)
        fixed, fixed_hashes = prepare_scene(scene_path(code_root, image_index), total_latents, vae, device)
        all_actions = sorted({action for _, old, new in directions for action in (old, new)})
        conditions = {
            action: make_condition(
                fixed, raw_frames=raw_frames, branch_start_raw=current_start_raw,
                action=action, device=device
            )
            for action in all_actions
        }
        prefix_noise = make_noise_bundle(70000 + ordinal)
        current_noise = make_noise_bundle(80000 + ordinal)
        next_noise = make_noise_bundle(90000 + ordinal)
        pipeline._initialize_kv_cache(1, torch.bfloat16, device)
        pipeline._initialize_kv_cache_mouse_and_keyboard(1, torch.bfloat16, device)
        pipeline._initialize_crossattn_cache(1, torch.bfloat16, device)
        prefix, current_start = generate_prefix(pipeline, prefix_noise, conditions[all_actions[0]])
        prefix_snapshot = snapshot_pipeline(pipeline, current_start)
        guard = make_history_guard(pipeline)
        globals_ = {
            name: [int(cache["global_end_index"].item()) for cache in caches]
            for name, caches in (
                ("visual", pipeline.kv_cache1),
                ("mouse", pipeline.kv_cache_mouse),
                ("keyboard", pipeline.kv_cache_keyboard),
            )
        }
        scene_report = {
            "source_image": str(scene_path(code_root, image_index)),
            "source_image_index": image_index,
            "prefix_variant": variant,
            "fixed_condition_hashes": fixed_hashes,
            "noise_hashes": {
                "prefix": noise_hashes(prefix_noise),
                "current": noise_hashes(current_noise),
                "next": noise_hashes(next_noise),
            },
            "directions": {},
        }
        report["scenes"][scene_name] = scene_report
        for direction_index, (direction, old_action, new_action) in enumerate(directions):
            transition = make_transition_condition(
                fixed, raw_frames=raw_frames, first_start_raw=current_start_raw,
                second_start_raw=next_start_raw, first_action=old_action,
                second_action=new_action, device=device
            )
            arrivals = stratified_arrivals(
                arrival_count, 20260824 + ordinal * 17 + direction_index
            )
            result, outputs, frames = run_direction(
                pipeline, vae, prefix=prefix, prefix_snapshot=prefix_snapshot,
                current_noise=current_noise,
                next_noise=next_noise, old_condition=conditions[old_action],
                new_condition=conditions[new_action], transition_condition=transition,
                current_start=current_start, guard=guard, globals_=globals_,
                arrivals=arrivals, device=device
            )
            result["old_action"] = old_action
            result["new_action"] = new_action
            for name, value in outputs.items():
                torch.save(value, scene_dir / f"{direction}_{name}.pt")
                save_video(scene_dir / f"{direction}_{name}.mp4", frames[name])
            scene_report["directions"][direction] = result
            atomic_json(report_path, report)
            del transition, outputs, frames
            torch.cuda.empty_cache()
        del fixed, conditions, prefix, prefix_snapshot, guard
        torch.cuda.empty_cache()

    report["status"] = "complete"
    report["finished_unix"] = time.time()
    atomic_json(report_path, report)


if __name__ == "__main__":
    main()
