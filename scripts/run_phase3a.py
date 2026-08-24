#!/usr/bin/env python3
"""Run one frozen Phase 3A keyboard commitment-calibration shard."""

from __future__ import annotations

import argparse
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
    condition_hashes,
    decoded_uint8,
    load_models,
    make_condition,
    prepare_scene,
    save_video,
    scene_path,
    signed_lateral_translation,
)
from lateact.runtime import (
    cache_bytes,
    generate_prefix,
    generate_switched_block,
    make_noise_bundle,
    noise_hashes,
    restore_pipeline,
    snapshot_pipeline,
    tensor_sha256,
    tree_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    return parser.parse_args()


def compact_indices(indices: dict) -> dict:
    return {
        name: {"unique": sorted({tuple(pair) for pair in pairs}), "layer_count": len(pairs)}
        for name, pairs in indices.items()
    }


def compact_trace(trace: dict) -> dict:
    return {
        "switch_after": trace["switch_after"],
        "nfe": trace["nfe"],
        "context_cache_writes": trace["context_cache_writes"],
        "actions_by_nfe": trace["actions_by_nfe"],
        "context_action": trace["context_action"],
        "indices_before": compact_indices(trace["indices_before"]),
        "indices_after_nfe": [compact_indices(value) for value in trace["indices_after_nfe"]],
        "indices_after_context_write": compact_indices(trace["indices_after_context_write"]),
        "history_exact_after_nfe": trace["history_exact_after_nfe"],
        "history_exact_after_context_write": trace["history_exact_after_context_write"],
    }


def run_specs() -> list[tuple[str, str, str, int]]:
    return [
        ("oracle_keyboard_left", "keyboard_left", "keyboard_left", 0),
        ("oracle_keyboard_right", "keyboard_right", "keyboard_right", 0),
        ("lr_s1", "keyboard_left", "keyboard_right", 1),
        ("lr_s2", "keyboard_left", "keyboard_right", 2),
        ("rl_s1", "keyboard_right", "keyboard_left", 1),
        ("rl_s2", "keyboard_right", "keyboard_left", 2),
    ]


def frame_ssim(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean([
        structural_similarity(a, b, data_range=255, channel_axis=2)
        for a, b in zip(first, second)
    ]))


def temporal_consistency(frames: np.ndarray, prefix_frames: int) -> float:
    segment = frames[prefix_frames - 1 :]
    return frame_ssim(segment[:-1], segment[1:])


def boundary_jump(frames: np.ndarray, prefix_frames: int) -> float:
    previous = frames[prefix_frames - 1].astype(np.float32)
    current = frames[prefix_frames].astype(np.float32)
    return float(np.mean(np.abs(current - previous)))


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    code_root, pipeline, vae = load_models(args.upstream, args.model_root, device)
    steps = len(pipeline.denoising_step_list)
    if steps != 3:
        raise RuntimeError(f"frozen Phase 3A requires 3 NFE, found {steps}")

    variants = [value for value in range(10) if value % args.shard_count == args.shard_index]
    report_path = output / f"phase3a_shard_{args.shard_index}.json"
    report = {
        "status": "running",
        "phase": "phase3a_keyboard_commitment_calibration",
        "source_image_index": 16,
        "prefix_seed_base": 120000,
        "future_seed_base": 130000,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "denoising_timesteps": [float(value) for value in pipeline.denoising_step_list],
        "contexts": {},
    }
    atomic_json(report_path, report)

    prefix_latents = 3
    total_latents = 6
    raw_frames = (total_latents - 1) * 4 + 1
    prefix_frames = (prefix_latents - 1) * 4 + 1
    image = scene_path(code_root, 16)

    for variant in variants:
        name = f"0016_c{variant:02d}"
        context_dir = output / name
        context_dir.mkdir(exist_ok=True)
        fixed, fixed_hashes = prepare_scene(image, total_latents, vae, device)
        conditions = {
            action: make_condition(
                fixed,
                raw_frames=raw_frames,
                branch_start_raw=prefix_frames,
                action=action,
                device=device,
            )
            for action in ("keyboard_left", "keyboard_right")
        }
        hashes = {action: condition_hashes(value) for action, value in conditions.items()}
        changed_keys = sorted(
            key for key in hashes["keyboard_left"]
            if hashes["keyboard_left"][key] != hashes["keyboard_right"][key]
        )
        if changed_keys != ["keyboard_cond"]:
            raise RuntimeError(f"conditions differ outside keyboard_cond: {changed_keys}")

        prefix_noise = make_noise_bundle(120000 + variant)
        future_noise = make_noise_bundle(130000 + variant)
        pipeline._initialize_kv_cache(1, torch.bfloat16, device)
        pipeline._initialize_kv_cache_mouse_and_keyboard(1, torch.bfloat16, device)
        pipeline._initialize_crossattn_cache(1, torch.bfloat16, device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        prefix, current_start = generate_prefix(pipeline, prefix_noise, conditions["keyboard_left"])
        torch.cuda.synchronize(device)
        prefix_seconds = time.perf_counter() - started
        snapshot = snapshot_pipeline(pipeline, current_start)
        snapshot_hashes = {
            "visual": tree_sha256(snapshot.visual),
            "mouse": tree_sha256(snapshot.mouse),
            "keyboard": tree_sha256(snapshot.keyboard),
            "cross": tree_sha256(snapshot.cross),
        }
        torch.save(prefix.cpu(), context_dir / "prefix.pt")
        context_report = {
            "source_image": str(image),
            "prefix_variant": variant,
            "prefix_seed": 120000 + variant,
            "future_seed": 130000 + variant,
            "fixed_condition_hashes": fixed_hashes,
            "condition_hashes": hashes,
            "condition_changed_keys": changed_keys,
            "prefix_noise_hashes": noise_hashes(prefix_noise),
            "future_noise_hashes": noise_hashes(future_noise),
            "prefix_seconds": prefix_seconds,
            "prefix_snapshot_hashes": snapshot_hashes,
            "runs": {},
        }
        report["contexts"][name] = context_report
        atomic_json(report_path, report)

        for run_name, old_action, new_action, switch_after in run_specs():
            restored_start = restore_pipeline(pipeline, snapshot, device)
            if restored_start != prefix_latents:
                raise RuntimeError("prefix current_start mismatch")
            at_fork_bytes = cache_bytes(pipeline)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            branch, end, trace = generate_switched_block(
                pipeline,
                noise=future_noise,
                old_condition=conditions[old_action],
                new_condition=conditions[new_action],
                current_start=restored_start,
                switch_after=switch_after,
                audit_history=True,
            )
            torch.cuda.synchronize(device)
            generation_seconds = time.perf_counter() - started
            if end != total_latents:
                raise RuntimeError("future current_start mismatch")
            branch_cpu = branch.cpu()
            torch.save(branch_cpu, context_dir / f"{run_name}.pt")
            decode_started = time.perf_counter()
            decoded = vae.decode(
                torch.cat([prefix.cpu(), branch_cpu], dim=2).to(torch.bfloat16),
                device=device,
                tiled=True,
                tile_size=[44, 80],
                tile_stride=[23, 38],
            )
            torch.cuda.synchronize(device)
            decode_seconds = time.perf_counter() - decode_started
            frames = decoded_uint8(decoded)
            save_video(context_dir / f"{run_name}.mp4", frames)
            context_report["runs"][run_name] = {
                "old_action": old_action,
                "new_action": new_action,
                "switch_after": switch_after,
                "latent_sha256": tensor_sha256(branch_cpu),
                "future_noise_hashes": noise_hashes(future_noise),
                "cache_bytes_at_fork": at_fork_bytes,
                "generation_seconds": generation_seconds,
                "decode_seconds": decode_seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "trace": compact_trace(trace),
                "lateral_translation": signed_lateral_translation(frames, prefix_frames),
                "temporal_consistency_ssim": temporal_consistency(frames, prefix_frames),
                "boundary_jump": boundary_jump(frames, prefix_frames),
                "decode_valid": bool(np.isfinite(frames).all() and np.std(frames) > 0),
            }
            atomic_json(report_path, report)
            del branch, branch_cpu, decoded, frames
            torch.cuda.empty_cache()

        left = context_report["runs"]["oracle_keyboard_left"]
        right = context_report["runs"]["oracle_keyboard_right"]
        context_report["engineering_checks"] = {
            "oracle_latents_distinct": left["latent_sha256"] != right["latent_sha256"],
            "all_noise_hashes_match": all(
                run["future_noise_hashes"] == context_report["future_noise_hashes"]
                for run in context_report["runs"].values()
            ),
            "only_keyboard_condition_differs": changed_keys == ["keyboard_cond"],
            "all_historical_cache_checks_passed": all(
                all(run["trace"]["history_exact_after_nfe"])
                and run["trace"]["history_exact_after_context_write"]
                for run in context_report["runs"].values()
            ),
            "all_decodes_valid": all(run["decode_valid"] for run in context_report["runs"].values()),
        }
        atomic_json(report_path, report)
        del fixed, conditions, snapshot, prefix
        torch.cuda.empty_cache()

    report["status"] = "complete"
    report["finished_unix"] = time.time()
    atomic_json(report_path, report)


if __name__ == "__main__":
    main()
