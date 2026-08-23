#!/usr/bin/env python3
"""Run a deterministic LateAct smoke or Gate 0 shard."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


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
    signed_camera_motion,
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
    parser.add_argument("--phase", choices=("smoke", "gate0"), required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def compact_indices(indices: dict) -> dict:
    return {
        name: {
            "unique": sorted({tuple(pair) for pair in pairs}),
            "layer_count": len(pairs),
        }
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


def run_specs(phase: str) -> list[tuple[str, str, str, int]]:
    specs = [
        ("oracle_mouse_left", "mouse_left", "mouse_left", 0),
        ("oracle_mouse_right", "mouse_right", "mouse_right", 0),
        ("lr_s1", "mouse_left", "mouse_right", 1),
        ("lr_s2", "mouse_left", "mouse_right", 2),
        ("rl_s1", "mouse_right", "mouse_left", 1),
        ("rl_s2", "mouse_right", "mouse_left", 2),
    ]
    if phase == "smoke":
        specs.extend(
            (f"control_left_s{s}", "mouse_left", "mouse_left", s)
            for s in range(4)
        )
    return specs


def action_maps() -> dict[str, dict[int, str]]:
    return {
        "left_to_right": {
            0: "oracle_mouse_right",
            1: "lr_s1",
            2: "lr_s2",
            3: "oracle_mouse_left",
        },
        "right_to_left": {
            0: "oracle_mouse_left",
            1: "rl_s1",
            2: "rl_s2",
            3: "oracle_mouse_right",
        },
    }


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
        raise RuntimeError(f"frozen design requires exactly 3 NFE, found {steps}")

    all_scenes = [0, 1] if args.phase == "smoke" else list(range(8))
    scenes = [index for index in all_scenes if index % args.shard_count == args.shard_index]
    report_path = output / f"{args.phase}_shard_{args.shard_index}.json"
    report = {
        "status": "running",
        "phase": args.phase,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "denoising_timesteps": [float(value) for value in pipeline.denoising_step_list],
        "nfe_per_block": steps,
        "context_cache_writes_per_block": 1,
        "prefix_blocks": 1,
        "future_blocks": 1,
        "latent_frames_per_block": 3,
        "action_maps": {name: {str(k): v for k, v in values.items()} for name, values in action_maps().items()},
        "scenes": {},
    }
    atomic_json(report_path, report)

    prefix_latents = 3
    future_latents = 3
    total_latents = prefix_latents + future_latents
    raw_frames = (total_latents - 1) * 4 + 1
    prefix_decoded_frames = (prefix_latents - 1) * 4 + 1

    for scene_index in scenes:
        name = f"{scene_index:04d}"
        scene_dir = output / name
        scene_dir.mkdir(exist_ok=True)
        image = scene_path(code_root, scene_index)
        fixed, fixed_hashes = prepare_scene(image, total_latents, vae, device)
        conditions = {
            action: make_condition(
                fixed,
                raw_frames=raw_frames,
                branch_start_raw=prefix_decoded_frames,
                action=action,
                device=device,
            )
            for action in ("mouse_left", "mouse_right")
        }
        hashes = {action: condition_hashes(value) for action, value in conditions.items()}
        changed_keys = sorted(
            key for key in hashes["mouse_left"] if hashes["mouse_left"][key] != hashes["mouse_right"][key]
        )
        if changed_keys != ["mouse_cond"]:
            raise RuntimeError(f"conditions differ outside mouse_cond: {changed_keys}")

        prefix_noise = make_noise_bundle(50000 + scene_index)
        future_noise = make_noise_bundle(60000 + scene_index)
        pipeline._initialize_kv_cache(1, torch.bfloat16, device)
        pipeline._initialize_kv_cache_mouse_and_keyboard(1, torch.bfloat16, device)
        pipeline._initialize_crossattn_cache(1, torch.bfloat16, device)

        torch.cuda.reset_peak_memory_stats(device)
        prefix_started = time.perf_counter()
        prefix, current_start = generate_prefix(
            pipeline, prefix_noise, conditions["mouse_left"]
        )
        torch.cuda.synchronize(device)
        prefix_seconds = time.perf_counter() - prefix_started
        snapshot = snapshot_pipeline(pipeline, current_start)
        snapshot_hashes = {
            "visual": tree_sha256(snapshot.visual),
            "mouse": tree_sha256(snapshot.mouse),
            "keyboard": tree_sha256(snapshot.keyboard),
            "cross": tree_sha256(snapshot.cross),
        }
        torch.save(prefix.detach().cpu(), scene_dir / "prefix.pt")
        del pipeline.kv_cache1, pipeline.kv_cache_mouse, pipeline.kv_cache_keyboard, pipeline.crossattn_cache
        torch.cuda.empty_cache()

        scene_report = {
            "image": str(image),
            "fixed_condition_hashes": fixed_hashes,
            "condition_hashes": hashes,
            "condition_changed_keys": changed_keys,
            "prefix_noise_hashes": noise_hashes(prefix_noise),
            "future_noise_hashes": noise_hashes(future_noise),
            "prefix_seconds": prefix_seconds,
            "prefix_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "prefix_snapshot_hashes": snapshot_hashes,
            "runs": {},
        }
        report["scenes"][name] = scene_report
        atomic_json(report_path, report)

        outputs: dict[str, torch.Tensor] = {}
        for run_name, old_action, new_action, switch_after in run_specs(args.phase):
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
            seconds = time.perf_counter() - started
            if end != total_latents:
                raise RuntimeError("future current_start mismatch")
            branch_cpu = branch.cpu()
            outputs[run_name] = branch_cpu
            torch.save(branch_cpu, scene_dir / f"{run_name}.pt")

            decode_started = time.perf_counter()
            decoded = vae.decode(
                torch.cat([prefix.detach().cpu(), branch_cpu], dim=2).to(torch.bfloat16),
                device=device,
                tiled=True,
                tile_size=[44, 80],
                tile_stride=[23, 38],
            )
            frames = decoded_uint8(decoded)
            decode_seconds = time.perf_counter() - decode_started
            save_video(scene_dir / f"{run_name}.mp4", frames)
            motion = signed_camera_motion(frames, prefix_decoded_frames)
            scene_report["runs"][run_name] = {
                "old_action": old_action,
                "new_action": new_action,
                "switch_after": switch_after,
                "latent_sha256": tensor_sha256(branch_cpu),
                "future_noise_hashes": noise_hashes(future_noise),
                "cache_bytes_at_fork": at_fork_bytes,
                "generation_seconds": seconds,
                "decode_seconds": decode_seconds,
                "nfe_per_second": steps / seconds,
                "latent_frames_per_second": future_latents / seconds,
                "decoded_frame_equivalent_per_second": (4 * future_latents) / seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "trace": compact_trace(trace),
                "motion": motion,
            }
            atomic_json(report_path, report)
            del branch, decoded, frames
            torch.cuda.empty_cache()

        left_hash = scene_report["runs"]["oracle_mouse_left"]["latent_sha256"]
        right_hash = scene_report["runs"]["oracle_mouse_right"]["latent_sha256"]
        checks = {
            "oracle_latents_distinct": left_hash != right_hash,
            "all_noise_hashes_match": all(
                run["future_noise_hashes"] == scene_report["future_noise_hashes"]
                for run in scene_report["runs"].values()
            ),
            "only_mouse_condition_differs": changed_keys == ["mouse_cond"],
            "all_historical_cache_checks_passed": all(
                all(run["trace"]["history_exact_after_nfe"])
                and run["trace"]["history_exact_after_context_write"]
                for run in scene_report["runs"].values()
            ),
        }
        if args.phase == "smoke":
            control_hashes = {
                scene_report["runs"][f"control_left_s{s}"]["latent_sha256"]
                for s in range(4)
            }
            checks["same_action_controls_exact"] = control_hashes == {left_hash}
            control_scores = [
                scene_report["runs"][f"control_left_s{s}"]["motion"]["lk_signed_sum"]
                for s in range(4)
            ]
            checks["same_action_motion_range"] = max(control_scores) - min(control_scores)
        scene_report["engineering_checks"] = checks
        atomic_json(report_path, report)
        del fixed, conditions, snapshot, prefix, outputs
        torch.cuda.empty_cache()

    report["status"] = "complete"
    report["wall_finished_unix"] = time.time()
    atomic_json(report_path, report)


if __name__ == "__main__":
    main()

