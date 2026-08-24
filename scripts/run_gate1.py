#!/usr/bin/env python3
"""Execute one deterministic LateAct Gate 1 shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    signed_camera_motion,
)
from lateact.rollback import (
    assert_checkpoint_live,
    capture_boundary,
    checkpoint_hashes,
    checkpoint_nbytes,
    commit_context,
    reset_to_prefix,
    restore_boundary,
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
    tensor_sha256,
    tree_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--gate0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    return parser.parse_args()


def gate0_scenes(root: Path) -> dict:
    scenes = {}
    for path in sorted(root.glob("gate0_shard_*.json")):
        value = json.loads(path.read_text())
        if value["status"] != "complete":
            raise RuntimeError(f"incomplete Gate 0 shard: {path}")
        scenes.update(value["scenes"])
    if len(scenes) != 8:
        raise RuntimeError("Gate 0 reference must contain exactly 8 scenes")
    return scenes


def frame_ssim(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.mean(
            [
                structural_similarity(a, b, data_range=255, channel_axis=2)
                for a, b in zip(first, second)
            ]
        )
    )


def temporal_consistency(frames: np.ndarray, prefix_frames: int) -> float:
    segment = frames[prefix_frames - 1 :]
    return frame_ssim(segment[:-1], segment[1:])


def frames_sha256(frames: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def sync_elapsed(device, started: float) -> float:
    torch.cuda.synchronize(device)
    return time.perf_counter() - started


def timed_full_trajectory(
    pipeline,
    *,
    initial: torch.Tensor,
    condition: dict,
    current_start: int,
    renoise: list[torch.Tensor],
    history_guard: dict,
    prefix_global_ends: dict,
    device,
) -> tuple[torch.Tensor, float, float]:
    assert_history_guard(pipeline, history_guard)
    reset_started = time.perf_counter()
    current = reset_to_prefix(
        pipeline,
        initial_latent=initial,
        history_guard=history_guard,
        prefix_global_ends=prefix_global_ends,
        verify=False,
    )
    reset_seconds = sync_elapsed(device, reset_started)
    suffix_started = time.perf_counter()
    output = run_steps(
        pipeline,
        noisy_input=current,
        condition=condition,
        current_start=current_start,
        start_step=0,
        renoise=renoise,
    )
    suffix_seconds = sync_elapsed(device, suffix_started)
    assert_history_guard(pipeline, history_guard)
    return output, reset_seconds, suffix_seconds


@torch.no_grad()
def execute_pair(
    pipeline,
    *,
    prefix: torch.Tensor,
    current_start: int,
    future_noise,
    old_condition: dict,
    new_condition: dict,
    history_guard: dict,
    prefix_global_ends: dict,
    device,
) -> tuple[dict[str, torch.Tensor], dict]:
    initial = future_noise.initial.to(device=device, dtype=torch.bfloat16)
    scheduler_before = scheduler_hash(pipeline)
    rng_before = rng_hashes(device)
    reset_to_prefix(
        pipeline,
        initial_latent=initial,
        history_guard=history_guard,
        prefix_global_ends=prefix_global_ends,
    )

    prearrival_times = []
    started = time.perf_counter()
    _, entering_nfe2 = run_nfe(
        pipeline,
        noisy_input=initial,
        condition=old_condition,
        current_start=current_start,
        step_index=0,
        renoise=future_noise.renoise,
    )
    prearrival_times.append(sync_elapsed(device, started))
    assert entering_nfe2 is not None
    checkpoint = capture_boundary(
        pipeline,
        entering_latent=entering_nfe2,
        history_guard=history_guard,
        step_entering=1,
    )

    started = time.perf_counter()
    _, entering_nfe3 = run_nfe(
        pipeline,
        noisy_input=entering_nfe2,
        condition=old_condition,
        current_start=current_start,
        step_index=1,
        renoise=future_noise.renoise,
    )
    prearrival_times.append(sync_elapsed(device, started))
    assert entering_nfe3 is not None

    # Direct continuation starts from the actual live post-NFE2 state: no restore.
    direct_started = time.perf_counter()
    direct = run_steps(
        pipeline,
        noisy_input=entering_nfe3,
        condition=new_condition,
        current_start=current_start,
        start_step=2,
        renoise=future_noise.renoise,
    )
    direct_seconds = sync_elapsed(device, direct_started)
    assert_history_guard(pipeline, history_guard)

    # Minimal rollback latency is split so equality auditing is excluded but the
    # actual GPU restore copies remain included.
    restore_started = time.perf_counter()
    restored_input = restore_boundary(
        pipeline, checkpoint, history_guard, verify=False
    )
    rollback_restore_seconds = sync_elapsed(device, restore_started)
    assert_checkpoint_live(pipeline, checkpoint)
    if not torch.equal(restored_input, checkpoint.entering_latent):
        raise RuntimeError("rollback latent restore mismatch")
    if tree_sha256(pipeline.crossattn_cache) != checkpoint.cross_hash:
        raise RuntimeError("cross state changed across rollback")
    if scheduler_hash(pipeline) != checkpoint.scheduler_hash:
        raise RuntimeError("scheduler state changed across rollback")
    rollback_started = time.perf_counter()
    minimal = run_steps(
        pipeline,
        noisy_input=restored_input,
        condition=new_condition,
        current_start=current_start,
        start_step=1,
        renoise=future_noise.renoise,
    )
    rollback_suffix_seconds = sync_elapsed(device, rollback_started)
    assert_history_guard(pipeline, history_guard)

    # OLD and NEW are independently reconstructed from the frozen prefix.
    old, _, _ = timed_full_trajectory(
        pipeline,
        initial=initial,
        condition=old_condition,
        current_start=current_start,
        renoise=future_noise.renoise,
        history_guard=history_guard,
        prefix_global_ends=prefix_global_ends,
        device=device,
    )
    new_oracle, _, _ = timed_full_trajectory(
        pipeline,
        initial=initial,
        condition=new_condition,
        current_start=current_start,
        renoise=future_noise.renoise,
        history_guard=history_guard,
        prefix_global_ends=prefix_global_ends,
        device=device,
    )

    # Full restart repeats NEW independently; reset plus suffix is timed from arrival.
    full_restart, full_reset_seconds, full_suffix_seconds = timed_full_trajectory(
        pipeline,
        initial=initial,
        condition=new_condition,
        current_start=current_start,
        renoise=future_noise.renoise,
        history_guard=history_guard,
        prefix_global_ends=prefix_global_ends,
        device=device,
    )
    rng_after = rng_hashes(device)
    if rng_after != rng_before:
        raise RuntimeError("generation consumed implicit CPU/CUDA RNG state")
    if scheduler_hash(pipeline) != scheduler_before:
        raise RuntimeError("scheduler mutated across Gate 1 trajectory")

    outputs = {
        "old": old.cpu(),
        "new_oracle": new_oracle.cpu(),
        "direct_late_bind": direct.cpu(),
        "minimal_rollback": minimal.cpu(),
        "full_restart": full_restart.cpu(),
    }
    audit = {
        "checkpoint_hashes": checkpoint_hashes(checkpoint),
        "checkpoint_bytes": checkpoint_nbytes(checkpoint),
        "scheduler_hash_before_after": [scheduler_before, scheduler_hash(pipeline)],
        "rng_hashes_before_after": [rng_before, rng_after],
        "latent_restore_exact": True,
        "checkpoint_live_exact_before_recompute": True,
        "historical_prefix_exact": True,
        "cross_attention_exact": True,
        "no_old_nfe2_state_leak": True,
        "latency_seconds": {
            "prearrival_nfe1": prearrival_times[0],
            "prearrival_nfe2": prearrival_times[1],
            "prearrival_total": sum(prearrival_times),
            "direct_from_arrival": direct_seconds,
            "minimal_restore": rollback_restore_seconds,
            "minimal_suffix": rollback_suffix_seconds,
            "minimal_from_arrival": rollback_restore_seconds + rollback_suffix_seconds,
            "full_reset": full_reset_seconds,
            "full_suffix": full_suffix_seconds,
            "full_from_arrival": full_reset_seconds + full_suffix_seconds,
        },
        "compute": {
            "direct": {"before_arrival": 2, "discarded": 0, "post_arrival": 1, "redone": 0, "total_nfe": 3, "block_equivalent": 1.0},
            "minimal": {"before_arrival": 2, "discarded": 1, "post_arrival": 2, "redone": 1, "total_nfe": 4, "block_equivalent": 4 / 3},
            "full_restart": {"before_arrival": 2, "discarded": 2, "post_arrival": 3, "redone": 2, "total_nfe": 5, "block_equivalent": 5 / 3},
        },
    }
    return outputs, audit


def decode_outputs(vae, prefix: torch.Tensor, outputs: dict[str, torch.Tensor], device) -> dict[str, np.ndarray]:
    decoded = {}
    for name, branch in outputs.items():
        video = vae.decode(
            torch.cat([prefix.cpu(), branch], dim=2).to(torch.bfloat16),
            device=device,
            tiled=True,
            tile_size=[44, 80],
            tile_stride=[23, 38],
        )
        decoded[name] = decoded_uint8(video)
        del video
    return decoded


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
    reference = gate0_scenes(args.gate0.resolve())
    code_root, pipeline, vae = load_models(args.upstream, args.model_root, device)
    if len(pipeline.denoising_step_list) != 3:
        raise RuntimeError("Gate 1 requires the frozen three-NFE sampler")

    scenes = [index for index in range(8) if index % args.shard_count == args.shard_index]
    report_path = output / f"gate1_shard_{args.shard_index}.json"
    report = {
        "status": "running",
        "phase": "gate1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "denoising_timesteps": [float(value) for value in pipeline.denoising_step_list],
        "scenes": {},
    }
    atomic_json(report_path, report)

    prefix_latents, total_latents = 3, 6
    raw_frames = (total_latents - 1) * 4 + 1
    prefix_decoded_frames = (prefix_latents - 1) * 4 + 1
    for scene_index in scenes:
        scene_name = f"{scene_index:04d}"
        scene_dir = output / scene_name
        scene_dir.mkdir(exist_ok=True)
        fixed, fixed_hashes = prepare_scene(scene_path(code_root, scene_index), total_latents, vae, device)
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
        prefix_noise = make_noise_bundle(50000 + scene_index)
        future_noise = make_noise_bundle(60000 + scene_index)
        pipeline._initialize_kv_cache(1, torch.bfloat16, device)
        pipeline._initialize_kv_cache_mouse_and_keyboard(1, torch.bfloat16, device)
        pipeline._initialize_crossattn_cache(1, torch.bfloat16, device)
        prefix, current_start = generate_prefix(pipeline, prefix_noise, conditions["mouse_left"])
        history_guard = make_history_guard(pipeline)
        prefix_global_ends = {
            name: [int(cache["global_end_index"].item()) for cache in caches]
            for name, caches in (
                ("visual", pipeline.kv_cache1),
                ("mouse", pipeline.kv_cache_mouse),
                ("keyboard", pipeline.kv_cache_keyboard),
            )
        }
        gate0_prefix = torch.load(args.gate0 / scene_name / "prefix.pt", weights_only=True)
        if tensor_sha256(prefix.cpu()) != tensor_sha256(gate0_prefix):
            raise RuntimeError("Gate 1 prefix does not reproduce frozen Gate 0")

        scene_report = {
            "fixed_condition_hashes": fixed_hashes,
            "condition_hashes": {name: condition_hashes(value) for name, value in conditions.items()},
            "prefix_noise_hashes": noise_hashes(prefix_noise),
            "future_noise_hashes": noise_hashes(future_noise),
            "prefix_reproduces_gate0": True,
            "directions": {},
            "same_action_control": None,
        }
        report["scenes"][scene_name] = scene_report

        directions = (
            ("left_to_right", "mouse_left", "mouse_right", "lr_s2", "lr_s1"),
            ("right_to_left", "mouse_right", "mouse_left", "rl_s2", "rl_s1"),
        )
        for direction, old_action, new_action, gate0_direct, gate0_minimal in directions:
            torch.cuda.reset_peak_memory_stats(device)
            outputs, audit = execute_pair(
                pipeline,
                prefix=prefix,
                current_start=current_start,
                future_noise=future_noise,
                old_condition=conditions[old_action],
                new_condition=conditions[new_action],
                history_guard=history_guard,
                prefix_global_ends=prefix_global_ends,
                device=device,
            )
            gate0_runs = reference[scene_name]["runs"]
            old_label = "oracle_mouse_left" if old_action == "mouse_left" else "oracle_mouse_right"
            new_label = "oracle_mouse_left" if new_action == "mouse_left" else "oracle_mouse_right"
            expected = {
                "old": gate0_runs[old_label]["latent_sha256"],
                "new_oracle": gate0_runs[new_label]["latent_sha256"],
                "direct_late_bind": gate0_runs[gate0_direct]["latent_sha256"],
                "minimal_rollback": gate0_runs[gate0_minimal]["latent_sha256"],
                "full_restart": gate0_runs[new_label]["latent_sha256"],
            }
            hashes = {name: tensor_sha256(value) for name, value in outputs.items()}
            if hashes != expected:
                raise RuntimeError(f"Gate 1 outputs do not reproduce Gate 0: {direction}")
            decoded = decode_outputs(vae, prefix, outputs, device)
            for name, value in outputs.items():
                torch.save(value, scene_dir / f"{direction}_{name}.pt")
                save_video(scene_dir / f"{direction}_{name}.mp4", decoded[name])
            metrics = {
                name: {
                    "motion": signed_camera_motion(frames, prefix_decoded_frames),
                    "frames_sha256": frames_sha256(frames),
                    "temporal_consistency_ssim": temporal_consistency(frames, prefix_decoded_frames),
                }
                for name, frames in decoded.items()
            }
            future = slice(prefix_decoded_frames, None)
            quality = {
                "minimal_vs_new_future_ssim": frame_ssim(decoded["minimal_rollback"][future], decoded["new_oracle"][future]),
                "direct_vs_new_future_ssim": frame_ssim(decoded["direct_late_bind"][future], decoded["new_oracle"][future]),
                "full_vs_new_future_ssim": frame_ssim(decoded["full_restart"][future], decoded["new_oracle"][future]),
                "minimal_temporal_minus_new": metrics["minimal_rollback"]["temporal_consistency_ssim"] - metrics["new_oracle"]["temporal_consistency_ssim"],
                "all_decodes_finite_and_nonconstant": all(np.isfinite(frames).all() and np.ptp(frames) > 0 for frames in decoded.values()),
            }
            audit["gate0_hashes_expected"] = expected
            audit["gate0_hashes_observed"] = hashes
            audit["all_outputs_reproduce_gate0"] = True
            audit["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            audit["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            scene_report["directions"][direction] = {
                "old_action": old_action,
                "new_action": new_action,
                "audit": audit,
                "metrics": metrics,
                "quality": quality,
            }
            atomic_json(report_path, report)
            del outputs, decoded
            torch.cuda.empty_cache()

        if scene_index in (0, 1):
            outputs, audit = execute_pair(
                pipeline,
                prefix=prefix,
                current_start=current_start,
                future_noise=future_noise,
                old_condition=conditions["mouse_left"],
                new_condition=conditions["mouse_left"],
                history_guard=history_guard,
                prefix_global_ends=prefix_global_ends,
                device=device,
            )
            hashes = {name: tensor_sha256(value) for name, value in outputs.items()}
            expected_hash = reference[scene_name]["runs"]["oracle_mouse_left"]["latent_sha256"]
            scene_report["same_action_control"] = {
                "action": "mouse_left",
                "hashes": hashes,
                "expected_gate0_hash": expected_hash,
                "all_exact": set(hashes.values()) == {expected_hash},
                "checkpoint_audit": audit,
            }
            if not scene_report["same_action_control"]["all_exact"]:
                raise RuntimeError("same-action rollback changed output")
            atomic_json(report_path, report)
            del outputs

        del fixed, conditions, prefix, history_guard
        torch.cuda.empty_cache()

    report["status"] = "complete"
    report["finished_unix"] = time.time()
    atomic_json(report_path, report)


if __name__ == "__main__":
    main()

