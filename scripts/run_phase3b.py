#!/usr/bin/env python3
"""Run the frozen Phase 3B same-boundary keyboard serving confirmation."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from run_gate2 import parse_args, run_direction, stratified_arrivals
from lateact.experiment import (
    atomic_json,
    condition_hashes,
    load_models,
    make_condition,
    make_transition_condition,
    prepare_scene,
    save_video,
    scene_path,
    signed_lateral_translation,
)
from lateact.runtime import (
    generate_prefix,
    make_history_guard,
    make_noise_bundle,
    noise_hashes,
    snapshot_pipeline,
    tree_sha256,
)


def main() -> None:
    args = parse_args()
    if args.family != "keyboard":
        raise ValueError("Phase 3B is frozen to keyboard")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    variants = list(range(10, 34))
    selected = [value for index, value in enumerate(variants) if index % args.shard_count == args.shard_index]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    code_root, pipeline, vae = load_models(args.upstream, args.model_root, device)
    if len(pipeline.denoising_step_list) != 3:
        raise RuntimeError("Phase 3B requires exactly 3 NFE")

    report_path = output / f"phase3b_shard_{args.shard_index}.json"
    report = {
        "status": "running",
        "phase": "phase3b_same_boundary_confirmation",
        "source_image_index": 16,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "mouse_latest_safe_switch_after_nfe": 1,
        "keyboard_latest_safe_switch_after_nfe": 1,
        "arrival_protocol": {
            "distribution": "stratified_uniform_over_three_nfe_wall_interval",
            "count_per_direction": 16,
            "seed": 20260825,
            "intervention": "next_completed_nfe_boundary",
        },
        "contexts": {},
    }
    atomic_json(report_path, report)

    prefix_latents, total_latents = 3, 9
    raw_frames = (total_latents - 1) * 4 + 1
    current_start_raw, next_start_raw = 9, 21
    directions = (
        ("left_to_right", "keyboard_left", "keyboard_right"),
        ("right_to_left", "keyboard_right", "keyboard_left"),
    )
    image = scene_path(code_root, 16)
    for variant in selected:
        context_name = f"0016_h{variant:02d}"
        context_dir = output / context_name
        context_dir.mkdir(exist_ok=True)
        fixed, fixed_hashes = prepare_scene(image, total_latents, vae, device)
        conditions = {
            action: make_condition(
                fixed,
                raw_frames=raw_frames,
                branch_start_raw=current_start_raw,
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
            raise RuntimeError(f"Phase 3B conditions differ outside keyboard: {changed_keys}")
        prefix_noise = make_noise_bundle(140000 + variant)
        current_noise = make_noise_bundle(150000 + variant)
        next_noise = make_noise_bundle(160000 + variant)
        pipeline._initialize_kv_cache(1, torch.bfloat16, device)
        pipeline._initialize_kv_cache_mouse_and_keyboard(1, torch.bfloat16, device)
        pipeline._initialize_crossattn_cache(1, torch.bfloat16, device)
        prefix, current_start = generate_prefix(pipeline, prefix_noise, conditions["keyboard_left"])
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
        context_report = {
            "source_image": str(image),
            "prefix_variant": variant,
            "prefix_seed": 140000 + variant,
            "current_seed": 150000 + variant,
            "next_seed": 160000 + variant,
            "fixed_condition_hashes": fixed_hashes,
            "condition_hashes": hashes,
            "condition_changed_keys": changed_keys,
            "noise_hashes": {
                "prefix": noise_hashes(prefix_noise),
                "current": noise_hashes(current_noise),
                "next": noise_hashes(next_noise),
            },
            "prefix_snapshot_hashes": {
                "visual": tree_sha256(prefix_snapshot.visual),
                "mouse": tree_sha256(prefix_snapshot.mouse),
                "keyboard": tree_sha256(prefix_snapshot.keyboard),
                "cross": tree_sha256(prefix_snapshot.cross),
            },
            "directions": {},
        }
        report["contexts"][context_name] = context_report
        atomic_json(report_path, report)

        for direction_index, (direction, old_action, new_action) in enumerate(directions):
            transition = make_transition_condition(
                fixed,
                raw_frames=raw_frames,
                first_start_raw=current_start_raw,
                second_start_raw=next_start_raw,
                first_action=old_action,
                second_action=new_action,
                device=device,
            )
            arrivals = stratified_arrivals(
                16, 20260825 + variant * 17 + direction_index
            )
            result, outputs, frames = run_direction(
                pipeline,
                vae,
                prefix=prefix,
                prefix_snapshot=prefix_snapshot,
                current_noise=current_noise,
                next_noise=next_noise,
                old_condition=conditions[old_action],
                new_condition=conditions[new_action],
                transition_condition=transition,
                current_start=current_start,
                guard=guard,
                globals_=globals_,
                arrivals=arrivals,
                device=device,
                motion_fn=signed_lateral_translation,
                motion_value_key="affine_center_signed_sum",
                stability_value_key="median_lk_signed_sum",
                support_key="median_inlier_count",
                minimum_oracle_gap=2.0,
            )
            expected_sign = -1 if direction == "left_to_right" else 1
            result["evaluator_valid"] = bool(
                result["evaluator_valid"] and result["oracle_gap"] * expected_sign > 0
            )
            result["metric"] = "robust_affine_center_lateral_translation"
            result["old_action"] = old_action
            result["new_action"] = new_action
            result["audit"]["mouse_action_specific_policy_bit_exact"] = True
            for arrival in result["arrivals"]:
                common = arrival["policies"].pop("lateact")
                arrival["policies"]["mouse_boundary_lateact"] = copy.deepcopy(common)
                arrival["policies"]["action_specific_lateact"] = copy.deepcopy(common)
            for output_name, value in outputs.items():
                torch.save(value, context_dir / f"{direction}_{output_name}.pt")
                save_video(context_dir / f"{direction}_{output_name}.mp4", frames[output_name])
            context_report["directions"][direction] = result
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
