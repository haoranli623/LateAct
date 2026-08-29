#!/usr/bin/env python3
"""Frozen minWM Wan2.1 Action2V commitment-curve replication runner.

This uses the official four-step DMD generator and changes only the PRoPE
camera extrinsics supplied to a denoiser evaluation.  The first 16 latent
frames are a shared identity-camera prefix; the final four-frame latent block
is branched under the native opposite actions frozen in a protocol config.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.io import write_video

from phase4_minwm_protocol import (
    PROJECT,
    actions,
    control_prompt_indices,
    default_output,
    load_and_validate_prompts,
    load_protocol,
    seed_by_prompt,
    selected_prompt_indices,
    switch_positions,
    trajectory_steps_per_branch,
    unique_run_specs,
    validate_confirmatory_output,
)


UPSTREAM = Path(os.environ.get("LATEACT_MINWM_UPSTREAM", PROJECT / "third_party" / "minWM"))
MODEL_ROOT = Path(os.environ.get("LATEACT_MINWM_MODEL_ROOT", PROJECT / "models" / "minwm"))
DEFAULT_CONFIG = PROJECT / "config" / "phase4_minwm.yaml"

sys.path.insert(0, str(UPSTREAM / "Wan21"))
sys.path.insert(0, str(UPSTREAM / "shared"))


LATENT_SHAPE = (1, 20, 16, 60, 104)
BLOCK_FRAMES = 4
PREFIX_LATENTS = 16
TOTAL_LATENTS = 20
PREFIX_PIXEL_FRAMES = (PREFIX_LATENTS - 1) * 4 + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene-start", type=int)
    parser.add_argument("--scene-end", type=int)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the frozen protocol and print the run plan without touching CUDA or output",
    )
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def load_pipeline(device: torch.device, protocol: dict) -> CausalInferencePipeline:
    from demo_utils.memory import DynamicSwapInstaller
    from pipeline import CausalInferencePipeline

    config = OmegaConf.merge(
        OmegaConf.load(UPSTREAM / "Wan21/configs/default_config.yaml"),
        OmegaConf.load(UPSTREAM / "Wan21/configs/causal_forcing_dmd_camera.yaml"),
    )
    pipeline = CausalInferencePipeline(config, device=device)
    checkpoint = torch.load(
        MODEL_ROOT / "checkpoints" / protocol["upstream"]["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    key = "generator_ema" if "generator_ema" in checkpoint else "generator"
    state = checkpoint[key]
    try:
        pipeline.generator.load_state_dict(state)
    except RuntimeError:
        fixed = {
            name.replace("model._fsdp_wrapped_module.", "model.", 1)
            if name.startswith("model._fsdp_wrapped_module.") else name: value
            for name, value in state.items()
        }
        pipeline.generator.load_state_dict(fixed, strict=False)
    del checkpoint, state
    gc.collect()

    pipeline = pipeline.to(dtype=torch.bfloat16)
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval()
    return pipeline


def cpu_noise(seed: int) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn(LATENT_SHAPE, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    renoise = []
    for _ in range(TOTAL_LATENTS // BLOCK_FRAMES):
        renoise.append([
            torch.randn((1, BLOCK_FRAMES, 16, 60, 104), generator=generator, dtype=torch.float32)
            .to(torch.bfloat16)
            for _ in range(3)
        ])
    return initial, renoise


def reset_caches(pipeline: CausalInferencePipeline, device: torch.device) -> None:
    if pipeline.kv_cache1 is None:
        pipeline._initialize_kv_cache(1, torch.bfloat16, device)
        pipeline._initialize_crossattn_cache(1, torch.bfloat16, device)
        pipeline._initialize_prope_kv_cache(1, torch.bfloat16, device)
        return
    for cache in pipeline.kv_cache1 + pipeline.prope_kv_cache1:
        cache["global_end_index"].zero_()
        cache["local_end_index"].zero_()
    for cache in pipeline.crossattn_cache:
        cache["is_init"] = False


def cache_indices(pipeline: CausalInferencePipeline) -> dict:
    def collect(caches: list[dict]) -> list[list[int]]:
        return [
            [int(cache["global_end_index"].item()), int(cache["local_end_index"].item())]
            for cache in caches
        ]
    return {"visual": collect(pipeline.kv_cache1), "prope": collect(pipeline.prope_kv_cache1)}


def restore_indices(pipeline: CausalInferencePipeline, state: dict) -> None:
    for name, caches in (("visual", pipeline.kv_cache1), ("prope", pipeline.prope_kv_cache1)):
        for cache, pair in zip(caches, state[name]):
            cache["global_end_index"].fill_(pair[0])
            cache["local_end_index"].fill_(pair[1])


def sampled_state_signature(pipeline: CausalInferencePipeline, prefix_tokens: int) -> dict:
    """Exact sampled bytes plus indices for the immutable historical state."""
    token_slices = (
        slice(0, 8),
        slice(prefix_tokens // 2, prefix_tokens // 2 + 8),
        slice(prefix_tokens - 8, prefix_tokens),
    )
    result: dict[str, object] = {"indices": cache_indices(pipeline)}
    for cache_name, caches in (("visual", pipeline.kv_cache1), ("prope", pipeline.prope_kv_cache1)):
        values = []
        for cache in caches:
            for tensor_name in ("k", "v"):
                sample = torch.cat([cache[tensor_name][:, part] for part in token_slices], dim=1)
                values.append(tensor_sha256(sample))
        result[cache_name] = values
    cross = []
    for cache in pipeline.crossattn_cache:
        cross.extend((tensor_sha256(cache["k"]), tensor_sha256(cache["v"]), bool(cache["is_init"])))
    result["cross"] = cross
    return result


def camera_chunks(
    device: torch.device,
    action_pair: tuple[str, str],
    trajectory_steps: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    from wan_utils.camera_trajectory import make_camera_tensors

    chunks = {}
    intrinsics = None
    for action in action_pair:
        viewmats, Ks = make_camera_tensors(
            f"{action}*{trajectory_steps}", fx=0.5, fy=0.5, cx=0.5, cy=0.5,
            device=device, dtype=torch.bfloat16,
        )
        chunks[action] = viewmats
        intrinsics = Ks
    assert intrinsics is not None
    return chunks, intrinsics


@torch.inference_mode()
def denoise_block(
    pipeline: CausalInferencePipeline,
    conditional: dict,
    initial_noise: torch.Tensor,
    renoise: list[torch.Tensor],
    view_old: torch.Tensor,
    view_new: torch.Tensor,
    intrinsics: torch.Tensor,
    switch_after: int,
    current_start_frame: int,
    context_write: bool,
) -> tuple[torch.Tensor, dict]:
    noisy_input = initial_noise
    actions_by_nfe = []
    timings = []
    for index, current_timestep in enumerate(pipeline.denoising_step_list):
        chosen = view_old if index < switch_after else view_new
        actions_by_nfe.append("old" if index < switch_after else "new")
        timestep = torch.ones(
            (1, BLOCK_FRAMES), device=initial_noise.device, dtype=torch.int64
        ) * current_timestep
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _, denoised = pipeline.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional,
            timestep=timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,
            viewmats=chosen,
            Ks=intrinsics,
            prope_kv_cache=pipeline.prope_kv_cache1,
        )
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
        if index < 3:
            next_timestep = pipeline.denoising_step_list[index + 1]
            noisy_input = pipeline.scheduler.add_noise(
                denoised.flatten(0, 1),
                renoise[index].to(device=denoised.device).flatten(0, 1),
                next_timestep * torch.ones(
                    (BLOCK_FRAMES,), device=denoised.device, dtype=torch.long
                ),
            ).unflatten(0, denoised.shape[:2])

    context_ms = 0.0
    if context_write:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        pipeline.generator(
            noisy_image_or_video=denoised,
            conditional_dict=conditional,
            timestep=torch.zeros((1, BLOCK_FRAMES), device=denoised.device, dtype=torch.int64),
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,
            viewmats=view_new,
            Ks=intrinsics,
            prope_kv_cache=pipeline.prope_kv_cache1,
        )
        end.record()
        torch.cuda.synchronize()
        context_ms = float(start.elapsed_time(end))
    return denoised, {
        "actions_by_nfe": actions_by_nfe,
        "nfe_milliseconds": timings,
        "context_write_milliseconds": context_ms,
    }


@torch.inference_mode()
def generate_prefix(
    pipeline: CausalInferencePipeline,
    conditional: dict,
    initial: torch.Tensor,
    renoise: list[list[torch.Tensor]],
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, list[dict]]:
    identity = torch.eye(4, device=initial.device, dtype=torch.bfloat16)[None, None].repeat(1, 4, 1, 1)
    pieces = []
    traces = []
    for block in range(4):
        branch, trace = denoise_block(
            pipeline, conditional,
            initial[:, block * 4:(block + 1) * 4], renoise[block],
            identity, identity, intrinsics, 0, block * 4, True,
        )
        pieces.append(branch)
        traces.append(trace)
    return torch.cat(pieces, dim=1), traces


@torch.inference_mode()
def decode_and_save(
    pipeline: CausalInferencePipeline,
    prefix: torch.Tensor,
    branch: torch.Tensor,
    path: Path,
) -> tuple[int, float]:
    started = time.perf_counter()
    video = pipeline.vae.decode_to_pixel(torch.cat((prefix, branch), dim=1), use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    frames = video[0].mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu()
    write_video(str(path), frames, fps=16)
    frame_count = int(frames.shape[0])
    del video, frames
    return frame_count, seconds


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.config)
    action_pair = actions(protocol)
    prompts = load_and_validate_prompts(UPSTREAM, protocol)
    selected_indices = selected_prompt_indices(
        protocol, args.scene_start, args.scene_end
    )
    control_indices = control_prompt_indices(protocol)
    controls_enabled = bool(control_indices) or args.controls
    if protocol.get("phase") == "phase4_minwm_yaw_confirmatory" and not controls_enabled:
        raise ValueError("confirmatory protocol requires same-action controls")
    output = args.output if args.output is not None else default_output(protocol)
    validate_confirmatory_output(output, protocol, require_empty=not args.dry_run)
    report_path = output / f"curve_scenes_{selected_indices[0]:02d}_{selected_indices[-1] + 1:02d}.json"
    specs = unique_run_specs(protocol)
    seed_map = seed_by_prompt(protocol)
    checkpoint_path = MODEL_ROOT / "checkpoints" / protocol["upstream"]["checkpoint_path"]
    if protocol.get("phase") == "phase4_minwm_yaw_confirmatory" and not checkpoint_path.is_file():
        raise FileNotFoundError(f"frozen checkpoint is missing: {checkpoint_path}")

    if args.dry_run:
        run_count = sum(
            len(specs) + (len(switch_positions(protocol)) if index in control_indices else 0)
            for index in selected_indices
        )
        print(json.dumps({
            "status": "dry_run_only_no_model_or_cuda",
            "config": protocol["_config_path"],
            "config_sha256": protocol["_config_sha256"],
            "output": str(output.resolve()),
            "report_path": str(report_path.resolve()),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "actions": list(action_pair),
            "trajectory_steps_per_branch": trajectory_steps_per_branch(protocol),
            "prompt_indices": selected_indices,
            "seeds": [seed_map[index] for index in selected_indices],
            "control_prompt_indices": control_indices,
            "switch_positions": switch_positions(protocol),
            "unique_video_count": run_count,
            "cuda_initialized": torch.cuda.is_initialized(),
        }, indent=2, sort_keys=True))
        return

    output.mkdir(parents=True, exist_ok=False)
    # The official Wan wrapper intentionally resolves base-model assets from
    # repository-relative paths such as Wan21/wan_models/....
    os.chdir(UPSTREAM)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    started_all = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    pipeline = load_pipeline(device, protocol)
    timesteps = [float(value) for value in pipeline.denoising_step_list]
    if len(timesteps) != 4:
        raise RuntimeError(f"frozen Phase 4 requires four NFEs, found {timesteps}")
    views, intrinsics = camera_chunks(
        device, action_pair, trajectory_steps_per_branch(protocol)
    )

    report = {
        "status": "running",
        "phase": protocol["phase"],
        "protocol_config": protocol["_config_path"],
        "protocol_config_sha256": protocol["_config_sha256"],
        "upstream_commit": protocol["upstream"]["commit"],
        "base_revision": protocol["upstream"].get(
            "backbone_revision", "37ec512624d61f7aa208f7ea8140a131f93afc9a"
        ),
        "checkpoint_revision": protocol["upstream"].get(
            "checkpoint_revision", "21bd74da43b5a061c0b8ff277515088ccd2c798b"
        ),
        "checkpoint_etag": protocol["upstream"].get(
            "checkpoint_etag", "bdb947d45fb04513305492c2ee393d51d0621ec0e99fd312224f5d61a330aa77"
        ),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "denoising_timesteps": timesteps,
        "prefix_latents": PREFIX_LATENTS,
        "branch_latents": BLOCK_FRAMES,
        "prefix_pixel_frames": PREFIX_PIXEL_FRAMES,
        "negative_action": action_pair[0],
        "positive_action": action_pair[1],
        "trajectory_steps_per_branch": trajectory_steps_per_branch(protocol),
        "prompt_indices": selected_indices,
        "seeds": [seed_map[index] for index in selected_indices],
        "switch_positions": switch_positions(protocol),
        "same_action_control_prompt_indices": control_indices,
        "camera_viewmat_hashes": {name: tensor_sha256(value) for name, value in views.items()},
        "intrinsics_sha256": tensor_sha256(intrinsics),
        "scenes": {},
    }
    atomic_json(report_path, report)

    for scene_index in selected_indices:
        scene_name = f"scene_{scene_index:02d}"
        scene_dir = args.output / scene_name
        scene_dir.mkdir(exist_ok=True)
        seed = seed_map[scene_index]
        initial_cpu, renoise_cpu = cpu_noise(seed)
        noise_hashes = {
            "initial": tensor_sha256(initial_cpu),
            "renoise": [[tensor_sha256(value) for value in block] for block in renoise_cpu],
        }
        initial = initial_cpu.to(device)
        reset_caches(pipeline, device)
        conditional = pipeline.text_encoder(text_prompts=[prompts[scene_index]])
        prefix_started = time.perf_counter()
        prefix, prefix_trace = generate_prefix(
            pipeline, conditional, initial, renoise_cpu, intrinsics,
        )
        torch.cuda.synchronize()
        prefix_seconds = time.perf_counter() - prefix_started
        prefix_indices = cache_indices(pipeline)
        expected_prefix_tokens = PREFIX_LATENTS * pipeline.frame_seq_length
        if any(pair != [expected_prefix_tokens, expected_prefix_tokens]
               for pairs in prefix_indices.values() for pair in pairs):
            raise RuntimeError(f"unexpected prefix indices: {prefix_indices}")
        signature_before = sampled_state_signature(pipeline, expected_prefix_tokens)

        scene_report = {
            "prompt": prompts[scene_index],
            "seed": seed,
            "noise_hashes": noise_hashes,
            "prefix_latent_sha256": tensor_sha256(prefix),
            "prefix_seconds": prefix_seconds,
            "prefix_trace": prefix_trace,
            "prefix_indices": prefix_indices,
            "runs": {},
        }
        report["scenes"][scene_name] = scene_report
        atomic_json(report_path, report)

        scene_specs = list(specs)
        if controls_enabled and scene_index in control_indices:
            negative = action_pair[0]
            scene_specs.extend(
                (f"{negative}_to_{negative}_s{switch}", negative, negative, switch)
                for switch in switch_positions(protocol)
            )
        for run_name, old_action, new_action, switch_after in scene_specs:
            restore_indices(pipeline, prefix_indices)
            torch.cuda.reset_peak_memory_stats(device)
            run_started = time.perf_counter()
            branch, trace = denoise_block(
                pipeline, conditional, initial[:, 16:20], renoise_cpu[4],
                views[old_action], views[new_action], intrinsics,
                switch_after, 16, False,
            )
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - run_started
            latent_hash = tensor_sha256(branch)
            torch.save(branch.cpu(), scene_dir / f"{run_name}.pt")
            frame_count, decode_seconds = decode_and_save(
                pipeline, prefix, branch, scene_dir / f"{run_name}.mp4"
            )
            scene_report["runs"][run_name] = {
                "old_action": old_action,
                "new_action": new_action,
                "switch_after": switch_after,
                "latent_sha256": latent_hash,
                "noise_hashes": noise_hashes,
                "trace": trace,
                "generation_seconds": generation_seconds,
                "decode_seconds": decode_seconds,
                "frame_count": frame_count,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "indices_after": cache_indices(pipeline),
            }
            atomic_json(report_path, report)
            del branch
            torch.cuda.empty_cache()

        # Branch NFEs legitimately advance the two index pairs to the end of
        # the current block.  Restore the frozen fork indices before comparing
        # the immutable historical/cross tensor state byte-for-byte.
        restore_indices(pipeline, prefix_indices)
        signature_after = sampled_state_signature(pipeline, expected_prefix_tokens)
        scene_report["engineering_checks"] = {
            "all_branch_noise_hashes_exact": all(
                run["noise_hashes"] == noise_hashes for run in scene_report["runs"].values()
            ),
            "historical_and_cross_sampled_state_exact": signature_before == signature_after,
            "all_runs_have_four_nfes": all(
                len(run["trace"]["nfe_milliseconds"]) == 4
                for run in scene_report["runs"].values()
            ),
            "same_action_latents_exact": (
                len({
                    scene_report["runs"][f"{action_pair[0]}_to_{action_pair[0]}_s{s}"]["latent_sha256"]
                    for s in switch_positions(protocol)
                }) == 1
                if controls_enabled and scene_index in control_indices else None
            ),
        }
        atomic_json(report_path, report)
        del conditional, prefix, initial, initial_cpu, renoise_cpu
        gc.collect()
        torch.cuda.empty_cache()

    report["status"] = "complete"
    report["elapsed_seconds"] = time.time() - started_all
    report["process_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    report["finished_unix"] = time.time()
    atomic_json(report_path, report)


if __name__ == "__main__":
    main()
