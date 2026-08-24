"""Exact current-block checkpointing for LateAct Gate 1."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange

from lateact.runtime import assert_history_guard, tensor_sha256, tree_sha256


@dataclass
class CacheSlice:
    start: int
    end: int
    key: torch.Tensor
    value: torch.Tensor
    global_end: int
    local_end: int


@dataclass
class BoundaryCheckpoint:
    entering_latent: torch.Tensor
    visual: list[CacheSlice]
    mouse: list[CacheSlice]
    keyboard: list[CacheSlice]
    cross_hash: str
    scheduler_hash: str
    cpu_rng_hash: str
    cuda_rng_hash: str
    step_entering: int


def rng_hashes(device: torch.device) -> dict[str, str]:
    return {
        "cpu": tensor_sha256(torch.get_rng_state()),
        "cuda": tensor_sha256(torch.cuda.get_rng_state(device)),
    }


def scheduler_state(pipeline) -> dict[str, Any]:
    scheduler = pipeline.scheduler
    return {
        "num_train_timesteps": scheduler.num_train_timesteps,
        "shift": scheduler.shift,
        "sigma_max": scheduler.sigma_max,
        "sigma_min": scheduler.sigma_min,
        "inverse_timesteps": scheduler.inverse_timesteps,
        "extra_one_step": scheduler.extra_one_step,
        "reverse_sigmas": scheduler.reverse_sigmas,
        "sigmas": scheduler.sigmas,
        "timesteps": scheduler.timesteps,
        "linear_timesteps_weights": scheduler.linear_timesteps_weights,
    }


def scheduler_hash(pipeline) -> str:
    return tree_sha256(scheduler_state(pipeline))


def _capture_slices(caches: list[dict], prefix_ends: list[int]) -> list[CacheSlice]:
    result = []
    for cache, start in zip(caches, prefix_ends):
        end = int(cache["local_end_index"].item())
        result.append(
            CacheSlice(
                start=start,
                end=end,
                key=cache["k"][:, start:end].clone(),
                value=cache["v"][:, start:end].clone(),
                global_end=int(cache["global_end_index"].item()),
                local_end=end,
            )
        )
    return result


def capture_boundary(
    pipeline,
    *,
    entering_latent: torch.Tensor,
    history_guard: dict[str, Any],
    step_entering: int,
) -> BoundaryCheckpoint:
    device = entering_latent.device
    assert_history_guard(pipeline, history_guard)
    hashes = rng_hashes(device)
    return BoundaryCheckpoint(
        entering_latent=entering_latent.clone(),
        visual=_capture_slices(pipeline.kv_cache1, history_guard["visual_ends"]),
        mouse=_capture_slices(pipeline.kv_cache_mouse, history_guard["mouse_ends"]),
        keyboard=_capture_slices(pipeline.kv_cache_keyboard, history_guard["keyboard_ends"]),
        cross_hash=tree_sha256(pipeline.crossattn_cache),
        scheduler_hash=scheduler_hash(pipeline),
        cpu_rng_hash=hashes["cpu"],
        cuda_rng_hash=hashes["cuda"],
        step_entering=step_entering,
    )


def _restore_slices(caches: list[dict], slices: list[CacheSlice]) -> None:
    for cache, saved in zip(caches, slices):
        cache["k"][:, saved.start:saved.end].copy_(saved.key)
        cache["v"][:, saved.start:saved.end].copy_(saved.value)
        cache["global_end_index"].fill_(saved.global_end)
        cache["local_end_index"].fill_(saved.local_end)


def restore_boundary(
    pipeline,
    checkpoint: BoundaryCheckpoint,
    history_guard: dict[str, Any],
    *,
    verify: bool = True,
) -> torch.Tensor:
    """Restore only current-block data; prefix and cross state must be untouched."""
    if verify:
        assert_history_guard(pipeline, history_guard)
        if tree_sha256(pipeline.crossattn_cache) != checkpoint.cross_hash:
            raise RuntimeError("cross-attention state changed before rollback")
        if scheduler_hash(pipeline) != checkpoint.scheduler_hash:
            raise RuntimeError("scheduler state changed before rollback")
    _restore_slices(pipeline.kv_cache1, checkpoint.visual)
    _restore_slices(pipeline.kv_cache_mouse, checkpoint.mouse)
    _restore_slices(pipeline.kv_cache_keyboard, checkpoint.keyboard)
    if verify:
        assert_history_guard(pipeline, history_guard)
    return checkpoint.entering_latent.clone()


def assert_checkpoint_live(pipeline, checkpoint: BoundaryCheckpoint) -> None:
    """Prove the live current-block slices and indices exactly match a checkpoint."""
    for name, caches, slices in (
        ("visual", pipeline.kv_cache1, checkpoint.visual),
        ("mouse", pipeline.kv_cache_mouse, checkpoint.mouse),
        ("keyboard", pipeline.kv_cache_keyboard, checkpoint.keyboard),
    ):
        for layer, (cache, saved) in enumerate(zip(caches, slices)):
            if not torch.equal(cache["k"][:, saved.start:saved.end], saved.key):
                raise RuntimeError(f"{name} checkpoint K mismatch in layer {layer}")
            if not torch.equal(cache["v"][:, saved.start:saved.end], saved.value):
                raise RuntimeError(f"{name} checkpoint V mismatch in layer {layer}")
            if int(cache["global_end_index"].item()) != saved.global_end:
                raise RuntimeError(f"{name} checkpoint global index mismatch in layer {layer}")
            if int(cache["local_end_index"].item()) != saved.local_end:
                raise RuntimeError(f"{name} checkpoint local index mismatch in layer {layer}")


def reset_to_prefix(
    pipeline,
    *,
    initial_latent: torch.Tensor,
    history_guard: dict[str, Any],
    prefix_global_ends: dict[str, list[int]],
    verify: bool = True,
) -> torch.Tensor:
    """Logically discard current-block cache state without touching the prefix."""
    if verify:
        assert_history_guard(pipeline, history_guard)
    for name, caches in (
        ("visual", pipeline.kv_cache1),
        ("mouse", pipeline.kv_cache_mouse),
        ("keyboard", pipeline.kv_cache_keyboard),
    ):
        for cache, local_end, global_end in zip(
            caches, history_guard[f"{name}_ends"], prefix_global_ends[name]
        ):
            cache["local_end_index"].fill_(local_end)
            cache["global_end_index"].fill_(global_end)
    if verify:
        assert_history_guard(pipeline, history_guard)
    return initial_latent.clone()


def checkpoint_nbytes(checkpoint: BoundaryCheckpoint) -> dict[str, int]:
    def slices_bytes(values: list[CacheSlice]) -> int:
        return sum(
            item.key.numel() * item.key.element_size()
            + item.value.numel() * item.value.element_size()
            for item in values
        )

    values = {
        "entering_latent": checkpoint.entering_latent.numel()
        * checkpoint.entering_latent.element_size(),
        "visual_current_slice": slices_bytes(checkpoint.visual),
        "mouse_current_slice": slices_bytes(checkpoint.mouse),
        "keyboard_current_slice": slices_bytes(checkpoint.keyboard),
    }
    values["tensor_total"] = sum(values.values())
    values["index_metadata_estimate"] = 30 * 3 * 2 * 8
    values["total"] = values["tensor_total"] + values["index_metadata_estimate"]
    return values


def checkpoint_hashes(checkpoint: BoundaryCheckpoint) -> dict[str, Any]:
    def slice_hashes(values: list[CacheSlice]) -> str:
        digest = hashlib.sha256()
        for item in values:
            digest.update(str((item.start, item.end, item.global_end, item.local_end)).encode())
            digest.update(tensor_sha256(item.key).encode())
            digest.update(tensor_sha256(item.value).encode())
        return digest.hexdigest()

    return {
        "entering_latent": tensor_sha256(checkpoint.entering_latent),
        "visual_current_slices": slice_hashes(checkpoint.visual),
        "mouse_current_slices": slice_hashes(checkpoint.mouse),
        "keyboard_current_slices": slice_hashes(checkpoint.keyboard),
        "cross": checkpoint.cross_hash,
        "scheduler": checkpoint.scheduler_hash,
        "cpu_rng": checkpoint.cpu_rng_hash,
        "cuda_rng": checkpoint.cuda_rng_hash,
    }


@torch.no_grad()
def run_nfe(
    pipeline,
    *,
    noisy_input: torch.Tensor,
    condition: dict,
    current_start: int,
    step_index: int,
    renoise: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from pipeline.causal_inference import cond_current

    device, dtype = noisy_input.device, noisy_input.dtype
    frames = pipeline.num_frame_per_block
    current_timestep = pipeline.denoising_step_list[step_index]
    timestep = torch.full(
        (noisy_input.shape[0], frames), current_timestep, dtype=torch.int64, device=device
    )
    _, denoised = pipeline.generator(
        noisy_image_or_video=noisy_input,
        conditional_dict=cond_current(condition, current_start, frames, mode="universal"),
        timestep=timestep,
        kv_cache=pipeline.kv_cache1,
        kv_cache_mouse=pipeline.kv_cache_mouse,
        kv_cache_keyboard=pipeline.kv_cache_keyboard,
        crossattn_cache=pipeline.crossattn_cache,
        current_start=current_start * pipeline.frame_seq_length,
    )
    if step_index == len(pipeline.denoising_step_list) - 1:
        return denoised.detach(), None
    explicit_noise = renoise[step_index].to(device=device, dtype=dtype)
    next_timestep = pipeline.denoising_step_list[step_index + 1]
    next_input = pipeline.scheduler.add_noise(
        rearrange(denoised, "b c f h w -> (b f) c h w"),
        explicit_noise,
        next_timestep
        * torch.ones(denoised.shape[0] * frames, dtype=torch.long, device=device),
    )
    next_input = rearrange(next_input, "(b f) c h w -> b c f h w", b=denoised.shape[0])
    return denoised.detach(), next_input


@torch.no_grad()
def commit_context(pipeline, denoised: torch.Tensor, condition: dict, current_start: int) -> None:
    from pipeline.causal_inference import cond_current

    frames = pipeline.num_frame_per_block
    timestep = torch.full(
        (denoised.shape[0], frames),
        pipeline.args.context_noise,
        dtype=torch.int64,
        device=denoised.device,
    )
    pipeline.generator(
        noisy_image_or_video=denoised,
        conditional_dict=cond_current(condition, current_start, frames, mode="universal"),
        timestep=timestep,
        kv_cache=pipeline.kv_cache1,
        kv_cache_mouse=pipeline.kv_cache_mouse,
        kv_cache_keyboard=pipeline.kv_cache_keyboard,
        crossattn_cache=pipeline.crossattn_cache,
        current_start=current_start * pipeline.frame_seq_length,
    )


@torch.no_grad()
def run_steps(
    pipeline,
    *,
    noisy_input: torch.Tensor,
    condition: dict,
    current_start: int,
    start_step: int,
    renoise: list[torch.Tensor],
    commit: bool = True,
) -> torch.Tensor:
    denoised = None
    current = noisy_input
    for step in range(start_step, len(pipeline.denoising_step_list)):
        denoised, next_input = run_nfe(
            pipeline,
            noisy_input=current,
            condition=condition,
            current_start=current_start,
            step_index=step,
            renoise=renoise,
        )
        if next_input is not None:
            current = next_input
    assert denoised is not None
    if commit:
        commit_context(pipeline, denoised, condition, current_start)
    return denoised
