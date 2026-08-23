"""Explicit-noise, action-switching Matrix-Game runtime primitives."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange


@dataclass
class NoiseBundle:
    initial: torch.Tensor
    renoise: list[torch.Tensor]


@dataclass
class RuntimeSnapshot:
    visual: list[dict]
    mouse: list[dict]
    keyboard: list[dict]
    cross: list[dict]
    current_start: int


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                update(child)
        else:
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def tree_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tree_nbytes(item) for item in value)
    return 0


def make_noise_bundle(
    seed: int,
    *,
    frames: int = 3,
    steps: int = 3,
    channels: int = 16,
    height: int = 44,
    width: int = 80,
) -> NoiseBundle:
    """Materialize every stochastic tensor on CPU for exact branch reuse."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn(
        (1, channels, frames, height, width), generator=rng, dtype=torch.float32
    ).to(torch.bfloat16)
    renoise = [
        torch.randn(
            (frames, channels, height, width), generator=rng, dtype=torch.float32
        ).to(torch.bfloat16)
        for _ in range(steps - 1)
    ]
    return NoiseBundle(initial=initial, renoise=renoise)


def noise_hashes(bundle: NoiseBundle) -> dict[str, Any]:
    return {
        "initial": tensor_sha256(bundle.initial),
        "renoise": [tensor_sha256(value) for value in bundle.renoise],
    }


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    return copy.deepcopy(value)


def _to_device(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return copy.deepcopy(value)


def snapshot_pipeline(pipeline, current_start: int) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        visual=_cpu_clone(pipeline.kv_cache1),
        mouse=_cpu_clone(pipeline.kv_cache_mouse),
        keyboard=_cpu_clone(pipeline.kv_cache_keyboard),
        cross=_cpu_clone(pipeline.crossattn_cache),
        current_start=current_start,
    )


def restore_pipeline(pipeline, snapshot: RuntimeSnapshot, device) -> int:
    pipeline.kv_cache1 = _to_device(snapshot.visual, device)
    pipeline.kv_cache_mouse = _to_device(snapshot.mouse, device)
    pipeline.kv_cache_keyboard = _to_device(snapshot.keyboard, device)
    pipeline.crossattn_cache = _to_device(snapshot.cross, device)
    return snapshot.current_start


def cache_bytes(pipeline) -> dict[str, int]:
    values = {
        "visual": tree_nbytes(pipeline.kv_cache1),
        "mouse": tree_nbytes(pipeline.kv_cache_mouse),
        "keyboard": tree_nbytes(pipeline.kv_cache_keyboard),
        "cross": tree_nbytes(pipeline.crossattn_cache),
    }
    values["total"] = sum(values.values())
    return values


def _active_ends(caches: list[dict]) -> list[int]:
    return [int(cache["local_end_index"].item()) for cache in caches]


def make_history_guard(pipeline) -> dict[str, Any]:
    """Clone active prefix regions on-device for exact post-call assertions."""
    guard: dict[str, Any] = {
        "visual_ends": _active_ends(pipeline.kv_cache1),
        "mouse_ends": _active_ends(pipeline.kv_cache_mouse),
        "keyboard_ends": _active_ends(pipeline.kv_cache_keyboard),
        "visual": [],
        "mouse": [],
        "keyboard": [],
        "cross": [],
    }
    for name, caches in (
        ("visual", pipeline.kv_cache1),
        ("mouse", pipeline.kv_cache_mouse),
        ("keyboard", pipeline.kv_cache_keyboard),
    ):
        ends = guard[f"{name}_ends"]
        guard[name] = [
            {
                "k": cache["k"][:, :end].clone(),
                "v": cache["v"][:, :end].clone(),
            }
            for cache, end in zip(caches, ends)
        ]
    guard["cross"] = [
        {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in cache.items()}
        for cache in pipeline.crossattn_cache
    ]
    return guard


def assert_history_guard(pipeline, guard: dict[str, Any]) -> None:
    for name, caches in (
        ("visual", pipeline.kv_cache1),
        ("mouse", pipeline.kv_cache_mouse),
        ("keyboard", pipeline.kv_cache_keyboard),
    ):
        for layer, (cache, expected, end) in enumerate(
            zip(caches, guard[name], guard[f"{name}_ends"])
        ):
            if not torch.equal(cache["k"][:, :end], expected["k"]):
                raise RuntimeError(f"{name} historical K changed in layer {layer}")
            if not torch.equal(cache["v"][:, :end], expected["v"]):
                raise RuntimeError(f"{name} historical V changed in layer {layer}")
    for layer, (cache, expected) in enumerate(zip(pipeline.crossattn_cache, guard["cross"])):
        for key in ("k", "v"):
            if not torch.equal(cache[key], expected[key]):
                raise RuntimeError(f"cross-attention {key} changed in layer {layer}")
        if cache["is_init"] != expected["is_init"]:
            raise RuntimeError(f"cross-attention init flag changed in layer {layer}")


def cache_index_trace(pipeline) -> dict[str, list[list[int]]]:
    return {
        name: [
            [int(cache["global_end_index"].item()), int(cache["local_end_index"].item())]
            for cache in caches
        ]
        for name, caches in (
            ("visual", pipeline.kv_cache1),
            ("mouse", pipeline.kv_cache_mouse),
            ("keyboard", pipeline.kv_cache_keyboard),
        )
    }


@torch.no_grad()
def generate_prefix(pipeline, noise: NoiseBundle, condition: dict) -> tuple[torch.Tensor, int]:
    """Generate one clean-context-committed prefix block."""
    return generate_switched_block(
        pipeline,
        noise=noise,
        old_condition=condition,
        new_condition=condition,
        current_start=0,
        switch_after=0,
        audit_history=False,
    )[:2]


@torch.no_grad()
def generate_switched_block(
    pipeline,
    *,
    noise: NoiseBundle,
    old_condition: dict,
    new_condition: dict,
    current_start: int,
    switch_after: int,
    audit_history: bool = True,
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    """Denoise one block while changing only action input after an NFE boundary."""
    from pipeline.causal_inference import cond_current

    steps = len(pipeline.denoising_step_list)
    if not 0 <= switch_after <= steps:
        raise ValueError(f"switch_after must be in [0,{steps}]")
    if len(noise.renoise) != steps - 1:
        raise ValueError("explicit re-noising tensor count mismatch")

    device = next(pipeline.generator.parameters()).device
    dtype = next(pipeline.generator.parameters()).dtype
    frames = pipeline.num_frame_per_block
    if noise.initial.shape[2] != frames:
        raise ValueError("LateAct runner accepts exactly one output block")

    noisy_input = noise.initial.to(device=device, dtype=dtype)
    guard = make_history_guard(pipeline) if audit_history else None
    trace: dict[str, Any] = {
        "switch_after": switch_after,
        "nfe": steps,
        "context_cache_writes": 1,
        "actions_by_nfe": [],
        "indices_before": cache_index_trace(pipeline),
        "indices_after_nfe": [],
        "history_exact_after_nfe": [],
    }

    denoised = None
    timestep = None
    for step_idx, current_timestep in enumerate(pipeline.denoising_step_list):
        action_name = "old" if step_idx < switch_after else "new"
        condition = old_condition if action_name == "old" else new_condition
        current_condition = cond_current(condition, current_start, frames, mode="universal")
        trace["actions_by_nfe"].append(action_name)
        timestep = torch.full(
            (noisy_input.shape[0], frames),
            current_timestep,
            dtype=torch.int64,
            device=device,
        )
        _, denoised = pipeline.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=current_condition,
            timestep=timestep,
            kv_cache=pipeline.kv_cache1,
            kv_cache_mouse=pipeline.kv_cache_mouse,
            kv_cache_keyboard=pipeline.kv_cache_keyboard,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start * pipeline.frame_seq_length,
        )
        if guard is not None:
            assert_history_guard(pipeline, guard)
        trace["history_exact_after_nfe"].append(True)
        trace["indices_after_nfe"].append(cache_index_trace(pipeline))

        if step_idx < steps - 1:
            explicit_noise = noise.renoise[step_idx].to(device=device, dtype=dtype)
            next_timestep = pipeline.denoising_step_list[step_idx + 1]
            noisy_input = pipeline.scheduler.add_noise(
                rearrange(denoised, "b c f h w -> (b f) c h w"),
                explicit_noise,
                next_timestep
                * torch.ones(
                    denoised.shape[0] * frames, dtype=torch.long, device=device
                ),
            )
            noisy_input = rearrange(
                noisy_input, "(b f) c h w -> b c f h w", b=denoised.shape[0]
            )

    assert denoised is not None and timestep is not None
    final_condition = old_condition if switch_after == steps else new_condition
    pipeline.generator(
        noisy_image_or_video=denoised,
        conditional_dict=cond_current(final_condition, current_start, frames, mode="universal"),
        timestep=torch.full_like(timestep, pipeline.args.context_noise),
        kv_cache=pipeline.kv_cache1,
        kv_cache_mouse=pipeline.kv_cache_mouse,
        kv_cache_keyboard=pipeline.kv_cache_keyboard,
        crossattn_cache=pipeline.crossattn_cache,
        current_start=current_start * pipeline.frame_seq_length,
    )
    if guard is not None:
        assert_history_guard(pipeline, guard)
    trace["history_exact_after_context_write"] = True
    trace["indices_after_context_write"] = cache_index_trace(pipeline)
    trace["context_action"] = "old" if switch_after == steps else "new"
    return denoised.detach(), current_start + frames, trace

