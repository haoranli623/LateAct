"""Typed storage operations for Matrix-Game visual causal KV caches.

Quantization is symmetric and per token/head vector. Scales are rounded to the
requested compute dtype before the integer payload is produced, so the writer
and reader use exactly the same stored scale.
"""

from __future__ import annotations

from typing import Iterable

import torch


SUPPORTED_FORMATS = {"bf16", "int8", "int4"}


def cache_format(cache: dict) -> str:
    return cache.get("cache_format", "bf16")


def make_cache(
    shape: tuple[int, int, int, int],
    *,
    storage_format: str,
    compute_dtype: torch.dtype,
    device: torch.device | str,
) -> dict:
    if storage_format not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported cache format: {storage_format}")
    if shape[-1] % 2 and storage_format == "int4":
        raise ValueError("INT4 packing requires an even head dimension")

    payload_shape = shape if storage_format != "int4" else (*shape[:-1], shape[-1] // 2)
    payload_dtype = compute_dtype if storage_format == "bf16" else (
        torch.int8 if storage_format == "int8" else torch.uint8
    )
    cache = {
        "k": torch.zeros(payload_shape, dtype=payload_dtype, device=device),
        "v": torch.zeros(payload_shape, dtype=payload_dtype, device=device),
        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "cache_format": storage_format,
        "compute_dtype": compute_dtype,
        "head_dim": shape[-1],
    }
    if storage_format != "bf16":
        scale_shape = (*shape[:-1], 1)
        cache["k_scale"] = torch.ones(scale_shape, dtype=compute_dtype, device=device)
        cache["v_scale"] = torch.ones(scale_shape, dtype=compute_dtype, device=device)
    return cache


def _quantize(value: torch.Tensor, levels: int, scale_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = value.float().abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / levels).clamp_min(torch.finfo(torch.float32).tiny).to(scale_dtype)
    quantized = torch.round(value.float() / scale.float()).clamp(-levels, levels).to(torch.int8)
    return quantized, scale


def _pack_int4(value: torch.Tensor) -> torch.Tensor:
    encoded = (value.to(torch.int16) + 8).to(torch.uint8)
    return encoded[..., 0::2] | (encoded[..., 1::2] << 4)


def _unpack_int4(value: torch.Tensor, head_dim: int) -> torch.Tensor:
    low = (value & 0x0F).to(torch.int16) - 8
    high = (value >> 4).to(torch.int16) - 8
    return torch.stack((low, high), dim=-1).flatten(-2)[..., :head_dim]


def write_range(cache: dict, name: str, start: int, end: int, value: torch.Tensor) -> None:
    fmt = cache_format(cache)
    if fmt == "bf16":
        cache[name][:, start:end] = value
        return

    levels = 127 if fmt == "int8" else 7
    quantized, scale = _quantize(value, levels, cache["compute_dtype"])
    if fmt == "int4":
        quantized = _pack_int4(quantized)
    cache[name][:, start:end] = quantized
    cache[f"{name}_scale"][:, start:end] = scale


def read_range(cache: dict, name: str, start: int, end: int, dtype: torch.dtype) -> torch.Tensor:
    fmt = cache_format(cache)
    payload = cache[name][:, start:end]
    if fmt == "bf16":
        return payload
    if fmt == "int4":
        payload = _unpack_int4(payload, cache["head_dim"])
    return payload.to(dtype) * cache[f"{name}_scale"][:, start:end].to(dtype)


def move_range(cache: dict, name: str, dst_start: int, src_start: int, length: int) -> None:
    if length <= 0:
        return
    cache[name][:, dst_start:dst_start + length] = cache[name][
        :, src_start:src_start + length
    ].clone()
    if cache_format(cache) != "bf16":
        scale_name = f"{name}_scale"
        cache[scale_name][:, dst_start:dst_start + length] = cache[scale_name][
            :, src_start:src_start + length
        ].clone()


def append_and_read(
    cache: dict,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    current_start: int,
    sink_tokens: int,
    max_attention_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update a causal cache and return the K/V used by attention.

    The special path where a whole query block is larger than the persistent
    cache makes matched-byte eviction well-defined: the current block remains
    available transiently to attention, while only the newest byte-budgeted
    tail persists afterward.
    """
    capacity = cache["k"].shape[1]
    new_tokens = key.shape[1]
    current_end = current_start + new_tokens
    old_global_end = int(cache["global_end_index"].item())
    old_local_end = int(cache["local_end_index"].item())

    if new_tokens > capacity:
        if current_end > old_global_end:
            history_limit = max(0, max_attention_size - new_tokens)
            history_start = max(0, old_local_end - history_limit)
            history_key = read_range(cache, "k", history_start, old_local_end, key.dtype)
            history_value = read_range(cache, "v", history_start, old_local_end, value.dtype)
        else:
            history_key = key[:, :0]
            history_value = value[:, :0]
        attention_key = torch.cat([history_key, key], dim=1)
        attention_value = torch.cat([history_value, value], dim=1)
        retained = min(capacity, attention_key.shape[1])
        write_range(cache, "k", 0, retained, attention_key[:, -retained:])
        write_range(cache, "v", 0, retained, attention_value[:, -retained:])
        local_end = retained
    else:
        if current_end > old_global_end and new_tokens + old_local_end > capacity:
            evicted = new_tokens + old_local_end - capacity
            rolled = old_local_end - evicted - sink_tokens
            move_range(cache, "k", sink_tokens, sink_tokens + evicted, rolled)
            move_range(cache, "v", sink_tokens, sink_tokens + evicted, rolled)
            local_end = old_local_end + current_end - old_global_end - evicted
        else:
            local_end = old_local_end + current_end - old_global_end
        local_start = local_end - new_tokens
        write_range(cache, "k", local_start, local_end, key)
        write_range(cache, "v", local_start, local_end, value)
        attention_start = max(0, local_end - max_attention_size)
        attention_key = read_range(cache, "k", attention_start, local_end, key.dtype)
        attention_value = read_range(cache, "v", attention_start, local_end, value.dtype)

    cache["global_end_index"].fill_(current_end)
    cache["local_end_index"].fill_(local_end)
    return attention_key, attention_value


def payload_tensors(cache: dict) -> Iterable[torch.Tensor]:
    for name in ("k", "v", "k_scale", "v_scale", "global_end_index", "local_end_index"):
        value = cache.get(name)
        if isinstance(value, torch.Tensor):
            yield value


def payload_nbytes(caches: Iterable[dict]) -> int:
    return sum(t.numel() * t.element_size() for cache in caches for t in payload_tensors(cache))
