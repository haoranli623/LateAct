import torch

from branchsafe.cache_ops import append_and_read, make_cache, move_range, payload_nbytes, read_range, write_range


def test_int8_roundtrip_and_bytes():
    torch.manual_seed(0)
    shape = (1, 7, 3, 8)
    source = torch.randn(shape, dtype=torch.bfloat16)
    bf16 = make_cache(shape, storage_format="bf16", compute_dtype=torch.bfloat16, device="cpu")
    int8 = make_cache(shape, storage_format="int8", compute_dtype=torch.bfloat16, device="cpu")
    write_range(bf16, "k", 0, 7, source)
    write_range(int8, "k", 0, 7, source)
    restored = read_range(int8, "k", 0, 7, torch.bfloat16)
    assert int8["k"].dtype == torch.int8
    assert torch.max(torch.abs(restored.float() - source.float())) < 0.02
    assert payload_nbytes([int8]) < payload_nbytes([bf16])


def test_int4_roundtrip_and_move_keeps_scale_aligned():
    torch.manual_seed(1)
    shape = (1, 6, 2, 8)
    source = torch.randn(shape, dtype=torch.bfloat16)
    cache = make_cache(shape, storage_format="int4", compute_dtype=torch.bfloat16, device="cpu")
    write_range(cache, "v", 0, 6, source)
    before = read_range(cache, "v", 2, 6, torch.bfloat16).clone()
    move_range(cache, "v", 0, 2, 4)
    after = read_range(cache, "v", 0, 4, torch.bfloat16)
    assert cache["v"].dtype == torch.uint8
    assert torch.equal(before, after)
    assert torch.max(torch.abs(after.float() - source[:, 2:6].float())) < 0.35


def test_query_larger_than_matched_eviction_cache_is_transient():
    cache = make_cache(
        (1, 2, 1, 4), storage_format="bf16", compute_dtype=torch.bfloat16, device="cpu"
    )
    history = torch.tensor([[[[1, 1, 1, 1]], [[2, 2, 2, 2]]]], dtype=torch.bfloat16)
    write_range(cache, "k", 0, 2, history)
    write_range(cache, "v", 0, 2, history + 10)
    cache["global_end_index"].fill_(2)
    cache["local_end_index"].fill_(2)
    current = torch.tensor(
        [[[[3, 3, 3, 3]], [[4, 4, 4, 4]], [[5, 5, 5, 5]]]], dtype=torch.bfloat16
    )
    attention_k, _ = append_and_read(
        cache, current, current + 10, current_start=2, sink_tokens=0, max_attention_size=6
    )
    assert attention_k.shape[1] == 5
    assert torch.equal(attention_k[:, :2], history)
    assert torch.equal(read_range(cache, "k", 0, 2, torch.bfloat16), current[:, -2:])
    assert cache["global_end_index"].item() == 5
    assert cache["local_end_index"].item() == 2

    replacement = current + 20
    attention_k, _ = append_and_read(
        cache, replacement, replacement + 10, current_start=2, sink_tokens=0, max_attention_size=6
    )
    assert torch.equal(attention_k, replacement)
    assert torch.equal(read_range(cache, "k", 0, 2, torch.bfloat16), replacement[:, -2:])
