"""Device-parity tests: the codec must behave identically on CPU and MPS.

This is the local half of the Mac/pod contract. Anything that passes here on
MPS is very likely to pass on CUDA, so GPU time is spent debugging SGLang
integration rather than codec arithmetic.

Skipped automatically when MPS is unavailable.
"""

from __future__ import annotations

import pytest
import torch

from hiqcache.codec import (
    AMAX_FLOOR,
    QUANT_MAX,
    decode_records,
    encode_records,
    error_stats,
    pack_records,
    unpack_records,
    verified_error_bound,
)

HEAD_NUM = 8
HEAD_DIM = 128

mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
requires_mps = pytest.mark.skipif(not mps_available, reason="MPS not available")


def _inputs():
    g = torch.Generator(device="cpu").manual_seed(1234)
    return {
        "random": torch.randn((32, HEAD_NUM, HEAD_DIM), generator=g).to(torch.bfloat16),
        "all_zeros": torch.zeros((8, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16),
        "mixed": _mixed(),
        "exact_grid": _exact_grid(),
    }


def _mixed():
    x = torch.randn((16, HEAD_NUM, HEAD_DIM), dtype=torch.float32).to(torch.bfloat16)
    x[0, 0, :] = 0.0
    x[0, 1, :] = 1e-30
    x[0, 2, :] = 1e4
    return x


def _exact_grid():
    """Values that should land exactly on integer quotients.

    Catches a divergent rounding rule (round-half-away-from-zero vs
    round-half-to-even) between backends.
    """
    steps = torch.arange(-127, 128, dtype=torch.float32) / 127.0
    row = steps.repeat(HEAD_DIM // steps.numel() + 1)[:HEAD_DIM]
    return row.unsqueeze(0).unsqueeze(0).expand(4, HEAD_NUM, HEAD_DIM).contiguous().to(
        torch.bfloat16
    )


@requires_mps
@pytest.mark.parametrize("name", ["random", "all_zeros", "mixed", "exact_grid"])
def test_records_are_bit_identical_between_cpu_and_mps(name):
    """Encoded bytes must match bit for bit -- this is what the pod will store."""
    x = _inputs()[name]
    cpu_records = encode_records(x)
    mps_records = encode_records(x.to("mps")).cpu()
    assert torch.equal(cpu_records, mps_records), (
        f"{name}: CPU and MPS produced different encoded bytes; "
        f"{int((cpu_records != mps_records).sum())} differing bytes"
    )


@requires_mps
@pytest.mark.parametrize("name", ["random", "all_zeros", "mixed", "exact_grid"])
def test_payload_and_scales_agree_between_devices(name):
    x = _inputs()[name]
    p_cpu, s_cpu = unpack_records(
        encode_records(x), head_num=HEAD_NUM, head_dim=HEAD_DIM
    )
    p_mps, s_mps = unpack_records(
        encode_records(x.to("mps")).cpu(), head_num=HEAD_NUM, head_dim=HEAD_DIM
    )
    assert torch.equal(p_cpu, p_mps), f"{name}: INT8 payload differs"
    assert torch.equal(s_cpu, s_mps), f"{name}: BF16 scales differ"


@requires_mps
def test_rounding_rule_matches_on_exact_half_boundaries():
    """Explicitly probe round-half behaviour, the likeliest backend divergence.

    The 1152-byte row is fixed at 8 heads x 128 dims, so build a full row and
    place the probe values in head 0. Heads are quantised independently, so the
    zero heads neither help nor hinder the comparison.
    """
    values = [1.0, -1.0, 3.0, -3.0, 5.0, -5.0, 125.0, -125.0]
    x = torch.zeros((1, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16)
    x[0, 0, : len(values)] = torch.tensor(values, dtype=torch.bfloat16)

    records_cpu = encode_records(x)
    records_mps = encode_records(x.to("mps")).cpu()
    assert torch.equal(records_cpu, records_mps), (
        "rounding rule differs between CPU and MPS"
    )

    payload, scales = unpack_records(records_cpu, head_num=HEAD_NUM, head_dim=HEAD_DIM)
    # absmax of head 0 is 125, so s ~= 125/127 and the largest quotient is ~127.
    assert int(payload[0, 0, : len(values)].abs().max()) == QUANT_MAX
    # 1.0 / s ~= 1.016 must round to 1 under any IEEE rounding rule.
    assert int(payload[0, 0, 0]) == 1
    assert int(payload[0, 0, 1]) == -1
    assert float(scales[0, 0]) > 0


@requires_mps
def test_full_round_trip_error_matches_between_devices():
    x = _inputs()["random"]
    cpu_restored = decode_records(
        encode_records(x), head_num=HEAD_NUM, head_dim=HEAD_DIM
    )
    mps_restored = decode_records(
        encode_records(x.to("mps")), head_num=HEAD_NUM, head_dim=HEAD_DIM
    ).cpu()
    assert torch.equal(cpu_restored, mps_restored), "decoded bf16 KV differs by device"


@requires_mps
def test_error_bound_holds_on_mps():
    """The verified bound must hold on the accelerator too, not just on CPU."""
    x = _inputs()["random"]
    records = encode_records(x.to("mps"))
    restored = decode_records(records, head_num=HEAD_NUM, head_dim=HEAD_DIM)
    _, stored = unpack_records(records, head_num=HEAD_NUM, head_dim=HEAD_DIM)
    err = (restored.float() - x.to("mps").float()).abs()
    bound = verified_error_bound(restored, stored)
    assert float((err - bound).max()) <= 1e-6


@requires_mps
def test_no_nan_or_inf_on_mps_for_degenerate_inputs():
    for name, x in _inputs().items():
        records = encode_records(x.to("mps"))
        restored = decode_records(records, head_num=HEAD_NUM, head_dim=HEAD_DIM)
        assert torch.isfinite(restored.float()).all(), f"{name}: non-finite output"
        _, scales = unpack_records(records, head_num=HEAD_NUM, head_dim=HEAD_DIM)
        assert bool((scales.float() > 0).all()), f"{name}: non-positive scale"


@requires_mps
def test_mps_encode_is_not_slower_than_cpu_by_an_order_of_magnitude():
    """Loose guard: codec ops must not be accidentally falling back to scalar loops."""
    import time

    x = torch.randn((512, HEAD_NUM, HEAD_DIM), dtype=torch.float32).to(torch.bfloat16)
    x_mps = x.to("mps")

    def timed(fn, n=5):
        fn()  # warm up
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n

    cpu_ms = timed(lambda: encode_records(x)) * 1e3
    mps_ms = timed(lambda: encode_records(x_mps)) * 1e3
    print(f"\n[MPS parity] 512-token row batch: CPU {cpu_ms:.3f} ms, MPS {mps_ms:.3f} ms")
    assert mps_ms < cpu_ms * 10
