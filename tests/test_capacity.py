"""Host-capacity arithmetic tests.

These assert the *fixed-GB L2 capacity increase* -- the headline claim of
Experiment B (``PROJECT.md`` Phase 13). They run locally with no GPU and
re-derive SGLang's sizing formula independently so a drift in either the
calculator or the codec format is caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from capacity_table import capacity_table, host_pool_size  # noqa: E402

from hiqcache.layout import (  # noqa: E402
    QWEN3_8B_HEAD_DIM,
    QWEN3_8B_KV_HEADS,
    QWEN3_8B_LAYER_NUM,
    V1_LAYOUT,
)

BASE_PER_TOKEN = 147_456
ENC_PER_TOKEN = 82_944


def test_size_per_token_constants():
    rows = capacity_table(hicache_size_gb=8.0)
    assert rows[0].config == "bf16_hicache"
    assert rows[1].config == "hiqcache_int8"
    assert rows[0].size_per_token == BASE_PER_TOKEN
    assert rows[1].size_per_token == ENC_PER_TOKEN
    assert rows[0].size_per_token == 144 * 1024
    assert rows[1].size_per_token == 81 * 1024


def test_matches_sglang_formula():
    """Re-implement base.py:200-208 inline and compare."""
    for size_per_token in (BASE_PER_TOKEN, ENC_PER_TOKEN):
        for host_size in (1.0, 4.0, 8.0, 16.0, 32.0, 64.0):
            expected_size = int(host_size * 1e9 // size_per_token)
            expected_page_num = expected_size // 1 + 1  # page_size == 1
            expected = expected_page_num * 1
            got = host_pool_size(
                size_per_token=size_per_token, host_size_gb=host_size, page_size=1
            )
            assert got == expected, (
                f"size_per_token={size_per_token} host_size={host_size}: "
                f"{got} != {expected}"
            )


def test_capacity_gain_is_approximately_compression_ratio():
    rows = capacity_table(hicache_size_gb=8.0)
    gain = rows[1].capacity_vs_baseline
    # Integer division and the +1 page slack make the realised gain marginally
    # below the ideal 1.7778x byte ratio.
    assert 1.77 < gain <= BASE_PER_TOKEN / ENC_PER_TOKEN
    assert gain == pytest.approx(BASE_PER_TOKEN / ENC_PER_TOKEN, rel=2e-3)


@pytest.mark.parametrize("host_size_gb", [1, 2, 4, 8, 16, 32])
def test_capacity_gain_is_stable_across_budgets(host_size_gb):
    rows = capacity_table(hicache_size_gb=host_size_gb)
    assert rows[1].capacity_vs_baseline == pytest.approx(1.7778, rel=3e-3)


def test_allocated_bytes_do_not_exceed_budget_materially():
    """The +1 page slack is the only overshoot SGLang permits."""
    for host_size_gb in (1, 8, 32):
        rows = capacity_table(hicache_size_gb=host_size_gb)
        budget = host_size_gb * 1e9
        for row in rows:
            assert row.host_bytes_allocated >= budget - row.size_per_token
            # page_size == 1 means at most one extra token of slack
            assert row.host_bytes_allocated <= budget + row.size_per_token * 2


def test_8gb_budget_concrete_numbers():
    """Pin the exact numbers the pod runs must reproduce."""
    rows = capacity_table(hicache_size_gb=8.0)
    baseline, encoded = rows
    assert baseline.token_capacity == 8_000_000_000 // BASE_PER_TOKEN + 1
    assert encoded.token_capacity == 8_000_000_000 // ENC_PER_TOKEN + 1
    assert baseline.token_capacity == 54_254
    assert encoded.token_capacity == 96_451
    # ~42,197 extra tokens of L2 capacity for the same 8 GB.
    assert encoded.token_capacity - baseline.token_capacity == 42_197


def test_bytes_per_layer_accounting_is_consistent():
    rows = capacity_table(hicache_size_gb=8.0)
    assert rows[0].bytes_per_token_per_layer == 4096  # 2 x 8 x 128 x 2
    assert rows[1].bytes_per_token_per_layer == 2304  # 2 x 1152
    assert rows[0].bytes_per_token_per_layer == 2 * QWEN3_8B_KV_HEADS * QWEN3_8B_HEAD_DIM * 2
    assert rows[1].bytes_per_token_per_layer == 2 * V1_LAYOUT.row_bytes
    assert rows[0].size_per_token == rows[0].bytes_per_token_per_layer * QWEN3_8B_LAYER_NUM


def test_ratio_sizing_matches_sglang():
    """``--hicache-ratio`` path: size = int(device_capacity * ratio)."""
    device_capacity = 100_000
    for ratio in (0.5, 1.0, 2.0, 4.0):
        got = host_pool_size(
            size_per_token=ENC_PER_TOKEN,
            host_to_device_ratio=ratio,
            device_capacity=device_capacity,
        )
        expected = int(device_capacity * ratio) // 1 + 1
        assert got == expected


def test_requires_one_sizing_mode():
    with pytest.raises(ValueError, match="must be positive"):
        host_pool_size(size_per_token=ENC_PER_TOKEN)


def test_page_size_above_one_rounds_up_to_whole_pages():
    got = host_pool_size(
        size_per_token=ENC_PER_TOKEN, host_size_gb=1.0, page_size=64
    )
    raw = int(1e9 // ENC_PER_TOKEN)
    assert got % 64 == 0
    assert got == (raw // 64 + 1) * 64
