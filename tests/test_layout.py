"""Packed-layout tests: sizes, alignment, region disjointness, compression math.

These lock in the published layout numbers so that any change to
the format is caught immediately.
"""

from __future__ import annotations

import pytest

from hiqcache.layout import (
    ALIGNMENT_BYTES,
    QWEN3_8B_HEAD_DIM,
    QWEN3_8B_KV_HEADS,
    QWEN3_8B_LAYER_NUM,
    V1_LAYOUT,
    RecordLayout,
    describe_compression,
    iter_region_spans,
    qwen3_8b_compression,
)


# ---------------------------------------------------------------------------
# Published layout numbers
# ---------------------------------------------------------------------------


def test_v1_row_is_1152_bytes():
    assert V1_LAYOUT.row_bytes == 1152


def test_v1_row_is_nine_128_byte_groups():
    assert V1_LAYOUT.row_bytes == 9 * 128
    assert V1_LAYOUT.is_mover_aligned


def test_v1_sub_region_offsets():
    assert V1_LAYOUT.payload_offset == 0
    assert V1_LAYOUT.payload_bytes == 1024
    assert V1_LAYOUT.scale_offset == 1024
    assert V1_LAYOUT.scale_bytes == 16
    assert V1_LAYOUT.padding_offset == 1040
    assert V1_LAYOUT.padding_bytes == 112
    assert V1_LAYOUT.padding_offset + V1_LAYOUT.padding_bytes == 1152


def test_bytes_per_token_per_layer():
    # 2 (K,V) x 1152
    assert V1_LAYOUT.bytes_per_token == 2304


def test_qwen3_8b_compression_matches_project_md():
    stats = qwen3_8b_compression()
    assert stats["baseline_bytes_per_token"] == 147_456
    assert stats["encoded_bytes_per_token"] == 82_944
    assert stats["reduction_fraction"] == pytest.approx(0.4375, abs=1e-12)
    assert stats["compression_ratio"] == pytest.approx(1.7777777, rel=1e-6)


def test_baseline_row_arithmetic():
    # 8 heads x 128 dims x 2 bytes = 2048 B per K row, same for V.
    k_row = QWEN3_8B_KV_HEADS * QWEN3_8B_HEAD_DIM * 2
    assert k_row == 2048
    assert k_row * 2 == 4096
    assert k_row * 2 * QWEN3_8B_LAYER_NUM == 147_456
    assert 147_456 // 1024 == 144  # 144 KiB


def test_compression_reduction_is_exactly_seven_sixteenths():
    # 1 - 2304/4096 = 1 - 9/16 = 7/16
    stats = describe_compression(
        V1_LAYOUT, layer_num=36, head_num=8, head_dim=128, itemsize=2
    )
    assert stats["reduction_fraction"] == 7 / 16


# ---------------------------------------------------------------------------
# Region accounting
# ---------------------------------------------------------------------------


def test_regions_are_disjoint_and_ordered():
    spans = list(iter_region_spans(V1_LAYOUT, QWEN3_8B_KV_HEADS, QWEN3_8B_HEAD_DIM))
    assert len(spans) == 2 * QWEN3_8B_KV_HEADS  # 8 payload + 8 scale

    covered = 0
    cursor = 0
    for name, start, stop in spans:
        assert start == cursor, f"{name} starts at {start}, expected {cursor}"
        assert stop > start, f"{name} is empty"
        covered += stop - start
        cursor = stop

    # Payload + scales tile the meaningful region exactly; padding is the rest.
    assert covered == V1_LAYOUT.payload_bytes + V1_LAYOUT.scale_bytes
    assert cursor == V1_LAYOUT.padding_offset
    assert covered + V1_LAYOUT.padding_bytes == V1_LAYOUT.row_bytes


def test_payload_head_slices_tile_the_payload_region():
    total = 0
    for head in range(QWEN3_8B_KV_HEADS):
        start, stop = V1_LAYOUT.payload_slice(head, QWEN3_8B_HEAD_DIM)
        assert stop - start == QWEN3_8B_HEAD_DIM
        total += stop - start
    assert total == V1_LAYOUT.payload_bytes


def test_scale_slices_tile_the_scale_region():
    total = 0
    for head in range(QWEN3_8B_KV_HEADS):
        start, stop = V1_LAYOUT.scale_slice(head)
        assert stop - start == 2, "one BF16 scale per head"
        total += stop - start
    assert total == V1_LAYOUT.scale_bytes


def test_record_offsets_are_contiguous():
    offsets = V1_LAYOUT.record_offsets(64)
    assert offsets[0] == 0
    assert offsets[1] == 1152
    assert all(
        b - a == V1_LAYOUT.row_bytes for a, b in zip(offsets, offsets[1:])
    )


# ---------------------------------------------------------------------------
# Fail-fast validation
# ---------------------------------------------------------------------------


def test_check_against_accepts_qwen3_8b():
    V1_LAYOUT.check_against(QWEN3_8B_KV_HEADS, QWEN3_8B_HEAD_DIM, 2)


def test_check_against_rejects_wrong_head_num():
    with pytest.raises(ValueError, match="payload_bytes"):
        V1_LAYOUT.check_against(16, 128, 2)  # 16 heads would need 2048 B payload


def test_check_against_rejects_wrong_head_dim():
    with pytest.raises(ValueError, match="payload_bytes"):
        V1_LAYOUT.check_against(8, 64, 2)


def test_check_against_rejects_odd_geometry_that_breaks_alignment():
    # 64 int8 bytes payload + 2 scales + 2 padding = 68 B, not a 128 multiple.
    weird = RecordLayout(payload_bytes=64, scale_bytes=2, padding_bytes=2)
    assert not weird.is_mover_aligned
    with pytest.raises(ValueError, match="not a multiple of 128"):
        weird.check_against(1, 64, 2)


def test_layout_rejects_oversized_header():
    with pytest.raises(ValueError, match="does not fit in padding"):
        RecordLayout(header_bytes=200)


def test_layout_rejects_zero_payload():
    with pytest.raises(ValueError, match="payload_bytes must be positive"):
        RecordLayout(payload_bytes=0)


def test_layout_rejects_negative_sizes():
    with pytest.raises(ValueError, match="scale_bytes"):
        RecordLayout(scale_bytes=-16)


def test_header_eats_into_padding_without_changing_row_size():
    with_header = RecordLayout(header_bytes=64)
    assert with_header.row_bytes == V1_LAYOUT.row_bytes
    assert with_header.payload_offset == 64
    assert with_header.scale_offset == 64 + 1024
    assert with_header.padding_offset == 64 + 1024 + 16
    assert with_header.is_mover_aligned


def test_aligned_row_bytes_rounds_up():
    assert V1_LAYOUT.aligned_row_bytes == V1_LAYOUT.row_bytes
    weird = RecordLayout(payload_bytes=64, scale_bytes=2, padding_bytes=2)
    assert weird.aligned_row_bytes == ALIGNMENT_BYTES


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_payload_must_exactly_match_head_geometry(head_dim):
    # A payload sized for head_num=8 must reject any other head_dim.
    if head_dim == 128:
        V1_LAYOUT.check_against(8, head_dim, 2)
    else:
        with pytest.raises(ValueError):
            V1_LAYOUT.check_against(8, head_dim, 2)
