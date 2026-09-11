"""Drift guard: the fork's self-contained codec must equal the reference codec.

The SGLang fork carries its own copy of the codec (``int8_codec.py``) so it stays
shippable without depending on the standalone ``hiqcache`` package. That
duplication is deliberate but it is also a correctness hazard: a fix applied to
one side and not the other produces a runtime that disagrees with every measured
result in this repo.

This test pins the two together over the vector set the project's error bounds
and capacity numbers are quoted from. It is the cheapest possible defence
against the most likely silent failure.
"""

from __future__ import annotations

import torch

from hiqcache.codec import AMAX_FLOOR as REF_AMAX_FLOOR
from hiqcache.codec import encode_rows as ref_encode_rows
from hiqcache.codec import decode_records as ref_decode_records
from hiqcache.codec import pack_records as ref_pack_records
from hiqcache.layout import V1_LAYOUT

import int8_codec as fork

HEAD_NUM = 8
HEAD_DIM = 128
REF_ROW_BYTES = V1_LAYOUT.row_bytes


def _vectors():
    g = torch.Generator().manual_seed(0xC0FFEE)
    out = {
        "random_unit": torch.randn((64, HEAD_NUM, HEAD_DIM), generator=g).to(
            torch.bfloat16
        ),
        "all_zeros": torch.zeros((16, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16),
        "small": (torch.randn((32, HEAD_NUM, HEAD_DIM), generator=g) * 1e-3).to(
            torch.bfloat16
        ),
        "large": (torch.randn((32, HEAD_NUM, HEAD_DIM), generator=g) * 1e3).to(
            torch.bfloat16
        ),
        "tiny": (torch.randn((16, HEAD_NUM, HEAD_DIM), generator=g) * 1e-30).to(
            torch.bfloat16
        ),
    }
    mixed = torch.randn((32, HEAD_NUM, HEAD_DIM), generator=g).to(torch.bfloat16)
    mixed[0, :, :] = 0.0
    mixed[1, 0, :] = 1e-30
    mixed[2, 1, :] = 1e4
    out["mixed"] = mixed

    steps = torch.arange(-127, 128, dtype=torch.float32) / 127.0
    row = steps.repeat(HEAD_DIM // steps.numel() + 1)[:HEAD_DIM]
    out["exact_grid"] = (
        row.unsqueeze(0).unsqueeze(0).expand(8, HEAD_NUM, HEAD_DIM).contiguous().to(
            torch.bfloat16
        )
    )
    return out


def test_format_constants_match():
    assert fork.ROW_BYTES == REF_ROW_BYTES == 1152
    assert fork.PAYLOAD_BYTES == V1_LAYOUT.payload_bytes == 1024
    assert fork.SCALE_BYTES == V1_LAYOUT.scale_bytes == 16
    assert fork.PADDING_BYTES == V1_LAYOUT.padding_bytes == 112
    assert fork.SCALE_OFFSET == V1_LAYOUT.scale_offset == 1024
    assert fork.AMAX_FLOOR == REF_AMAX_FLOOR
    assert fork.QUANT_MAX == 127


def test_bytes_per_token_matches():
    for layer_num in (1, 8, 36, 80):
        assert fork.bytes_per_token(layer_num) == V1_LAYOUT.bytes_per_token * layer_num


def test_quantised_payloads_are_bit_identical():
    for name, x in _vectors().items():
        ref_payload, ref_scales = ref_encode_rows(x)
        fork_scales = fork.compute_scales(x)
        fork_payload = fork.quantize_rows(x, fork_scales)

        assert torch.equal(ref_scales, fork_scales), f"{name}: scales differ"
        assert torch.equal(ref_payload, fork_payload), (
            f"{name}: {int((ref_payload != fork_payload).sum())} payload elements differ"
        )


def test_encoded_records_are_bit_identical():
    for name, x in _vectors().items():
        ref_payload, ref_scales = ref_encode_rows(x)
        ref_records = ref_pack_records(ref_payload, ref_scales)

        fork_records = torch.zeros((x.shape[0], fork.ROW_BYTES), dtype=torch.uint8)
        fork.write_record(x, fork_records)

        assert torch.equal(ref_records, fork_records), (
            f"{name}: {int((ref_records != fork_records).sum())} record bytes differ"
        )


def test_decoded_values_are_bit_identical():
    for name, x in _vectors().items():
        ref_payload, ref_scales = ref_encode_rows(x)
        ref_restored = ref_decode_records(
            ref_pack_records(ref_payload, ref_scales),
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
        )
        fork_records = torch.zeros((x.shape[0], fork.ROW_BYTES), dtype=torch.uint8)
        fork.write_record(x, fork_records)
        fork_restored = fork.decode_records(
            fork_records,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
        )
        assert torch.equal(ref_restored, fork_restored), f"{name}: decoded values differ"


def test_fork_geometry_validation_matches_reference():
    """Both sides must accept the same geometries, or fail the same way."""
    for head_num, head_dim, itemsize in (
        (8, 128, 2),
        (16, 128, 2),
        (8, 64, 2),
        (4, 128, 2),
        (8, 128, 1),
    ):
        ref_error = fork_error = None
        try:
            fork.check_layout(head_num, head_dim, itemsize)
        except ValueError as exc:
            fork_error = str(exc)
        try:
            V1_LAYOUT.check_against(head_num, head_dim, itemsize)
        except ValueError as exc:
            ref_error = str(exc)
        assert (ref_error is None) == (fork_error is None), (
            f"geometry ({head_num},{head_dim},{itemsize}): reference "
            f"{'accepted' if ref_error is None else 'rejected'} but fork "
            f"{'accepted' if fork_error is None else 'rejected'}"
        )


def test_fork_error_bound_holds_over_all_vectors():
    """The reference bound is the project's contract; the fork must satisfy it."""
    for name, x in _vectors().items():
        records = torch.zeros((x.shape[0], fork.ROW_BYTES), dtype=torch.uint8)
        fork.write_record(x, records)
        restored = fork.decode_records(
            records, head_num=HEAD_NUM, head_dim=HEAD_DIM, dtype=torch.bfloat16
        )
        scales = fork.compute_scales(x)
        s = scales.float().unsqueeze(-1)
        bound = (0.5 + 2**-8) * s + 2**-8 * restored.float().abs()
        err = (restored.float() - x.float()).abs()
        assert int((err > bound).sum()) == 0, f"{name}: exceeds the verified bound"
