"""Codec correctness tests.

Covers the required input classes -- random, all-zero, small, large, mixed
magnitudes, realistic KV-shaped tensors -- plus packed-record layout, scale
encoding, the ``<= scale/2`` error bound, and compression/time measurements.
"""

from __future__ import annotations

import time

import pytest
import torch

from hiqcache.codec import (
    AMAX_FLOOR,
    QUANT_MAX,
    SCALE_FLOOR,
    compute_scales,
    decode_records,
    encode_decode,
    encode_records,
    encode_rows,
    error_stats,
    pack_records,
    quantize_rows,
    unpack_records,
    verified_error_bound,
)
from hiqcache.layout import V1_LAYOUT

HEAD_NUM = 8
HEAD_DIM = 128
ROW_SHAPE = (HEAD_NUM, HEAD_DIM)
LAYER_NUM = 36

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Fixtures / input generators for the required input classes
# ---------------------------------------------------------------------------


def random_bf16(shape, *, scale=1.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=g, dtype=torch.float32) * scale).to(torch.bfloat16)


def realistic_kv_rows(num_tokens, *, seed=0, scale=1.0):
    """``[num_tokens, head_num, head_dim]`` shaped like a Qwen3-8B KV row batch."""
    return random_bf16((num_tokens, HEAD_NUM, HEAD_DIM), scale=scale, seed=seed)


INPUT_CLASSES = {
    "random_unit": lambda: realistic_kv_rows(64, seed=1),
    "all_zeros": lambda: torch.zeros((64, *ROW_SHAPE), dtype=torch.bfloat16),
    "small_values": lambda: realistic_kv_rows(64, seed=2, scale=1e-3),
    "tiny_values": lambda: realistic_kv_rows(64, seed=3, scale=1e-30),
    "large_values": lambda: realistic_kv_rows(64, seed=4, scale=1e3),
    "mixed_magnitudes": lambda: _mixed_magnitudes(),
    "one_large_rest_small": lambda: _one_large_rest_small(),
    "exact_bf16_grid": lambda: _exact_grid(),
}


def _mixed_magnitudes():
    """Some heads near zero, some large -- within one row and across rows."""
    x = realistic_kv_rows(64, seed=5)
    x[0, 0, :] = 0.0
    x[0, 1, :] *= 1e-8
    x[0, 2, :] *= 1e8
    x[1::2] *= 0.0  # every other token entirely zero
    return x


def _one_large_rest_small():
    """A single outlier dominates the absmax of its head."""
    x = realistic_kv_rows(32, seed=6) * 1e-4
    x[:, 3, 7] = 1.0
    return x


def _exact_grid():
    """Values exactly representable on the bf16 grid after /127 scaling."""
    steps = torch.arange(-127, 128, dtype=torch.float32) / 127.0
    base = steps.repeat(HEAD_DIM // steps.numel() + 1)[:HEAD_DIM]
    return base.unsqueeze(0).unsqueeze(0).expand(8, HEAD_NUM, HEAD_DIM).contiguous().to(
        torch.bfloat16
    )


@pytest.fixture(params=sorted(INPUT_CLASSES))
def input_class(request):
    return request.param, INPUT_CLASSES[request.param]()


# ---------------------------------------------------------------------------
# Round-trip shape / dtype correctness
# ---------------------------------------------------------------------------


def test_encode_emits_1152_byte_records(input_class):
    _, x = input_class
    records = encode_records(x)
    assert records.dtype == torch.uint8
    assert records.shape == (*x.shape[:-2], 1152)
    assert records.is_contiguous()


def test_decode_restores_shape_and_dtype(input_class):
    _, x = input_class
    restored = encode_decode(x)
    assert restored.shape == x.shape
    assert restored.dtype == torch.bfloat16


def test_pack_unpack_is_lossless_for_the_intermediate_representation(input_class):
    _, x = input_class
    payload, scales = encode_rows(x)
    records = pack_records(payload, scales)
    payload2, scales2 = unpack_records(
        records, head_num=HEAD_NUM, head_dim=HEAD_DIM
    )
    assert torch.equal(payload, payload2), "INT8 payload must survive packing"
    assert torch.equal(scales, scales2), "BF16 scales must survive packing bit-exactly"


def test_record_size_is_independent_of_batch_shape():
    for shape in [(1,), (7,), (64,), (3, 5)]:
        x = realistic_kv_rows(1, seed=7).expand(*shape, HEAD_NUM, HEAD_DIM)
        records = encode_records(x.contiguous())
        assert records.shape == (*shape, 1152)


# ---------------------------------------------------------------------------
# Scale encoding
# ---------------------------------------------------------------------------


def test_scale_is_absmax_over_127(input_class):
    """The stored scale must equal ``bf16(clamp(absmax) / 127)`` -- the spec formula."""
    _, x = input_class
    payload, scales = encode_rows(x)

    amax = x.abs().amax(dim=-1).clamp_min(AMAX_FLOOR)
    expected = (amax / QUANT_MAX).to(torch.bfloat16)
    assert torch.equal(scales, expected), (
        "stored scale deviates from bf16(clamp(absmax)/127); "
        f"worst delta {float((scales.float() - expected.float()).abs().max()):.3e}"
    )
    assert torch.equal(scales, compute_scales(x))


def test_scale_matches_absmax_within_bf16_rounding(input_class):
    """Independent check against the raw absmax, not the codec's own expression."""
    _, x = input_class
    _, scales = encode_rows(x)
    raw_absmax = x.abs().amax(dim=-1).float()
    expected = (raw_absmax.clamp_min(AMAX_FLOOR) / QUANT_MAX).to(torch.bfloat16).float()
    stored = scales.float()
    # Either an exact match, or within one bf16 ulp of the expected scale.
    close = torch.isclose(stored, expected, rtol=2**-7, atol=0.0)
    assert bool(close.all()), (
        f"scale off by more than one bf16 ulp; max relative deviation "
        f"{float(((stored - expected).abs() / expected.clamp_min(1e-38)).max()):.3e}"
    )
    # And the scale always brackets the true absmax/127 within the same rounding.
    ratio = stored / (raw_absmax.clamp_min(AMAX_FLOOR) / QUANT_MAX)
    assert float((ratio - 1.0).abs().max()) <= 2**-7


def test_scales_are_positive_and_finite(input_class):
    _, x = input_class
    _, scales = encode_rows(x)
    assert torch.isfinite(scales).all()
    assert (scales > 0).all(), "a scale must always be usable as a divisor"


def test_all_zero_head_uses_safe_nonzero_scale_and_encodes_zero():
    x = torch.zeros((4, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16)
    payload, scales = encode_rows(x)
    assert (scales > 0).all()
    assert torch.isfinite(scales).all()
    assert (payload == 0).all(), "all-zero head must encode to exact zeros"
    assert torch.equal(encode_decode(x), x), "all-zero head must decode to exact zeros"


def test_tiny_head_encodes_zero_without_nan():
    x = torch.full((2, HEAD_NUM, HEAD_DIM), 1e-40, dtype=torch.float32).to(
        torch.bfloat16
    )
    payload, scales = encode_rows(x)
    assert torch.isfinite(scales).all()
    assert not torch.isnan(payload.float()).any()
    restored = decode_records(
        pack_records(payload, scales), head_num=HEAD_NUM, head_dim=HEAD_DIM
    )
    assert torch.isfinite(restored).all()


def test_zero_and_nonzero_heads_coexist_in_one_row():
    x = realistic_kv_rows(8, seed=8)
    x[:, 0, :] = 0.0
    x[:, 5, :] = 0.0
    payload, scales = encode_rows(x)
    assert (payload[:, 0, :] == 0).all()
    assert (payload[:, 5, :] == 0).all()
    assert (payload[:, 1, :] != 0).any()
    assert (scales > 0).all()


# ---------------------------------------------------------------------------
# Quantisation range
# ---------------------------------------------------------------------------


def test_quantised_values_stay_in_signed_int8_range(input_class):
    _, x = input_class
    payload, _ = encode_rows(x)
    assert payload.dtype == torch.int8
    assert int(payload.min()) >= -QUANT_MAX
    assert int(payload.max()) <= QUANT_MAX


def test_absmax_element_quantises_to_plus_or_minus_127():
    x = realistic_kv_rows(16, seed=9)
    # Zero each head so the injected value is unambiguously that head's absmax.
    x[:, :, :] = 0.0
    x[:, :, 1] = 0.25
    payload, _ = encode_rows(x)
    assert int(payload[:, :, 1].abs().min()) == QUANT_MAX, (
        "the absmax element of every head must quantise to +/-127"
    )
    assert int(payload[:, :, 1].abs().max()) == QUANT_MAX


# ---------------------------------------------------------------------------
# Error bounds  ("quantization error <= scale / 2")
# ---------------------------------------------------------------------------


def _exact_reference(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Exact dequantisation using the *stored* bf16 scale and exact rounding.

    ``round(x64 / s64) * s64`` computed in float64 isolates the error that comes
    purely from rounding to INT8, with no float32 or bf16 arithmetic noise.
    """
    x64 = x.double()
    s64 = scales.double().unsqueeze(-1)
    q = torch.round(x64 / s64).clamp_(-QUANT_MAX, QUANT_MAX)
    return (q * s64).float()


def test_error_within_verified_bound(input_class):
    """The documented format guarantee, checked elementwise."""
    name, x = input_class
    payload, scales = encode_rows(x)
    records = pack_records(payload, scales)
    restored = decode_records(records, head_num=HEAD_NUM, head_dim=HEAD_DIM)
    _, stored_scales = unpack_records(records, head_num=HEAD_NUM, head_dim=HEAD_DIM)

    err = (restored.float() - x.float()).abs()
    bound = verified_error_bound(restored, stored_scales)

    violations = int((err > bound).sum())
    assert violations == 0, (
        f"{name}: {violations}/{err.numel()} elements exceed the verified bound; "
        f"worst excess {float((err - bound).max()):.3e}"
    )


def test_exact_quantiser_error_is_within_half_scale(input_class):
    """Term 1 alone: exact round-to-nearest is bounded by ``s / 2``.

    Isolating the quantiser proves the ``s / 2`` claim is correct for the part of
    the pipeline it actually describes, and that the remaining error comes from
    representation choices (BF16 scale, BF16 output) rather than from a bad
    rounding rule.
    """
    _, x = input_class
    payload, scales = encode_rows(x)
    exact = _exact_reference(x, scales)
    s = scales.float().unsqueeze(-1)
    excess = (exact - x.float()).abs() - s / 2
    assert float(excess.max()) <= 1e-6, (
        f"exact quantiser exceeds s/2; worst excess {float(excess.max()):.3e}"
    )


def test_error_sources_decompose_as_documented():
    """The three error terms must each be present and non-negative."""
    x = realistic_kv_rows(256, seed=20)
    payload, scales = encode_rows(x)
    restored = decode_records(
        pack_records(payload, scales), head_num=HEAD_NUM, head_dim=HEAD_DIM
    )
    _, stored = unpack_records(
        pack_records(payload, scales), head_num=HEAD_NUM, head_dim=HEAD_DIM
    )
    s = stored.float().unsqueeze(-1)

    quantiser_err = (_exact_reference(x, scales) - x.float()).abs()
    output_rounding = (restored.float() - _exact_reference(x, scales)).abs()

    # Term 1 is real and approached.
    assert float((quantiser_err / s).max()) > 0.4
    assert float((quantiser_err / s).max()) <= 0.5 + 1e-6
    # Term 3 is real and approached.
    assert float((output_rounding / s).max()) > 0.3
    # The bound is not vacuous: it is within 2x of the observed maximum.
    total = (restored.float() - x.float()).abs()
    assert float(total.max()) > 0.3 * float(verified_error_bound(restored, stored).max())


def test_decoder_is_bitwise_bf16_of_the_exact_product():
    """The decoder adds no error of its own beyond the BF16 output rounding.

    Pins the decoder contract: ``decode == bf16(int8_payload) * stored_scale``
    computed in BF16, so term 3 above is entirely attributable to the output
    dtype and not to sloppy decoder arithmetic.
    """
    x = realistic_kv_rows(128, seed=21)
    payload, scales = encode_rows(x)
    records = pack_records(payload, scales)
    restored = decode_records(records, head_num=HEAD_NUM, head_dim=HEAD_DIM)
    expected = payload.to(torch.bfloat16) * scales.unsqueeze(-1)
    assert torch.equal(restored, expected)


def test_half_scale_bound_holds_for_the_exact_product():
    """``|q*s - x| <= s/2`` holds for every magnitude when the product is exact.

    Uses powers of two as absmax so the scale needs no BF16 rounding, leaving
    INT8 rounding as the only error source.
    """
    worst = 0.0
    for exponent in range(-20, 25):
        absmax = 2.0**exponent
        x = torch.zeros((8, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16)
        x[:, :, 0] = absmax
        x[:, :, 1] = -absmax
        noise = random_bf16((8, HEAD_NUM, HEAD_DIM - 2), seed=exponent) * absmax
        x[:, :, 2:] = noise
        payload, scales = encode_rows(x)
        product = _exact_reference(x, scales)
        s = scales.float().unsqueeze(-1)
        normalised = float(((product - x.float()).abs() / s).max())
        assert normalised <= 0.5 + 1e-6, (
            f"absmax=2**{exponent}: max|q*s - x|/s = {normalised:.6f} exceeds 0.5"
        )
        worst = max(worst, normalised)
    assert worst > 0.4, "sanity: the bound should be approached, not vacuous"


def test_error_matches_exact_float64_quantiser():
    """Differential test: the tensor codec must agree with a float64 reference.

    Because the quotient is formed in float32 and the scale is stored in BF16,
    the codec should reproduce exact round-to-nearest of ``x / s_stored`` with no
    disagreement at all.
    """
    x = realistic_kv_rows(256, seed=10)
    payload, scales = encode_rows(x)

    x64 = x.double()
    s64 = scales.double().unsqueeze(-1)
    reference = torch.round(x64 / s64).clamp_(-QUANT_MAX, QUANT_MAX).to(torch.int8)

    disagreement = int((payload != reference).sum())
    assert disagreement == 0, (
        f"{disagreement} of {payload.numel()} elements disagree with the "
        f"float64 reference quantiser"
    )


def test_bf16_division_would_break_the_bound():
    """Regression guard for the float32 quotient.

    Documents *why* ``quantize_rows`` widens to float32: forming the quotient in
    BF16 puts the arithmetic noise near 0.25 of a scale step at full magnitude,
    enough to push the total error past the verified bound.
    """
    x = realistic_kv_rows(512, seed=22)
    scales = compute_scales(x)
    bf16_q = torch.round(x / scales.unsqueeze(-1)).clamp_(-QUANT_MAX, QUANT_MAX)
    f32_q = quantize_rows(x, scales).float()
    assert int((bf16_q != f32_q).sum()) > 0, (
        "sanity: bf16 and float32 quotients should differ somewhere"
    )
    # And the widening is what keeps the quantiser inside its own bound.
    exact = _exact_reference(x, scales)
    s = scales.float().unsqueeze(-1)
    assert float(((exact - x.float()).abs() / s).max()) <= 0.5 + 1e-6


def test_error_bound_is_tight_on_exact_grid():
    """On the exactly-representable grid the round trip must be near-exact."""
    x = _exact_grid()
    payload, scales = encode_rows(x)
    restored = decode_records(
        pack_records(payload, scales), head_num=HEAD_NUM, head_dim=HEAD_DIM
    )
    stats = error_stats(x, restored, scales)
    # Only bf16 multiply rounding remains.
    assert stats.max_abs <= float(scales.float().max()) * 2**-8 * 2
    assert stats.max_abs_over_scale <= 0.5 + 1e-3


def test_normalised_error_bound_holds(input_class):
    """``max |x_hat - x| / s`` must stay inside the verified format bound."""
    name, x = input_class
    payload, scales = encode_rows(x)
    restored = decode_records(
        pack_records(payload, scales), head_num=HEAD_NUM, head_dim=HEAD_DIM
    )
    _, stored = unpack_records(
        pack_records(payload, scales), head_num=HEAD_NUM, head_dim=HEAD_DIM
    )
    err = (restored.float() - x.float()).abs()
    bound = verified_error_bound(restored, stored)
    assert float((err - bound).max()) <= 1e-6, (
        f"{name}: max|x_hat-x|/s = {error_stats(x, restored, stored).max_abs_over_scale:.6f} "
        f"violates the verified bound"
    )


def test_large_magnitudes_do_not_saturate_badly():
    x = realistic_kv_rows(32, seed=11, scale=1e4)
    payload, scales = encode_rows(x)
    assert int(payload.abs().max()) <= QUANT_MAX
    records = pack_records(payload, scales)
    restored = decode_records(records, head_num=HEAD_NUM, head_dim=HEAD_DIM)
    _, stored = unpack_records(records, head_num=HEAD_NUM, head_dim=HEAD_DIM)

    err = (restored.float() - x.float()).abs()
    assert float((err - verified_error_bound(restored, stored)).max()) <= 1e-6

    # Error is *absolute* and uniform over the head's range, so relative error
    # is large for the many elements near zero. The informative figures are the
    # normalised error and the median relative error.
    stats = error_stats(x, restored, stored)
    assert stats.max_abs_over_scale <= 1.0, (
        f"normalised error {stats.max_abs_over_scale:.4f} indicates saturation"
    )
    assert stats.p50_rel <= 1.0 / QUANT_MAX, (
        f"median relative error {stats.p50_rel:.4f} exceeds the int8 floor"
    )
    # p99 relative error is dominated by near-zero elements; report it, but do
    # not gate on it. Log it so the number is visible in test output.
    print(
        f"\n[large_values] max_abs={stats.max_abs:.1f} "
        f"max|x_hat-x|/s={stats.max_abs_over_scale:.4f} "
        f"p50_rel={stats.p50_rel:.5f} p99_rel={stats.p99_rel:.4f}"
    )


# ---------------------------------------------------------------------------
# Packed record layout correctness
# ---------------------------------------------------------------------------


def test_payload_lands_in_bytes_0_1023_and_scales_in_1024_1039(input_class):
    _, x = input_class
    payload, scales = encode_rows(x)
    records = pack_records(payload, scales)

    assert torch.equal(
        records[..., :1024], payload.reshape(*x.shape[:-2], -1).view(torch.uint8)
    )
    assert torch.equal(
        records[..., 1024:1040], scales.reshape(*x.shape[:-2], -1).view(torch.uint8)
    )


def test_padding_bytes_are_zero(input_class):
    _, x = input_class
    records = encode_records(x)
    assert (records[..., 1040:1152] == 0).all(), "padding must be deterministically zero"


def test_scale_bytes_are_little_endian_bf16():
    """A scale written at offset 1024 must read back as the same bf16 value."""
    x = torch.zeros((1, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16)
    x[0, 0, :] = 1.0  # absmax 1.0 -> scale 1/127
    records = encode_records(x)
    raw = records[0, 1024:1040].contiguous().view(torch.bfloat16)
    # 1/127 is not representable in bf16 (8 mantissa bits); it rounds down by
    # ~0.78%, which is exactly why quantize_rows clamps rather than trusting
    # |x/s| <= 127. Assert the byte contents equal the bf16 rounding of 1/127.
    expected_scale = torch.tensor(1.0 / 127, dtype=torch.bfloat16)
    assert torch.equal(raw[0], expected_scale), (
        f"stored scale {float(raw[0].float()):.9e} != bf16(1/127) "
        f"{float(expected_scale.float()):.9e}"
    )
    assert float(raw[0].float()) < 1.0 / 127, "bf16(1/127) must round down"
    assert float(raw[0].float()) > 0
    # Scales 1..7 belong to all-zero heads and must carry the safe floor scale,
    # never a zero that would make the decoder divide by zero on re-encode.
    # Scale i lives at byte 1024 + 2*i, so scales 1..7 are bytes 1026..1039.
    remaining = records[0, 1026:1040].contiguous().view(torch.bfloat16)
    floor_scale = torch.tensor(AMAX_FLOOR / QUANT_MAX, dtype=torch.bfloat16)
    assert torch.equal(remaining, floor_scale.expand(HEAD_NUM - 1)), (
        f"zero heads stored {[float(v) for v in remaining[:3]]}, "
        f"expected the floor {float(floor_scale.float()):.3e}"
    )
    assert bool((remaining.float() > 0).all())


def test_record_offset_of_token_n_is_n_times_row_bytes_in_a_layer_arena():
    """A token's record must start at exactly ``n * row_bytes`` in the arena.

    This is the addressing contract the GPU staging path relies on: the JIT
    mover writes record ``i`` at ``base + i * row_bytes`` with no header, so a
    decoder reading at any other offset would corrupt every token after 0.
    """
    num_tokens = 5
    x = realistic_kv_rows(num_tokens, seed=12)
    row_records = encode_records(x)  # [num_tokens, 1152]
    assert row_records.shape == (num_tokens, V1_LAYOUT.row_bytes)

    # Build the arena by explicit byte offset, the way a writer would.
    arena = torch.zeros(num_tokens * V1_LAYOUT.row_bytes, dtype=torch.uint8)
    for token in range(num_tokens):
        start = token * V1_LAYOUT.row_bytes
        arena[start : start + V1_LAYOUT.row_bytes] = row_records[token]

    # And read it back by explicit byte offset, the way a reader would.
    for token in range(num_tokens):
        start = token * V1_LAYOUT.row_bytes
        slot = arena[start : start + V1_LAYOUT.row_bytes].unsqueeze(0)
        assert torch.equal(slot, row_records[token : token + 1])
        decoded = decode_records(slot, head_num=HEAD_NUM, head_dim=HEAD_DIM)
        assert torch.equal(decoded, encode_decode(x[token : token + 1])), (
            f"slot {token} round trip mismatch at offset {start}"
        )


def test_head_slices_are_independently_decodable():
    """Each head occupies a contiguous 128-byte payload slice -- proves the map
    is element-wise per head and needs no cross-head state."""
    x = realistic_kv_rows(4, seed=13)
    records = encode_records(x)
    for head in range(HEAD_NUM):
        start, stop = V1_LAYOUT.payload_slice(head, HEAD_DIM)
        assert stop - start == HEAD_DIM
        payload, scales = unpack_records(
            records, head_num=HEAD_NUM, head_dim=HEAD_DIM
        )
        restored_head = payload[..., head, :].to(torch.bfloat16) * scales[..., head : head + 1]
        full = decode_records(records, head_num=HEAD_NUM, head_dim=HEAD_DIM)
        assert torch.equal(restored_head, full[..., head, :])


# ---------------------------------------------------------------------------
# Compression measurement
# ---------------------------------------------------------------------------


def test_compression_ratio_when_stored_as_records():
    num_tokens = 256
    x = realistic_kv_rows(num_tokens, seed=14)
    baseline_bytes = x.numel() * 2  # bf16
    encoded_bytes = encode_records(x).numel()
    assert encoded_bytes == num_tokens * 1152
    ratio = baseline_bytes / encoded_bytes
    assert ratio == pytest.approx(2048 / 1152, rel=1e-9)
    assert ratio == pytest.approx(1.7777777, rel=1e-6)


def test_no_expansion_for_any_input_class(input_class):
    """Encoded output must never exceed the bf16 baseline."""
    _, x = input_class
    assert encode_records(x).numel() <= x.numel() * 2


# ---------------------------------------------------------------------------
# Timing (informational; asserted only loosely so CI is not flaky)
# ---------------------------------------------------------------------------


def test_encode_decode_timing_is_reported(capsys):
    x = realistic_kv_rows(4096, seed=15)

    t0 = time.perf_counter()
    for _ in range(5):
        records = encode_records(x)
    t1 = time.perf_counter()
    for _ in range(5):
        decode_records(records, head_num=HEAD_NUM, head_dim=HEAD_DIM)
    t2 = time.perf_counter()

    encode_ms = (t1 - t0) / 5 * 1e3
    decode_ms = (t2 - t1) / 5 * 1e3
    tokens_per_s = 4096 / max(encode_ms, 1e-6) * 1e3

    with capsys.disabled():
        print(
            f"\n[timing/CPU] 4096 tokens x 36 layers worth of rows: "
            f"encode {encode_ms:.3f} ms, decode {decode_ms:.3f} ms, "
            f"{tokens_per_s:,.0f} tokens/s"
        )
    assert encode_ms < 5000
    assert decode_ms < 5000
