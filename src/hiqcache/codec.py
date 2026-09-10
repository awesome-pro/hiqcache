"""HiQCache V1 codec: symmetric per-head INT8 KV quantisation.

Pure ``torch``. **No SGLang imports, no CUDA-only ops, no host syncs** — the
identical module runs under CPU, MPS and CUDA so the same conformance tests can
be executed locally on a Mac and again on a GPU pod.

Scope of V1 (see ``PROJECT.md`` Phase 1):

    For every (token, layer, K|V, KV head) compute a per-head scale

        s = max(|x|) / 127
        q = clamp(round(x / s), -127, 127)
        x_hat = q * s

    and pack ``q`` (INT8) plus ``s`` (BF16) into one aligned 1152-byte record.

Everything here is row-local: a record depends only on the 1024 BF16 values of
one ``(token, layer, K|V)`` row, so the transform is position-independent and
needs no sequence metadata. That is what makes it usable as a drop-in L2
representation without touching SGLang's caching policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hiqcache.layout import V1_LAYOUT, RecordLayout

#: Quantiser full-scale. ``q`` occupies ``[-127, 127]``; -128 is unused so the
#: range stays symmetric and sign handling cannot drift.
QUANT_MAX = 127

#: Lower bound applied to the per-head absmax, as a power of two so it is
#: exactly representable in BF16 on every backend.
#:
#: This is the "safe nonzero scale" of ``PROJECT.md`` Phase 1. A power of two is
#: chosen so the clamp is exact and the same bits are produced on CPU, MPS and
#: CUDA -- a clamp against a non-representable value would round differently per
#: backend and make cross-device conformance testing meaningless.
#:
#: The value is deliberately tiny: ``2**-112 / 127`` is still a *normal* BF16
#: (normal range starts at ``2**-126``), so the stored scale is never subnormal.
#: A head is affected only when its absmax is below ``2**-112``, and since
#: ``2**-112 / s_floor == 127`` exactly, such a head quantises to all zeros and
#: decodes back to exact zero -- the correct answer for negligible input.
AMAX_FLOOR = 2.0**-112

#: Smallest scale the codec will ever store, ``AMAX_FLOOR / QUANT_MAX``.
#: Normal (not subnormal) in both BF16 and FP16.
SCALE_FLOOR = AMAX_FLOOR / 127


def compute_scales(x: torch.Tensor) -> torch.Tensor:
    """Per-head BF16 scales for a ``[..., head_num, head_dim]`` row batch.

    Returns ``[..., head_num]`` in the same dtype as ``x``. The result is the
    value that is *stored*, so callers can use it directly as the exact
    reference for error-bound checks.
    """
    if x.shape[-1] == 0:
        raise ValueError("cannot compute scales for a zero-width head_dim")
    # Clamp the absmax (not the scale) so the subsequent division happens in a
    # normal range and cannot produce a subnormal result.
    amax = x.abs().amax(dim=-1).clamp_min(AMAX_FLOOR)
    return amax / QUANT_MAX


def quantize_rows(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Quantise ``x`` to INT8 using per-head ``scales``.

    ``x``: ``[..., head_num, head_dim]``; ``scales``: ``[..., head_num]``.
    Returns ``[..., head_num, head_dim]`` ``torch.int8``.

    The quotient is formed in **float32**. This is not cosmetic: a bf16 division
    near full scale has ~0.25 absolute granularity (8 mantissa bits at a
    magnitude of 127), which would push the normalised reconstruction error from
    ``<= 0.5`` up to ``~0.75``. Widening makes the quotient correctly rounded to
    well under 0.5, which is what restores the ``scale / 2`` guarantee.

    The clamp is load-bearing rather than defensive. With ``s`` rounded to bf16,
    ``|x / s|`` can reach ``127 * (1 + 2**-8) < 128``, so the clamp to 127 is
    mathematically necessary and -128 is never produced.
    """
    quotient = (x.float() / scales.float().unsqueeze(-1)).round()
    return quotient.clamp_(-QUANT_MAX, QUANT_MAX).to(torch.int8)


def dequantize_rows(q: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Restore ``q * s`` as ``dtype`` (normally ``torch.bfloat16``)."""
    return q.to(dtype) * scales.unsqueeze(-1)


def encode_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise a row batch, returning ``(int8 payload, bf16 scales)``."""
    scales = compute_scales(x)
    return quantize_rows(x, scales), scales


# ---------------------------------------------------------------------------
# Record pack / unpack
# ---------------------------------------------------------------------------


def _flat_uint8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8).reshape(-1)


def pack_records(
    payload: torch.Tensor,
    scales: torch.Tensor,
    *,
    layout: RecordLayout = V1_LAYOUT,
) -> torch.Tensor:
    """Pack payload + scales into ``[*batch, row_bytes]`` ``uint8`` records.

    ``payload``: ``[*batch, head_num, head_dim]`` int8.
    ``scales``:  ``[*batch, head_num]`` bf16.
    """
    if payload.dtype != torch.int8:
        raise TypeError(f"payload must be int8, got {payload.dtype}")
    if scales.dtype != torch.bfloat16:
        raise TypeError(f"scales must be bfloat16, got {scales.dtype}")
    if payload.shape[:-2] != scales.shape[:-1]:
        raise ValueError(
            f"payload batch {tuple(payload.shape[:-2])} does not match "
            f"scales batch {tuple(scales.shape[:-1])}"
        )

    batch = payload.shape[:-2]
    head_dim = payload.shape[-1]
    head_num = payload.shape[-2]
    if layout.payload_bytes != head_num * head_dim:
        raise ValueError(
            f"payload is {head_num * head_dim} bytes but layout row reserves "
            f"{layout.payload_bytes}"
        )
    if layout.scale_bytes != scales.shape[-1] * 2:
        raise ValueError(
            f"scales are {scales.shape[-1] * 2} bytes but layout row reserves "
            f"{layout.scale_bytes}"
        )

    buf = torch.zeros(
        (*batch, layout.row_bytes), dtype=torch.uint8, device=payload.device
    )
    # NB: `.view(torch.uint8)` on a 2-byte dtype consumes the last dimension
    # (8 bf16 -> 16 uint8), so flatten first and reshape the byte view back to
    # `batch` before the slice assignment.
    payload_bytes = payload.reshape(*batch, -1).view(torch.uint8)
    buf[..., layout.payload_offset : layout.payload_offset + payload_bytes.shape[-1]] = (
        payload_bytes.reshape(*batch, -1)
    )
    scale_bytes = scales.reshape(*batch, -1).view(torch.uint8)
    buf[..., layout.scale_offset : layout.scale_offset + scale_bytes.shape[-1]] = (
        scale_bytes.reshape(*batch, -1)
    )
    return buf


def unpack_records(
    records: torch.Tensor,
    *,
    head_num: int,
    head_dim: int,
    layout: RecordLayout = V1_LAYOUT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pack_records`.

    Returns ``(payload [*batch, head_num, head_dim] int8,
    scales [*batch, head_num] bf16)``.
    """
    if records.dtype != torch.uint8:
        raise TypeError(f"records must be uint8, got {records.dtype}")
    if records.shape[-1] != layout.row_bytes:
        raise ValueError(
            f"record width {records.shape[-1]} != layout.row_bytes {layout.row_bytes}"
        )
    batch = records.shape[:-1]
    payload = records[
        ..., layout.payload_offset : layout.payload_offset + head_num * head_dim
    ].contiguous()
    payload = payload.view(torch.int8).reshape(*batch, head_num, head_dim)
    scale_bytes = records[..., layout.scale_offset : layout.scale_offset + head_num * 2]
    scales = scale_bytes.contiguous().view(torch.bfloat16).reshape(*batch, head_num)
    return payload, scales


def encode_records(
    x: torch.Tensor, *, layout: RecordLayout = V1_LAYOUT
) -> torch.Tensor:
    """``[..., head_num, head_dim]`` BF16 -> ``[..., row_bytes]`` ``uint8``."""
    payload, scales = encode_rows(x)
    return pack_records(payload, scales, layout=layout)


def decode_records(
    records: torch.Tensor,
    *,
    head_num: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    layout: RecordLayout = V1_LAYOUT,
) -> torch.Tensor:
    """``[..., row_bytes]`` ``uint8`` -> ``[..., head_num, head_dim]`` ``dtype``."""
    payload, scales = unpack_records(
        records, head_num=head_num, head_dim=head_dim, layout=layout
    )
    return dequantize_rows(payload, scales, dtype)


def encode_decode(x: torch.Tensor, *, layout: RecordLayout = V1_LAYOUT) -> torch.Tensor:
    """Round-trip convenience used by tests and the local error harness."""
    return decode_records(
        encode_records(x, layout=layout),
        head_num=x.shape[-2],
        head_dim=x.shape[-1],
        dtype=x.dtype,
        layout=layout,
    )


# ---------------------------------------------------------------------------
# Error accounting
# ---------------------------------------------------------------------------


def verified_error_bound(
    restored: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Upper bound on ``|x_hat - x|`` derived from the stored scale alone.

    ::

        |x_hat - x|  <=  (0.5 + 2**-8) * s  +  2**-8 * |x_hat|

    Three terms, each traceable to one implementation decision:

    1. ``0.5 * s`` -- round-to-nearest to INT8, in units of the stored scale.
    2. ``2**-8 * s`` -- the scale is BF16 (8 mantissa bits) and its relative
       error is amplified by ``|x / s| <= 127``, giving ``127 * 2**-8 * s``.
       This is exactly why :func:`quantize_rows` forms the quotient in float32
       and why the clamp to 127 is mathematically required, not defensive.
    3. ``2**-8 * |x_hat|`` -- the decoder emits BF16, because L1 attention
       consumes BF16, so the final product is rounded to BF16.

    Term 3 dominates at large magnitudes. It is the reason the naive
    "error <= scale / 2" claim in ``PROJECT.md`` does not survive contact with a
    BF16 output dtype, and the reason the honest claim is stated in three parts.

    ``restored`` is the decoded tensor (BF16); ``scales`` is the stored
    ``[..., head_num]`` BF16 scale. Both are broadcast against each other.
    """
    s = scales.float().unsqueeze(-1)
    return (0.5 + 2**-8) * s + 2**-8 * restored.float().abs()


@dataclass
class ErrorStats:
    """Reconstruction-error summary for one round trip."""

    numel: int
    mean_abs: float
    max_abs: float
    p50_abs: float
    p95_abs: float
    p99_abs: float
    mean_rel: float
    p50_rel: float
    p99_rel: float
    max_rel: float
    max_abs_over_scale: float
    rel_numel: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "numel": self.numel,
            "mean_abs": self.mean_abs,
            "max_abs": self.max_abs,
            "p50_abs": self.p50_abs,
            "p95_abs": self.p95_abs,
            "p99_abs": self.p99_abs,
            "mean_rel": self.mean_rel,
            "p50_rel": self.p50_rel,
            "p99_rel": self.p99_rel,
            "max_rel": self.max_rel,
            "max_abs_over_scale": self.max_abs_over_scale,
            "rel_numel": self.rel_numel,
        }


def error_stats(
    original: torch.Tensor,
    restored: torch.Tensor,
    scales: torch.Tensor | None = None,
    *,
    rel_floor_scale: float = 1e-2,
) -> ErrorStats:
    """Summarise ``|restored - original|``.

    ``scales`` is the ``[..., head_num]`` stored scale; when given, the
    normalised error ``max|x_hat - x| / s`` is reported, which is the quantity
    the ``<= 0.5`` bound in ``PROJECT.md`` refers to.

    Relative error is only informative where ``|x|`` is a meaningful fraction of
    its own head's scale. For an element near zero, ``|x_hat - x| / |x|`` is
    dominated by the denominator and reads ~1.0 for *any* quantiser, exact or
    not -- an artefact, not a quality signal. Elements below
    ``rel_floor_scale * s`` are therefore excluded from the relative
    statistics, and ``p99_rel`` is reported alongside ``max_rel`` so callers can
    quote a figure that is not set by a single near-zero sample.
    """
    if original.shape != restored.shape:
        raise ValueError(
            f"shape mismatch: {tuple(original.shape)} vs {tuple(restored.shape)}"
        )
    diff = (restored.float() - original.float()).abs()
    quantiles = torch.tensor([0.5, 0.95, 0.99], dtype=torch.float32)
    q = torch.quantile(diff.reshape(-1).float(), quantiles)

    max_over_scale = float("nan")
    if scales is not None:
        per_head = diff.amax(dim=-1) / scales.float()
        max_over_scale = float(per_head.max().item())
        threshold = rel_floor_scale * scales.float().unsqueeze(-1)
    else:
        # No scale available: fall back to a magnitude floor relative to the
        # overall dynamic range so the statistic stays meaningful.
        threshold = rel_floor_scale * original.float().abs().max()
    mask = original.float().abs() > threshold

    rel_numel = int(mask.sum())
    if rel_numel:
        rel = (diff[mask] / original.float().abs()[mask]).reshape(-1)
        rel_q = torch.quantile(rel, quantiles)
        mean_rel = float(rel.mean().item())
        p50_rel = float(rel_q[0].item())
        p99_rel = float(rel_q[2].item())
        max_rel = float(rel.max().item())
    else:
        mean_rel = p50_rel = p99_rel = max_rel = float("nan")

    return ErrorStats(
        numel=original.numel(),
        mean_abs=float(diff.mean().item()),
        max_abs=float(diff.max().item()),
        p50_abs=float(q[0].item()),
        p95_abs=float(q[1].item()),
        p99_abs=float(q[2].item()),
        mean_rel=mean_rel,
        p50_rel=p50_rel,
        p99_rel=p99_rel,
        max_rel=max_rel,
        max_abs_over_scale=max_over_scale,
        rel_numel=rel_numel,
    )
