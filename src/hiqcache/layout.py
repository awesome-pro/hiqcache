"""Byte layout of a HiQCache encoded KV record.

This module is the **single source of truth** for every byte offset used by the
codec. The CUDA/pod implementation and the local CPU/MPS tests both derive their
addressing from these helpers, so a layout bug cannot pass one and fail the
other silently.

Encoded record (one per ``(layer, K|V, token)`` row)::

    offset 0                        1023
    |        1024 INT8 payload       |      head_dim bytes per head, 8 heads
    1024                            1039
    |      8 BF16 scales (16 B)      |      one scale per KV head
    1040                            1151
    |        112 bytes padding       |      unused, keeps row == 9 x 128 B

``ROW_BYTES = 1152 = 9 x 128`` is deliberate: SGLang's CUDA JIT HiCache mover is
a pure element-wise byte copier that requires ``element_size % 128 == 0`` on
CUDA, so a packed row is admissible with zero kernel changes.

See ``docs/design.md`` and ``docs/hicache-internals.md`` for the kernel-side
proof that ``element_size = 1152`` lands on the fast path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# ---------------------------------------------------------------------------
# Fixed format constants
# ---------------------------------------------------------------------------

#: Bytes of one INT8 payload: ``HEAD_NUM * HEAD_DIM`` one-byte quantised values.
DEFAULT_PAYLOAD_BYTES = 1024

#: One BF16 scale per KV head, stored verbatim.
DEFAULT_SCALE_BYTES = 16

#: Padding, chosen so that a row is a whole number of 128-byte groups.
DEFAULT_PADDING_BYTES = 112

#: Bytes of padding reserved at the front of the payload region for future
#: per-row metadata. Zero today; the remaining padding absorbs it without
#: changing ``ROW_BYTES``.
DEFAULT_HEADER_BYTES = 0

#: CUDA JIT mover requirement: ``element_size % ALIGNMENT_BYTES == 0``.
ALIGNMENT_BYTES = 128

KVCACHE_K = 0
KVCACHE_V = 1


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return -(-value // multiple) * multiple


@dataclass(frozen=True)
class RecordLayout:
    """Byte layout of one encoded MHA row.

    Defaults reproduce the V1 format: a 1152-byte row holding a 1024-byte INT8
    payload and 8 BF16 scales, sized for 8 local KV heads at TP=1.
    """

    payload_bytes: int = DEFAULT_PAYLOAD_BYTES
    scale_bytes: int = DEFAULT_SCALE_BYTES
    padding_bytes: int = DEFAULT_PADDING_BYTES
    header_bytes: int = DEFAULT_HEADER_BYTES

    def __post_init__(self) -> None:
        for name in ("payload_bytes", "scale_bytes", "padding_bytes", "header_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if self.payload_bytes == 0:
            raise ValueError("payload_bytes must be positive")
        if self.header_bytes > self.padding_bytes:
            raise ValueError(
                f"header_bytes ({self.header_bytes}) does not fit in "
                f"padding_bytes ({self.padding_bytes})"
            )

    # -- sizes -------------------------------------------------------------

    @property
    def row_bytes(self) -> int:
        """Total bytes of one encoded K-or-V row for one token."""
        return self.payload_bytes + self.scale_bytes + self.padding_bytes

    @property
    def payload_offset(self) -> int:
        return self.header_bytes

    @property
    def scale_offset(self) -> int:
        """Byte offset where the BF16 scale block starts."""
        return self.header_bytes + self.payload_bytes

    @property
    def padding_offset(self) -> int:
        """Byte offset where padding starts (== end of meaningful data)."""
        return self.scale_offset + self.scale_bytes

    @property
    def aligned_row_bytes(self) -> int:
        """Row size rounded up to the mover's group multiple."""
        return _round_up(self.row_bytes, ALIGNMENT_BYTES)

    @property
    def is_mover_aligned(self) -> bool:
        """True when the row can ride SGLang's CUDA JIT byte mover unchanged."""
        return self.row_bytes % ALIGNMENT_BYTES == 0

    # -- per-token KV accounting ------------------------------------------

    @property
    def bytes_per_token(self) -> int:
        """Encoded bytes for one token across both K and V of one layer."""
        return 2 * self.row_bytes

    def bytes_per_token_all_layers(self, layer_num: int) -> int:
        return self.bytes_per_token * layer_num

    def uncompressed_bytes_per_token_all_layers(
        self, layer_num: int, head_num: int, head_dim: int, itemsize: int
    ) -> int:
        return 2 * layer_num * head_num * head_dim * itemsize

    # -- validation --------------------------------------------------------

    def check_against(self, head_num: int, head_dim: int, itemsize: int) -> None:
        """Fail fast if the format cannot represent the given KV geometry.

        ``head_num * head_dim`` must equal the payload width exactly, and the
        scale block must hold one BF16 per head. Anything else means the caller
        built a pool whose K/V geometry the format silently cannot express.
        """
        expected_payload = head_num * head_dim * itemsize // 2
        if self.payload_bytes != expected_payload:
            raise ValueError(
                f"payload_bytes={self.payload_bytes} does not match "
                f"head_num*head_dim={head_num * head_dim} "
                f"(expected {expected_payload} bytes for INT8 payload)"
            )
        expected_scales = head_num * 2
        if self.scale_bytes != expected_scales:
            raise ValueError(
                f"scale_bytes={self.scale_bytes} does not match one BF16 scale "
                f"per head_num={head_num} (expected {expected_scales})"
            )
        if not self.is_mover_aligned:
            raise ValueError(
                f"row_bytes={self.row_bytes} is not a multiple of "
                f"{ALIGNMENT_BYTES}; SGLang's CUDA JIT HiCache mover requires "
                f"element_size % {ALIGNMENT_BYTES} == 0"
            )

    # -- addressing --------------------------------------------------------

    def payload_slice(self, head: int, head_dim: int) -> tuple[int, int]:
        """``[start, stop)`` byte offsets of one head's INT8 payload."""
        start = self.payload_offset + head * head_dim
        return start, start + head_dim

    def scale_slice(self, head: int) -> tuple[int, int]:
        """``[start, stop)`` byte offsets of one head's BF16 scale."""
        start = self.scale_offset + head * 2
        return start, start + 2

    def record_offsets(self, slots: int) -> list[int]:
        """Byte offsets of ``slots`` consecutive records inside a K/V arena."""
        return [slot * self.row_bytes for slot in range(slots)]


#: The V1 format: one 1152-byte row per (layer, K|V, token).
V1_LAYOUT = RecordLayout()


def describe_compression(
    layout: RecordLayout,
    *,
    layer_num: int,
    head_num: int,
    head_dim: int,
    itemsize: int = 2,
) -> dict[str, float | int]:
    """Compute the storage-reduction numbers for the given KV geometry."""
    layout.check_against(head_num, head_dim, itemsize)
    baseline = layout.uncompressed_bytes_per_token_all_layers(
        layer_num, head_num, head_dim, itemsize
    )
    encoded = layout.bytes_per_token_all_layers(layer_num)
    return {
        "layer_num": layer_num,
        "head_num": head_num,
        "head_dim": head_dim,
        "row_bytes": layout.row_bytes,
        "baseline_bytes_per_token": baseline,
        "encoded_bytes_per_token": encoded,
        "reduction_fraction": 1.0 - encoded / baseline,
        "compression_ratio": baseline / encoded,
    }


# ---------------------------------------------------------------------------
# Qwen3-8B geometry, used by tests and the capacity calculator
# ---------------------------------------------------------------------------

QWEN3_8B_LAYER_NUM = 36
QWEN3_8B_KV_HEADS = 8
QWEN3_8B_HEAD_DIM = 128
BF16_ITEMSIZE = 2


def qwen3_8b_layout_check(layout: RecordLayout = V1_LAYOUT) -> None:
    layout.check_against(QWEN3_8B_KV_HEADS, QWEN3_8B_HEAD_DIM, BF16_ITEMSIZE)


def qwen3_8b_compression(layout: RecordLayout = V1_LAYOUT) -> dict[str, float | int]:
    return describe_compression(
        layout,
        layer_num=QWEN3_8B_LAYER_NUM,
        head_num=QWEN3_8B_KV_HEADS,
        head_dim=QWEN3_8B_HEAD_DIM,
        itemsize=BF16_ITEMSIZE,
    )


def iter_region_spans(
    layout: RecordLayout, head_num: int, head_dim: int
) -> Iterable[tuple[str, int, int]]:
    """Yield ``(region, start, stop)`` for every meaningful byte span of a row.

    Used by tests to assert that regions never overlap and that every byte is
    either accounted for or explicitly padding.
    """
    for head in range(head_num):
        start, stop = layout.payload_slice(head, head_dim)
        yield f"payload[{head}]", start, stop
    for head in range(head_num):
        start, stop = layout.scale_slice(head)
        yield f"scale[{head}]", start, stop
