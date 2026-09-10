"""Reproduce SGLang's HiCache host-pool sizing arithmetic exactly.

Mirrors, byte for byte, what ``HostKVCache.__init__`` does at
``python/sglang/srt/mem_cache/pool_host/base.py:193-208``::

    self.dtype = device_pool.store_dtype
    self.size_per_token = self.get_size_per_token()
    device_capacity = getattr(device_pool, "host_capacity_tokens", None) or device_pool.size
    if host_size > 0:
        self.size = sync_fixed_hicache_size(int(host_size * 1e9 // self.size_per_token), host_size)
    else:
        self.size = int(device_capacity * host_to_device_ratio)
    self.page_num = self.size // self.page_size + 1
    self.size = self.page_num * self.page_size

Kept in sync by ``tests/test_capacity.py::test_matches_sglang_formula``, which
re-derives the numbers independently. Running this on the Mac gives the exact
token-capacity numbers the pod runs must reproduce, before any GPU time is
spent.

Usage::

    python scripts/capacity_table.py --hicache-size 8
    python scripts/capacity_table.py --hicache-size 8 --json results/capacity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

# Allow running as a plain script from a fresh checkout.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from hiqcache.layout import (  # noqa: E402
    BF16_ITEMSIZE,
    QWEN3_8B_HEAD_DIM,
    QWEN3_8B_KV_HEADS,
    QWEN3_8B_LAYER_NUM,
    V1_LAYOUT,
    RecordLayout,
)


def host_pool_size(
    *,
    size_per_token: int,
    host_size_gb: float = 0.0,
    host_to_device_ratio: float = 0.0,
    device_capacity: int = 0,
    page_size: int = 1,
) -> int:
    """Token capacity SGLang would compute for a host pool."""
    if host_size_gb > 0:
        size = int(host_size_gb * 1e9 // size_per_token)
    elif host_to_device_ratio > 0:
        size = int(device_capacity * host_to_device_ratio)
    else:
        raise ValueError("one of host_size_gb or host_to_device_ratio must be positive")
    page_num = size // page_size + 1
    return page_num * page_size


@dataclass
class CapacityRow:
    config: str
    size_per_token: int
    bytes_per_token_per_layer: int
    token_capacity: int
    host_bytes_allocated: int
    requested_bytes: int
    capacity_vs_baseline: float
    compression_ratio: float


def capacity_table(
    *,
    hicache_size_gb: float = 8.0,
    layer_num: int = QWEN3_8B_LAYER_NUM,
    head_num: int = QWEN3_8B_KV_HEADS,
    head_dim: int = QWEN3_8B_HEAD_DIM,
    itemsize: int = BF16_ITEMSIZE,
    page_size: int = 1,
    layout: RecordLayout = V1_LAYOUT,
) -> list[CapacityRow]:
    """Baseline BF16 vs HiQCache-encoded host capacity under a fixed GB budget."""
    layout.check_against(head_num, head_dim, itemsize)

    baseline_per_token = 2 * layer_num * head_num * head_dim * itemsize
    encoded_per_token = layout.bytes_per_token_all_layers(layer_num)

    rows: list[CapacityRow] = []
    for name, size_per_token in (
        ("bf16_hicache", baseline_per_token),
        ("hiqcache_int8", encoded_per_token),
    ):
        tokens = host_pool_size(
            size_per_token=size_per_token,
            host_size_gb=hicache_size_gb,
            page_size=page_size,
        )
        rows.append(
            CapacityRow(
                config=name,
                size_per_token=size_per_token,
                bytes_per_token_per_layer=size_per_token // layer_num,
                token_capacity=tokens,
                host_bytes_allocated=tokens * size_per_token,
                requested_bytes=int(hicache_size_gb * 1e9),
                capacity_vs_baseline=float("nan"),  # filled below
                compression_ratio=baseline_per_token / size_per_token,
            )
        )

    base_tokens = rows[0].token_capacity
    for row in rows:
        row.capacity_vs_baseline = row.token_capacity / base_tokens
    return rows


def format_table(rows: list[CapacityRow], hicache_size_gb: float) -> str:
    lines = [
        f"Qwen3-8B  TP=1  page_size=1  --hicache-size={hicache_size_gb:g} (decimal GB)",
        "",
        f"{'config':<16}{'B/token':>10}{'B/layer/tok':>13}{'capacity':>12}"
        f"{'allocated GiB':>15}{'x baseline':>12}",
        "-" * 78,
    ]
    for row in rows:
        lines.append(
            f"{row.config:<16}{row.size_per_token:>10,}"
            f"{row.bytes_per_token_per_layer:>13,}{row.token_capacity:>12,}"
            f"{row.host_bytes_allocated / 2**30:>15.3f}"
            f"{row.capacity_vs_baseline:>11.3f}x"
        )
    encoded = rows[-1]
    lines += [
        "",
        f"storage reduction : {1 - encoded.size_per_token / rows[0].size_per_token:.4%}",
        f"compression ratio : {encoded.compression_ratio:.4f}x",
        f"capacity gain     : {encoded.capacity_vs_baseline:.4f}x",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hicache-size", type=float, default=8.0, help="decimal GB")
    parser.add_argument("--json", type=str, default=None, help="write JSON here")
    args = parser.parse_args()

    rows = capacity_table(hicache_size_gb=args.hicache_size)
    print(format_table(rows, args.hicache_size))

    if args.json:
        from pathlib import Path

        payload = {
            "hicache_size_gb": args.hicache_size,
            "model": "Qwen3-8B",
            "tp": 1,
            "page_size": 1,
            "layer_num": QWEN3_8B_LAYER_NUM,
            "kv_heads": QWEN3_8B_KV_HEADS,
            "head_dim": QWEN3_8B_HEAD_DIM,
            "results": [asdict(row) for row in rows],
        }
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
