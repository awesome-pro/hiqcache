"""Measure codec and PCIe cost of one L2 restore on the GPU.

Times encode, D2H, H2D, and decode plus scatter with CUDA events, and the BF16
transfer of the same tokens as a control. Reports p50/p95/p99 and GB/s.

The transfer phases use a bulk ``copy_``. Stream overlap and the scheduler's
critical path are not measured.

Usage::

    /workspace/venv/bin/python scripts/microbench_codec.py \\
        --sglang-root /workspace/sglang

Outputs ``results/validation-2026-10/raw/microbench.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

LAYERS = 36
HEADS = 8
HEAD_DIM = 128
ROW_BYTES = 1152
BF16_BYTES_PER_TOKEN = 2 * LAYERS * HEADS * HEAD_DIM * 2
INT8_BYTES_PER_TOKEN = 2 * LAYERS * ROW_BYTES


def load_codec(sglang_root: Path):
    import importlib.util

    path = sglang_root / "python/sglang/srt/mem_cache/pool_host/int8_codec.py"
    if not path.is_file():
        raise SystemExit(f"int8_codec.py not found under {sglang_root}")
    spec = importlib.util.spec_from_file_location("int8_codec_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q / 100.0 * (len(ordered) - 1))))
    return ordered[idx]


def timed(fn, *, warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def stats_ms(times: list[float]) -> dict:
    return {
        "n": len(times),
        "mean_ms": statistics.mean(times),
        "p50_ms": pct(times, 50),
        "p95_ms": pct(times, 95),
        "p99_ms": pct(times, 99),
        "min_ms": min(times),
    }


def gbps(nbytes: int, ms: float) -> float:
    return nbytes / (ms / 1000.0) / 1e9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", default=str(REPO_ROOT.parent / "sglang"))
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[512, 2048, 2152, 4096]
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "results" / "validation-2026-10" / "raw" / "microbench.json"
        ),
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    codec = load_codec(Path(args.sglang_root).expanduser().resolve())
    device = torch.device("cuda")

    # A device pool with room for the largest restore, so index_select reads
    # from a realistic layout rather than a tiny contiguous tensor.
    capacity = max(args.tokens) + 1024
    pool = torch.randn(
        (2 * LAYERS, capacity, HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16
    )

    pcie = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pcie.link.gen.max,pcie.link.width.max",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()

    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "pcie_max": pcie,
        "layers": LAYERS,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "row_bytes": ROW_BYTES,
        "bf16_bytes_per_token": BF16_BYTES_PER_TOKEN,
        "int8_bytes_per_token": INT8_BYTES_PER_TOKEN,
        "theoretical_pcie_gen4_x16_gbps": 31.5,
        "runs": {},
    }

    for n_tokens in args.tokens:
        n_rows = 2 * LAYERS * n_tokens
        indices = torch.randperm(capacity, device=device)[:n_tokens]
        scatter = indices.to(torch.int64)

        staging = torch.empty((n_rows, ROW_BYTES), device=device, dtype=torch.uint8)
        pinned = torch.empty((n_rows, ROW_BYTES), dtype=torch.uint8, pin_memory=True)
        gathered = torch.empty(
            (2 * LAYERS, n_tokens, HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16
        )

        def index_select_only():
            for layer in range(2 * LAYERS):
                gathered[layer].copy_(pool[layer].index_select(0, indices))

        def encode_only():
            for layer in range(2 * LAYERS):
                codec.write_record(gathered[layer], staging[layer * n_tokens : (layer + 1) * n_tokens])

        def encode_full():
            for layer in range(2 * LAYERS):
                rows = pool[layer].index_select(0, indices)
                codec.write_record(
                    rows, staging[layer * n_tokens : (layer + 1) * n_tokens]
                )

        def d2h_bulk():
            pinned.copy_(staging, non_blocking=True)

        def h2d_bulk():
            staging.copy_(pinned, non_blocking=True)

        def d2h_per_layer():
            for layer in range(2 * LAYERS):
                pinned[layer * n_tokens : (layer + 1) * n_tokens].copy_(
                    staging[layer * n_tokens : (layer + 1) * n_tokens], non_blocking=True
                )

        def h2d_per_layer():
            for layer in range(2 * LAYERS):
                staging[layer * n_tokens : (layer + 1) * n_tokens].copy_(
                    pinned[layer * n_tokens : (layer + 1) * n_tokens], non_blocking=True
                )

        def decode_and_scatter():
            for layer in range(2 * LAYERS):
                out = codec.decode_records(
                    staging[layer * n_tokens : (layer + 1) * n_tokens],
                    head_num=HEADS,
                    head_dim=HEAD_DIM,
                    dtype=torch.bfloat16,
                )
                pool[layer][scatter] = out.reshape(n_tokens, HEADS, HEAD_DIM)

        phases = {
            "index_select": index_select_only,
            "bf16_d2h": d2h_bulk,
            "bf16_h2d": h2d_bulk,
            "int8_d2h_bulk": d2h_bulk,
            "int8_h2d_bulk": h2d_bulk,
            "int8_d2h_per_layer": d2h_per_layer,
            "int8_h2d_per_layer": h2d_per_layer,
            "encode_only": encode_only,
            "encode_full": encode_full,
            "decode_and_scatter": decode_and_scatter,
        }

        entry = {"tokens": n_tokens, "phases": {}}
        for name, fn in phases.items():
            times = timed(fn, warmup=args.warmup, reps=args.reps)
            entry["phases"][name] = stats_ms(times)

        # Bandwidth is only meaningful for the copy phases.
        int8_bytes = n_rows * ROW_BYTES
        bf16_bytes = 2 * n_rows * HEADS * HEAD_DIM * 2
        for name in ("int8_d2h_bulk", "int8_h2d_bulk", "int8_d2h_per_layer", "int8_h2d_per_layer"):
            entry["phases"][name]["gbps"] = gbps(int8_bytes, entry["phases"][name]["p50_ms"])
            entry["phases"][name]["bytes"] = int8_bytes
        for name in ("bf16_d2h", "bf16_h2d"):
            entry["phases"][name]["gbps"] = gbps(bf16_bytes, entry["phases"][name]["p50_ms"])
            entry["phases"][name]["bytes"] = bf16_bytes

        # End-to-end restore estimate: H2D + decode, from the measured phases.
        h2d = entry["phases"]["int8_h2d_bulk"]["p50_ms"]
        dec = entry["phases"]["decode_and_scatter"]["p50_ms"]
        entry["estimated_restore_ms"] = h2d + dec
        entry["estimated_restore_gbps"] = gbps(int8_bytes, h2d)

        result["runs"][str(n_tokens)] = entry
        print(
            f"{n_tokens:>5} tokens  encode_full {entry['phases']['encode_full']['p50_ms']:7.2f} ms  "
            f"d2h {entry['phases']['int8_d2h_bulk']['p50_ms']:7.2f} ms "
            f"({entry['phases']['int8_d2h_bulk']['gbps']:5.1f} GB/s)  "
            f"h2d {h2d:7.2f} ms ({entry['phases']['int8_h2d_bulk']['gbps']:5.1f} GB/s)  "
            f"decode {dec:7.2f} ms  restore {entry['estimated_restore_ms']:7.2f} ms"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
