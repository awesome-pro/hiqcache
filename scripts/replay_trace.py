"""Replay the exact 128-request trace and record per-request cache/latency data.

The published benchmark reports only aggregates. This client sends the same
tokenized prompts, in the same arrival order, at the same concurrency, and
records for every request:

* ``cached_tokens`` and ``cached_tokens_details`` (device / host split) from
  the server's own ``meta_info``;
* TTFT, end-to-end latency, and the full ITL series;
* the group id from the trace, so per-group residency can be checked against
  the 25/32-prefix capacity model.

That is what separates "the hit rate is higher" from "the extra hits are L2
restores that shorten TTFT", and it is the per-request correlation the brief
asks for.

The workload is the published Experiment B one: 8 GB host pool, L1 capped at
16,384 tokens, concurrency 8. One configuration per invocation, so the caller
alternates BF16 and INT8 rather than pooling them.

Usage::

    /workspace/venv/bin/python scripts/replay_trace.py \\
        --config bf16 --trace results/validation-2026-10/trace/trace-seed42.jsonl

Outputs ``results/validation-2026-10/raw/replay_<config>.json`` and the server
log beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_experiment import build_configs, http_get, scrape_metrics, wait_for_server  # noqa: E402


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q / 100.0 * (len(ordered) - 1))))
    return ordered[idx]


def load_trace(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["arrival_index"])
    return rows


def one_request(
    url: str, row: dict, *, max_new_tokens: int, timeout: float
) -> dict:
    """Send one prompt, stream the response, timestamp first token and ITLs."""
    import requests

    payload = {
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "return_logprob": False,
    }
    # The trace is token ids; the native /generate endpoint accepts input_ids,
    # which skips tokenization and makes the replay exact.
    if "token_ids" in row:
        payload["input_ids"] = row["token_ids"]
    else:
        payload["text"] = row["prompt"]

    st = time.perf_counter()
    ttft = None
    last_ts = st
    last_completion = 0
    itls: list[float] = []
    meta: dict = {}
    error = None

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                error = f"{resp.status_code}: {resp.text[:500]}"
            else:
                for raw in resp.iter_lines(decode_unicode=False):
                    if not raw:
                        continue
                    if raw.startswith(b"data: "):
                        raw = raw[6:]
                    if raw == b"[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    chunk_meta = data.get("meta_info") or {}
                    for key, value in chunk_meta.items():
                        if value is not None:
                            meta[key] = value
                    if data.get("text"):
                        now = time.perf_counter()
                        completion = int(chunk_meta.get("completion_tokens") or 0)
                        if ttft is None:
                            ttft = now - st
                        elif completion > last_completion:
                            gap = now - last_ts
                            n = completion - last_completion
                            itls.extend([gap / n] * n)
                        last_ts = now
                        last_completion = completion
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    e2e = time.perf_counter() - st
    details = meta.get("cached_tokens_details") or {}
    return {
        "request_id": row["request_id"],
        "arrival_index": row["arrival_index"],
        "group": row["group"],
        "dataset_prompt_len": row["prompt_len"],
        "prompt_tokens": int(meta.get("prompt_tokens") or 0),
        "completion_tokens": int(meta.get("completion_tokens") or 0),
        "cached_tokens": int(meta.get("cached_tokens") or 0),
        "device_cached_tokens": details.get("device"),
        "host_cached_tokens": details.get("host"),
        "ttft_ms": (ttft or 0.0) * 1000.0,
        "e2e_ms": e2e * 1000.0,
        "itl_ms": [x * 1000.0 for x in itls],
        "success": error is None,
        "error": error,
    }


def summarize(requests: list[dict]) -> dict:
    ok = [r for r in requests if r["success"]]
    ttfts = [r["ttft_ms"] for r in ok]
    e2es = [r["e2e_ms"] for r in ok]
    itls = [x for r in ok for x in r["itl_ms"]]
    tpots = [
        (r["e2e_ms"] - r["ttft_ms"]) / max(1, r["completion_tokens"] - 1) for r in ok
    ]
    total_prompt = sum(r["prompt_tokens"] for r in ok)
    total_cached = sum(r["cached_tokens"] for r in ok)
    output_tokens = sum(r["completion_tokens"] for r in ok)
    device = sum(r["device_cached_tokens"] or 0 for r in ok)
    host = sum(r["host_cached_tokens"] or 0 for r in ok)

    buckets: dict[str, list[float]] = {}
    for r in ok:
        frac = r["cached_tokens"] / r["prompt_tokens"] if r["prompt_tokens"] else 0.0
        if frac == 0:
            key = "0 (miss)"
        elif frac < 0.5:
            key = "(0, 0.5)"
        elif frac < 0.9:
            key = "[0.5, 0.9)"
        else:
            key = "[0.9, 1] (prefix hit)"
        buckets.setdefault(key, []).append(r["ttft_ms"])

    groups: dict[int, list[dict]] = {}
    for r in ok:
        groups.setdefault(r["group"], []).append(r)

    return {
        "requests_total": len(requests),
        "requests_ok": len(ok),
        "requests_failed": len(requests) - len(ok),
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
        "hit_rate": total_cached / total_prompt if total_prompt else None,
        "device_cached_tokens": device,
        "host_cached_tokens": host,
        "device_share_of_hits": device / total_cached if total_cached else None,
        "host_share_of_hits": host / total_cached if total_cached else None,
        "requests_with_any_hit": sum(1 for r in ok if r["cached_tokens"] > 0),
        "requests_with_prefix_hit": sum(
            1
            for r in ok
            if r["prompt_tokens"] and r["cached_tokens"] / r["prompt_tokens"] >= 0.9
        ),
        "ttft_mean_ms": statistics.mean(ttfts) if ttfts else None,
        "ttft_p50_ms": pct(ttfts, 50),
        "ttft_p95_ms": pct(ttfts, 95),
        "ttft_p99_ms": pct(ttfts, 99),
        "e2e_mean_ms": statistics.mean(e2es) if e2es else None,
        "tpot_mean_ms": statistics.mean(tpots) if tpots else None,
        "tpot_p95_ms": pct(tpots, 95),
        "itl_p50_ms": pct(itls, 50),
        "itl_p95_ms": pct(itls, 95),
        "itl_p99_ms": pct(itls, 99),
        "output_tokens": output_tokens,
        "ttft_by_cached_fraction": {
            key: {
                "n": len(vals),
                "mean_ms": statistics.mean(vals),
                "p50_ms": pct(vals, 50),
            }
            for key, vals in sorted(buckets.items())
        },
        "per_group": [
            {
                "group": group,
                "n": len(rows),
                "prefix_hits": sum(
                    1
                    for r in rows
                    if r["prompt_tokens"] and r["cached_tokens"] / r["prompt_tokens"] >= 0.9
                ),
                "mean_cached_fraction": statistics.mean(
                    r["cached_tokens"] / r["prompt_tokens"]
                    for r in rows
                    if r["prompt_tokens"]
                ),
                "mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in rows),
            }
            for group, rows in sorted(groups.items())
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=["baseline", "bf16", "int8"], required=True)
    parser.add_argument(
        "--trace",
        default=str(
            REPO_ROOT / "results" / "validation-2026-10" / "trace" / "trace-seed42.jsonl"
        ),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--host-size", type=float, default=8.0)
    parser.add_argument("--max-total-tokens", type=int, default=16_384)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--sglang-root", default=str(REPO_ROOT.parent / "sglang")
    )
    parser.add_argument(
        "--out-dir", default=str(REPO_ROOT / "results" / "validation-2026-10" / "raw")
    )
    args = parser.parse_args(argv)

    from concurrent.futures import ThreadPoolExecutor

    trace_path = Path(args.trace)
    rows = load_trace(trace_path)
    sglang_root = Path(args.sglang_root).expanduser().resolve()
    base_url = f"http://127.0.0.1:{args.port}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs(
        args.model,
        args.host_size,
        1,
        1,
        max_total_tokens=args.max_total_tokens,
    )
    config = configs[args.config]
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        *config.server_args,
        "--port",
        str(args.port),
    ]
    env = dict(os.environ)
    env.update(config.env)
    env["PYTHONPATH"] = f"{sglang_root / 'python'}:{env.get('PYTHONPATH', '')}"
    log_path = out_dir / f"replay_{args.config}.server.log"

    print(f"=== {config.label}")
    print(f"    {len(rows)} requests, concurrency {args.concurrency}, trace {trace_path.name}")
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=sglang_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    result: dict = {
        "config": args.config,
        "label": config.label,
        "sglang_sha": None,
        "trace": str(trace_path),
        "trace_requests": len(rows),
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "host_size_gb": args.host_size,
        "max_total_tokens": args.max_total_tokens,
        "server_args": cmd,
        "env": config.env,
        "started_at": started,
        "server_log": str(log_path),
    }
    try:
        waited = wait_for_server(base_url, args.startup_timeout, proc)
        print(f"    healthy after {waited:.0f}s")
        result["startup_seconds"] = waited
        try:
            result["sglang_sha"] = subprocess.check_output(
                ["git", "-C", str(sglang_root), "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:  # noqa: BLE001
            pass

        try:
            http_get(f"{base_url}/flush_cache", timeout=120)
            result["flushed_cache"] = True
        except Exception as exc:  # noqa: BLE001
            result["flushed_cache"] = False
            result["flush_error"] = str(exc)

        before = scrape_metrics(base_url)
        result["l2_capacity_absolute"] = before.get(
            "sglang:hicache_host_total_tokens"
        )

        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            requests = list(
                pool.map(
                    lambda row: one_request(
                        f"{base_url}/generate",
                        row,
                        max_new_tokens=args.max_new_tokens,
                        timeout=args.request_timeout,
                    ),
                    rows,
                )
            )
        wall = time.perf_counter() - wall_start
        after = scrape_metrics(base_url)

        result["wall_seconds"] = wall
        result["ended_at"] = datetime.now(timezone.utc).isoformat()
        result["requests"] = requests
        result["summary"] = summarize(requests)
        result["summary"]["request_throughput"] = len(requests) / wall
        result["summary"]["output_throughput"] = (
            result["summary"]["output_tokens"] / wall
        )
        result["metrics_before"] = before
        result["metrics_after"] = after
        result["metrics_delta"] = {
            key: after.get(key, 0.0) - before.get(key, 0.0)
            for key in set(before) | set(after)
        }
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    out = out_dir / f"replay_{args.config}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    s = result.get("summary", {})
    if not s or not s.get("requests_ok"):
        print(f"    no successful requests; wrote {out}")
        return 1
    print(
        f"    hit rate {s.get('hit_rate'):.3%} "
        f"({s.get('total_cached_tokens'):,}/{s.get('total_prompt_tokens'):,}); "
        f"device {s.get('device_cached_tokens'):,} host {s.get('host_cached_tokens'):,}"
    )
    print(
        f"    TTFT mean {s.get('ttft_mean_ms'):.0f} p50 {s.get('ttft_p50_ms'):.0f} "
        f"p95 {s.get('ttft_p95_ms'):.0f} p99 {s.get('ttft_p99_ms'):.0f} ms; "
        f"TPOT mean {s.get('tpot_mean_ms'):.1f} ms"
    )
    print(f"    wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
