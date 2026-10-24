"""Trace-derived hit-rate ceiling, capacity model, and observed comparison.

The oracle walks the exact token trace in arrival order and counts the longest
prefix that already exists in a trie of earlier requests. That gives the reuse
ceiling for the workload.

Outputs ``oracle.json``, ``oracle.md`` and ``oracle_per_request.jsonl``. The
capacity model and the reuse-multiplicity table are described in
``docs/theory.md``.

Usage::

    python scripts/oracle.py \\
        --trace results/validation-2026-10/trace/trace-seed42.jsonl \\
        --out results/validation-2026-10/oracle.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Qwen3-8B TP=1 geometry. Cross-checked against the codec layout.
LAYERS = 36
KV_HEADS = 8
HEAD_DIM = 128
BF16_BYTES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * 2  # 147,456
INT8_ROW_BYTES = 1152
INT8_BYTES_PER_TOKEN = 2 * LAYERS * INT8_ROW_BYTES  # 82,944


def _display_path(path: Path) -> str:
    """Repo-relative path when possible; a bare name otherwise.

    The artifact is published, so it must not carry the author's home
    directory. The trace itself is identified by its sha256 in the manifest.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def load_trace(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["arrival_index"])
    return rows


def trie_hits(sequences: list[list[int]]) -> list[int]:
    """Longest prefix of each sequence already present from an earlier one."""
    root: dict = {}
    hits: list[int] = []
    for seq in sequences:
        node = root
        n = 0
        for token in seq:
            nxt = node.get(token)
            if nxt is None:
                break
            node = nxt
            n += 1
        hits.append(n)
        node = root
        for token in seq:
            node = node.setdefault(token, {})
    return hits


def lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def group_prefix_lengths(sequences: list[list[int]], groups: list[int]) -> dict[int, int]:
    by_group: dict[int, list[list[int]]] = {}
    for seq, g in zip(sequences, groups):
        by_group.setdefault(g, []).append(seq)
    out = {}
    for g, seqs in by_group.items():
        out[g] = (
            min(lcp(seqs[i], seqs[j]) for i in range(len(seqs)) for j in range(i + 1, len(seqs)))
            if len(seqs) > 1
            else 0
        )
    return out


def capacity_tokens(host_gb: float, bytes_per_token: int) -> int:
    """Mirror of HostKVCache sizing, including the +1 page."""
    return int(host_gb * 1e9 // bytes_per_token) + 1


def fit_groups(capacity: int, prefix_lens: dict[int, int]) -> tuple[list[int], int]:
    """Greedy best-case set of groups whose shared prefixes fit in ``capacity``.

    Sorted smallest-first, because that maximises the number that fit. The
    worst-case (largest-first) count is also returned by the caller as a range.
    """
    order = sorted(prefix_lens, key=lambda g: prefix_lens[g])
    used = 0
    fitted = []
    for g in order:
        if used + prefix_lens[g] > capacity:
            break
        used += prefix_lens[g]
        fitted.append(g)
    return fitted, used


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        default=str(
            REPO_ROOT / "results" / "validation-2026-10" / "trace" / "trace-seed42.jsonl"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "results" / "validation-2026-10" / "oracle.json"),
    )
    parser.add_argument("--md", default=None, help="also write a markdown summary")
    parser.add_argument("--observed-dir", default=str(REPO_ROOT / "results"))
    args = parser.parse_args(argv)

    trace_path = Path(args.trace)
    rows = load_trace(trace_path)
    sequences = [r["token_ids"] for r in rows]
    groups = [r["group"] for r in rows]
    total = sum(len(s) for s in sequences)

    hits = trie_hits(sequences)
    ideal_hits = sum(hits)
    ideal_rate = ideal_hits / total
    unique_tokens = sum(len(s) - h for s, h in zip(sequences, hits))
    prefix_lens = group_prefix_lengths(sequences, groups)
    mean_prefix = statistics.mean(prefix_lens.values())
    mean_q = statistics.mean(len(s) - prefix_lens[g] for s, g in zip(sequences, groups))

    # Per-group reuse value: what the group contributes to the ideal oracle.
    group_hits = {g: 0 for g in prefix_lens}
    for h, g in zip(hits, groups):
        group_hits[g] += h

    # Reuse distance in *unique inserted tokens* per group, and per request.
    last_index: dict[int, int] = {}
    unique_after: list[int] = []
    u = 0
    per_request = []
    for i, (seq, g) in enumerate(zip(sequences, groups)):
        u += len(seq) - hits[i]
        unique_after.append(u)
        gap_tokens = None
        gap_requests = None
        if g in last_index:
            j = last_index[g]
            gap_tokens = u - unique_after[j]
            gap_requests = i - j - 1
        per_request.append(
            {
                "request_id": rows[i]["request_id"],
                "arrival_index": i,
                "group": g,
                "prompt_len": len(seq),
                "ideal_hit_tokens": hits[i],
                "unique_tokens_after": u,
                "reuse_gap_tokens": gap_tokens,
                "reuse_gap_requests": gap_requests,
            }
        )
        last_index[g] = i

    gaps = [p["reuse_gap_tokens"] for p in per_request if p["reuse_gap_tokens"]]

    # ------------------------------------------------------------------
    # Capacity model: all-or-nothing shared prefixes
    # ------------------------------------------------------------------
    def capacity_model(capacity: int) -> dict:
        # Maximum number of prefixes that fit: smallest-first is the best case.
        fitted, used = fit_groups(capacity, prefix_lens)
        n = len(fitted)
        smallest_hits = sum(group_hits[g] for g in fitted)

        # Same count, but the n largest prefixes: the least room left over.
        ordered_big = sorted(prefix_lens, key=lambda g: prefix_lens[g], reverse=True)
        largest = ordered_big[:n]
        largest_hits = sum(group_hits[g] for g in largest)
        largest_bytes = sum(prefix_lens[g] for g in largest)

        # Representative case: the retained groups are a random subset of the
        # same size, so their share of the ideal hits is n / groups.
        proportional = n / len(prefix_lens) * ideal_hits
        return {
            "capacity_tokens": capacity,
            "groups_fit": n,
            "groups_total": len(prefix_lens),
            "prefix_bytes_used": used,
            "predicted_hits_proportional": proportional,
            "predicted_rate_proportional": proportional / total,
            "predicted_rate_smallest_prefixes": smallest_hits / total,
            "predicted_rate_largest_prefixes": largest_hits / total,
            "largest_selection_bytes": largest_bytes,
        }

    bf16_cap = capacity_tokens(8, BF16_BYTES_PER_TOKEN)
    int8_cap = capacity_tokens(8, INT8_BYTES_PER_TOKEN)
    model_bf16 = capacity_model(bf16_cap)
    model_int8 = capacity_model(int8_cap)

    # ------------------------------------------------------------------
    # Roofline vs reuse multiplicity
    # ------------------------------------------------------------------
    prefix_share = mean_prefix / (mean_prefix + mean_q)
    roofline = [
        {
            "prompts_per_group": p,
            "roofline_hit_rate": (p - 1) / p * prefix_share,
        }
        for p in (2, 4, 8, 16, 32, 64, 128, 256)
    ]

    # ------------------------------------------------------------------
    # Capacity sweep (model, for host sizes at the same concurrency)
    # ------------------------------------------------------------------
    sweep = []
    for gb in (1, 2, 4, 8, 16, 32):
        for name, bpt, cap in (
            ("bf16", BF16_BYTES_PER_TOKEN, capacity_tokens(gb, BF16_BYTES_PER_TOKEN)),
            ("int8", INT8_BYTES_PER_TOKEN, capacity_tokens(gb, INT8_BYTES_PER_TOKEN)),
        ):
            m = capacity_model(cap)
            sweep.append(
                {
                    "host_size_gb": gb,
                    "config": name,
                    **m,
                    "observed": False,
                }
            )

    # ------------------------------------------------------------------
    # Observed runs (published Experiment B, same trace)
    # ------------------------------------------------------------------
    def parse_observed(name: str) -> dict | None:
        path = Path(args.observed_dir) / f"exp_{name}_exp-b-8gb.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        out = data.get("benchmark_stdout", "")

        def num(pattern: str, as_float: bool = True):
            m = re.search(pattern, out)
            if not m:
                return None
            text = m.group(1).replace(",", "")
            return float(text) if as_float else int(text)

        absolute = data.get("metrics_absolute_after", {})
        delta = data.get("metrics_delta", {})
        return {
            "file": path.name,
            "sglang_sha": data.get("sglang_sha"),
            "hit_rate": (num(r"Cache hit rate:\s+([0-9.]+)%") or 0) / 100,
            "total_cached_tokens": num(r"Total cached tokens:\s+([0-9,]+)"),
            "total_prompt_tokens": num(r"Total prompt tokens:\s+([0-9,]+)"),
            "device_cached_tokens": num(r"\n\s+Device:\s+([0-9,]+)"),
            "host_cached_tokens": num(r"\n\s+Host:\s+([0-9,]+)"),
            "ttft_mean_ms": num(r"Mean TTFT \(ms\):\s+([0-9.]+)"),
            "ttft_p99_ms": num(r"P99 TTFT \(ms\):\s+([0-9.]+)"),
            "tpot_mean_ms": num(r"Mean TPOT \(ms\):\s+([0-9.]+)"),
            "itl_p99_ms": num(r"P99 ITL \(ms\):\s+([0-9.]+)"),
            "output_throughput": num(r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
            "duration_s": num(r"Benchmark duration \(s\):\s+([0-9.]+)"),
            "load_back_tokens": delta.get("sglang:load_back_tokens_total"),
            "backup_tokens": delta.get("sglang:hicache_backup_tokens_total"),
            "dropped_tokens": delta.get("sglang:hicache_dropped_tokens_total"),
            # Gauges: the delta is a difference of two gauges (0 by
            # construction); the post-run absolute is the capacity.
            "l2_capacity_gauge_absolute": absolute.get("sglang:hicache_host_total_tokens"),
            "l2_used_gauge_absolute": absolute.get("sglang:hicache_host_used_tokens"),
            "l2_capacity_gauge_delta": delta.get("sglang:hicache_host_total_tokens"),
            "server_cache_hit_rate": delta.get("sglang:cache_hit_rate"),
            "prompt_tokens_total_server": delta.get("sglang:prompt_tokens_total"),
        }

    observed = {name: parse_observed(name) for name in ("baseline", "bf16", "int8")}

    result = {
        "trace": {
            "path": _display_path(trace_path),
            "requests": len(rows),
            "total_prompt_tokens": total,
            "groups": len(prefix_lens),
            "mean_prompt_len": total / len(rows),
            "mean_shared_prefix_tokens": mean_prefix,
            "mean_question_tokens": mean_q,
            "prefix_share": prefix_share,
            "unique_tokens": unique_tokens,
            "unique_tokens_ratio": unique_tokens / total,
        },
        "oracle": {
            "definition": "unlimited capacity, sequential, longest prior token prefix",
            "ideal_hits": ideal_hits,
            "ideal_hit_rate": ideal_rate,
            "mean_hit_tokens_per_request": ideal_hits / len(rows),
            "reuse_events": len(gaps),
            "reuse_gap_tokens": {
                "min": min(gaps),
                "median": statistics.median(gaps),
                "mean": statistics.mean(gaps),
                "max": max(gaps),
            },
        },
        "capacity_model": {
            "bf16_8gb": model_bf16,
            "int8_8gb": model_int8,
            "effective_prefix_capacity": {
                "bf16_groups": model_bf16["groups_fit"],
                "int8_groups": model_int8["groups_fit"],
                "groups_total": len(prefix_lens),
            },
        },
        "roofline_by_reuse_multiplicity": roofline,
        "capacity_sweep_model": sweep,
        "observed_experiment_b": observed,
        "definitions": {
            "hit_rate": "sum(cached prompt tokens) / sum(prompt tokens), benchmark denominator",
            "ideal_oracle": "upper bound; omits concurrency, L1 residency, eviction policy, completions",
            "capacity_model": "all-or-nothing shared-prefix residency; a model, not a measurement",
            "observed": "published results/exp_*_exp-b-8gb.json; one run per config",
        },
        "omissions": [
            "concurrency (8 in-flight requests) and request-completion timing",
            "L1 device residency and the per-layer restore path",
            "radix-tree eviction granularity and lock_ref protection",
            "generated continuations and decode scheduling",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    per_request_path = out.with_name("oracle_per_request.jsonl")
    with per_request_path.open("w") as f:
        for row in per_request:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    lines = [
        "# Trace-derived hit-rate ceiling",
        "",
        f"Trace: `{trace_path.name}`. {len(rows)} requests, {total:,} prompt tokens, "
        f"{len(prefix_lens)} groups.",
        "",
        "## Ceiling",
        "",
        "| quantity | value |",
        "| --- | ---: |",
        f"| ideal hits (unlimited capacity, sequential) | {ideal_hits:,} |",
        f"| total prompt tokens | {total:,} |",
        f"| **ideal hit rate** | **{ideal_rate:.2%}** |",
        f"| unique token working set | {unique_tokens:,} ({unique_tokens / total:.1%} of traffic) |",
        f"| mean shared prefix / question | {mean_prefix:,.1f} / {mean_q:,.1f} tokens |",
        "",
        "## Observed vs ceiling",
        "",
        "| config | observed hit rate | ideal / observed | hit tokens | L2 capacity |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, cap in (("baseline", 0), ("bf16", bf16_cap), ("int8", int8_cap)):
        o = observed.get(name)
        if not o:
            continue
        ratio = o["hit_rate"] / ideal_rate if ideal_rate else 0
        lines.append(
            f"| {name} | {o['hit_rate']:.2%} | {ratio:.1%} of ideal | "
            f"{o['total_cached_tokens']:,.0f} | {cap:,} |"
        )

    lines += [
        "",
        "## Why the measured rate is not 95-99%",
        "",
        f"The shared prefix is {prefix_share:.1%} of a prompt. With `P` requests per shared prefix,",
        "the best possible hit rate is `(P-1)/P x prefix share`:",
        "",
        "| requests per prefix | roofline |",
        "| ---: | ---: |",
    ]
    for row in roofline:
        lines.append(f"| {row['prompts_per_group']} | {row['roofline_hit_rate']:.2%} |")
    lines += [
        "",
        "Production 95-99% figures come from workloads with tens to hundreds of requests",
        "per prefix (or multi-turn reuse of a long history). This workload has `P=4`.",
        "",
        "## Capacity model (8 GB host pool)",
        "",
        "| config | L2 tokens | groups fitting | predicted rate (range) | observed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, model in (("bf16", model_bf16), ("int8", model_int8)):
        o = observed.get(name)
        obs = f"{o['hit_rate']:.2%}" if o else "n/a"
        lo = min(
            model["predicted_rate_smallest_prefixes"],
            model["predicted_rate_largest_prefixes"],
        )
        hi = max(
            model["predicted_rate_smallest_prefixes"],
            model["predicted_rate_largest_prefixes"],
        )
        lines.append(
            f"| {name} | {model['capacity_tokens']:,} | {model['groups_fit']}/"
            f"{model['groups_total']} | {lo:.2%} to {hi:.2%} "
            f"(representative {model['predicted_rate_proportional']:.2%}) | {obs} |"
        )
    lines += [
        "",
        "The model is all-or-nothing per shared prefix: a prefix is reusable only if it",
        "fits in L2. It tracks the measured rates without tuning. It is a model, not",
        "a substitute for the per-request counters.",
        "",
        "## Omissions",
        "",
    ]
    lines += [f"- {item}" for item in result["omissions"]]
    lines.append("")

    md = "\n".join(lines)
    if args.md:
        Path(args.md).write_text(md + "\n")
    print(md)
    print(f"wrote {out}")
    print(f"wrote {per_request_path}")
    if args.md:
        print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
