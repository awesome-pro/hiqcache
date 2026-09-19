"""Phase 14/18 analysis: turn pod result JSON into the PROJECT.md tables.

Runs on the Mac, on results copied back from the pod. The pod is for measuring,
not for analysing.

Usage::

    python scripts/analyse.py results/
    python scripts/analyse.py results/ --json results/analysis.json --markdown
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

#: Metrics that answer each PROJECT.md Phase 14 category.
CATEGORIES = {
    "storage": [
        ("sglang:hicache_host_total_tokens", "L2 token capacity", "count"),
        ("sglang:hicache_host_used_tokens", "L2 tokens used", "count"),
        ("sglang:hicache_dropped_tokens_total", "L2 evictions", "count"),
    ],
    "d2h": [
        ("sglang:hicache_backup_bytes_total", "backup bytes", "bytes"),
        ("sglang:hicache_backup_tokens_total", "backup tokens", "count"),
        ("sglang:hicache_backup_duration_seconds", "backup time", "seconds"),
    ],
    "h2d": [
        ("sglang:load_back_bytes_total", "restore bytes", "bytes"),
        ("sglang:load_back_tokens_total", "restore tokens", "count"),
        ("sglang:load_back_duration_seconds", "restore time", "seconds"),
    ],
    "serving": [
        ("sglang:cache_hit_rate", "prefix cache hit rate", "ratio"),
        ("sglang:prompt_tokens_total", "prefill tokens", "count"),
        ("sglang:generation_tokens_total", "decode tokens", "count"),
    ],
}


def human(value: float, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "bytes":
        for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
            if abs(value) >= scale:
                return f"{value / scale:,.2f} {unit}"
        return f"{value:,.0f} B"
    if kind == "seconds":
        return f"{value:,.4f} s"
    if kind == "ratio":
        return f"{value:.4f}"
    return f"{value:,.0f}"


@dataclass
class Run:
    path: Path
    config: str
    tag: str
    workload: str
    ok: bool
    sglang_sha: str
    host_size_gb: float
    delta: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)
    absolute: dict = field(default_factory=dict)
    capacity_budget: dict = field(default_factory=dict)
    check: dict | None = None
    error: str | None = None
    benchmark_stdout: str = ""
    bench: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.config}/{self.tag}/{self.workload}"


def load_runs(root: Path) -> list[Run]:
    runs: list[Run] = []
    for path in sorted(root.glob("exp_*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipping {path.name}: {exc}")
            continue
        runs.append(
            Run(
                path=path,
                config=payload.get("config", "?"),
                tag=payload.get("tag", "?"),
                workload=payload.get("workload", {}).get("name", "?"),
                ok=bool(payload.get("ok")),
                sglang_sha=payload.get("sglang_sha", "?")[:12],
                host_size_gb=payload.get("host_size_gb", 0.0),
                delta=payload.get("metrics_delta", {}),
                derived=payload.get("derived", {}),
                absolute=payload.get("metrics_absolute_after", {}),
                capacity_budget=payload.get("capacity_budget", {}),
                check=payload.get("encoded_bytes_per_token_check"),
                error=payload.get("error"),
                benchmark_stdout=payload.get("benchmark_stdout", ""),
                bench=parse_benchmark(payload.get("benchmark_stdout", "")),
            )
        )
    return runs


#: Metrics the benchmark itself reports, which are the ones that answer
#: PROJECT.md's serving questions. The Prometheus counters above are cumulative
#: process-lifetime totals and are useful for the codec's byte arithmetic, but
#: they say nothing about cache effectiveness -- an earlier version of this file
#: compared prefill tokens across configs and concluded "comparable
#: recomputation" while the benchmark's own report showed a 16.6-point cache-hit
#: gap. Read the benchmark's numbers for serving, the counters for bytes.
_BENCH_FIELDS = {
    "hit_rate_pct": r"Cache hit rate:\s*([0-9.]+)%",
    "ttft_mean_ms": r"Mean TTFT \(ms\):\s*([0-9.]+)",
    "ttft_p99_ms": r"P99 TTFT \(ms\):\s*([0-9.]+)",
    "tpot_mean_ms": r"Mean TPOT \(ms\):\s*([0-9.]+)",
    "itl_p99_ms": r"P99 ITL \(ms\):\s*([0-9.]+)",
    "out_tok_per_s": r"Output token throughput \(tok/s\):\s*([0-9.]+)",
    "prompt_tokens": r"Total prompt tokens:\s*([0-9]+)",
    "cached_total": r"Total cached tokens:\s*([0-9]+)",
}


def parse_benchmark(stdout: str) -> dict:
    """Pull the serving metrics out of sglang.benchmark.serving's report."""
    import re as _re

    out = {}
    for key, pattern in _BENCH_FIELDS.items():
        m = _re.search(pattern, stdout or "")
        if m:
            out[key] = float(m.group(1))
    # The Host/Device split appears twice (tokens, then rate); take the token
    # block, which is the first occurrence after "Total cached tokens".
    block = (stdout or "").split("Total cached tokens:", 1)
    if len(block) == 2:
        tail = block[1]
        dm = _re.search(r"Device:\s*([0-9]+)", tail)
        hm = _re.search(r"Host:\s*([0-9]+)", tail)
        if dm:
            out["cached_device_tokens"] = float(dm.group(1))
        if hm:
            out["cached_host_tokens"] = float(hm.group(1))
    return out


def group(runs: list[Run]) -> dict[tuple[str, str], dict[str, Run]]:
    """Group by (tag, workload) so configs are compared within a sweep."""
    out: dict[tuple[str, str], dict[str, Run]] = {}
    for run in runs:
        out.setdefault((run.tag, run.workload), {})[run.config] = run
    return out


def print_run(run: Run) -> None:
    print(f"\n--- {run.label}")
    print(f"    sglang {run.sglang_sha}  host_size={run.host_size_gb:g} GB  "
          f"ok={run.ok}")
    if run.error:
        print(f"    ERROR: {run.error}")
    for category, metrics in CATEGORIES.items():
        rows = [(label, run.delta.get(key, 0.0), kind) for key, label, kind in metrics]
        if not any(value for _, value, _ in rows):
            continue
        print(f"    {category}:")
        for label, value, kind in rows:
            print(f"      {label:<26} {human(value, kind):>16}")
    for key, value in run.derived.items():
        print(f"    derived.{key:<28} {value:>16,.4f}")
    if run.check:
        verdict = "PASS" if run.check.get("pass") else "FAIL"
        print(f"    encoded bytes/token check: {verdict} "
              f"(measured {run.check['measured']:,.0f} vs "
              f"expected {run.check['expected']:,})")


def print_comparison(tag: str, workload: str, configs: dict[str, Run]) -> None:
    print(f"\n{'=' * 78}")
    print(f"SWEEP {tag} / workload={workload}")
    print(f"{'=' * 78}")

    shas = {run.sglang_sha for run in configs.values()}
    if len(shas) > 1:
        print(f"  WARNING: configs ran on different SGLang SHAs {sorted(shas)}; "
              f"results are not comparable")
    order = [c for c in ("baseline", "bf16", "int8") if c in configs]
    if not order:
        return

    def row(label: str, pick, kind: str = "count") -> None:
        """One comparison row; every picker takes the run and returns a number."""
        cells = []
        for name in order:
            run = configs.get(name)
            value = pick(run) if run else None
            cells.append(human(value, kind))
        print(f"  {label:<30}" + "".join(f"{c:>15}" for c in cells))

    def num(value, _kind=None, fmt="{:,.0f}"):
        """Numeric formatter for derived ratios (no unit scaling)."""
        return fmt.format(value) if value else "-"

    print(f"  {'metric':<30}" + "".join(f"{c:>15}" for c in order))
    print("  " + "-" * (30 + 15 * len(order)))

    # L2 capacity is a GAUGE, not a counter: its delta across the measured phase
    # is zero by definition. Read the absolute value -- it is the fixed
    # allocation the entire experiment turns on.
    row("L2 capacity (tokens)",
        lambda r: r.absolute.get("sglang:hicache_host_total_tokens"))
    row("L2 tokens used (final)",
        lambda r: r.absolute.get("sglang:hicache_host_used_tokens"))
    row("backup bytes",
        lambda r: r.delta.get("sglang:hicache_backup_bytes_total"), "bytes")
    row("measured B/token (backup)",
        lambda r: r.derived.get("measured_backup_bytes_per_token"))
    row("restore bytes",
        lambda r: r.delta.get("sglang:load_back_bytes_total"), "bytes")
    row("backup time",
        lambda r: r.delta.get("sglang:hicache_backup_duration_seconds"), "seconds")
    row("restore time",
        lambda r: r.delta.get("sglang:load_back_duration_seconds"), "seconds")
    # ---- serving metrics, from the benchmark's own report ----------------
    # These are the numbers PROJECT.md's serving table asks for. The
    # Prometheus counters above only describe bytes.
    print(f"  {'serving (benchmark report)':<30}" + "".join(f"{c:>15}" for c in order))
    for label, key, fmt in (
        ("cache hit rate %", "hit_rate_pct", "{:,.1f}"),
        ("cached from host (tokens)", "cached_host_tokens", "{:,.0f}"),
        ("cached from device (tok)", "cached_device_tokens", "{:,.0f}"),
        ("TTFT mean (ms)", "ttft_mean_ms", "{:,.1f}"),
        ("TTFT p99 (ms)", "ttft_p99_ms", "{:,.1f}"),
        ("TPOT mean (ms)", "tpot_mean_ms", "{:,.1f}"),
        ("ITL p99 (ms)", "itl_p99_ms", "{:,.1f}"),
        ("output tok/s", "out_tok_per_s", "{:,.1f}"),
    ):
        cells = []
        for name in order:
            run = configs.get(name)
            value = run.bench.get(key) if run and run.bench else None
            cells.append(fmt.format(value) if value is not None else "-")
        print(f"  {label:<30}" + "".join(f"{c:>15}" for c in cells))

    # ---- Experiment B verdict, from the metrics that actually bear on it ---
    if "bf16" in configs and "int8" in configs:
        b = configs["bf16"].bench or {}
        i = configs["int8"].bench or {}
        print()
        print("  Experiment B reading:")
        if b.get("hit_rate_pct") is None or i.get("hit_rate_pct") is None:
            print("    benchmark report missing; cannot compare cache effectiveness")
            return
        dh = i["hit_rate_pct"] - b["hit_rate_pct"]
        bh, ih = b.get("cached_host_tokens", 0), i.get("cached_host_tokens", 0)
        print(f"    cache hit rate      : bf16 {b['hit_rate_pct']:.1f}%  ->  "
              f"int8 {i['hit_rate_pct']:.1f}%   ({dh:+.1f} points)")
        print(f"    tokens served by L2 : bf16 {bh:,.0f}  ->  int8 {ih:,.0f}"
              + (f"   ({(ih / bh - 1) * 100:+.0f}%)" if bh else ""))
        for key, label in (("ttft_mean_ms", "TTFT mean"),
                           ("ttft_p99_ms", "TTFT p99"),
                           ("out_tok_per_s", "output tok/s")):
            bv, iv = b.get(key), i.get(key)
            if bv and iv:
                print(f"    {label:<19} : bf16 {bv:,.1f}  ->  int8 {iv:,.1f}"
                      f"   ({(iv / bv - 1) * 100:+.1f}%)")
        if dh > 1.0:
            print(f"    -> int8 serves MORE of the workload from L2: the capacity")
            print(f"       gain converts into fewer prefills ({dh:+.1f} hit points).")
        elif dh < -1.0:
            print(f"    -> int8 served LESS from L2 ({dh:+.1f} points): investigate")
        else:
            print(f"    -> hit rates are within noise ({dh:+.1f} points); the")
            print(f"       workload may fit both L2s or neither. Re-check the")
            print(f"       workload budget printed by run_experiment.py.")
        print()
        print("    NOTE: single runs. Confirm any difference with repeats before")
        print("    quoting it; the pod-side spread is not characterised yet.")


def print_markdown(runs: list[Run]) -> None:
    print("\n\n### Markdown (paste into the README)\n")
    print("| config | workload | L2 tokens | L2 B/token | backup B | "
          "restore B | hit rate | prefill tokens |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for run in runs:
        per_token = run.derived.get("measured_backup_bytes_per_token")
        l2_tokens = run.delta.get("sglang:hicache_host_total_tokens", 0.0)
        backup = run.delta.get("sglang:hicache_backup_bytes_total", 0.0)
        restore = run.delta.get("sglang:load_back_bytes_total", 0.0)
        hit = run.delta.get("sglang:cache_hit_rate", 0.0)
        prefill = run.delta.get("sglang:prompt_tokens_total", 0.0)
        per_token_cell = f"{per_token:,.0f}" if per_token else "-"
        print(
            f"| {run.config} | {run.workload} | {l2_tokens:,.0f} | "
            f"{per_token_cell} | {backup:,.0f} | {restore:,.0f} | "
            f"{hit:.4f} | {prefill:,.0f} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory of exp_*.json")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    if not args.results.is_dir():
        print(f"ERROR: {args.results} is not a directory")
        return 2

    runs = load_runs(args.results)
    if not runs:
        print(f"no exp_*.json found in {args.results}")
        print("run scripts/run_experiment.py on the pod first, then copy results/ back")
        return 1

    print(f"HiQCache analysis: {len(runs)} run(s) from {args.results}")
    for run in runs:
        print_run(run)

    sweeps = group(runs)
    for (tag, workload), configs in sorted(sweeps.items()):
        print_comparison(tag, workload, configs)

    failed_checks = [
        r.label for r in runs if r.check and not r.check.get("pass")
    ]
    if failed_checks:
        print("\nWARNING: encoded bytes/token check failed for:")
        for label in failed_checks:
            print(f"  - {label}")

    if args.markdown:
        print_markdown(runs)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "runs": [
                        {
                            "path": str(r.path),
                            "config": r.config,
                            "tag": r.tag,
                            "workload": r.workload,
                            "ok": r.ok,
                            "sglang_sha": r.sglang_sha,
                            "host_size_gb": r.host_size_gb,
                            "delta": r.delta,
                            "derived": r.derived,
                            "capacity_budget": r.capacity_budget,
                            "check": r.check,
                        }
                        for r in runs
                    ]
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
