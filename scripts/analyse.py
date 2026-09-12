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
            )
        )
    return runs


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

    row("L2 capacity (tokens)",
        lambda r: r.delta.get("sglang:hicache_host_total_tokens"))
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
    row("prefix cache hit rate",
        lambda r: r.delta.get("sglang:cache_hit_rate"), "ratio")
    row("prefill tokens (recompute)",
        lambda r: r.delta.get("sglang:prompt_tokens_total"))
    row("L2 evictions",
        lambda r: r.delta.get("sglang:hicache_dropped_tokens_total"))

    # Effective bandwidth, shown separately because it is a derived ratio rather
    # than a counter and needs different formatting.
    print(f"  {'effective bandwidth':<30}" + "".join(f"{c:>15}" for c in order))
    for label, key in (("  backup GB/s", "backup_GB_per_s"),
                       ("  restore GB/s", "load_GB_per_s")):
        cells = []
        for name in order:
            run = configs.get(name)
            value = run.derived.get(key) if run else None
            cells.append(num(value, fmt="{:,.3f}"))
        print(f"  {label:<30}" + "".join(f"{c:>15}" for c in cells))

    # The Experiment B verdict, stated explicitly.
    if "baseline" in configs and "bf16" in configs and "int8" in configs:
        def prefill(name):
            run = configs.get(name)
            return run.delta.get("sglang:prompt_tokens_total", 0.0) if run else 0.0

        base, bf16, int8 = prefill("baseline"), prefill("bf16"), prefill("int8")
        print()
        print(f"  Experiment B reading:")
        print(f"    baseline prefill tokens : {base:,.0f}")
        print(f"    bf16     prefill tokens : {bf16:,.0f}")
        print(f"    int8     prefill tokens : {int8:,.0f}")
        if bf16 > 0 and int8 > 0:
            if int8 < bf16 * 0.9:
                print("    -> int8 avoids recomputation that bf16 cannot: "
                      "capacity win confirmed")
            elif int8 > bf16 * 1.1:
                print("    -> int8 recomputes MORE than bf16: investigate "
                      "(workload may not exceed the bf16 L2)")
            else:
                print("    -> comparable recomputation: the workload probably "
                      "fits both L2s; raise --gsp-question-len or --num-groups")


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
