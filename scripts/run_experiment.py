"""Phase 12/13 experiment driver: run one configuration, capture labelled metrics.

Design goal: make the pod run *one command per configuration*, with the
phase boundary (idsle vs active) handled correctly. SGLang's HiCache counters are
cumulative process-lifetime values, so comparing two configurations naively
includes server startup and warmup. This driver snapshots ``/metrics`` around the
measured phase and reports deltas, while keeping absolute totals for context.

Configurations (PROJECT.md Phase 12):

    baseline   HiCache disabled entirely -- pure prefill recomputation
    bf16       standard SGLang HiCache, BF16 L2
    int8       HiQCache, compressed INT8 L2

Usage::

    for cfg in int8 bf16 baseline; do
      python scripts/run_experiment.py --config $cfg --tag exp-b-8gb \\
          --workload reusable-prefixes
    done

    # Experiment A -- codec cost at equal logical capacity.
    python scripts/run_experiment.py --config int8 --tag exp-a --workload small

Results go to ``results/exp_<config>_<tag>.json``.

Experiment B only says anything if the aggregate reusable prefixes land BETWEEN
the two L2 capacities. Below the BF16 capacity both configs hold everything;
above the INT8 capacity both evict. Either way the comparison is empty, so the
budget is checked and reported before a server starts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hiqcache.layout import V1_LAYOUT  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration definitions
# ---------------------------------------------------------------------------

HF_CACHE_HINT = "set HF_HOME / HF_HUB_CACHE to a persistent volume before running"


@dataclass
class Config:
    name: str
    label: str
    env: dict
    server_args: list
    host_size_gb: float = 8.0


def _row_bytes(config_name: str) -> int:
    """Encoded row width for a config: 1152 for int8, 2048 for bf16.

    Derived from the same constants the pools use, so the equal-tokens sizing
    cannot drift from the codec layout.
    """
    if config_name == "int8":
        return V1_LAYOUT.row_bytes
    # BF16: head_num * head_dim * itemsize, from the Qwen3-8B TP=1 geometry the
    # INT8 format is defined against.
    return 2 * V1_LAYOUT.payload_bytes


def build_configs(
    model: str,
    host_size_gb: float,
    tp: int,
    page_size: int,
    max_total_tokens: int | None = None,
    sizing: str = "fixed-size",
    host_ratio: float = 2.0,
    target_l2_tokens: int | None = None,
    layer_num: int = 36,
) -> dict:
    """The three Phase 12 configurations, sharing every non-codec setting."""
    common = [
        "--model-path", model,
        "--tp-size", str(tp),
        "--page-size", str(page_size),
        "--trust-remote-code",
        "--enable-metrics",
    ]
    # An L1 cap is REQUIRED for these experiments, not optional. The default
    # device pool on a 48 GB card holds millions of tokens, so a workload that
    # fits entirely in L1 never reaches L2 at all and the experiment measures
    # nothing. The cap has to be small enough that the reusable working set
    # overflows L1 into L2 -- that overflow is the thing being measured.
    if max_total_tokens:
        common += ["--max-total-tokens", str(max_total_tokens)]
    hicache_common = [
        "--enable-hierarchical-cache",
        "--hicache-io-backend", "kernel",
        "--hicache-mem-layout", "layer_first",
        # write_back is REQUIRED for L2 to be useful. Under write_through (and
        # write_through_selective) SGLang sets is_write_back=False, and
        # evict_device_leaf then DELETES an unbacked device leaf from the tree
        # instead of demoting it to a host-only node. L2 fills up with KV that
        # no tree node references, so host_hit_length stays 0, load-back never
        # fires, and every revisit re-prefills -- which is what the smoke test
        # measured: 21.8 GB backed up, 0 tokens ever restored.
        "--hicache-write-policy", "write_back",
    ]
    if sizing == "equal-tokens":
        # Experiment A. Equal LOGICAL capacity: both tiers hold the same number
        # of L2 tokens, so any remaining difference is codec cost rather than the
        # capacity the codec buys. INT8 reaching that capacity in ~56% of the
        # bytes is the end-to-end claim.
        #
        # Sized with --hicache-ratio, NOT --hicache-size. Two independent
        # reasons, both load-bearing:
        #
        #   * hicache_size is declared `int` (arg_groups/fields/memory.py:
        #     `hicache_size: A[int, ...]`, "in gigabytes"). Equal token capacity
        #     needs sub-GB precision -- BF16 needs 8.000078 GB -- so argparse
        #     rejects the value and the server exits with code 2 before it ever
        #     loads the model. That is not a crash to debug; it is an
        #     unrepresentable argument.
        #   * hicache_ratio is a *token* ratio (pool_host/base.py:
        #     `self.size = int(device_capacity * host_to_device_ratio)`), then
        #     rounded up a page (`page_num = size // page_size + 1`). With
        #     --page-size 1 that lands on `int(device*ratio) + 1` tokens, hence
        #     the -0.5 that makes the floor one below the target.
        #
        # A device pool pinned by --max-total-tokens is what makes the ratio
        # exact, and the achieved capacity is checked against the pool's own
        # gauge after startup rather than trusted.
        if not target_l2_tokens:
            raise ValueError("--sizing equal-tokens requires --target-l2-tokens")
        if not max_total_tokens:
            raise ValueError(
                "--sizing equal-tokens requires --max-total-tokens: the host pool "
                "is sized as a ratio of the device pool, so the device pool has "
                "to be pinned to a known token count."
            )
    elif sizing == "ratio":
        hicache_common += ["--hicache-ratio", str(host_ratio)]
    else:
        hicache_common += ["--hicache-size", f"{host_size_gb:g}"]

    def hicache_args(config_name: str) -> list:
        """Per-config HiCache args, because the host pool is per-config."""
        args = list(hicache_common)
        if sizing == "equal-tokens":
            ratio = (target_l2_tokens - 0.5) / max_total_tokens
            args += ["--hicache-ratio", f"{ratio:.12f}"]
        return args
    return {
        # A: no reusable L2 at all -> every L1 miss is a prefill recomputation.
        "baseline": Config(
            name="baseline",
            label="A: HiCache disabled (prefill recompute)",
            env={},
            server_args=list(common),
            host_size_gb=0.0,
        ),
        # B: standard BF16 L2.
        "bf16": Config(
            name="bf16",
            label="B: standard HiCache, BF16 L2",
            env={"SGLANG_EXPERIMENTAL_HICACHE_INT8": "0"},
            server_args=common + hicache_args("bf16"),
            host_size_gb=host_size_gb,
        ),
        # C: HiQCache, compressed INT8 L2.
        "int8": Config(
            name="int8",
            label="C: HiQCache, compressed INT8 L2",
            env={"SGLANG_EXPERIMENTAL_HICACHE_INT8": "1"},
            server_args=common + hicache_args("int8"),
            host_size_gb=host_size_gb,
        ),
    }


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


@dataclass
class Workload:
    name: str
    bench_args: list
    description: str


def build_workloads(args) -> dict:
    """The Phase 13 workloads."""
    return {
        # Baseline sanity: modest reusable prefixes that fit either L2.
        "small": Workload(
            name="small",
            description="reusable prefixes that fit a BF16 L2 (cache-hit TTFT)",
            bench_args=[
                "--dataset-name", "generated-shared-prefix",
                "--gsp-num-groups", "16",
                "--gsp-prompts-per-group", "8",
                "--gsp-system-prompt-len", "1024",
                "--gsp-question-len", "256",
                "--gsp-output-len", "64",
                "--num-prompts", "128",
                "--max-concurrency", "16",
                "--cache-report",
            ],
        ),
        # Experiment B: aggregate reusable prefixes that do NOT fit a BF16 L2
        # but DO fit the INT8 L2. The token budget is computed from the codec
        # layout so this cannot drift from the measured size_per_token.
        "reusable-prefixes": Workload(
            name="reusable-prefixes",
            description=(
                "aggregate reusable prefixes sized between the BF16 and INT8 "
                "L2 capacities (the real end-to-end value proposition)"
            ),
            bench_args=[
                "--dataset-name", "generated-shared-prefix",
                # system-prompt-len is the REUSABLE PER-GROUP PREFIX: the whole
                # point of this workload. num_groups of them at that length is
                # the aggregate working set the budget check sizes.
                "--gsp-num-groups", str(args.num_groups),
                "--gsp-prompts-per-group", str(args.prompts_per_group),
                "--gsp-system-prompt-len", str(args.gsp_prefix_len),
                "--gsp-question-len", str(args.gsp_question_len),
                "--gsp-output-len", "32",
                "--num-prompts", str(args.num_groups * args.prompts_per_group),
                "--max-concurrency", str(args.max_concurrency),
                "--cache-report",
            ],
        ),
    }


def capacity_budget(host_size_gb: float, layer_num: int) -> dict:
    """Token capacities for this host budget, from the codec layout.

    Mirrors ``HostKVCache.__init__``: ``size = int(host_size * 1e9 // size_per_token)``
    then ``page_num = size // page_size + 1``. page_size is 1 here.
    """
    baseline_per_token = 147_456
    encoded_per_token = V1_LAYOUT.bytes_per_token_all_layers(layer_num)
    out = {}
    for name, per_token in (("bf16", baseline_per_token), ("int8", encoded_per_token)):
        raw = int(host_size_gb * 1e9 // per_token)
        out[name] = {"size_per_token": per_token, "token_capacity": raw + 1}
    out["gain"] = out["int8"]["token_capacity"] / out["bf16"]["token_capacity"]
    return out


def workload_budget(args, budget: dict) -> dict:
    """Estimate the reusable working set and say whether it discriminates.

    Experiment B only says anything if the aggregate reusable prefixes sit
    BETWEEN the two L2 capacities:

        below bf16 capacity  -> both configs hold it, no difference, no result
        above int8 capacity  -> both configs evict, no difference, no result
        in between           -> bf16 evicts and recomputes, int8 hits

    This is printed before the run because getting it wrong wastes the run, and
    the arithmetic is cheap: the shared-prefix dataset gives each group a
    distinct prefix of ``gsp_question_len`` tokens, reused by
    ``prompts_per_group`` prompts.
    """
    prefix_tokens = args.num_groups * args.gsp_prefix_len
    l1 = args.max_total_tokens or 0
    bf16_cap = budget["bf16"]["token_capacity"] if budget else 0
    int8_cap = budget["int8"]["token_capacity"] if budget else 0

    if not budget:
        verdict = "no fixed host budget (--hicache-ratio sizing); cannot predict"
    elif prefix_tokens < bf16_cap:
        verdict = (
            f"TOO SMALL: {prefix_tokens:,} reusable tokens fit the BF16 L2 "
            f"({bf16_cap:,}), so both configs hold everything and the comparison "
            f"is uninformative. Raise --num-groups or --gsp-question-len."
        )
    elif prefix_tokens > int8_cap:
        verdict = (
            f"TOO LARGE: {prefix_tokens:,} reusable tokens exceed even the INT8 L2 "
            f"({int8_cap:,}), so both configs evict. Lower --num-groups."
        )
    else:
        verdict = (
            f"DISCRIMINATING: {prefix_tokens:,} reusable tokens exceed the BF16 L2 "
            f"({bf16_cap:,}) but fit the INT8 L2 ({int8_cap:,}). BF16 must evict "
            f"and recompute where INT8 can hit."
        )
    if l1 and l1 >= prefix_tokens:
        verdict += (
            f" WARNING: L1 cap {l1:,} >= working set, so nothing reaches L2 at "
            f"all; lower --max-total-tokens."
        )
    return {
        "reusable_prefix_tokens": prefix_tokens,
        "l1_capacity_cap": l1,
        "bf16_l2_capacity": bf16_cap,
        "int8_l2_capacity": int8_cap,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def http_get(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def wait_for_server(base_url: str, timeout_s: float, proc: subprocess.Popen) -> float:
    """Block until /health responds. Returns seconds waited."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            # Print the LOG, not just the command. The previous version pointed
            # at proc.args, which is the argv the caller already knows; the
            # reason the server died is only ever in its log (an argparse
            # rejection, an OOM, a busy port), so quote it here instead of
            # making the operator go and find it.
            log = getattr(proc, "_hiqcache_log_path", None)
            tail = _tail(log, 25) if log else "(log path unknown)"
            raise RuntimeError(
                f"server exited during startup with code {proc.returncode}.\n"
                f"--- last 25 lines of {log} ---\n{tail}\n"
                f"--- end of log ---\n"
                f"command: {' '.join(proc.args)}"
            )
        try:
            http_get(f"{base_url}/health", timeout=5)
            return timeout_s - (deadline - time.time())
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    raise TimeoutError(f"server not healthy after {timeout_s:.0f}s")


METRIC_RE = re.compile(r"^(sglang:[a-z_0-9]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$")


def scrape_metrics(base_url: str) -> dict:
    """Parse the Prometheus endpoint into ``{metric_name: summed_value}``.

    Sums across label sets, which is what we want for totals; per-label detail is
    preserved in the raw text saved alongside the JSON.
    """
    text = http_get(f"{base_url}/metrics", timeout=30)
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        match = METRIC_RE.match(line.strip())
        if not match:
            continue
        name, _, value = match.groups()
        try:
            totals[name] = totals.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return totals


#: Metrics that matter, with the PROJECT.md phase each answers.
TRACKED = {
    "sglang:hicache_backup_bytes_total": "Phase 14 D2H: total backup bytes",
    "sglang:hicache_backup_tokens_total": "Phase 14 D2H: tokens backed up",
    "sglang:hicache_backup_duration_seconds": "Phase 14 D2H: total backup time",
    "sglang:load_back_bytes_total": "Phase 14 H2D: total restore bytes",
    "sglang:load_back_tokens_total": "Phase 14 H2D: tokens restored",
    "sglang:load_back_duration_seconds": "Phase 14 H2D: total restore time",
    "sglang:hicache_host_used_tokens": "Phase 14 storage: L2 tokens in use",
    "sglang:hicache_host_total_tokens": "Phase 14 storage: L2 capacity",
    "sglang:hicache_dropped_tokens_total": "Phase 14 storage: L2 evictions",
    "sglang:cache_hit_rate": "Phase 14 serving: prefix cache hit rate",
    "sglang:prompt_tokens_total": "Phase 14 serving: prefill tokens",
    "sglang:generation_tokens_total": "Phase 14 serving: decode tokens",
}


def derive(metrics: dict) -> dict:
    """Turn raw counters into the derived figures PROJECT.md asks for."""
    out = {}
    backup_bytes = metrics.get("sglang:hicache_backup_bytes_total", 0.0)
    backup_tokens = metrics.get("sglang:hicache_backup_tokens_total", 0.0)
    backup_time = metrics.get("sglang:hicache_backup_duration_seconds", 0.0)
    load_bytes = metrics.get("sglang:load_back_bytes_total", 0.0)
    load_tokens = metrics.get("sglang:load_back_tokens_total", 0.0)
    load_time = metrics.get("sglang:load_back_duration_seconds", 0.0)

    if backup_tokens > 0:
        out["measured_backup_bytes_per_token"] = backup_bytes / backup_tokens
    if load_tokens > 0:
        out["measured_load_bytes_per_token"] = load_bytes / load_tokens
    if backup_time > 0:
        out["backup_GB_per_s"] = backup_bytes / backup_time / 1e9
    if load_time > 0:
        out["load_GB_per_s"] = load_bytes / load_time / 1e9
    used = metrics.get("sglang:hicache_host_used_tokens", 0.0)
    total = metrics.get("sglang:hicache_host_total_tokens", 0.0)
    if total > 0:
        out["l2_utilisation"] = used / total
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=["baseline", "bf16", "int8"])
    parser.add_argument("--tag", required=True, help="label for this run, e.g. exp-b-8gb")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--host-size", type=float, default=8.0, help="decimal GB")
    parser.add_argument(
        "--sizing",
        choices=["fixed-size", "ratio", "equal-tokens"],
        default="fixed-size",
        help=(
            "fixed-size: --hicache-size, the SAME PHYSICAL budget for every "
            "config (Experiment B -- what a fixed amount of host RAM buys). "
            "ratio: --hicache-ratio, the SAME LOGICAL token capacity for every "
            "config (Experiment A -- isolates the codec's cost from its "
            "capacity benefit, since both tiers then cache the same tokens)."
        ),
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=16384,
        help=(
            "Cap the L1 (device) KV pool. Required for these experiments: the "
            "default L1 on a 48 GB card holds millions of tokens, so a workload "
            "that fits in L1 never reaches L2 and nothing is measured. The "
            "reusable working set must overflow this."
        ),
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--host-ratio", type=float, default=2.0,
                        help="host:device capacity ratio when --sizing ratio")
    parser.add_argument(
        "--target-l2-tokens",
        type=int,
        default=None,
        help=(
            "Required by --sizing equal-tokens: the L2 token capacity both "
            "configs should end up with. Each config's host bytes are computed "
            "from its own size_per_token, so BF16 and INT8 cache the same number "
            "of tokens -- which is what isolates codec cost from capacity gain. "
            "Use the BF16 capacity for the budget in question (54,254 at 8 GB)."
        ),
    )
    parser.add_argument("--layer-num", type=int, default=36)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--workload", default="small", choices=["small", "reusable-prefixes"])
    parser.add_argument(
        "--num-groups",
        type=int,
        default=32,
        help=(
            "Number of distinct reusable prefixes. 32 x 2048 = 65,536 tokens, "
            "which sits between the 8 GB BF16 (54,254) and INT8 (96,451) L2 "
            "capacities -- the range where the comparison means something."
        ),
    )
    parser.add_argument("--prompts-per-group", type=int, default=4)
    parser.add_argument(
        "--gsp-prefix-len",
        type=int,
        default=2048,
        help=(
            "Reusable per-group prefix length (--gsp-system-prompt-len). The "
            "aggregate working set is num_groups * this, and Experiment B only "
            "discriminates when that lands between the BF16 and INT8 L2 "
            "capacities -- checked and reported before the run."
        ),
    )
    parser.add_argument(
        "--gsp-question-len",
        type=int,
        default=64,
        help="Per-prompt question length, appended after the shared prefix.",
    )
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--flush-cache", action="store_true", default=True)
    parser.add_argument("--bench-python", default=sys.executable)
    parser.add_argument("--sglang-root", default=str(REPO_ROOT.parent / "sglang"))
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    args = parser.parse_args()

    sglang_root = Path(args.sglang_root).expanduser().resolve()
    config = build_configs(
        args.model, args.host_size, args.tp, args.page_size,
        max_total_tokens=args.max_total_tokens,
        sizing=args.sizing,
        host_ratio=args.host_ratio,
        target_l2_tokens=args.target_l2_tokens,
        layer_num=args.layer_num,
    )[args.config]
    workload = build_workloads(args)[args.workload]
    budget = capacity_budget(args.host_size, args.layer_num) if args.host_size > 0 else {}

    base_url = f"http://127.0.0.1:{args.port}"
    server_cmd = [sys.executable, "-m", "sglang.launch_server", *config.server_args,
                  "--port", str(args.port)]
    bench_cmd = [args.bench_python, "-m", "sglang.benchmark.serving",
                 "--backend", "sglang", "--base-url", base_url,
                 "--model", args.model, *workload.bench_args]

    if args.dry_run:
        print(f"config   : {config.label}")
        print(f"workload : {workload.name} -- {workload.description}")
        print(f"env      : {config.env}")
        print(f"server   : cd {sglang_root} && {' '.join(server_cmd)}")
        print(f"bench    : cd {sglang_root} && {' '.join(bench_cmd)}")
        if budget:
            print(f"budget   : bf16={budget['bf16']['token_capacity']:,} tokens, "
                  f"int8={budget['int8']['token_capacity']:,} tokens "
                  f"({budget['gain']:.4f}x)")
            wb = workload_budget(args, budget)
            print(f"workload : {wb['reusable_prefix_tokens']:,} reusable prefix "
                  f"tokens | L1 cap {wb['l1_capacity_cap']:,}")
            if args.sizing == "equal-tokens":
                print(f"verdict  : equal tokens by construction "
                      f"({args.target_l2_tokens:,} in both tiers); the comparison "
                      f"is the bytes each tier needs, not the hit rate")
            else:
                print(f"verdict  : {wb['verdict']}")
        return 0

    if not (sglang_root / "python" / "sglang").is_dir():
        print(f"ERROR: SGLang fork not found at {sglang_root}", file=sys.stderr)
        return 2

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(config.env)
    env["PYTHONPATH"] = f"{sglang_root / 'python'}:{env.get('PYTHONPATH', '')}"
    # Codec phase timing for Phase 17. The INT8 pool records CUDA events around
    # encode / decode / each mover call and writes totals on teardown, so this
    # file appears during shutdown rather than during the run. It decomposes the
    # L2 path: if encode+decode is negligible next to the D2H copy, fused Triton
    # kernels are not worth writing.
    timing_path = out_dir / f"codec_timing_{config.name}_{args.tag}.json"
    if config.name == "int8":
        env["SGLANG_HICACHE_INT8_TIMING"] = "1"
        env["SGLANG_HICACHE_INT8_TIMING_PATH"] = str(timing_path)
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    stem = f"exp_{config.name}_{args.tag}"
    server_log = out_dir / f"{stem}.server.log"

    print(f"=== {config.label}")
    print(f"=== workload: {workload.name} -- {workload.description}")
    if budget and args.sizing == "equal-tokens":
        # The fixed-size budget below describes Experiment B. Printing it here
        # would mislabel what this run actually does: under equal-tokens BOTH
        # tiers are pinned to the same token count, and INT8 gets by on ~56% of
        # the bytes.
        print(f"=== L2 capacity target: {args.target_l2_tokens:,} tokens in BOTH "
              f"tiers (equal logical capacity, different bytes)")
    elif budget:
        print(f"=== L2 budget @ {args.host_size:g} GB: "
              f"bf16 {budget['bf16']['token_capacity']:,} tokens | "
              f"int8 {budget['int8']['token_capacity']:,} tokens "
              f"({budget['gain']:.4f}x)")
    print(f"=== server log: {server_log}")
    if budget:
        wb = workload_budget(args, budget)
        print(f"=== workload budget: {wb['reusable_prefix_tokens']:,} reusable "
              f"prefix tokens | L1 cap {wb['l1_capacity_cap']:,}")
        if args.sizing == "equal-tokens":
            print("=== both tiers hold the same capacity by construction, so equal "
                  "hit rates are the EXPECTED result; the measured difference is "
                  "the memory each tier needs to hold them")
        else:
            print(f"=== {wb['verdict']}")

    with open(server_log, "w") as log:
        proc = subprocess.Popen(
            server_cmd, cwd=sglang_root, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    # So wait_for_server() can report log growth while the first-run model
    # download is in progress; without it the wait looks like a hang.
    proc._hiqcache_log_path = server_log
    result: dict = {
        "config": config.name,
        "config_label": config.label,
        "tag": args.tag,
        "workload": asdict(workload),
        "host_size_gb": args.host_size,
        "tp": args.tp,
        "page_size": args.page_size,
        "model": args.model,
        "env": config.env,
        "server_args": server_cmd,
        "sglang_sha": _git_sha(sglang_root),
        "capacity_budget": budget,
        "ok": False,
    }

    try:
        waited = wait_for_server(base_url, args.startup_timeout, proc)
        print(f"=== server healthy after {waited:.0f}s")
        result["startup_seconds"] = waited

        before = scrape_metrics(base_url)
        result["metrics_absolute_before"] = {
            k: before.get(k, 0.0) for k in TRACKED
        }

        # Equal-tokens sizing is pure arithmetic (device_capacity * ratio, then
        # rounded up to a page). Off by one token and the experiment quietly
        # measures a capacity other than the one it claims, so check it against
        # the pool's own gauge instead of trusting the formula.
        if args.sizing == "equal-tokens":
            achieved = int(before.get("sglang:hicache_host_total_tokens", 0.0))
            result["target_l2_tokens"] = args.target_l2_tokens
            result["achieved_l2_tokens"] = achieved
            if achieved != int(args.target_l2_tokens):
                raise RuntimeError(
                    f"equal-tokens sizing missed: asked for "
                    f"{args.target_l2_tokens:,} L2 tokens, the pool reports "
                    f"{achieved:,}. The ratio assumed a device pool of "
                    f"{args.max_total_tokens:,} tokens; check --max-total-tokens."
                )
            print(f"=== L2 capacity verified against the pool's gauge: "
                  f"{achieved:,} tokens")

        # Flush the prefix cache so the measured phase starts cold; otherwise a
        # previous run's L1/L2 state leaks in and the hit rate is meaningless.
        if args.flush_cache:
            try:
                http_get(f"{base_url}/flush_cache", timeout=120)
                print("=== flushed prefix cache")
                result["flushed_cache"] = True
            except Exception as exc:  # noqa: BLE001
                print(f"=== WARNING: flush_cache failed ({exc}); results may be warm")
                result["flushed_cache"] = False

        print(f"=== running benchmark")
        t0 = time.time()
        bench = subprocess.run(
            bench_cmd, cwd=sglang_root, env=env, capture_output=True, text=True
        )
        result["benchmark_seconds"] = time.time() - t0
        result["benchmark_stdout"] = bench.stdout
        result["benchmark_stderr"] = bench.stderr[-8000:]
        result["benchmark_returncode"] = bench.returncode
        print(bench.stdout[-4000:] if bench.stdout else "(no benchmark stdout)")
        if bench.returncode != 0:
            print(f"=== benchmark FAILED (rc={bench.returncode})", file=sys.stderr)
            print(bench.stderr[-4000:], file=sys.stderr)

        after = scrape_metrics(base_url)
        result["metrics_absolute_after"] = {k: after.get(k, 0.0) for k in TRACKED}
        delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in TRACKED}
        result["metrics_delta"] = delta
        result["derived"] = derive(delta)
        result["ok"] = bench.returncode == 0

        print("\n=== measured deltas (this workload only)")
        for key, value in delta.items():
            print(f"  {key:<48} {value:>18.4f}   {TRACKED[key]}")
        print("\n=== derived")
        for key, value in result["derived"].items():
            print(f"  {key:<48} {value:>18.4f}")

        # The headline check: does the measured encoded bytes/token match the
        # layout? If not, the compression is not actually being applied.
        per_token = result["derived"].get("measured_backup_bytes_per_token")
        if config.name == "int8" and per_token:
            expected = V1_LAYOUT.bytes_per_token_all_layers(args.layer_num)
            err = abs(per_token - expected) / expected
            result["encoded_bytes_per_token_check"] = {
                "expected": expected,
                "measured": per_token,
                "relative_error": err,
                "pass": err < 0.02,
            }
            verdict = "PASS" if err < 0.02 else "FAIL"
            print(f"\n=== encoded bytes/token: measured {per_token:,.0f} vs layout "
                  f"{expected:,} ({err:.2%}) -> {verdict}")
        elif config.name == "bf16" and per_token:
            expected = 147_456
            print(f"\n=== baseline bytes/token: measured {per_token:,.0f} vs "
                  f"expected {expected:,}")

    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"=== ERROR: {result['error']}", file=sys.stderr)
    finally:
        _terminate(proc)
        tail = _tail(server_log, 40)
        result["server_log_tail"] = tail

    if config.name == "int8":
        try:
            result["codec_timing"] = json.loads(timing_path.read_text())
            phases = result["codec_timing"].get("phases", {})
            if phases:
                print("\n=== codec phase timing (GPU ms, from the pool's teardown)")
                for phase, stats in phases.items():
                    print(f"  {phase:<12} total {stats['total_ms']:>12,.1f} ms  "
                          f"calls {stats['calls']:>8,}  "
                          f"mean {stats['mean_ms']:>8,.4f} ms")
        except (OSError, json.JSONDecodeError):
            print(f"\n=== codec timing file missing ({timing_path.name}); the pool "
                  f"writes it during teardown, which may have been killed")

    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n=== wrote {path}")
    return 0 if result.get("ok") else 1


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    # SIGTERM was not enough -- unpinning a multi-GB host pool can stall teardown
    # for tens of seconds. Escalate, but WAIT for the kill to actually land:
    # returning immediately leaves the process tree holding the port, the GPU and
    # up to --hicache-size GB of pinned host memory, so the NEXT config's server
    # fails at startup for a reason that looks unrelated to this one.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        print(
            f"=== WARNING: process group {pgid} survived SIGKILL for 60s; "
            f"the next server may fail to bind its port",
            file=sys.stderr,
        )


def _git_sha(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _tail(path: Path, n: int) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


if __name__ == "__main__":
    raise SystemExit(main())
