"""Phase 11 smoke test: prove the compressed L2 path actually executes.

"Server didn't crash" is not evidence. This script drives a real server through a
cold-populate-then-reuse cycle and asserts on measured counters that

  1. a request populates L1,
  2. a cache entry is evicted to L2,
  3. L2 holds *compressed* bytes at the encoded rate,
  4. a later request restores from L2,
  5. generation still completes.

The decisive check is (3): SGLang exposes ``sglang:hicache_backup_bytes_total``
and ``hicache_backup_tokens_total``, so the measured encoded bytes per token can
be compared against the codec layout. The BF16 pool reports 147,456 B/token; the
INT8 pool must report 82,944. Anything in between means the record is being
padded or the wrong pool was selected.

The expectation is derived from ``--config``: ``int8`` must report the encoded
82,944 B/token in L2, ``bf16`` must report the 147,456 baseline. Run both, as the
bf16 run is the control that proves the measurement is sensitive.

Usage::

    # from the hiqcache checkout, using the python that has SGLang installed
    python scripts/smoke_test.py --config int8 --host-size 2
    python scripts/smoke_test.py --config bf16 --host-size 2

Notes:
  * First run downloads Qwen3-8B (~16 GB). Set HF_HUB_ENABLE_HF_TRANSFER=1 and a
    persistent HF_HOME if you have a volume mounted.
  * --sglang-root defaults to the sibling ``sglang/`` checkout; override it if
    the fork lives elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hiqcache.layout import V1_LAYOUT  # noqa: E402
from run_experiment import (  # noqa: E402
    build_configs,
    http_get,
    scrape_metrics,
    wait_for_server,
)

LAYER_NUM = 36


class Step:
    def __init__(self, name: str):
        self.name = name
        self.ok: bool | None = None
        self.detail = ""

    def record(self, ok: bool, detail: str = "") -> bool:
        self.ok = bool(ok)
        self.detail = detail
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {self.name}" + (f"\n         {detail}" if detail else ""))
        return self.ok


def post_generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int = 16,
    timeout: float = 300,
    label: str = "",
):
    """One deterministic generation; returns the response JSON.

    On an HTTP error the server's response body is included in the raised
    exception. SGLang explains rejected requests there -- for example an input
    longer than the KV pool can budget -- and without it the caller only sees
    "HTTP Error 400: Bad Request", which says nothing about which limit was hit.
    """
    body = json.dumps(
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode()[:600]
        except Exception:  # noqa: BLE001
            pass
        prefix = f"{label}: " if label else ""
        raise RuntimeError(
            f"{prefix}HTTP {exc.code} from /generate "
            f"(prompt {len(prompt)} chars). Server said: {detail or '(no body)'}"
        ) from exc


_PREFILL_RE = re.compile(
    r"Prefill batch.*?#new-seq:\s*(?P<seqs>\d+).*?"
    r"#new-token:\s*(?P<newtok>\d+).*?"
    r"#cached-token:\s*(?P<cached>\d+)"
)


def prefill_history(log_path: Path) -> list[dict]:
    """Per-request prefill stats scraped from the server log.

    ``#cached-token`` is how many prefix tokens the request matched, and
    ``#new-token`` is how many it had to prefill. Together they distinguish the
    two failure modes for load-back:

      * cached ~= whole prompt, backup/load counters flat -> the prefix never
        left L1, so nothing was asked of L2;
      * cached ~= 0, prompt long -> the prefix was gone from L1 AND was not
        restored from L2.

    Without this the two look identical from the metrics alone, which is exactly
    the ambiguity that made the previous failure hard to read.
    """
    if not log_path.is_file():
        return []
    out = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = _PREFILL_RE.search(line)
        if m:
            out.append(
                {
                    "seqs": int(m.group("seqs")),
                    "new_tokens": int(m.group("newtok")),
                    "cached_tokens": int(m.group("cached")),
                }
            )
    return out


def build_prompt(prefix: str, question: str) -> str:
    return f"{prefix}\n\n{question}\n\nAnswer:"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=["bf16", "int8"])
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--host-size", type=float, default=2.0)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=8192,
        help=(
            "Cap the L1 (device) KV pool. Load-back only fires when a prefix is "
            "in L2 but NOT in L1, so the cap has to leave little headroom above "
            "the shared prefix. The test prefix measures ~5,900 tokens, so the "
            "default 8192 leaves ~2,300 tokens -- about four filler requests -- "
            "before eviction starts, while still comfortably admitting any single "
            "request. Raise it and the chain fits, the prefix is never evicted, "
            "and there is nothing to restore."
        ),
    )
    parser.add_argument(
        "--filler-words",
        type=int,
        default=900,
        help=(
            "Words per filler request. Must be comfortably under the L1 cap: "
            "demotion under write_back needs free device memory, and a tree "
            "bursting past the cap can leave nodes evicted-but-unbacked, which "
            "makes them unreachable. ~900 words is roughly 1200 tokens against a "
            "16,384-token cap."
        ),
    )
    parser.add_argument(
        "--filler-requests",
        type=int,
        default=80,
        help=(
            "How many small filler requests to send in step 3. Volume is what "
            "overflows L1, and it has to comfortably EXCEED the cap rather than "
            "merely approach it: SGLang's default radix eviction policy is LRU, "
            "so the shared prefix is the most recently used node and survives as "
            "long as the filler fits. Keep each request well under the L1 cap or "
            "the server rejects it with HTTP 400. 80 x ~1500 words is roughly "
            "100k+ tokens against a 16,384-token cap."
        ),
    )
    parser.add_argument("--prefix-repeats", type=int, default=400,
                        help="shared prefix length in repeated sentences")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--sglang-root", default=str(REPO_ROOT.parent / "sglang"))
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    sglang_root = Path(args.sglang_root).expanduser().resolve()
    config = build_configs(args.model, args.host_size, 1, 1)[args.config]
    base_url = f"http://127.0.0.1:{args.port}"
    extra = []
    if args.max_total_tokens:
        extra += ["--max-total-tokens", str(args.max_total_tokens)]
    server_cmd = [
        sys.executable, "-m", "sglang.launch_server",
        *config.server_args, "--port", str(args.port), *extra,
    ]

    env = dict(os.environ)
    env.update(config.env)
    # The match-walk diagnostic is the only thing that explains WHY
    # host_hit_length is 0 when it should not be; it is a no-op unless set.
    env["SGLANG_HICACHE_DEBUG_MATCH"] = "1"
    env["PYTHONPATH"] = f"{sglang_root / 'python'}:{env.get('PYTHONPATH', '')}"

    expected_encoded = V1_LAYOUT.bytes_per_token_all_layers(LAYER_NUM)
    expected_baseline = 147_456
    expected = expected_encoded if args.config == "int8" else expected_baseline

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"smoke_{args.config}.server.log"

    print(f"=== Phase 11 smoke: {config.label}")
    print(f"=== expected bytes/token: {expected:,}")

    steps: list[Step] = []
    report: dict = {
        "config": config.name,
        "expected_bytes_per_token": expected,
        "sglang_root": str(sglang_root),
        "env": config.env,
    }

    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            server_cmd, cwd=sglang_root, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    # So wait_for_server() can report log growth while the first-run model
    # download is in progress; without it the wait looks like a hang.
    proc._hiqcache_log_path = log_path

    try:
        # --- 1. boots -------------------------------------------------
        step = Step("server boots and HiCache is enabled")
        steps.append(step)
        waited = wait_for_server(base_url, args.startup_timeout, proc)
        info = json.loads(http_get(f"{base_url}/get_model_info", timeout=30))
        step.record(True, f"healthy after {waited:.0f}s; model={info.get('model_path')}")
        report["startup_seconds"] = waited
        report["model_info"] = info

        # The L2 arena must exist with the encoded capacity. If the dispatch
        # picked the BF16 pool, hicache_host_total_tokens will be ~1.78x smaller.
        before = scrape_metrics(base_url)
        total_tokens = before.get("sglang:hicache_host_total_tokens", 0.0)
        l1_tokens = before.get("sglang:max_total_num_tokens", 0.0)
        report["l1_token_capacity"] = l1_tokens
        if l1_tokens:
            print(
                f"    L1 (device) capacity {l1_tokens:,.0f} tokens; "
                f"L2 (host) capacity {total_tokens:,.0f} tokens"
            )
            if l1_tokens > total_tokens:
                print(
                    "    NOTE: L1 > L2, so filler traffic must exceed L1 before "
                    "the shared prefix is evicted to L2. If the load-back step "
                    "fails, pass --max-total-tokens to cap L1."
                )
        step = Step("L2 host pool allocated with the expected token capacity")
        steps.append(step)
        if total_tokens <= 0:
            step.record(False, "sglang:hicache_host_total_tokens is 0 or missing")
        else:
            raw_capacity = int(args.host_size * 1e9 // expected) + 1
            err = abs(total_tokens - raw_capacity) / raw_capacity
            step.record(
                err < 0.02,
                f"L2 capacity {total_tokens:,.0f} tokens vs expected "
                f"{raw_capacity:,} for {expected:,} B/token ({err:.2%} off)",
            )
            report["l2_token_capacity"] = total_tokens
            report["l2_expected_capacity"] = raw_capacity

        # --- 2. populate L1 -------------------------------------------
        shared_prefix = " ".join(
            f"The quick brown fox jumps over the lazy dog number {i}."
            for i in range(args.prefix_repeats)
        )
        step = Step("a request populates L1 and generation completes")
        steps.append(step)
        first = post_generate(base_url, build_prompt(shared_prefix, "Say hello."))
        text = (first.get("text") or "").strip()
        step.record(len(text) > 0, f"generated {len(text)} chars: {text[:80]!r}")
        report["first_generation"] = {"text": text[:200]}

        # --- 3. L1 -> L2 ----------------------------------------------
        # Enough distinct traffic to evict the shared prefix out of L1 into L2.
        #
        # Each filler request must be small enough for the capped L1 to admit,
        # and there must be enough of them in total to overflow it. Sizing these
        # too large makes the server reject them with HTTP 400 (the L1 cap cannot
        # budget a request nearly as big as itself), which is a test bug, not a
        # HiCache one. Volume, not size, is what forces eviction.
        step = Step("a cache entry is evicted from L1 to L2 (backup executed)")
        steps.append(step)
        # Each filler request is a SHARED-PREFIX chain, not a set of unrelated
        # prompts. Two reasons, both learned the hard way:
        #
        #  1. Unrelated prompts share ~nothing (the pod log showed
        #     #cached-token=4), so they never form a deep chain and eviction
        #     pressure never targets the shared prefix.
        #  2. Demotion under write_back needs free device memory: the eviction
        #     path runs the D->H backup and only then calls _demote. A tree
        #     bursting far past the cap can leave nodes evicted-but-unbacked,
        #     which is a DEAD node -- `_match_prefix_helper` refuses to descend
        #     past `child.evicted and not child.backuped`, so the walk stops and
        #     host_hit_length can never be positive. Keeping each request
        #     comfortably under the cap keeps eviction in the steady state the
        #     demote path is designed for.
        filler_words = max(64, int(args.filler_words))
        shared = " ".join(str(i) for i in range(max(64, filler_words // 2)))
        filler = f"{shared} "
        filler_sent = 0
        filler_errors = 0
        for i in range(args.filler_requests):
            prompt = f"{filler}tail {i}\n\nQ: count?\n\nAnswer:"
            try:
                post_generate(base_url, prompt, label=f"filler {i}")
                filler_sent += 1
            except Exception as exc:  # noqa: BLE001
                filler_errors += 1
                if filler_errors == 1:
                    print(f"    filler request {i} failed: {type(exc).__name__}: {exc}")
        print(
            f"    sent {filler_sent}/{args.filler_requests} filler requests of "
            f"~{filler_words:,} words sharing a ~{filler_words // 2:,}-word "
            f"prefix ({filler_sent * filler_words:,} words total)"
        )
        if filler_errors:
            print(
                f"    WARNING: {filler_errors} filler request(s) were rejected; "
                f"L1 may not have been overflowed. Reduce --filler-requests size "
                f"or raise --max-total-tokens."
            )
        report["filler_requests_sent"] = filler_sent
        report["filler_requests_failed"] = filler_errors
        mid = scrape_metrics(base_url)
        backup_tokens = mid.get("sglang:hicache_backup_tokens_total", 0.0)
        backup_bytes = mid.get("sglang:hicache_backup_bytes_total", 0.0)
        step.record(
            backup_tokens > 0,
            f"backup_tokens={backup_tokens:,.0f} backup_bytes={backup_bytes:,.0f}",
        )
        report["backup_tokens"] = backup_tokens
        report["backup_bytes"] = backup_bytes
        if backup_tokens > 0:
            per_token = backup_bytes / backup_tokens
            report["measured_backup_bytes_per_token"] = per_token

            # --- 4. the decisive check: what width does L2 actually hold? ----
            # SGLang computes the measured bytes as
            #     len(device_indices) * mem_pool_host.size_per_token
            # so this ratio IS size_per_token: 82,944 for the INT8 pool,
            # 147,456 for BF16. The BF16 control run matters because it proves
            # the measurement tracks the pool class rather than reporting a
            # constant that happens to match.
            compressed = args.config == "int8"
            step = Step(
                "L2 holds the compressed representation (82,944 B/token)"
                if compressed
                else "L2 holds the baseline BF16 representation (147,456 B/token)"
            )
            steps.append(step)
            err = abs(per_token - expected) / expected
            step.record(
                err < 0.02,
                f"measured {per_token:,.0f} B/token vs expected {expected:,} "
                f"({err:.2%} off). INT8 pool must report {expected_encoded:,}; "
                f"BF16 pool must report {expected_baseline:,}.",
            )

        # --- 5. L2 -> L1 restore --------------------------------------
        step = Step("a later request restores from L2 (load-back executed)")
        steps.append(step)
        load_before = mid.get("sglang:load_back_tokens_total", 0.0)
        second = post_generate(base_url, build_prompt(shared_prefix, "Say hello."))
        second_text = (second.get("text") or "").strip()
        after = scrape_metrics(base_url)
        load_delta = after.get("sglang:load_back_tokens_total", 0.0) - load_before
        history = prefill_history(log_path)
        last = history[-1] if history else None
        report["final_request_prefill"] = last
        report["prefill_history_tail"] = history[-3:]
        if load_delta > 0:
            detail = f"loaded {load_delta:,.0f} tokens back from L2"
            if last:
                detail += (
                    f"; final request cached {last['cached_tokens']:,} tokens, "
                    f"prefilled {last['new_tokens']:,}"
                )
            step.record(True, detail)
        else:
            # Distinguish the two causes rather than reporting a bare zero.
            # Any prefix match at all means the request was served from L1; the
            # only question is whether it was a near-total hit or a partial one.
            # A strict threshold here misreads a partial match as "gone from L1".
            if last and last["cached_tokens"] > 0:
                why = (
                    f"no L2 load-back was attempted: the final request matched "
                    f"{last['cached_tokens']:,} tokens and prefilled only "
                    f"{last['new_tokens']:,}, so the prefix was still in L1. The "
                    f"filler did not evict it -- lower --max-total-tokens."
                )
            elif last and last["cached_tokens"] == 0 and last["new_tokens"] > 0:
                why = (
                    f"the prefix was gone from L1 (cached 0) AND was not restored "
                    f"from L2: the request prefilled {last['new_tokens']:,} tokens. "
                    f"This is a real load-back failure, not a test-setup problem."
                )
            else:
                why = "loaded 0 tokens back from L2"
            step.record(False, why)
        report["load_back_tokens_delta"] = load_delta

        # --- 6. generation is still coherent --------------------------
        step = Step("regenerated text matches the first run (deterministic decoding)")
        steps.append(step)
        same = text == second_text
        detail = f"first={text[:60]!r} second={second_text[:60]!r}"
        if not same:
            detail = (
                f"len {len(text)} vs {len(second_text)}; "
                + (
                    f"first difference at {next(i for i, (a, b) in enumerate(zip(text, second_text)) if a != b)}; "
                    if len(text) == len(second_text) and text != second_text
                    else "lengths differ; "
                )
                + detail
            )
        step.record(same, detail)
        report["second_generation"] = {"text": second_text[:200]}

        report["metrics_after"] = {
            k: after.get(k, 0.0)
            for k in (
                "sglang:hicache_backup_bytes_total",
                "sglang:hicache_backup_tokens_total",
                "sglang:load_back_bytes_total",
                "sglang:load_back_tokens_total",
                "sglang:hicache_host_used_tokens",
                "sglang:hicache_host_total_tokens",
                "sglang:hicache_dropped_tokens_total",
                "sglang:max_total_num_tokens",
            )
        }
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"\n=== ERROR: {report['error']}", file=sys.stderr)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        lines = log_path.read_text(errors="replace").splitlines()
        report["server_log_tail"] = "\n".join(lines[-40:])
        # Pull out the match-walk lines specifically: they carry the tree state
        # behind each request's cache decision.
        report["match_walk"] = [
            ln for ln in lines if "HiCache match walk" in ln
        ][-6:]

    report["steps"] = [{"name": s.name, "ok": s.ok, "detail": s.detail} for s in steps]
    failed = [s.name for s in steps if s.ok is False]
    # A step is only a pass if it actually recorded a result. Treating
    # ok is None as "not a failure" made the verdict print PASS when an
    # exception had aborted the run before four of the six checks executed --
    # a harness that lies is worse than one that fails.
    not_run = [s.name for s in steps if s.ok is None]
    report["steps_not_run"] = not_run
    report["verdict"] = "PASS" if steps and not failed and not not_run else "FAIL"

    print(f"\n=== verdict: {report['verdict']}")
    if failed:
        print("=== failed steps:")
        for name in failed:
            print(f"  - {name}")
    if not_run:
        print("=== steps that never ran (aborted before recording a result):")
        for name in not_run:
            print(f"  - {name}")
    if report.get("match_walk"):
        print("\n=== HiCache match walk (why each request did or did not hit L2) ===")
        for line in report["match_walk"]:
            print("  " + line.split("HiCache match walk: ", 1)[-1])
        print(f"\n=== server log tail ({log_path}):")
        print(report["server_log_tail"])

    path = Path(args.json) if args.json else out_dir / f"smoke_{args.config}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n=== wrote {path}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
