"""Phase 15 — numerical and generation quality of the INT8 L2 codec.

The capacity and latency results say nothing about what the lossy codec does to
model output. This measures it directly, by running identical prompts through the
BF16 and INT8 configurations and comparing:

  * generation agreement — first token, and the whole sequence
  * chosen-token logprob deltas — mean / p95 / p99 / max
  * (optionally) a small task subset

Getting the L2 path exercised matters. Prefix caching on a warm server can serve
everything from L1, in which case both configs return identical logits and the
comparison is vacuous. So each prompt is:
  1. sent once to populate the cache,
  2. followed by filler traffic that evicts it from L1 into L2,
  3. flushed of L1 only, if the server supports it,
  4. re-sent, so the logprobs come from a prefix restored out of the INT8 L2.
Step 3 is skipped with a warning when /flush_cache is unavailable rather than
silently producing a vacuous pass.

Usage::

    python scripts/quality_compare.py --host-size 2 --max-total-tokens 8192
    python scripts/quality_compare.py --prompts 40 --json results/quality.json
"""

from __future__ import annotations

import argparse
import json
import os
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

from run_experiment import build_configs, http_get, scrape_metrics, wait_for_server  # noqa: E402

#: Deterministic prompts. Identical text is sent to both configs, so any
#: difference in output is attributable to the KV representation.
FACTS = [
    ("List the first five prime numbers.", "2, 3, 5, 7, 11"),
    ("What is the capital of France?", "Paris"),
    ("Explain in one sentence why the sky is blue.", "Rayleigh scattering"),
    ("What year did the Berlin Wall fall?", "1989"),
    ("Name the largest planet in the solar system.", "Jupiter"),
    ("What is 17 multiplied by 23?", "391"),
    ("Who wrote the play Hamlet?", "Shakespeare"),
    ("What is the chemical symbol for gold?", "Au"),
    ("How many continents are there?", "Seven"),
    ("What is the boiling point of water in Celsius at sea level?", "100"),
    ("Name the process by which plants make food from sunlight.", "Photosynthesis"),
    ("What is the longest river in the world?", "Nile or Amazon"),
    ("How many sides does a hexagon have?", "Six"),
    ("What gas do humans need to breathe to survive?", "Oxygen"),
    ("What is the freezing point of water in Fahrenheit?", "32"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the square root of 144?", "12"),
    ("Name the three primary colours of light.", "Red, green, blue"),
    ("What is the currency of Japan?", "Yen"),
    ("How many minutes are there in a day?", "1440"),
]


def build_prompt(subject: str, question: str) -> str:
    return (
        f"You are a precise assistant answering questions about {subject}.\n\n"
        f"Question: {question}\n\nAnswer concisely:"
    )


def post_generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    logprob: bool,
    timeout: float = 300,
) -> dict:
    """One deterministic generation, optionally returning token logprobs."""
    body = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if logprob:
        body["return_logprob"] = True
        body["logprob_start_len"] = 0
        body["top_logprobs_num"] = 1
    req = urllib.request.Request(
        f"{base_url}/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode()[:400]
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def extract_chosen_logprobs(payload: dict) -> list[float]:
    """Chosen-token logprobs, in order, from a /generate response.

    SGLang returns ``meta_info.input_token_logprobs`` as
    ``[[logprob, token_id, text], ...]``.
    """
    meta = payload.get("meta_info") or {}
    raw = meta.get("input_token_logprobs")
    if raw is None:
        raw = meta.get("output_token_logprobs")
    out: list[float] = []
    for entry in raw or []:
        if isinstance(entry, (list, tuple)) and entry:
            try:
                out.append(float(entry[0]))
            except (TypeError, ValueError):
                continue
    return out


def generate_all(url: str, prompts: list[str], *, max_new_tokens: int) -> list[dict]:
    results = []
    for prompt in prompts:
        payload = post_generate(
            url, prompt, max_new_tokens=max_new_tokens, logprob=True
        )
        meta = payload.get("meta_info") or {}
        results.append(
            {
                "text": payload.get("text", ""),
                "token_ids": list(meta.get("output_ids") or []),
                "logprobs": extract_chosen_logprobs(payload),
            }
        )
    return results


def compare(a: list[dict], b: list[dict]) -> dict:
    """Agreement and logprob deltas between two result sets."""
    n = min(len(a), len(b))
    first_agree = 0
    full_agree = 0
    token_total = 0
    token_match = 0
    deltas: list[float] = []
    length_deltas: list[int] = []
    diverged_at: list[tuple[int, int, int, int]] = []

    for i in range(n):
        ta, tb = a[i]["token_ids"], b[i]["token_ids"]
        if ta and tb and ta[0] == tb[0]:
            first_agree += 1
        if ta == tb:
            full_agree += 1
        token_total += max(len(ta), len(tb))
        token_match += sum(1 for x, y in zip(ta, tb) if x == y)
        length_deltas.append(len(tb) - len(ta))
        # Compare position by position, and only while the token sequences still
        # agree. Once they diverge, position k is a different token in each run
        # and a logprob difference there measures the divergence, not the codec.
        common = 0
        for x, y in zip(ta, tb):
            if x != y:
                break
            common += 1
        for la, lb in zip(a[i]["logprobs"][:common], b[i]["logprobs"][:common]):
            deltas.append(abs(lb - la))
        if common < len(a[i]["logprobs"]):
            diverged_at.append((i, common, len(ta), len(tb)))

    deltas.sort()
    stats = {}
    if deltas:
        def pct(p: float) -> float:
            idx = min(len(deltas) - 1, int(p * len(deltas)))
            return deltas[idx]

        stats = {
            "mean_abs_delta": sum(deltas) / len(deltas),
            "p50_abs_delta": pct(0.50),
            "p95_abs_delta": pct(0.95),
            "p99_abs_delta": pct(0.99),
            "max_abs_delta": deltas[-1],
            "n_logprobs": len(deltas),
        }

    return {
        "prompts": n,
        "first_token_agreement": first_agree / n if n else 0.0,
        "full_sequence_agreement": full_agree / n if n else 0.0,
        "token_level_agreement": token_match / token_total if token_total else 0.0,
        "mean_length_delta": (
            sum(length_deltas) / len(length_deltas) if length_deltas else 0.0
        ),
        # How far the two runs tracked each other before producing a different
        # token. Means "the codec changed nothing for N tokens" and is a more
        # informative summary than a binary sequence match.
        "mean_agreed_prefix_tokens": (
            sum(d[1] for d in diverged_at) / len(diverged_at)
            if len(diverged_at) == n and n
            else float(sum(len(x["token_ids"]) for x in a[:n])) / n
            if n
            else 0.0
        ),
        **stats,
    }


def run_config(args, config, prompts, *, tag: str) -> dict:
    """Boot one config, populate then L2-restore, and collect generations."""
    sglang_root = Path(args.sglang_root).expanduser().resolve()
    base_url = f"http://127.0.0.1:{args.port}"
    extra = ["--max-total-tokens", str(args.max_total_tokens)]
    server_cmd = [
        sys.executable, "-m", "sglang.launch_server",
        *config.server_args, "--port", str(args.port), *extra,
    ]
    env = dict(os.environ)
    env.update(config.env)
    env["PYTHONPATH"] = f"{sglang_root / 'python'}:{env.get('PYTHONPATH', '')}"
    env["SGLANG_HICACHE_DEBUG_MATCH"] = "1"

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"quality_{config.name}.server.log"

    print(f"\n=== {config.label}  (port {args.port})")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            server_cmd, cwd=sglang_root, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    proc._hiqcache_log_path = log_path

    report: dict = {"config": config.name, "label": config.label}
    try:
        waited = wait_for_server(base_url, args.startup_timeout, proc)
        print(f"    healthy after {waited:.0f}s")

        # 1. populate
        warm = build_prompt("general knowledge", FACTS[0][0])
        post_generate(base_url, warm, max_new_tokens=4, logprob=False)

        # 2. filler that shares a prefix, to push things into L2
        shared = " ".join(str(i) for i in range(args.filler_words // 2))
        for i in range(args.filler_requests):
            try:
                post_generate(
                    base_url,
                    f"{shared} tail {i}\n\nQ: count?\n\nAnswer:",
                    max_new_tokens=4,
                    logprob=False,
                )
            except Exception:  # noqa: BLE001
                pass

        # 3. drop L1 so the measured prompts must come from L2
        flushed = False
        try:
            http_get(f"{base_url}/flush_cache", timeout=120)
            flushed = True
        except Exception as exc:  # noqa: BLE001
            print(f"    WARNING: /flush_cache unavailable ({exc}); the prompts "
                  f"may be served from L1 and the comparison could be vacuous")
        report["flushed_l1"] = flushed

        # 4. measure
        prompts_to_send = [
            build_prompt("general knowledge", q) for q, _ in FACTS[: args.prompts]
        ]
        report["generations"] = generate_all(
            base_url, prompts_to_send, max_new_tokens=args.max_new_tokens
        )
        metrics = scrape_metrics(base_url)
        report["load_back_tokens"] = metrics.get("sglang:load_back_tokens_total", 0.0)
        report["backup_tokens"] = metrics.get("sglang:hicache_backup_tokens_total", 0.0)
        print(f"    L2 restores: {report['load_back_tokens']:,.0f} tokens")
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
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--host-size", type=float, default=2.0)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--prompts", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--filler-words", type=int, default=900)
    parser.add_argument("--filler-requests", type=int, default=40)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--sglang-root", default=str(REPO_ROOT.parent / "sglang"))
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    configs = build_configs(
        args.model, args.host_size, 1, 1, max_total_tokens=args.max_total_tokens
    )
    prompts = [build_prompt("general knowledge", q) for q, _ in FACTS[: args.prompts]]

    bf16 = run_config(args, configs["bf16"], prompts, tag="bf16")
    int8 = run_config(args, configs["int8"], prompts, tag="int8")

    result = compare(bf16["generations"], int8["generations"])
    result["bf16_load_back_tokens"] = bf16.get("load_back_tokens", 0.0)
    result["int8_load_back_tokens"] = int8.get("load_back_tokens", 0.0)

    print("\n" + "=" * 74)
    print("QUALITY: BF16 L2 vs INT8 L2, identical deterministic prompts")
    print("=" * 74)
    print(f"  L2 tokens restored   bf16 {result['bf16_load_back_tokens']:>12,.0f}"
          f"   int8 {result['int8_load_back_tokens']:>12,.0f}")
    print(f"  prompts compared     {result['prompts']:>12,}")
    print(f"  first-token agreement{result['first_token_agreement']:>12.1%}")
    print(f"  full-seq agreement   {result['full_sequence_agreement']:>12.1%}")
    print(f"  token-level agreement{result['token_level_agreement']:>12.1%}")
    print(f"  mean length delta    {result['mean_length_delta']:>12.2f}")
    if "mean_abs_delta" in result:
        print(f"  |dlogprob| mean      {result['mean_abs_delta']:>12.5f}")
        print(f"  |dlogprob| p95       {result['p95_abs_delta']:>12.5f}")
        print(f"  |dlogprob| p99       {result['p99_abs_delta']:>12.5f}")
        print(f"  |dlogprob| max       {result['max_abs_delta']:>12.5f}")
    else:
        print("  logprobs unavailable -- agreement only")
    if not bf16.get("flushed_l1") or not int8.get("flushed_l1"):
        print("\n  WARNING: L1 was not flushed on at least one config, so these")
        print("  generations may not have exercised the L2 path.")

    out = Path(args.json) if args.json else REPO_ROOT / "results" / "quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": result, "bf16": bf16, "int8": int8}, indent=2) + "\n")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
