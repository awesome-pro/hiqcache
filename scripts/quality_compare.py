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


def shared_preamble(repeats: int, seed: int = 0) -> str:
    """A long, deterministic preamble.

    ``seed`` makes each prompt's preamble DISTINCT. That matters more than it
    looks: when every prompt shared one preamble the radix tree held a constant
    ~4,905 tokens regardless of prompt count, so the working set never grew,
    the filler's one-shot nodes were evicted ahead of the prefix, and the prefix
    was backed up (backuped=True) but never actually left the device
    (evicted=False). A re-send therefore matched in L1 and no restore ever
    happened. Experiment B restored 187,989 tokens precisely because its working
    set genuinely exceeded L1; distinct preambles reproduce that.

    Without this the measured prompts are ~30 tokens each and share nothing, so
    the radix tree holds a few hundred tiny nodes: nothing substantial is ever
    demoted to L2 and the re-send has nothing to restore. That is why every
    earlier attempt recorded 0 restores no matter how the filler was sized.

    smoke_test.py gets restores reliably with exactly this trick
    (``--prefix-repeats 400``), so the quality harness mirrors it. Sharing a long
    prefix also makes each restore a meaningful amount of KV rather than ~30
    tokens, which is what makes the comparison worth running at all.
    """
    return " ".join(
        f"Reference {seed}-{i}: archived record {i} of series {seed}."
        for i in range(repeats)
    )


def build_prompts(count: int, prefix_repeats: int) -> list[str]:
    """The measured prompts: one long shared preamble plus a distinct question.

    Built in exactly one place on purpose. An earlier version built them here for
    sending and again inside run_config for populating, and the second copy
    dropped the preamble -- so a 3,500-token prefix silently became ~25 tokens
    and nothing was ever large enough to demote into L2. Two builders for the
    same list is the bug; there is now one.
    """
    return [
        shared_preamble(prefix_repeats, seed=i) + "\n\n" + build_prompt("general knowledge", q)
        for i, (q, _) in enumerate(FACTS[:count])
    ]


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


def extract_output_ids(payload: dict) -> list[int]:
    """Generated token ids, wherever this build happens to put them.

    SGLang has moved this field around: ``/generate`` may report it top-level, or
    inside ``meta_info``, or not at all -- ``return_output_ids`` applies to the
    OpenAI chat endpoint, not here. The chosen-token logprobs carry the id as
    their middle element (``[logprob, token_id, text]``), so fall back to those
    instead of silently returning nothing.

    This matters more than it looks. The first real run of this harness read only
    ``meta_info.output_ids``, got an empty list for every prompt, and went on to
    report "100% full-sequence agreement" -- which is exactly what ``[] == []``
    scores. An empty capture must never again read as a perfect match.
    """
    meta = payload.get("meta_info") or {}
    for candidate in (payload.get("output_ids"), meta.get("output_ids")):
        if candidate:
            return [int(i) for i in candidate]
    ids: list[int] = []
    for entry in meta.get("output_token_logprobs") or []:
        if isinstance(entry, (list, tuple)) and len(entry) > 1:
            try:
                ids.append(int(entry[1]))
            except (TypeError, ValueError):
                continue
    return ids


def generate_all(
    url: str,
    prompts: list[str],
    *,
    max_new_tokens: int,
    raw_dump: Path | None = None,
) -> list[dict]:
    results = []
    for idx, prompt in enumerate(prompts):
        payload = post_generate(
            url, prompt, max_new_tokens=max_new_tokens, logprob=True
        )
        if idx == 0 and raw_dump is not None:
            # Record the response shape once so a future mismatch is diagnosed
            # from data rather than inferred from missing keys.
            try:
                raw_dump.write_text(json.dumps(payload, indent=2)[:20000])
            except (OSError, TypeError, ValueError):
                pass
        results.append(
            {
                "text": payload.get("text", ""),
                "token_ids": extract_output_ids(payload),
                "logprobs": extract_chosen_logprobs(payload),
                # How much of this prompt the server matched in the radix cache.
                # This is the number that separates "matched in L1, so no restore
                # was needed" from "matched nothing, the nodes are gone" -- two
                # opposite faults that both show up as load_back == 0.
                "cached_tokens": int((payload.get("meta_info") or {}).get("cached_tokens") or 0),
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
        # Zero here means nothing was captured and every ratio below is
        # undefined rather than perfect. Callers must check it first.
        "tokens_compared": token_total,
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

        # 1. populate: send the measured prompts once so the shared prefix enters
        #    the radix tree and can later be demoted to L2. The first response's
        #    prompt_tokens is the server's own measure of how large that shared
        #    prefix really is -- the filler sizing below depends on it.
        #
        #    Use the prompts we were GIVEN. An earlier version rebuilt them here
        #    from FACTS, which silently dropped the shared preamble main()
        #    prepends -- the run then logged "shared prefix: ~25 tokens per
        #    prompt" (a 300-repeat preamble is ~3,500) and had nothing large
        #    enough to demote or restore. The bug was invisible because both
        #    paths produced plausible-looking prompts.
        prompts_to_send = list(prompts)
        prefix_tokens = 0.0
        for idx, prompt in enumerate(prompts_to_send):
            payload = post_generate(base_url, prompt, max_new_tokens=2, logprob=False)
            if idx == 0:
                meta = payload.get("meta_info") or {}
                prefix_tokens = float(meta.get("prompt_tokens") or 0)
        print(f"    shared prefix: ~{prefix_tokens:,.0f} tokens per prompt")

        # 2. overflow L1 with DISTINCT filler, sized from the SERVER's own token
        #    accounting rather than a words-to-tokens guess.
        #    Both earlier attempts failed on sizing. The first reused one shared
        #    prefix, so it added almost nothing and never approached the L1 cap.
        #    The second guessed ~1.3 tokens per synthetic word; these tokenize to
        #    several tokens each, so the filler came out ~3x over target, flooded
        #    L2 as well, and pushed the measured prompts out of it -- still 0
        #    restores, for the opposite reason.
        #    The window is narrow and must be hit exactly: enough filler to evict
        #    the measured prompts from L1 into L2, not enough to evict them from
        #    L2 as well. meta_info.prompt_tokens is the ground truth for volume.
        cap = scrape_metrics(base_url)
        l1_cap = float(args.max_total_tokens or 0)
        l2_cap = float(cap.get("sglang:hicache_host_total_tokens", 0.0))
        # If the populate ALREADY overflows L1, that is the condition Experiment B
        # restored 187,989 tokens under, and no filler is needed or wanted: the
        # earlier prompts are demoted into L2 and their host nodes are retained,
        # so re-sending them must restore. Filler on top only pushes them back
        # out of L2 -- measured: 116,142 tokens demoted, L2 down to 12,210 of
        # 13,564, and every re-sent prompt matched a node with no host value.
        skip_filler = prefix_tokens * len(prompts_to_send) > l1_cap
        if skip_filler:
            print(f"    populate overflows L1 "
                  f"({prefix_tokens * len(prompts_to_send):,.0f} > {l1_cap:,.0f} "
                  f"tokens); skipping the filler so demoted prompts stay in L2")
        # Two-sided constraint, and it is narrow:
        #   filler > L1 - prefix   so the shared prefix is evicted from L1
        #   filler < L2 - prefix   so it survives in L2 long enough to restore
        # Target the middle of that window. Keying off L1 alone (as before)
        # ignored that the prefix already occupies most of L1, so the target sat
        # above the L2 ceiling and pushed the prefix straight back out again.
        lo = max(0.0, l1_cap - prefix_tokens)
        hi = max(lo + 1.0, l2_cap - prefix_tokens - 256)
        stop_at = (lo + hi) / 2
        per_request = max(20, args.filler_words)
        sent_tokens = lo if skip_filler else 0.0
        requests_sent = 0
        for i in range(0 if skip_filler else args.filler_requests):
            body = " ".join(f"f{i}w{j}" for j in range(per_request))
            try:
                payload = post_generate(
                    base_url,
                    f"{body}\n\nQ: reply OK\n\nAnswer:",
                    max_new_tokens=1,
                    logprob=False,
                )
            except Exception:  # noqa: BLE001
                continue
            requests_sent = i + 1
            meta = payload.get("meta_info") or {}
            sent_tokens += int(meta.get("prompt_tokens") or 0)
            if sent_tokens >= stop_at:
                break
        report["filler"] = {
            "l1_cap_tokens": l1_cap,
            "l2_cap_tokens": l2_cap,
            "stop_at_tokens": stop_at,
            "sent_tokens": sent_tokens,
            "requests_sent": requests_sent,
            "words_per_request": per_request,
        }
        print(f"    filler: {requests_sent} requests, {sent_tokens:,} prompt "
              f"tokens, stopped at {stop_at:,.0f} (L1 {l1_cap:,.0f} -> L2 "
              f"{l2_cap:,.0f})")
        if sent_tokens < lo:
            raise RuntimeError(
                f"filler only reached {sent_tokens:,} tokens but {lo:,.0f} is "
                f"needed to evict a {prefix_tokens:,.0f}-token prefix from an L1 "
                f"cap of {l1_cap:,.0f}: nothing was demoted, so the re-sent "
                f"prompts would come from L1. Raise --filler-requests."
            )

        # Record what the filler actually achieved. If restores still come back
        # zero this is what separates "nothing was ever demoted" (backup 0) from
        # "demoted, then evicted from L2 as well" (backup > 0, L2 full) -- two
        # opposite faults that look identical from the restore counter alone.
        after_filler = scrape_metrics(base_url)
        report["after_filler"] = {
            "demoted_tokens": after_filler.get("sglang:hicache_backup_tokens_total", 0.0),
            "l2_used_tokens": after_filler.get("sglang:hicache_host_used_tokens", 0.0),
            # L2 evictions. If the prefix matched nothing on the re-send, this
            # says whether L2 threw it out or whether it never became a
            # restorable node in the first place.
            "l2_evicted_tokens": after_filler.get("sglang:hicache_dropped_tokens_total", 0.0),
            "l1_used_estimate": prefix_tokens + sent_tokens,
        }
        print(f"    after filler: demoted {report['after_filler']['demoted_tokens']:,.0f} "
              f"tokens, L2 holds {report['after_filler']['l2_used_tokens']:,.0f} "
              f"of {l2_cap:,.0f}, L2 evicted "
              f"{report['after_filler']['l2_evicted_tokens']:,.0f}")

        # 3. /flush_cache is deliberately NOT called. It flushes the radix cache,
        #    and if that reaches the host tier it deletes the very L2 state this
        #    measurement exists to read back from -- the most likely reason an
        #    earlier run recorded 0 restores on both configs. The filler above has
        #    already evicted L1 by LRU: the measured prompts were sent first, so
        #    they are the oldest nodes and are evicted first.
        report["flushed_l1"] = False

        # 4. measure: re-send, so the shared prefix is restored out of L2
        report["generations"] = generate_all(
            base_url,
            prompts_to_send,
            max_new_tokens=args.max_new_tokens,
            raw_dump=REPO_ROOT / "results" / f"quality_raw_response_{tag}.json",
        )
        metrics = scrape_metrics(base_url)
        report["load_back_tokens"] = metrics.get("sglang:load_back_tokens_total", 0.0)
        report["backup_tokens"] = metrics.get("sglang:hicache_backup_tokens_total", 0.0)
        # The cache matched on the re-send is the missing half of the picture:
        # load_back == 0 on its own cannot tell "served from L1" apart from
        # "matched nothing and prefilled from scratch".
        cached_total = sum(g.get("cached_tokens", 0) for g in report["generations"])
        report["reused_prefix_tokens"] = cached_total
        print(f"    re-send matched {cached_total:,} cached prompt tokens; "
              f"L2 supplied {report['load_back_tokens']:,.0f}")

        # Refuse to report agreement from a run that never touched L2. Without
        # this the harness will happily compare two identical cold prefills and
        # print a perfect score, which is the most misleading output it could
        # produce: it looks like the codec was validated when it never ran.
        if report["load_back_tokens"] <= 0:
            if cached_total > 0:
                why = (
                    f"the re-sent prompts still matched {cached_total:,} tokens, "
                    f"so they were served from L1 and nothing needed restoring -- "
                    f"the filler did not evict the prefix"
                )
            else:
                why = (
                    "the re-sent prompts matched nothing at all, so their nodes "
                    "were evicted from L1 and are no longer in L2 either"
                )
            raise RuntimeError(
                f"no L2 restores for {tag}: {why}. Any agreement figure here "
                f"would describe two cold prefills, not the codec."
            )
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
    parser.add_argument(
        "--prefix-repeats",
        type=int,
        default=300,
        help=(
            "Length of the deterministic preamble shared by every measured "
            "prompt. It must be a large fraction of the L1 cap: short distinct "
            "prompts give the radix tree nothing substantial to demote into L2, "
            "so no restore ever happens. smoke_test.py uses 400 for exactly this "
            "reason."
        ),
    )
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--sglang-root", default=str(REPO_ROOT.parent / "sglang"))
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    configs = build_configs(
        args.model, args.host_size, 1, 1, max_total_tokens=args.max_total_tokens
    )
    prompts = build_prompts(args.prompts, args.prefix_repeats)

    bf16 = run_config(args, configs["bf16"], prompts, tag="bf16")
    int8 = run_config(args, configs["int8"], prompts, tag="int8")

    result = compare(bf16["generations"], int8["generations"])
    result["bf16_load_back_tokens"] = bf16.get("load_back_tokens", 0.0)
    result["int8_load_back_tokens"] = int8.get("load_back_tokens", 0.0)

    out = Path(args.json) if args.json else REPO_ROOT / "results" / "quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": result, "bf16": bf16, "int8": int8}, indent=2) + "\n")

    # Validity before numbers. An earlier run captured no tokens at all and still
    # printed "100% full-sequence agreement", because [] == [] is a match and
    # there was no first token to disagree on. Never print a verdict that reads
    # as a pass when nothing was compared.
    if result["tokens_compared"] == 0:
        print("\n" + "=" * 74)
        print("QUALITY: INVALID -- no generated tokens were captured")
        print("=" * 74)
        print("  Agreement is undefined here, NOT 100%. Raw responses were dumped")
        print("  so the field carrying output ids can be found:")
        print("    results/quality_raw_response_bf16.json")
        print("    results/quality_raw_response_int8.json")
        print(f"\n  wrote {out}")
        return 2

    print("\n" + "=" * 74)
    print("QUALITY: BF16 L2 vs INT8 L2, identical deterministic prompts")
    print("=" * 74)
    print(f"  L2 tokens restored   bf16 {result['bf16_load_back_tokens']:>12,.0f}"
          f"   int8 {result['int8_load_back_tokens']:>12,.0f}")
    print(f"  prompts compared     {result['prompts']:>12,}")
    print(f"  tokens compared      {result['tokens_compared']:>12,}")
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
        print("  logprobs unavailable -- agreement only (see the raw dumps)")
    print("\n  Both configs restored from L2 before these generations were")
    print("  recorded, so the comparison is of codec-round-tripped KV.")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
