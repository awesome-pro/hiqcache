"""Export the exact token-id trace a benchmark run served.

Regenerates the benchmark request list with SGLang's own
``GeneratedSharedPrefixDataset``, the same seed and tokenizer, then writes every
token id in submit order.

Outputs ``<out>.jsonl`` (one request per line) and ``<out>.manifest.json``
(tokenizer hash, seed, args, token totals, trace hash).

Usage::

    python scripts/export_trace.py \\
        --tokenizer-dir /path/to/Qwen3-8B-tokenizer \\
        --sglang-root ../sglang \\
        --out results/validation-2026-10/trace/trace-seed42.jsonl

``--tokenizer`` (a HF repo id) can be used instead of ``--tokenizer-dir``. The
script loads the dataset modules by path and does not import ``sglang``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The published Experiment B workload. Changing these changes the trace.
DEFAULT_SEED = 42
DEFAULT_NUM_GROUPS = 32
DEFAULT_PROMPTS_PER_GROUP = 4
DEFAULT_SYSTEM_PROMPT_LEN = 2048
DEFAULT_QUESTION_LEN = 64
DEFAULT_OUTPUT_LEN = 32
#: Published benchmark total for the workload above; a mismatch is a hard error.
DEFAULT_EXPECTED_TOTAL = 284_198


# ---------------------------------------------------------------------------
# Load SGLang's dataset code without importing the sglang package (needs torch)
# ---------------------------------------------------------------------------


def load_gsp_module(sglang_root: Path):
    """Import ``sglang.benchmark.datasets.generated_shared_prefix`` by path.

    ``sglang/__init__.py`` imports torch, which the trace exporter does not
    need. Registering a stub package with the real ``__path__`` lets the real
    module (and its ``common`` dependency) load with only numpy/transformers.
    """
    pkg = sglang_root / "python" / "sglang"
    if not (pkg / "benchmark" / "datasets" / "generated_shared_prefix.py").is_file():
        raise SystemExit(
            f"not an SGLang checkout: {sglang_root} "
            f"(expected {pkg / 'benchmark/datasets/generated_shared_prefix.py'})"
        )

    for name, path in (
        ("sglang", pkg),
        ("sglang.benchmark", pkg / "benchmark"),
        ("sglang.benchmark.datasets", pkg / "benchmark" / "datasets"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules.setdefault(name, module)

    for name, rel in (
        ("sglang.benchmark.datasets.common", "benchmark/datasets/common.py"),
        (
            "sglang.benchmark.datasets.generated_shared_prefix",
            "benchmark/datasets/generated_shared_prefix.py",
        ),
    ):
        if name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(name, pkg / rel)
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load {name} from {pkg / rel}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

    return sys.modules["sglang.benchmark.datasets.generated_shared_prefix"]


# ---------------------------------------------------------------------------
# Trace construction
# ---------------------------------------------------------------------------


def _lcp(a: list[int], b: list[int]) -> int:
    """Longest common token prefix of two sequences."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def derive_groups(sequences: list[list[int]], key_len: int = 64) -> list[int]:
    """Assign each request a group id from its shared system prompt.

    The generator shuffles request order, so the group label is not in the row.
    Prompts of one group share a ~2,152-token random prefix, while two groups
    are unrelated from token 0; the first ``key_len`` ids are therefore a safe
    group key. Ids are assigned in first-appearance order, which is
    deterministic for a fixed trace.
    """
    ids: dict[tuple[int, ...], int] = {}
    groups: list[int] = []
    for seq in sequences:
        key = tuple(seq[:key_len])
        if key not in ids:
            ids[key] = len(ids)
        groups.append(ids[key])
    return groups


def group_prefix_lengths(
    sequences: list[list[int]], groups: list[int]
) -> dict[int, int]:
    """Shared prefix length per group = minimum pairwise LCP in that group.

    The minimum (not the mean) is the prefix every member of the group can
    reuse; a question that happens to share a few opening tokens is not a
    guaranteed shared prefix for the others.
    """
    by_group: dict[int, list[list[int]]] = {}
    for seq, group in zip(sequences, groups):
        by_group.setdefault(group, []).append(seq)
    out: dict[int, int] = {}
    for group, seqs in by_group.items():
        if len(seqs) < 2:
            out[group] = 0
            continue
        out[group] = min(
            _lcp(seqs[i], seqs[j])
            for i in range(len(seqs))
            for j in range(i + 1, len(seqs))
        )
    return out


def _git_sha(path: Path) -> str | None:
    """Commit SHA of a checkout, or None when it is not a git tree.

    The manifest must be portable: a local absolute path says nothing to a
    reader and leaks the author's filesystem. The SHA is the reproducible
    identifier.
    """
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_rows(args, tokenizer):
    """Regenerate the benchmark request list, in submit order."""
    gsp = load_gsp_module(Path(args.sglang_root).expanduser().resolve())
    ns = SimpleNamespace(
        seed=args.seed,
        gsp_num_groups=args.num_groups,
        gsp_prompts_per_group=args.prompts_per_group,
        gsp_system_prompt_len=args.system_prompt_len,
        gsp_question_len=args.question_len,
        gsp_output_len=args.output_len,
        gsp_range_ratio=1.0,
        gsp_fast_prepare=False,
        gsp_send_routing_key=False,
        gsp_num_turns=1,
        gsp_ordered=False,
        gsp_group_distribution="uniform",
        gsp_zipf_alpha=None,
        tokenize_prompt=False,
    )

    # Force fresh generation. A stale pickle keyed only on the tokenizer *class*
    # could otherwise pin the trace to different tokenizer files.
    tmp = Path(tempfile.mkdtemp(prefix="hiqcache-trace-"))
    gsp.get_gen_prefix_cache_path = lambda *a, **k: tmp / "trace.pkl"

    # Same seeding as sglang/benchmark/serving.py before it loads the dataset.
    random.seed(args.seed)
    import numpy as np

    np.random.seed(args.seed)
    return gsp.GeneratedSharedPrefixDataset.from_args(ns).load(tokenizer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", default=str(REPO_ROOT.parent / "sglang"))
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-groups", type=int, default=DEFAULT_NUM_GROUPS)
    parser.add_argument(
        "--prompts-per-group", type=int, default=DEFAULT_PROMPTS_PER_GROUP
    )
    parser.add_argument(
        "--system-prompt-len", type=int, default=DEFAULT_SYSTEM_PROMPT_LEN
    )
    parser.add_argument("--question-len", type=int, default=DEFAULT_QUESTION_LEN)
    parser.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    parser.add_argument(
        "--expected-total",
        type=int,
        default=DEFAULT_EXPECTED_TOTAL,
        help="0 disables the total-token assertion",
    )
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "results" / "validation-2026-10" / "trace" / "trace-seed42.jsonl"
        ),
    )
    args = parser.parse_args(argv)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit("transformers is required: pip install transformers") from exc

    source = args.tokenizer_dir or args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(source)

    rows = build_rows(args, tokenizer)
    sequences = [tokenizer.encode(row.prompt) for row in rows]
    lengths = [len(seq) for seq in sequences]

    # The generator's own prompt_len must equal the re-encoded length; if it
    # does not, the benchmark and this exporter disagree about the prompt.
    for i, (row, length) in enumerate(zip(rows, lengths)):
        if row.prompt_len != length:
            raise SystemExit(
                f"prompt {i}: generator reports {row.prompt_len} tokens but "
                f"re-encoding gives {length}; tokenizer or code drift"
            )

    total = sum(lengths)
    if args.expected_total and total != args.expected_total:
        raise SystemExit(
            f"total prompt tokens {total:,} != expected {args.expected_total:,}; "
            "refusing to write a trace that does not match the published run"
        )

    groups = derive_groups(sequences)
    prefix_lens = group_prefix_lengths(sequences, groups)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, (seq, group, row) in enumerate(zip(sequences, groups, rows)):
            f.write(
                json.dumps(
                    {
                        "request_id": f"req-{i:03d}",
                        "arrival_index": i,
                        "group": group,
                        "prompt_len": len(seq),
                        "output_len": int(row.output_len),
                        "token_ids": seq,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

    tokenizer_file = Path(source) / "tokenizer.json"
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "exact token-id trace for the Experiment B workload",
        "source": {
            "dataset": "generated-shared-prefix",
            # Record the commit, not the local checkout path: a manifest is
            # provenance for other people, not a snapshot of one machine.
            "sglang_sha": _git_sha(Path(args.sglang_root).expanduser().resolve()),
            "tokenizer_id": args.tokenizer,
            "tokenizer_files": "local directory" if args.tokenizer_dir else "hub",
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_vocab_size": len(tokenizer),
            "tokenizer_json_sha256": (
                hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
                if tokenizer_file.is_file()
                else None
            ),
        },
        "generation_args": {
            "seed": args.seed,
            "num_groups": args.num_groups,
            "prompts_per_group": args.prompts_per_group,
            "system_prompt_len": args.system_prompt_len,
            "question_len": args.question_len,
            "output_len": args.output_len,
            "range_ratio": 1.0,
            "ordered": False,
            "group_distribution": "uniform",
        },
        "requests": len(rows),
        "total_prompt_tokens": total,
        "expected_total": args.expected_total,
        "mean_prompt_len": total / len(rows),
        "min_prompt_len": min(lengths),
        "max_prompt_len": max(lengths),
        "groups": len(prefix_lens),
        "group_prefix_tokens": {
            "min": min(prefix_lens.values()),
            "max": max(prefix_lens.values()),
            "mean": sum(prefix_lens.values()) / len(prefix_lens),
        },
        "trace_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "note": (
            "Nominal 2048+64 differs from the measured 284,198 because gen_prompt "
            "samples token ids, decodes to text, and re-encoding is not "
            "length-preserving; the trace records the real ids."
        ),
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {out}  ({total:,} prompt tokens, {len(rows)} requests)")
    print(
        f"  prompt len {min(lengths):,}-{max(lengths):,} "
        f"(mean {total / len(rows):,.1f}); groups {len(prefix_lens)}; "
        f"shared prefix {min(prefix_lens.values())}-{max(prefix_lens.values())} tokens"
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
