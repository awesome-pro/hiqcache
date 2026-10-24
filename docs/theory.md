# Theory and trace-derived ceiling

This document derives the cache hit-rate ceiling for the Experiment B workload
from the exact token trace, and compares it with the measured runs.

Artifacts: [`oracle.md`](../results/validation-2026-10/oracle.md),
[`oracle.json`](../results/validation-2026-10/oracle.json),
[`trace manifest`](../results/validation-2026-10/trace/trace-seed42.manifest.json).

## 1. Geometry and capacity

Qwen3-8B at TP=1 has 36 layers, 8 KV heads and head_dim 128. K and V are stored,
and L1 is BF16.

| quantity | derivation | value |
| --- | --- | ---: |
| BF16 K or V row | 8 heads x 128 dims x 2 B | 2,048 B |
| BF16 per token | 2 x 36 layers x 2,048 B | 147,456 B |
| INT8 encoded row | 1,024 payload + 16 scales + 112 pad | 1,152 B |
| INT8 per token | 2 x 36 layers x 1,152 B | 82,944 B |
| bytes saved | 1 - 82,944 / 147,456 | 43.75% |
| storage density | 147,456 / 82,944 | 1.7778x |

SGLang sizes the host pool as follows (`pool_host/base.py`, mirrored by
`scripts/capacity_table.py`):

```python
size = int(host_size_gb * 1e9 // bytes_per_token)
page_num = size // page_size + 1
size = page_num * page_size
```

At `--hicache-size 8` and `--page-size 1`:

| pool | capacity | allocated |
| --- | ---: | ---: |
| BF16 | 54,254 tokens | 8,000,077,824 B |
| INT8 | 96,451 tokens | 8,000,031,744 B |

These numbers come from the layout and the sizing rule, not from a benchmark.

## 2. Exact trace

`scripts/export_trace.py` regenerates the request list with SGLang's own
`GeneratedSharedPrefixDataset`, seed 42 and the Qwen3-8B tokenizer. It writes
every token id in submit order.

The trace has 128 requests and 284,198 prompt tokens. This is the same total as
the published benchmark.

The nominal 2,048 + 64 per prompt does not equal 284,198. `gen_prompt` samples
2,048 token ids, decodes them to text, and re-encodes the text. Re-encoding gives
2,121 to 2,181 tokens per shared prefix (mean 2,151.8) and about 67.5 tokens per
question. The real prompt is 2,185 to 2,254 tokens, mean 2,220.3.

The server's `prompt_tokens_total` is 286,423. That is 284,198 plus one
2,225-token warmup request. The benchmark's 206,753 cached tokens is the exact
sum of the 59 `Prefill batch` `#cached-token` lines in the server log. The
`sglang:cache_hit_rate` gauge reads 97.0%, but it measures a different quantity.

## 3. Oracle

The oracle walks the trace in arrival order. For each request it finds the
longest token prefix that exists in a trie of earlier requests, counts those
tokens as reused, and then inserts the full prompt.

This is an upper bound. It assumes unlimited L2 capacity and one request at a
time. It does not model concurrency, request completion order, L1 residency,
eviction policy, or generated continuations.

| quantity | value |
| --- | ---: |
| requests | 128 |
| total prompt tokens | 284,198 |
| ideal reused tokens | 206,670 |
| ideal hit rate | 72.72% |
| unique token working set | 77,528 (27.3% of traffic) |
| mean shared prefix / question | 2,152.8 / 67.5 |

### Measured against the ceiling

| config | observed | share of ideal | cached tokens | L2 capacity |
| --- | ---: | ---: | ---: | ---: |
| baseline (L1 only) | 12.2% | 16.8% | 34,547 | 0 |
| BF16 HiCache | 56.1% (repeats 57.6%) | 77.1% | 159,388 | 54,254 |
| HiQCache INT8 | 72.7% (repeats 72.7 to 73.4%) | 100.0% | 206,753 | 96,451 |

INT8 serves 206,753 tokens against an ideal of 206,670. The difference is 83
tokens (0.04%), which is consistent with the warmup request and rounding. BF16
reaches 77.1% of the same ceiling.

### Independent replay

The trace was replayed against the current fork tip (SGLang `89e0023`, torch
2.14.1+cu130) with the exact token ids at concurrency 8 and no warmup.

| config | replay hit rate | published |
| --- | ---: | ---: |
| baseline (L1 only) | 12.05% | 12.2% |
| BF16 HiCache | 57.57% | 56.1% (repeats 57.6%) |
| HiQCache INT8 | 71.97% | 72.7% (repeats 72.7 to 73.4%) |

The replay is within 0.75 points of the published rates. INT8 served 95 of 96
reuse events. The one miss is a concurrency overlap, not a codec loss. The
published run served 96.

## 4. Reuse multiplicity

The shared prefix is 97.0% of a prompt (2,152.8 of 2,220.3 tokens). With `P`
requests per shared prefix, the best possible hit rate is:

```
(P - 1) / P  x  prefix share
```

| requests per prefix | ceiling |
| ---: | ---: |
| 2 | 48.48% |
| 4 | 72.72% |
| 8 | 84.84% |
| 16 | 90.90% |
| 32 | 93.93% |
| 64 | 95.45% |
| 128 | 96.20% |
| 256 | 96.58% |

This workload has P = 4. A 95% hit rate needs about 64 requests per prefix, or
multi-turn reuse of a long history. The cache cannot exceed the reuse that the
workload contains.

## 5. Capacity model

The model assumes that a shared prefix is reusable only if it fits in L2. With 32
prefixes of mean 2,152.8 tokens:

| pool | L2 tokens | prefix slots | predicted | measured |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 54,254 | about 25 / 32 | 56.8% | 57.57% |
| INT8 | 96,451 | 32 / 32 | 72.72% | 71.97% |

BF16 served 76 of 96 reuse events (79.2%, against 78.1% predicted). The misses
are spread across groups:

| prefix hits per group (of 4) | BF16 groups | INT8 groups |
| ---: | ---: | ---: |
| 0 | 1 | 0 |
| 1 | 1 | 0 |
| 2 | 15 | 1 |
| 3 | 15 | 31 |
| 4 | 0 | 0 |

BF16 loses one or two reuses from many groups as LRU evicts prefixes between
uses. INT8 loses one reuse in one group. That miss is a concurrency overlap: two
prompts of the same group can be in flight at once, and the later one arrives
before the earlier one's prefix is complete. The oracle does not model this.

The model ignores L1 residency (`--max-total-tokens 16384`), concurrency,
`lock_ref` protection and radix-tree eviction granularity. The replay's per
request `cached_tokens_details` is what checks it.

A capacity sweep is included in `oracle.json` for 1 to 32 GB host pools. It is a
model, not a measurement, and is marked `"observed": false`.

## 6. Limits

- The oracle is an upper bound, not a prediction of a real run.
- The capacity model is a model, not a measurement.
- The hit-rate result does not cover latency or quality.
- The replay uses the current fork tip, not the published revision.

## 7. Reproduce

```bash
python scripts/export_trace.py \
    --tokenizer-dir /path/to/Qwen3-8B-tokenizer \
    --sglang-root ../sglang \
    --out results/validation-2026-10/trace/trace-seed42.jsonl

python scripts/oracle.py \
    --trace results/validation-2026-10/trace/trace-seed42.jsonl \
    --out results/validation-2026-10/oracle.json \
    --md results/validation-2026-10/oracle.md
```

The exporter fails if the trace total is not 284,198, if the generator's prompt
length disagrees with re-encoding, or if the tokenizer files change.
