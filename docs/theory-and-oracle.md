# Theory and trace-derived oracle

This is the theoretical baseline the project did not have. It answers, with
arithmetic on the exact token trace rather than with a benchmark run:

- what the cache-hit-rate **ceiling** is for the Experiment B workload;
- why the measured rate is **57–73%** when production prefix caches report
  **95–99%**;
- how much of the BF16 → INT8 gap is **capacity**, and how much is the codec.

Artifacts: [`oracle.md`](../results/validation-2026-10/oracle.md),
[`oracle.json`](../results/validation-2026-10/oracle.json),
[`trace manifest`](../results/validation-2026-10/trace/trace-seed42.manifest.json).

---

## 1. Geometry and byte arithmetic

Qwen3-8B at TP=1 (`config.json`): **36 layers**, **8 KV heads**, **head_dim 128**,
K and V both stored, L1 in BF16.

| quantity | derivation | value |
| --- | --- | ---: |
| BF16 K or V row | `8 heads x 128 dims x 2 B` | 2,048 B |
| BF16 per token | `2 x 36 layers x 2,048 B` | **147,456 B** |
| INT8 encoded row | `1,024 payload + 16 scales + 112 pad` | 1,152 B |
| INT8 per token | `2 x 36 layers x 1,152 B` | **82,944 B** |
| bytes saved | `1 - 82,944 / 147,456` | **43.75%** |
| storage density | `147,456 / 82,944` | **1.7778x** |

SGLang sizes the host pool as (`pool_host/base.py`, mirrored by
`scripts/capacity_table.py`):

```python
size = int(host_size_gb * 1e9 // bytes_per_token)
page_num = size // page_size + 1      # one extra page
size = page_num * page_size
```

At `--hicache-size 8` (decimal GB) and `--page-size 1`:

| pool | capacity | allocated |
| --- | ---: | ---: |
| BF16 | **54,254 tokens** | 8,000,077,824 B |
| INT8 | **96,451 tokens** | 8,000,031,744 B |

These are the numbers the previous results quoted; the derivation is an
identity over the layout, not a measurement.

---

## 2. The exact trace

`scripts/export_trace.py` regenerates the benchmark's request list with
SGLang's own `GeneratedSharedPrefixDataset` (loaded by path, no torch), the
same seed (`42`), and the same tokenizer. It writes every token id in submit
order plus a manifest.

**Verification:** the regenerated trace totals **284,198 prompt tokens**, which
is exactly the published `Total input tokens`. All 128 rows also satisfy
`generator prompt_len == re-encoded length`.

### Why nominal 2048+64 does not equal 284,198

`gen_prompt(tokenizer, 2,048)` samples 2,048 token ids, **decodes them to
text**, and that text re-encodes to a different length: 2,121–2,181 tokens per
group, mean **2,151.8**. The 64-token questions re-encode to mean **67.5**.
So the real prompt is 2,185–2,254 tokens (mean **2,220.3**), not 2,112.

```
128 x (2,048 + 64)          = 270,336   nominal
128 x 2,220.3               = 284,198   actual tokenized
server prompt_tokens_total  = 286,423   actual + one 2,225-token warmup
```

The 2,225-token difference between the benchmark total (284,198) and the
server counter (286,423) is exactly the benchmark's single warmup request.
This is not a discrepancy; it is the warmup.

The benchmark's cached-token total also reconciles: **206,753** is the exact
sum of the 59 `Prefill batch` `#cached-token` lines in the server log, and the
`sglang:cache_hit_rate` gauge (97.0%) is a different quantity from the
benchmark's 72.7% (see §5).

---

## 3. The idealized oracle

**Definition.** For each request in arrival order, count the longest token
prefix that already exists in a trie of all earlier requests; insert the full
prompt. That is the hit an unlimited-capacity, strictly sequential cache would
serve.

It omits: concurrency (8 in flight), request-completion timing, L1 residency and
per-layer restores, eviction policy, and generated continuations. It is an
**upper bound**, not a prediction of a real run.

| quantity | value |
| --- | ---: |
| requests | 128 |
| total prompt tokens | 284,198 |
| ideal reused tokens | **206,670** |
| **ideal hit rate** | **72.72%** |
| unique token working set | 77,528 (27.3% of traffic) |
| mean shared prefix / question | 2,152.8 / 67.5 |

### Observed against the ceiling

| config | observed | share of ideal | cached tokens | L2 capacity |
| --- | ---: | ---: | ---: | ---: |
| baseline (L1 only) | 12.2% | 16.8% | 34,547 | 0 |
| BF16 HiCache | 56.1% (repeats 57.6%) | 77.1% | 159,388 | 54,254 |
| HiQCache INT8 | **72.7%** (repeats 72.7–73.4%) | **100.0%** | **206,753** | 96,451 |

INT8 is at the ceiling: 206,753 observed versus 206,670 ideal, a difference of
**83 tokens (0.04%)**, consistent with the warmup request and the benchmark's
rounding. BF16 reaches only ~77% of the same ceiling.

This is the defensible form of the hit-rate claim: *the INT8 configuration
saturates the reuse available in this trace; the BF16 configuration does not.*

---

## 4. Why 57–73% is not 95–99%

The shared prefix is **97.0%** of a prompt (2,152.8 of 2,220.3 tokens). With
`P` requests per shared prefix, the best possible hit rate is

```
(P - 1) / P  x  prefix share
```

| requests per prefix `P` | roofline |
| ---: | ---: |
| 2 | 48.48% |
| **4 (this workload)** | **72.72%** |
| 8 | 84.84% |
| 16 | 90.90% |
| 32 | 93.93% |
| 64 | 95.45% |
| 128 | 96.20% |
| 256 | 96.58% |

A cache cannot exceed the reuse its workload contains. This workload has
**P = 4**, so 72.72% is the ceiling *by construction*; 95% would require roughly
64 requests per shared prefix (or multi-turn reuse of a long history, where each
turn reuses the whole preceding context).

Published production figures in the 95–99% range are consistent with traffic
that has this shape — a long system prompt or conversation history reused
dozens to hundreds of times. They are not measured on this 32-group × 4-prompt
synthetic trace, and the denominators differ (request-level vs token-level,
cached-token share vs total prompt tokens). **The 72.7% is low because the
workload has four requests per prefix, not because the cache is poor.**

The same conclusion from the other direction: the roofline arithmetic gives
≥90% at 16 reuses per prefix — e.g. a 16-turn chat over the same history —
without changing a byte of the codec.

---

## 5. Capacity model: where the BF16 gap comes from

The all-or-nothing model: a group's shared prefix is reusable only if it fits in
L2. With 32 prefixes of mean 2,152.8 tokens:

| pool | L2 tokens | prefixes that fit | predicted | observed |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 54,254 | **25 / 32** | 56.8% | 56.1% (repeats 57.6%) |
| INT8 | 96,451 | **32 / 32** | 72.72% | 72.7% |

The model uses no fitted parameters and lands within ~0.7 pp of both measured
rates. It explains the gap without invoking codec speed:

- BF16 can hold ~25 of the 32 shared prefixes, so ~7 groups' prefixes are not
  resident when their later prompts arrive; those prompts recompute.
- INT8 holds all 32, so every available reuse is served.
- The 16.6-point hit-rate difference is therefore a **prefix-capacity** effect,
  not a codec-accuracy effect.

Caveat: this is a model. It ignores L1 residency (`--max-total-tokens 16384`),
concurrency, `lock_ref` protection, and radix-tree eviction granularity. The
per-request counters on the GPU are what validate it; the published single runs
cannot.

A capacity sweep is included in `oracle.json` for 1–32 GB host pools. It is a
model, not a measurement, and is labeled `"observed": false`.

---

## 6. What this establishes, and what it does not

**Establishes**

- The theoretical hit-rate ceiling for the published trace is **72.72%**.
- INT8's 72.7% is at that ceiling (within 0.04%), so the codec does not lose
  cache effectiveness in this workload.
- BF16's 56.1–57.6% is consistent with fitting only ~25 of 32 prefixes, i.e. a
  capacity effect of the expected direction and magnitude.
- The 2048+64 vs 284,198 discrepancy is fully explained by decode/re-encode
  (mean +103.8 tokens per prefix, +3.5 per question) and one warmup request.

**Does not establish**

- That the same ceiling holds at other concurrency, L1 caps, or prefix lengths.
- That the per-request distribution matches the model (only the aggregate does).
- Any latency or quality claim. The hit rate is a count; TTFT/TPOT and the
  codec's numerical loss need separate measurements (Gates 2 and 1).
- That the model's 25/32 split is causally exact; it is a fit to two aggregates.

---

## 7. Reproduce

```bash
# needs transformers + numpy, and an SGLang checkout for the dataset code
python scripts/export_trace.py \
    --tokenizer-dir /path/to/Qwen3-8B-tokenizer \
    --sglang-root ../sglang \
    --out results/validation-2026-10/trace/trace-seed42.jsonl

python scripts/oracle.py \
    --trace results/validation-2026-10/trace/trace-seed42.jsonl \
    --out results/validation-2026-10/oracle.json \
    --md results/validation-2026-10/oracle.md
```

The exporter fails loudly if the trace total is not 284,198, if the generator's
prompt length disagrees with re-encoding, or if the tokenizer files change.
