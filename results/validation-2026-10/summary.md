# Validation 2026-10 — results summary

Evidence-first rerun of the HiQCache claim on the current fork tip, with the
missing theoretical baseline, a per-request replay, a real forced-L2 quality
check, and an isolated overhead measurement.

**One-line answer:** on this workload the cache hit rate has an exact ceiling of
**72.72%**; HiQCache INT8 reaches **71.97%** (98.97% of the ceiling), BF16
reaches **57.57%** because its L2 holds only ~25 of the 32 shared prefixes, and
the codec's cost is **16.1 ms** per 2,152-token restore against **1,352 ms** to
recompute it.

---

## 1. Environment

| item | value |
| --- | --- |
| GPU | NVIDIA RTX A6000, 46,068 MiB, sm_86, PCIe Gen4 x16 |
| driver | 595.91.07 |
| CUDA toolkit | 13.0, V13.0.88 (`/usr/local/cuda/bin`) |
| torch | **2.14.1+cu130** (in a venv; see image note) |
| SGLang | `89e0023e215a6d80b6cff0e3075e11067f2ff90d` (`hiqcache/int8-l2`) |
| model | `Qwen/Qwen3-8B`, snapshot `b968826d9c46dd6066d109eabc6255188de91218` |
| host | 96 vCPU, 503 GB RAM, Ubuntu 24.04.5, 100 GB container |
| trace | seed 42, 128 requests, 284,198 prompt tokens, sha256 `f5711019…` |

**Image note.** The rented image was not the documented
`runpod/pytorch:1.4.0-rc.164-cu1300-torch2130-ubuntu2404`: it shipped torch
2.8.0+cu129, no `nvcc` on `PATH`, and PEP 668. SGLang was installed into a clean
venv, which replaced torch with the fork's pin 2.14.1+cu130 (the docs said 2.13;
the fork pin moved). CUDA 13.0 and the driver matched; the 25 codec + 36 pool
tests passed. This is a recorded deviation, not a silent one.

---

## 2. Hit rate: ceiling, replay, published

Ceiling from the exact trace (`docs/theory-and-oracle.md`): **206,670 /
284,198 = 72.72%** (unlimited capacity, sequential, longest prior prefix).

| config | replay hit rate | cached tokens | published | share of ceiling |
| --- | ---: | ---: | ---: | ---: |
| baseline (L1 only) | 12.05% | 34,236 | 12.2% | 16.6% |
| BF16 HiCache | **57.57%** | 163,624 | 56.1% (repeats 57.6%) | **79.2%** |
| HiQCache INT8 | **71.97%** | 204,529 | 72.7% (repeats 72.7–73.4%) | **98.97%** |

The replay used the exact token ids via the native `/generate` endpoint, no
warmup, concurrency 8. It reproduces the published rates within 0.75 pp.

**Where the BF16 loss is.** 96 reuse events exist (128 prompts − 32
first-of-group). BF16 served **76/96**, INT8 **95/96**. The misses are spread
across groups, not whole groups:

| prefix hits per group (of 4) | BF16 groups | INT8 groups |
| ---: | ---: | ---: |
| 0 | 1 | 0 |
| 1 | 1 | 0 |
| 2 | 15 | 1 |
| 3 | 15 | 31 |

BF16's 54,254-token L2 holds ~25 prefixes' worth of bytes, so ~79% of reuse
events survive; INT8's 96,451 holds all 32, and its one miss is a concurrency
overlap (two prompts of a group in flight at once), not capacity.

---

## 3. TTFT / TPOT / ITL (per-request replay)

Aggregate, same 128 requests:

| metric | baseline | BF16 | INT8 |
| --- | ---: | ---: | ---: |
| TTFT mean | 2,046 ms | 1,229 ms | **990 ms** |
| TTFT p50 | 2,074 ms | 1,028 ms | **777 ms** |
| TTFT p95 | 3,936 ms | 2,488 ms | 2,755 ms |
| TTFT p99 | 4,319 ms | 3,806 ms | 4,130 ms |
| TPOT mean | 60.1 ms | 42.7 ms | **39.9 ms** |
| ITL p99 | 1,043.9 ms | 444.4 ms | **118.8 ms** |
| output throughput | 64.1 tok/s | 98.3 tok/s | **112.4 tok/s** |

These are one replay per config; the pod/version is slower than the published
runs (published BF16 1,244 ms / 37.0 ms, INT8 922 ms / 33.3 ms), so compare
within this table, not across documents.

### Paired decomposition (same request id)

| subset | n | BF16 TTFT | INT8 TTFT | Δ (INT8−BF16) | bootstrap 95% CI | INT8 faster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| both hit | 75 | 961 ms | 699 ms | **−262 ms** | [−426, −93] | 63% |
| both miss (control) | 32 | 1,795 ms | 1,851 ms | +56 ms | [−235, +341] | 41% |
| **INT8 hit, BF16 miss** | **20** | 1,280 ms | 613 ms | **−667 ms** | **[−862, −477]** | **90%** |
| BF16 hit, INT8 miss | 1 | 2,225 ms | 2,747 ms | +523 ms | — | 0% |

The 20 requests that only INT8 serves from L2 are the capacity benefit:
**667 ms lower TTFT**, CI excludes zero. When neither hits, the difference is
+56 ms with a CI spanning zero — the systems are equivalent, as they should be.

The −262 ms on requests both hit is **not** the codec's own restore cost (16 ms).
It is queueing: INT8 removes 20 recomputes (~43k prompt tokens) from the same
window, so every request waits less. That is why hit rate → TTFT is non-linear:
the benefit is a system-congestion effect, not a per-request timer.

---

## 4. Overhead (isolated, CUDA events, 10 reps)

| tokens | encode | D2H | H2D | decode+scatter | restore |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 16.40 ms | 2.19 ms | 2.17 ms | 8.30 ms | 10.47 ms |
| 2,048 | 14.63 ms | 7.28 ms | 7.27 ms | 7.36 ms | 14.63 ms |
| **2,152** | **15.11 ms** | **7.68 ms** | **7.64 ms** | **8.46 ms** | **16.10 ms** |
| 4,096 | 27.58 ms | 14.61 ms | 14.53 ms | 9.32 ms | 23.86 ms |

D2H 23.3 GB/s, H2D 23.4 GB/s (74% of theoretical 31.5 GB/s Gen4 x16); the
per-layer H2D pattern is 22.8 GB/s, a ~2.5% penalty.

**Break-even.** Recomputing 2,152 tokens takes **1,352 ms** (median 1,592 tok/s
cold prefill, n=22 log lines). A restore is **16.1 ms = 1.2%** of that; backup +
one restore is 38.9 ms = 2.9%. Even a 10x slower codec would be ~12% of a
recompute. **The codec is not the bottleneck.**

---

## 5. Quality: forced L2 restore, BF16 vs INT8

Proven smoke workload: one 5,895-token prompt, 80 shared-prefix filler requests,
re-send with `return_logprob=True`, `logprob_start_len=-1`. Both configs
reported `cached_tokens = 5,894` (device 4,097 + host **1,797**), `#new-token:
1`, and `load_back_tokens = 1,797` — a real codec round trip.

| comparison | ids | text | mean \|Δlogprob\| | max \|Δlogprob\| |
| --- | --- | --- | ---: | ---: |
| BF16 cold vs restored (restore-path noise) | 16/16 | identical | 0.00318 | 0.01989 |
| INT8 cold vs restored | 16/16 | identical | 0.00354 | 0.01965 |
| BF16 cold vs INT8 cold (control) | 16/16 | identical | **0.00000** | **0.00000** |
| **BF16 restored vs INT8 restored** | **16/16** | **identical** | 0.00219 | **0.01117** |

The cold path is bit-identical across boots. Even lossless BF16 restore moves
logprobs by up to 0.02 versus a cold prefill, because the restored prefix takes a
different attention/batching path. INT8's restored-prefix delta (max 0.011) is
**below that floor**, with identical token ids and text. On this sample the
codec's loss is not distinguishable from the restore path's own nondeterminism.

**Caveat:** one prompt, 16 greedy tokens. This is a spot-check, not a benchmark
accuracy result. The multi-prompt `quality_compare.py` path failed (zero
restores from `logprob_start_len=0`, then SGLang dead-node eviction); failures
are recorded in the manifest.

---

## 6. Denominator definitions

- **Hit rate** = Σ `prompt_tokens` cached / Σ `prompt_tokens`, per request, from
  `meta_info.cached_tokens`; denominator **284,198** for every config.
- **Device / host** = `meta_info.cached_tokens_details.device` / `.host`.
- **TTFT** = client-side send → first non-empty streaming chunk.
- **TPOT** = (end-to-end − TTFT) / (completion_tokens − 1).
- **ITL** = gaps between streaming chunks, normalised by new tokens per chunk.
- **Overhead** = p50 over 10 CUDA-event repetitions, 3 warmups.
- **Recompute** = 1,592 tok/s median cold-prefill rate × 2,152 tokens.

---

## 7. Failures and limitations

1. **Image mismatch.** torch 2.8.0+cu129, no nvcc on PATH, PEP 668. Resolved
   with a venv + torch 2.14.1+cu130; recorded as a deviation from the runbook.
2. **Quality harness, four failed attempts.** (a) stale pod copy lacked the new
   flags; (b) `logprob_start_len=0` forced a full prefill and bypassed the cache;
   (c) unrelated filler produced dead nodes (`evicted and not backuped`) so the
   match walk stopped; (d) the 4,096-token L1 thrashed 1,895-token prompts. The
   valid result was obtained on the proven smoke path with `-1`.
3. **One replay per config.** Hit rates are reproducible (57.57 vs 56.1/57.6
   published) and the paired CIs are reported; latency still has run-to-run
   variance from the shared pod.
4. **One quality sample.** No broad task accuracy, no long-form generation.
5. **Version drift.** SGLang 89e0023 and torch 2.14.1 differ from the published
   3bb2ef/2.13; the hit-rate reproduction is close but not byte-exact.
6. **Baseline teardown `SIGQUIT`.** The server logs a crash handler during
   SIGTERM shutdown; the replay JSON was written and the metrics are complete.
   It is a teardown artifact, not a measurement failure.

---

## 8. Reproduce

```bash
# trace + ceiling (CPU, no GPU)
python scripts/export_trace.py --tokenizer-dir <Qwen3-8B> --sglang-root ../sglang \
    --out results/validation-2026-10/trace/trace-seed42.jsonl
python scripts/oracle.py --trace results/validation-2026-10/trace/trace-seed42.jsonl

# on the pod
/workspace/venv/bin/python scripts/replay_trace.py --config int8 \
    --trace results/validation-2026-10/trace/trace-seed42.jsonl
/workspace/venv/bin/python scripts/microbench_codec.py --sglang-root /workspace/sglang
/workspace/venv/bin/python scripts/smoke_test.py --config int8 --host-size 2
```

Raw artifacts: [`raw/`](raw/), manifest: [`manifest.json`](manifest.json).
