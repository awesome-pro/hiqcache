# Validation 2026-10: results summary

This is a rerun of the HiQCache claim on the current fork tip. It adds an exact
trace, a hit-rate ceiling, a per-request replay, isolated overhead timings, and a
forced-L2 quality check.

Main result: the hit-rate ceiling for this workload is 72.72%. INT8 reaches
71.97% and BF16 reaches 57.57%. A 2,152-token L2 restore takes 16.1 ms against
1,352 ms to recompute the same prefix.

## 1. Environment

| item | value |
| --- | --- |
| GPU | NVIDIA RTX A6000, 46,068 MiB, sm_86, PCIe Gen4 x16 |
| driver | 595.91.07 |
| CUDA toolkit | 13.0, V13.0.88 |
| torch | 2.14.1+cu130, in a venv |
| SGLang | `89e0023e215a6d80b6cff0e3075e11067f2ff90d`, branch `hiqcache/int8-l2` |
| model | Qwen/Qwen3-8B, snapshot `b968826d9c46dd6066d109eabc6255188de91218` |
| host | 96 vCPU, 503 GB RAM, Ubuntu 24.04.5, 100 GB container |
| trace | seed 42, 128 requests, 284,198 prompt tokens, sha256 `f5711019` |

The image was not the documented one. It shipped torch 2.8.0+cu129, no `nvcc` on
`PATH`, and PEP 668. SGLang was installed into a clean venv, which replaced torch
with the fork's current pin, 2.14.1+cu130. CUDA 13.0 and the driver matched, and
the 25 codec and 36 pool tests passed.

## 2. Hit rate

The ceiling comes from the exact trace: 206,670 of 284,198 tokens, or 72.72%.
The oracle assumes unlimited capacity and one request at a time.

| config | replay | cached tokens | published | share of ceiling |
| --- | ---: | ---: | ---: | ---: |
| baseline (L1 only) | 12.05% | 34,236 | 12.2% | 16.6% |
| BF16 HiCache | 57.57% | 163,624 | 56.1% (repeats 57.6%) | 79.2% |
| HiQCache INT8 | 71.97% | 204,529 | 72.7% (repeats 72.7 to 73.4%) | 98.97% |

The replay uses the exact token ids, the native `/generate` endpoint,
concurrency 8, and no warmup. It gives 57.57% and 71.97%. The published runs
are BF16 56.1% (repeats 57.6) and INT8 72.7% (repeats 72.7 to 73.4).

There are 96 reuse events: 128 prompts minus 32 first-of-group prompts. BF16
served 76 of them. INT8 served 95. The misses are spread across groups:

| prefix hits per group (of 4) | BF16 groups | INT8 groups |
| ---: | ---: | ---: |
| 0 | 1 | 0 |
| 1 | 1 | 0 |
| 2 | 15 | 1 |
| 3 | 15 | 31 |

BF16's 54,254-token L2 holds about 25 prefixes' worth of bytes, so about 79% of
reuse events survive. INT8's 96,451 tokens hold all 32 prefixes. Its one miss is
a concurrency overlap, not capacity.

## 3. Latency

Same 128 requests, one replay per config:

| metric | baseline | BF16 | INT8 |
| --- | ---: | ---: | ---: |
| TTFT mean | 2,046 ms | 1,229 ms | 990 ms |
| TTFT p50 | 2,074 ms | 1,028 ms | 777 ms |
| TTFT p95 | 3,936 ms | 2,488 ms | 2,755 ms |
| TTFT p99 | 4,319 ms | 3,806 ms | 4,130 ms |
| TPOT mean | 60.1 ms | 42.7 ms | 39.9 ms |
| ITL p99 | 1,043.9 ms | 444.4 ms | 118.8 ms |
| output throughput | 64.1 tok/s | 98.3 tok/s | 112.4 tok/s |

The pod is slower than the published runs. Published BF16 is 1,244 ms TTFT and
37.0 ms TPOT; published INT8 is 922 ms and 33.3 ms. Compare within this table.

### Paired by request

| subset | n | BF16 TTFT | INT8 TTFT | difference | bootstrap 95% CI | INT8 faster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| both hit | 75 | 961 ms | 699 ms | -262 ms | -426 to -93 | 63% |
| both miss (control) | 32 | 1,795 ms | 1,851 ms | +56 ms | -235 to +341 | 41% |
| INT8 hit, BF16 miss | 20 | 1,280 ms | 613 ms | -667 ms | -862 to -477 | 90% |
| BF16 hit, INT8 miss | 1 | 2,225 ms | 2,747 ms | +523 ms | not reported | 0% |

The 20 requests that only INT8 serves from L2 are 667 ms faster, and the
interval excludes zero. When neither hits, the difference is 56 ms and the
interval contains zero.

The 262 ms on requests that both hit is not the codec's restore cost (16 ms). It
comes from queueing: INT8 removes 20 recomputes from the same window, so every
request waits less. This is why hit rate to TTFT is not linear.

## 4. Overhead

| tokens | encode | D2H | H2D | decode + scatter | restore |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 16.40 ms | 2.19 ms | 2.17 ms | 8.30 ms | 10.47 ms |
| 2,048 | 14.63 ms | 7.28 ms | 7.27 ms | 7.36 ms | 14.63 ms |
| 2,152 | 15.11 ms | 7.68 ms | 7.64 ms | 8.46 ms | 16.10 ms |
| 4,096 | 27.58 ms | 14.61 ms | 14.53 ms | 9.32 ms | 23.86 ms |

D2H reaches 23.3 GB/s and H2D reaches 23.4 GB/s, or 74% of the 31.5 GB/s Gen4
x16 theoretical link. The per-layer H2D path reaches 22.8 GB/s.

Recomputing 2,152 tokens takes 1,352 ms at 1,592 tok/s. A restore is 16.1 ms,
or 1.2% of that. Backup plus one restore is 38.9 ms, or 2.9%. Within the
restore, decode is 8.5 ms of the 16.1 ms. The in-server critical path was not
measured.

## 5. Quality

Proven smoke workload: one 5,895-token prompt, 80 shared-prefix filler requests,
then a re-send with output logprobs. Both configs reported `cached_tokens =
5,894` (device 4,097 + host 1,797), one new token, and `load_back_tokens =
1,797`.

| comparison | ids | text | mean abs(dlogprob) | max abs(dlogprob) |
| --- | --- | --- | ---: | ---: |
| BF16 cold vs BF16 restored | 16/16 | identical | 0.00318 | 0.01989 |
| INT8 cold vs INT8 restored | 16/16 | identical | 0.00354 | 0.01965 |
| BF16 cold vs INT8 cold | 16/16 | identical | 0.00000 | 0.00000 |
| BF16 restored vs INT8 restored | 16/16 | identical | 0.00219 | 0.01117 |

The cold path is identical across the two server boots. A lossless BF16 restore
changes logprobs by up to 0.0199 against a cold prefill, because the restored
prefix takes a different attention and batching path. That is the reference
difference on this sample. INT8's restored-prefix difference is 0.01117, below
that reference. All 16 token ids and the full text are
identical.

This is one prompt and 16 greedy tokens. It is a spot-check, not a benchmark
accuracy result.

## 6. Definitions

- Hit rate: sum of cached prompt tokens divided by sum of prompt tokens, from
  `meta_info.cached_tokens`. The denominator is 284,198 for every config.
- Device and host: `meta_info.cached_tokens_details.device` and `.host`.
- TTFT: client send time to the first non-empty streaming chunk.
- TPOT: end-to-end time minus TTFT, divided by completion tokens minus one.
- ITL: gaps between streaming chunks, normalised by new tokens per chunk.
- Overhead: p50 over 10 CUDA-event repetitions, with 3 warmups.
- Recompute: 1,592 tok/s median cold-prefill rate times 2,152 tokens.

## 7. Failures and limits

1. The image did not match the runbook. Fixed with a venv and torch
   2.14.1+cu130; recorded in the manifest.
2. The quality harness failed four times. The main cause was
   `logprob_start_len=0`, which bypassed the prefix cache. Unrelated filler also
   left dead nodes, and a 4,096-token L1 thrashed the long prompts. The valid run
   uses the smoke path with `-1`.
3. One replay per config. Hit rates are repeatable; latency still varies with
   the shared pod.
4. One quality sample. No broad accuracy result.
5. SGLang 89e0023 and torch 2.14.1 differ from the published 3bb2ef and 2.13.
   The hit rates are close to the published repeat ranges, not byte for byte.
6. The raw server logs were not kept in the repository. The failures are
   recorded in the manifest.

## 8. Reproduce

```bash
python scripts/export_trace.py --tokenizer-dir <Qwen3-8B> --sglang-root ../sglang \
    --out results/validation-2026-10/trace/trace-seed42.jsonl
python scripts/oracle.py --trace results/validation-2026-10/trace/trace-seed42.jsonl

/workspace/venv/bin/python scripts/replay_trace.py --config int8 \
    --trace results/validation-2026-10/trace/trace-seed42.jsonl
/workspace/venv/bin/python scripts/microbench_codec.py --sglang-root /workspace/sglang
/workspace/venv/bin/python scripts/smoke_test.py --config int8 --host-size 2
```

Raw artifacts: [`raw/`](raw/). Manifest: [`manifest.json`](manifest.json).
