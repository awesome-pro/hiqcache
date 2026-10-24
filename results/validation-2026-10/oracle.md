# Trace-derived hit-rate ceiling

Trace: `trace-seed42.jsonl` — 128 requests, 284,198 prompt tokens, 32 groups.

## Ceiling

| quantity | value |
| --- | ---: |
| ideal hits (unlimited capacity, sequential) | 206,670 |
| total prompt tokens | 284,198 |
| **ideal hit rate** | **72.72%** |
| unique token working set | 77,528 (27.3% of traffic) |
| mean shared prefix / question | 2,152.8 / 67.5 tokens |

## Observed vs ceiling

| config | observed hit rate | ideal / observed | hit tokens | L2 capacity |
| --- | ---: | ---: | ---: | ---: |
| baseline | 12.20% | 16.8% of ideal | 34,547 | 0 |
| bf16 | 56.10% | 77.1% of ideal | 159,388 | 54,254 |
| int8 | 72.70% | 100.0% of ideal | 206,753 | 96,451 |

## Why the measured rate is not 95-99%

The shared prefix is 97.0% of a prompt. With `P` requests per shared prefix,
the best possible hit rate is `(P-1)/P x prefix share`:

| requests per prefix | roofline |
| ---: | ---: |
| 2 | 48.48% |
| 4 | 72.72% |
| 8 | 84.84% |
| 16 | 90.90% |
| 32 | 93.93% |
| 64 | 95.45% |
| 128 | 96.20% |
| 256 | 96.58% |

Production 95-99% figures come from workloads with tens to hundreds of requests
per prefix (or multi-turn reuse of a long history). This workload has `P=4`.

## Capacity model (8 GB host pool)

| config | L2 tokens | groups fitting | predicted rate (range) | observed |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 54,254 | 25/32 | 56.67%–56.96% (representative 56.81%) | 56.10% |
| int8 | 96,451 | 32/32 | 72.72%–72.72% (representative 72.72%) | 72.70% |

The model is all-or-nothing per shared prefix: a prefix is reusable only if it
fits in L2. It tracks the measured rates without tuning, and it is a model — not
a substitute for the per-request counters.

## Omissions

- concurrency (8 in-flight requests) and request-completion timing
- L1 device residency and the per-layer restore path
- radix-tree eviction granularity and lock_ref protection
- generated continuations and decode scheduling

