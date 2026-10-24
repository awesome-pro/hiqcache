# Overhead and quality

This document measures the cost of the codec and the effect of an INT8 L2
restore on generation.

Artifacts: [`microbench.json`](../results/validation-2026-10/raw/microbench.json),
[`replay_*.json`](../results/validation-2026-10/raw/),
[`quality_smoke_bf16.json`](../results/validation-2026-10/raw/quality_smoke_bf16.json),
[`quality_smoke_int8.json`](../results/validation-2026-10/raw/quality_smoke_int8.json).

## 1. Method

`scripts/microbench_codec.py` runs on the RTX A6000 with the Qwen3-8B geometry
(36 layers, 8 KV heads, head_dim 128, K and V). Each phase is timed with CUDA
events over 3 warmups and 10 repetitions. Host buffers are pinned.

| phase | what it does |
| --- | --- |
| `encode_full` | per layer, `index_select`, per-head absmax, round, clamp, cast, record packing |
| `int8_d2h` | encoded GPU staging to pinned host arena, bulk and per layer |
| `int8_h2d` | pinned host arena to GPU staging, bulk and per layer |
| `decode_and_scatter` | unpack, dequantise, scatter into the BF16 device pool |
| `bf16_d2h` / `bf16_h2d` | the same tokens as BF16, the control |

Not measured: stream overlap with concurrent requests, the scheduler's critical
path, and the JIT mover's instruction mix. A bulk `copy_` is used for bandwidth.

## 2. Results (p50, 10 reps)

| tokens | encode | D2H | H2D | decode + scatter | restore (H2D + decode) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 16.40 ms | 2.19 ms | 2.17 ms | 8.30 ms | 10.47 ms |
| 2,048 | 14.63 ms | 7.28 ms | 7.27 ms | 7.36 ms | 14.63 ms |
| 2,152 (one prefix) | 15.11 ms | 7.68 ms | 7.64 ms | 8.46 ms | 16.10 ms |
| 4,096 | 27.58 ms | 14.61 ms | 14.53 ms | 9.32 ms | 23.86 ms |

| direction | measured | theoretical PCIe 4.0 x16 |
| --- | ---: | ---: |
| D2H | 23.3 GB/s | 31.5 GB/s |
| H2D, bulk | 23.4 GB/s | 31.5 GB/s |
| H2D, per layer (72 calls) | 22.8 GB/s | 31.5 GB/s |

A 2,152-token restore takes 16.1 ms: 7.6 ms PCIe and 8.5 ms
unpack/dequantise/scatter. The backup takes 22.8 ms.

The 512-token encode is slower than the 2,048-token encode because launch
overhead dominates at that size. The per-layer H2D path is 2.5% below the bulk
path.

## 3. Break-even

Recomputing 2,152 tokens takes 1,352 ms. This uses the median cold-prefill rate
of 1,592 tok/s for a 2,225-token prompt (n=22 log lines from the baseline
replay).

| operation | cost | relative to recompute |
| --- | ---: | ---: |
| restore only (H2D + decode) | 16.1 ms | 1.2% |
| backup + one restore | 38.9 ms | 2.9% |
| measured TTFT saving on an INT8-only hit | 667 ms | 49% |

One reuse costs 38.9 ms to back up and restore, against 1,352 ms to recompute.
Even a codec 10 times slower would be about 12% of the recompute cost. The
measured 667 ms saving is smaller than 1,352 ms because the hit request still
waits in the same queue.

A fused Triton kernel would save at most the 8.5 ms of decode, or 15.1 ms of
encode, per event. The codec is not the bottleneck.

## 4. Quality

### The cache bypass

`quality_compare.py` used `logprob_start_len = 0`, which asks for full input
logprobs. SGLang then prefills the prompt to produce them and does not use the
prefix cache. The re-sent prompt matches nothing and `load_back_tokens` stays 0.
This caused every earlier zero-restore quality run.

The comparison only needs output logprobs. Setting `logprob_start_len = -1`
requests those, keeps the cache path, and restores the prefix. The same failure
and fix were reproduced on the smoke workload: with `0`, the restore step failed
(cached 0, prefilled 1,799 tokens); with `-1`, it passed.

### Method

The smoke workload forces a real restore: one 5,895-token deterministic prompt,
80 shared-prefix filler requests, then the same prompt re-sent with
`return_logprob=True` and `logprob_start_len=-1`. Host pool 2 GB, L1 8,192
tokens. BF16 and INT8, one run each.

Both configurations reported `cached_tokens = 5,894` (device 4,097 + host
1,797), `#new-token: 1`, and `load_back_tokens = 1,797`. The re-sent text is
identical to the cold text. This is a real codec round trip.

### Results

| comparison | token ids | text | mean abs(dlogprob) | max abs(dlogprob) |
| --- | --- | --- | ---: | ---: |
| BF16 cold vs BF16 restored (restore noise) | 16/16 | identical | 0.00318 | 0.01989 |
| INT8 cold vs INT8 restored (codec + restore) | 16/16 | identical | 0.00354 | 0.01965 |
| BF16 cold vs INT8 cold (run control) | 16/16 | identical | 0.00000 | 0.00000 |
| BF16 restored vs INT8 restored (codec) | 16/16 | identical | 0.00219 | 0.01117 |

### Reading

- The cold path is identical across the two server boots. The difference is
  exactly 0 for all 16 tokens.
- A lossless BF16 restore changes logprobs by up to 0.02 against a cold prefill.
  The restored prefix goes through a different attention and batching path
  (5,894 cached tokens plus 1 new token). That is the noise floor.
- INT8 restored against BF16 restored has a maximum difference of 0.01117, below
  the 0.01989 floor. All 16 token ids and the full text are identical.
- The codec's elementwise bound in the README is an input-space bound. This test
  checks whether the perturbation changes the output on this sample.

### Limits

- One prompt, 16 generated tokens, one sample. This is a spot-check, not a
  benchmark accuracy result.
- The multi-prompt `quality_compare.py` path did not produce a valid run. Its
  failures are recorded in the manifest and the commit history.
- Greedy decoding means token-id agreement is the strongest observable. A
  sampled decode would need many more samples.

## 5. Limits

- The microbench runs each phase on one stream. The server overlaps backup,
  restore and prefill, so the in-situ critical path can differ.
- The quality sample is small.
- The 32-prefix working set fits in INT8 capacity, so this workload cannot show
  behaviour above 96,451 tokens.

## 6. Next steps

1. Do not fuse the codec. Decode is 8.5 ms against a 1,352 ms recompute.
2. Cold prefill and queueing dominate TTFT. INT8 already serves 95 of 96 reuse
   events. The 33 misses are the 32 first-of-group prompts plus one concurrency
   overlap.
3. Capacity changes only matter above the working set. Removing the 112-byte
   padding would add about 10.8% tokens, and 4-bit payloads would double
   capacity. Neither helps this workload.
4. The concurrency overlap costs one reuse event (0.75 points). A prefetch or
   coalescing policy could recover it, but it is small next to the cold-prefill
   cost.
