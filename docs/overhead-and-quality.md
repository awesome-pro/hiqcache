# Overhead, break-even, and quality

This answers the two questions the previous results could not:

- **What does the codec cost?** The fork's `SGLANG_HICACHE_INT8_TIMING`
  instrumentation is not present at the current tip, so the phases were measured
  directly, outside the server, with CUDA events.
- **What is the loss?** Every earlier logprob-based quality run silently
  bypassed the prefix cache; the cause is identified and fixed below.

Artifacts: [`microbench.json`](../results/validation-2026-10/raw/microbench.json),
[`replay_*.json`](../results/validation-2026-10/raw/),
[`quality_smoke_bf16.json`](../results/validation-2026-10/raw/quality_smoke_bf16.json), [`quality_smoke_int8.json`](../results/validation-2026-10/raw/quality_smoke_int8.json).

---

## 1. Overhead method

`scripts/microbench_codec.py`, on the RTX A6000, Qwen3-8B geometry
(36 layers, 8 KV heads, head_dim 128, K and V). Each phase is timed with CUDA
events, 3 warmups + 10 repetitions, reporting p50/p95/p99; host buffers are
pinned.

| phase | what it does |
| --- | --- |
| `encode_full` | per layer, `index_select` + per-head absmax + round/clamp/cast + 1,152-byte record packing |
| `int8_d2h` | encoded GPU staging → pinned host arena (bulk, and per-layer) |
| `int8_h2d` | pinned host arena → GPU staging (bulk, and per-layer) |
| `decode_and_scatter` | unpack + dequantise + scatter into the BF16 device pool |
| `bf16_d2h` / `bf16_h2d` | the same tokens as BF16, the control |

Not measured: stream overlap with concurrent requests, the scheduler's critical
path, and the JIT mover's exact instruction mix (a bulk `copy_` is used for
bandwidth). Those need the server; this is the lower bound that decides whether
a fused kernel is worth writing.

---

## 2. Measured cost (p50, 10 reps)

| tokens | encode | D2H | H2D | decode + scatter | restore (H2D+decode) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 16.40 ms | 2.19 ms | 2.17 ms | 8.30 ms | 10.47 ms |
| 2,048 | 14.63 ms | 7.28 ms | 7.27 ms | 7.36 ms | 14.63 ms |
| **2,152** (one prefix) | **15.11 ms** | **7.68 ms** | **7.64 ms** | **8.46 ms** | **16.10 ms** |
| 4,096 | 27.58 ms | 14.61 ms | 14.53 ms | 9.32 ms | 23.86 ms |

Transfer bandwidth:

| direction | realised | theoretical PCIe 4.0 x16 |
| --- | ---: | ---: |
| D2H | 23.3 GB/s | 31.5 GB/s |
| H2D (bulk) | 23.4 GB/s | 31.5 GB/s |
| H2D (per-layer, 72 calls) | 22.8 GB/s | 31.5 GB/s |

The per-layer penalty is ~2.5%, so the 36-call H2D pattern is not the problem.
The 512-token encode (16.4 ms) is *slower* than the 2,048-token encode (14.6 ms):
that size is launch-overhead dominated, not a larger fixed cost.

**Where a restore's 16.1 ms goes:** 7.6 ms PCIe, 8.5 ms unpack/dequantise/scatter.
The codec compute is roughly half the restore, so it is not negligible per event
— but see the next section.

---

## 3. Break-even

Recomputing the same 2,152-token prefix on this pod takes **1,352 ms** (median
prefill throughput 1,592 tok/s for a 2,225-token cold prompt; n=22 log lines from
the replay's baseline run).

| operation | cost | relative to recompute |
| --- | ---: | ---: |
| restore only (H2D + decode) | 16.1 ms | **1.2%** |
| backup + one restore | 22.8 + 16.1 = 38.9 ms | **2.9%** |
| measured TTFT saving on an INT8-only hit | 667 ms | 49% of recompute |

Even if the codec were 10x slower, a single reuse would still be ~12% of the
recompute cost. The measured 667 ms saving is smaller than the 1,352 ms prefill
because the hit request still waits in the same queue; the point is the direction
and the order of magnitude.

**This is why a fused Triton kernel is not the next optimization.** It could
save at most the ~8 ms of decode (or ~15 ms of encode) per event, against a
1,352 ms recompute. The codec is not the bottleneck.

---

## 4. Quality

### The bug that invalidated every earlier logprob run

`quality_compare.py` requested `logprob_start_len = 0`, i.e. **full input
logprobs**. SGLang then performs a full prefill to produce those logprobs and
does not use the prefix cache, so a re-sent prompt matches nothing and
`load_back_tokens` stays 0. The harness correctly refused to report agreement —
but the zero was caused by the request shape, not by HiCache.

The comparison only needs the **generated** tokens' logprobs. `logprob_start_len
= -1` requests exactly those, keeps the cache path, and restores the prefix. This
was verified by reproducing the same failure on the proven smoke workload: with
`0` the smoke's restore step failed (`cached 0`, prefilled 1,799 tokens); with
`-1` it passes.

### Method

The proven smoke workload, which reliably forces a real restore: one
5,895-token deterministic prompt, 80 shared-prefix filler requests, then the
same prompt re-sent with `return_logprob=True` and `logprob_start_len=-1`.
Host pool 2 GB, L1 8,192 tokens. BF16 and INT8, one run each.

**Validity checks (both configurations):** the final request reported
`cached_tokens = 5,894` (device 4,097 + host **1,797**), `#new-token: 1`, and
`load_back_tokens = 1,797`. The re-sent text is identical to the cold text. This
is a real codec round trip, not a warm L1 hit.

### Results

| comparison | token ids | text | mean \|Δlogprob\| | max \|Δlogprob\| |
| --- | --- | --- | ---: | ---: |
| BF16 cold vs BF16 restored — **restore-path noise floor** | 16/16 | identical | 0.00318 | **0.01989** |
| INT8 cold vs INT8 restored — codec + restore | 16/16 | identical | 0.00354 | 0.01965 |
| BF16 cold vs INT8 cold — run-to-run control | 16/16 | identical | **0.00000** | **0.00000** |
| **BF16 restored vs INT8 restored — codec** | **16/16** | **identical** | 0.00219 | **0.01117** |

### Reading it

- The **cold path is bit-identical across the two server boots** (Δ = 0 exactly):
  same tokenizer, same greedy decode, no run-to-run noise to confuse the result.
- Even the **lossless BF16 restore** changes logprobs by up to 0.02 versus a cold
  prefill. The restored prefix is consumed through a different attention/batching
  path (cached 5,894 + 1 new token), so BF16 arithmetic is not bit-reproducible
  between the two paths. That is the floor against which INT8 must be judged.
- **INT8 adds no measurable error above that floor.** BF16-vs-INT8 on the
  restored prefix has max |Δlogprob| **0.0112**, below the BF16 restore-path
  floor of 0.0199, and its cold-vs-restored delta (0.0197) is indistinguishable
  from BF16's (0.0199). All 16 generated token ids and the full text are
  identical.
- The codec's theoretical elementwise bound (README) is an input-space bound; this
  is the end-to-end check that the perturbation does not change the output on
  this sample.

### Limitations

- **One prompt, 16 generated tokens, one sample.** This is a deterministic
  spot-check, not a benchmark accuracy evaluation. It cannot bound the error on
  adversarial or long-form prompts.
- The multi-prompt `quality_compare.py` path did not produce a valid run; its
  failures are recorded in the manifest and the commit history. The zero-restore
  failures were caused by `logprob_start_len=0` bypassing the cache and by
  SGLang's dead-node eviction under unrelated filler. The fix and the diagnosis
  are in the commit history.
- Greedy decoding at temperature 0 means token-id agreement is the strongest
  observable; a sampled decode would need many more samples to detect a shift.

---

## 5. What this establishes

- A 2,152-token L2 restore costs **16.1 ms**, of which 7.6 ms is PCIe and 8.5 ms
  is decode/scatter. The backup costs 22.8 ms.
- Realised PCIe bandwidth is **23.3–23.4 GB/s** each way, 74% of the 31.5 GB/s
  theoretical Gen4 x16 link.
- Restore is **1.2% of the recompute it replaces** (1,352 ms), so the codec is
  not the bottleneck at this prefix length.
- The end-to-end TTFT effect is capacity-driven, not codec-driven: on the 20
  requests INT8 serves from L2 but BF16 cannot, TTFT is **667 ms lower** (95% CI
  −862 to −477 ms). On requests both configurations hit, the difference is
  −262 ms (CI −426 to −93), and on requests both miss it is +56 ms (CI −235 to
  +341, i.e. no difference — the control).

## 6. What it does not establish

- Stream overlap: the microbench runs each phase on one stream; the server
  overlaps backup, restore and prefill, so the in-situ critical path can be
  longer or shorter than the sum.
- Quality on a broad task set: the forced-L2 comparison is a small deterministic
  sample, not an accuracy benchmark.
- Behaviour above the INT8 capacity: the 32-prefix working set fits, so this
  workload cannot show a capacity cliff above 96,451 tokens.

## 7. Next optimization, from the data

1. **Do not fuse the codec.** ~8–15 ms per event against a 1,352 ms recompute.
2. **The remaining TTFT is cold prefill and queueing.** INT8 already serves 95 of
   96 reuse events; the 33 misses are the 32 first-of-group prompts plus one
   concurrency overlap. No cache can avoid the first prompt's prefill.
3. **If the working set grows beyond L2**, capacity still matters: removing the
   112-byte padding would add ~10.8% tokens, and 4-bit payloads would double it.
   Neither helps *this* workload, where all 32 prefixes already fit.
4. **Concurrency overlap** costs one reuse event (0.75 pp). A prefetch/coalescing
   policy for in-flight duplicate prefixes would recover it, but it is a small
   win next to the cold-prefill cost.
