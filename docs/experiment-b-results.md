# Experiment B — measured results

**Same physical CPU-memory budget (`--hicache-size 8`), same GPU, same SGLang SHA,
same workload.** Three configurations, one run each.

Hardware: NVIDIA RTX A6000 48 GB, 62 GB RAM. SGLang `9a7ac7978f49` on
`hiqcache/int8-l2`. Workload: 32 shared-prefix groups x 2048 tokens = 65,536
reusable prefix tokens, 128 prompts, concurrency 8. L1 capped at 16,384 tokens.

---

## Provenance

Every number below is reproduced from the raw JSON and server logs under
`results/`, which are committed alongside this document.

| item | value |
| --- | --- |
| GPU | NVIDIA RTX A6000, 46,068 MiB, sm_86 |
| driver | 580.159.03 |
| CUDA (image toolkit) | 13.0 (nvcc V13.0.98) |
| image | `runpod/pytorch:1.4.0-rc.164-cu1300-torch2130-ubuntu2404` |
| torch | 2.14.0+cu130 (codec venv); 2.13.0+cu130 (server, matching SGLang's pin) |
| host | Linux 6.8.0-139-generic x86_64, 62 GB RAM, 16 vCPU |
| **SGLang SHA measured** | **`9a7ac7978f49e9280f41fb10ec3ee6fb0e49b1c1`** |
| branch | `hiqcache/int8-l2` |
| hiqcache SHA | `3bbe653` |

### The fork has advanced since these measurements

Two commits landed after the measured SHA:

```
38e366694d  Add per-phase codec timing to the INT8 pool
ecb6278a0a  Match-walk diagnostic: also report is_write_back
```

**They do not affect these results.** Both add opt-in instrumentation only: the
codec timing is gated on `SGLANG_HICACHE_INT8_TIMING` (off by default) and merely
records CUDA events when on, and the match-walk diagnostic is gated on
`SGLANG_HICACHE_DEBUG_MATCH` and only logs. Neither alters the codec, the
mover calls, or the host-pool representation, so the capacity and byte figures --
which are arithmetic identities over `size_per_token` -- are unaffected.

Any *future* run must re-record its own SHA, which `run_experiment.py` does
automatically in every result JSON.

---

## Headline

| metric | baseline (no L2) | BF16 HiCache | **HiQCache INT8** |
| --- | --- | --- | --- |
| L2 capacity (tokens) | 0 | 54,254 | **96,451** |
| L2 tokens used | 0 | 53,585 (98.8%) | 80,833 (83.8%) |
| measured B/token in L2 | – | 147,456 | **82,944** |
| backup bytes | 0 | 31.08 GiB | 6.24 GiB |
| restore bytes | 0 | 17.44 GiB | 12.97 GiB |
| **cache hit rate** | 12.2% | 56.1% | **72.7%** |
| tokens served from L2 | 0 | 126,997 | **167,899** |
| TTFT mean | 1,643.0 ms | 1,244.3 ms | **922.3 ms** |
| TTFT p99 | 3,626.3 ms | 3,450.8 ms | 3,369.6 ms |
| TPOT mean | 47.9 ms | 37.0 ms | **33.3 ms** |
| ITL p99 | 1,021.4 ms | 390.9 ms | **112.6 ms** |
| output throughput | 80.0 tok/s | 105.1 tok/s | **127.8 tok/s** |

**vs the BF16 baseline at the same host memory:** +16.6 cache-hit points,
+32% tokens served from L2, −25.9% mean TTFT, +21.6% output throughput.

## What each number establishes

**The capacity claim is measured, not modelled.** `hicache_host_total_tokens`
reports exactly 54,254 and 96,451 — the two figures the codec layout predicts for
8 GB. The BF16 pool reached 98.8% utilisation, i.e. it was genuinely full.

**The encoding rate is exact.** Backup bytes divided by backup tokens is
147,456.00 for BF16 and **82,944.00** for INT8 — the layout value to the byte,
over 283 MB and 6.24 GiB of real traffic respectively.

**The extra capacity converts into served work.** INT8 restored 167,899 tokens
from L2 against BF16's 126,997 (+32%), which is what produces the hit-rate and
TTFT differences. The mechanism is the one PROJECT.md hypothesised: at a fixed
host budget, more of the working set stays resident, so fewer prefixes are
recomputed.

**Phase 11 mechanism checks (separate runs):** the INT8 pool was confirmed to
store 82,944 B/token and to restore from L2 (1,797 tokens loaded back, with the
tree node observed as `evicted=True backuped=True host_value=1797`). The BF16
control reported 147,456 B/token, so the measurement tracks the pool class rather
than reporting a constant.

## Limitations — read these before quoting the numbers

1. **Single runs.** The capacity and bytes/token figures are deterministic
   (arithmetic identities, not measurements), but the *latency and throughput*
   differences are single samples. TPOT and throughput differed by ~0.2% between
   two identical smoke-test runs, but TTFT/ITL at p99 are spiky on this hardware
   (note the baseline's 1,021 ms ITL p99). Repeats are needed before quoting
   latency deltas as established. The cache-hit-rate difference is the most
   trustworthy of the serving metrics, since it is a count, not a timing.

2. **The baseline still had L1.** It shows a 12.2% hit rate, entirely
   device-side, because SGLang always caches in L1. It is a "no reusable L2"
   baseline, not a "no cache" baseline — which is the correct comparison for
   isolating L2's contribution, but it is not a cold-recompute control.

3. **Backup/restore durations read 0.00 s.** `timing_enabled` was not set on the
   acks, so the duration histograms stayed empty. Byte and token volumes are
   unaffected. Enabling timing is required before quoting codec overhead.

4. **`--hicache-ratio` was not exercised.** Only fixed `--hicache-size`.
   Experiment A (equal logical capacity) is still outstanding.

5. **One workload shape.** 65,536 reusable tokens at concurrency 8. The
   crossover point where INT8 stops helping (workload exceeding the INT8 L2) is
   not characterised.

## What would falsify the result

The honest test of the claim is that INT8 *does not* simply inherit BF16's
behaviour. It does not: at an identical host budget, INT8 holds 1.7778x the
tokens and serves 32% more of this workload from L2. If the encoding were being
padded or the wrong pool selected, `hicache_host_total_tokens` would read ~54k
for INT8 and bytes/token would read 147,456. Both were checked directly.
