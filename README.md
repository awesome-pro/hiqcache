# HiQCache

An INT8-quantised hierarchical KV cache for SGLang's HiCache. Active KV stays BF16
in the GPU L1 tier; on eviction it is compressed to INT8 in the CPU L2 tier and
decoded back to BF16 on an L2 hit, so attention always reads the dtype it expects.
SGLang's caching policy, radix tree, transfer engine and CUDA kernels are unchanged:
the fork adds three files and edits two lines.

All figures are Qwen3-8B at TP=1 on one RTX A6000 48 GB, at SGLang `26e6d78dba`
(experiment A) and `3bb2ef6602` (experiment B); commit ids were rewritten after
measurement, the artifacts were not. Raw data: `results/`,
[experiment A](docs/experiment-a-results.md), [experiment B](docs/experiment-b-results.md).

## Results

### Experiment A: equal logical capacity

Both tiers pinned to the same 54,254 L2 tokens, so the comparison is bytes.

| | BF16 | INT8 |
| --- | --- | --- |
| L2 capacity (tokens) | 54,254 | 54,254 |
| L2 used (tokens) | 53,710 | 53,514 |
| measured bytes/token in L2 | 147,456 | 82,944 |
| L2 bytes for that capacity | 8,000,077,824 (8.000 GB) | 4,500,043,776 (4.500 GB) |
| tokens backed up (D2H) | 216,144 | 222,628 |
| tokens restored (H2D) | 122,678 | 126,979 |
| benchmark cache hit rate | 54.6% | 58.2% |
| L2 evictions | 0 | 0 |

The same logical cache in 56.25% of the host memory, a 43.75% reduction and 1.7778x
the density; both configurations reported `target == achieved == 54,254`. The bytes
are arithmetic, and the [hit-rate difference](docs/experiment-a-results.md) is one run.

### Experiment B: equal physical budget

| metric | baseline (no L2) | BF16 HiCache | HiQCache INT8 |
| --- | --- | --- | --- |
| L2 capacity (tokens) | 0 | 54,254 | 96,451 |
| L2 tokens used | 0 | 53,585 (98.8%) | 80,833 (83.8%) |
| measured B/token in L2 | - | 147,456 | 82,944 |
| backup bytes | 0 | 31.08 GiB | 6.24 GiB |
| restore bytes | 0 | 17.44 GiB | 12.97 GiB |
| cache hit rate | 12.2% | 56.1% | 72.7% |
| tokens served from L2 | 0 | 126,997 | 167,899 |
| TTFT mean | 1,643.0 ms | 1,244.3 ms | 922.3 ms |
| TTFT p99 | 3,626.3 ms | 3,450.8 ms | 3,369.6 ms |
| TPOT mean | 47.9 ms | 37.0 ms | 33.3 ms |
| output throughput | 80.0 tok/s | 105.1 tok/s | 127.8 tok/s |

Against BF16 at the same host memory: +16.6 cache-hit points, +32% tokens served
from L2, -25.9% mean TTFT and +21.6% output throughput.

Capacity and bytes/token are exact: at `--hicache-size 8`,
`hicache_host_total_tokens` reports 54,254 and 96,451, and backup bytes over backup
tokens is 147,456.00 and 82,944.00 over 31 GiB and 6.24 GiB of traffic. Hit rates
are the benchmark's own Cache Hit Details report.

Three further paired runs at the same budget hold the capacity and hit-rate figures
steady (BF16 56.8 to 57.6%, INT8 72.7 to 73.4% across all four) but not the latency:
BF16 mean TTFT ranged 1,193 to 2,558 ms against INT8's 1,202 to 1,259 ms, and one of
the four pairs was 6% worse. The table is one pair; read
[experiment B](docs/experiment-b-results.md) first.

## How it works

An L2 slot holds one 1152-byte record per `(layer, K or V, token)` row:

```
L1  GPU, BF16                        L2  CPU, INT8, 82,944 B/token
┌──────────────────────┐             ┌────────────────────────────────────┐
│ K,V bf16             │ ──encode──► │   0 .. 1023  1024 B INT8 (8 x 128) │
│ 2048 B per row       │             │ 1024 .. 1039  16 B: 8 BF16 scales  │
│ 147,456 B per token  │ ◄─decode─── │ 1040 .. 1151  112 B padding        │
└──────────────────────┘             └────────────────────────────────────┘
```

Quantisation is per-head and symmetric: `s = max(|x|) / 127`, `q = round(x / s)`, and
the decoder returns `q · s` in BF16. The format guarantees

```
|x̂ − x|  ≤  (0.5 + 2⁻⁸) · s  +  2⁻⁸ · |x̂|
```

`0.5 · s` is round-to-nearest, `2⁻⁸ · s` is the BF16 scale with its relative error
amplified by `|x/s| ≤ 127`, and `2⁻⁸ · |x̂|` is the BF16 decode that L1 attention
requires. `codec.verified_error_bound()` checks the bound elementwise against an exact
float64 quantizer, with zero violations on CPU, CUDA and MPS; the worst observed
`max |x̂−x|/s` on a Qwen3-8B-shaped 512-token batch was 0.934, inside the bound.

`1152 = 9 × 128` is the point of the layout: SGLang's CUDA HiCache mover is a byte
copier requiring `element_size % 128 == 0`, so a packed row rides the existing JIT
kernel with no CUDA changes. A BF16 row is 2048 bytes; a Qwen3-8B token is
36 layers × K,V × 1152 = 82,944 bytes against 147,456, or 1.78x the density.
[design.md](docs/design.md) has the derivation, alignment proof and stream safety.

## Limitations

- End-to-end generation quality is not established. `scripts/quality_compare.py`
  compares BF16 and INT8 generation for first-token agreement, sequence agreement
  and logprob deltas, but it could not force the L2 read path: HiCache retained the
  device copy in every configuration tested, so no restore occurred.
  `bf16_load_back_tokens` and `int8_load_back_tokens` are both 0 in
  `results/quality_INVALID_no_l2_restores.json`, so the agreement figures describe two cold prefills, and
  the script now refuses to report in that state (`no L2 restores for <tag>`). A
  smoke run did confirm the compressed path executes: 1,797 tokens loaded back from
  L2 with the tree node at `evicted=True backuped=True host_value=1797`, and
  82,944 B/token against the BF16 control's 147,456. How per-element error
  propagates through 36 layers of attention into token choice is not measured.
- Latency is not a stable result, because the BF16 baseline is bimodal. Across four
  paired runs at the same budget, three improved by 46 to 51% and one was 6% worse.
  INT8 was tight in all four (mean TTFT 1,202 to 1,259 ms) while BF16 ranged 1,193 to
  2,558 ms, which is the interesting part: the compressed tier is the predictable
  one. Quote the median pair (2,286 ms to 1,242 ms) or the range, never a single
  run's delta. One workload shape throughout: 32 shared-prefix groups × 2048 tokens
  at concurrency 8, L1 capped at 16,384 tokens. The crossover point where INT8's
  extra capacity stops helping is not characterised.
- Codec phase timing was not collected: the per-ack timing flag was unset, so the
  backup and restore duration histograms read 0.00 s. Byte and token volumes are
  unaffected, but no codec-overhead figure is claimed, so the codec stays pure
  `torch`. Per-phase timing is opt-in via `SGLANG_HICACHE_INT8_TIMING`.
- TP=1 with 8 KV heads only: the record is a fixed 1024-byte payload plus a 16-byte
  scale block sized for Qwen3-8B at TP=1, so at TP=2 (4 local heads = 512 + 8 bytes)
  the padding, alignment proof, capacity arithmetic and JIT geometry all change, and
  other local head counts are rejected at construction.

## Reproducing

[docs/reproducing-on-a-pod.md](docs/reproducing-on-a-pod.md) has the image, install
and measurement commands in order, with expected runtimes and a failure table.
Locally, no GPU:

```bash
.venv/bin/python -m pytest tests/ -q          # 332 tests, no GPU
.venv/bin/python scripts/capacity_table.py --hicache-size 8
.venv/bin/python scripts/conformance.py generate --device cpu
```

Tests find the fork checkout through the sibling `sglang/` directory
(`HIQCACHE_SGLANG_ROOT` overrides it), and `tests/test_codec_drift.py` pins the
fork's codec to this repo's reference implementation. Enable the cache with
`SGLANG_EXPERIMENTAL_HICACHE_INT8=1`; the server flags, the three-file diff and the
fail-fast matrix are in [docs/sglang-integration.md](docs/sglang-integration.md).
Upstream internals: [docs/hicache-internals.md](docs/hicache-internals.md). Image
matrix: [docs/image-compatibility.md](docs/image-compatibility.md).

## Licence

Apache-2.0. See `LICENSE`.
