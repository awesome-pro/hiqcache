# HiQCache

**Quantized hierarchical KV caching for SGLang HiCache.**

HiQCache keeps active KV in BF16 on the GPU, but stores evicted KV in an INT8
representation inside SGLang's CPU HiCache tier. On an L2 hit, the cached KV is
restored to BF16 before attention consumes it.

This changes the representation stored in L2 without changing SGLang's caching
policy, radix tree, transfer engine, or existing CUDA HiCache mover.

## Key results

Measured with **Qwen3-8B, TP=1 on one NVIDIA RTX A6000 48 GB**:

- **43.75% less host KV memory per token:** 147,456 → 82,944 bytes
- **1.78x L2 capacity at the same host-memory budget:** 54,254 → 96,451 tokens
- In repeated fixed-budget runs:
  - BF16 HiCache: ~57% cache-hit rate
  - HiQCache INT8: ~73% cache-hit rate
- HiQCache restored roughly **168K tokens from L2** versus ~125–127K with BF16
  in the repeated fixed-budget workload
- Real GPU → CPU → GPU compressed cache restores were validated end-to-end
- The codec's elementwise numerical error bound was verified on CPU, CUDA, and MPS

The main result is the **capacity/reuse effect**. Serving latency improved
substantially in 3 of 4 paired measurements and was near parity/slightly worse
in one, so latency is reported as a secondary result rather than as a universal
performance claim.

---

## Results

### Experiment A: equal logical capacity

Both configurations were pinned to the same **54,254-token L2 capacity**.

This isolates the amount of host memory required to hold the same logical cache.

| Metric | BF16 HiCache | HiQCache INT8 |
| --- | ---: | ---: |
| L2 capacity | 54,254 tokens | 54,254 tokens |
| L2 used | 53,710 | 53,514 |
| Measured L2 bytes/token | 147,456 | **82,944** |
| Host memory for that capacity | 8,000,077,824 B | **4,500,043,776 B** |
| Host memory | 8.000 GB | **4.500 GB** |
| Tokens backed up to L2 | 216,144 | 222,628 |
| Tokens restored from L2 | 122,678 | 126,979 |
| Benchmark cache-hit rate | 54.6% | 58.2% |
| L2 evictions | 0 | 0 |

The same logical cache fits in **56.25% of the host memory**:

- **43.75% fewer bytes**
- **1.7778x storage density**

Both configurations reported:

```text
target L2 capacity == achieved L2 capacity == 54,254 tokens
```

The storage result is determined directly by the encoded representation.
The cache-hit-rate difference above is from one run and is not presented as an
established performance effect.

Full experiment:
[docs/experiment-a-results.md](docs/experiment-a-results.md)

---

### Experiment B: equal physical budget

Both HiCache configurations used the same:

```text
--hicache-size 8
```

| Metric | No L2 | BF16 HiCache | HiQCache INT8 |
| --- | ---: | ---: | ---: |
| L2 capacity | 0 | 54,254 | **96,451** |
| L2 tokens used | 0 | 53,585 (98.8%) | 80,833 (83.8%) |
| Measured L2 bytes/token |:  | 147,456 | **82,944** |
| Backup bytes | 0 | 31.08 GiB | **6.24 GiB** |
| Restore bytes | 0 | 17.44 GiB | **12.97 GiB** |
| Cache-hit rate | 12.2% | 56.1% | **72.7%** |
| Tokens served from L2 | 0 | 126,997 | **167,899** |
| Mean TTFT | 1,643.0 ms | 1,244.3 ms | 922.3 ms |
| p99 TTFT | 3,626.3 ms | 3,450.8 ms | 3,369.6 ms |
| Mean TPOT | 47.9 ms | 37.0 ms | 33.3 ms |
| Output throughput | 80.0 tok/s | 105.1 tok/s | 127.8 tok/s |

In this representative run, compared with BF16 HiCache at the same host-memory
budget, HiQCache:

- increased L2 capacity from **54,254 → 96,451 tokens**
- improved cache hit rate by **16.6 percentage points**
- served **32% more tokens from L2**
- reduced mean TTFT by 25.9%
- increased output throughput by 21.6%

The capacity and byte-density results are exact properties of the two
representations.

The serving-latency result is more variable.

Across four paired fixed-budget measurements:

```text
BF16 cache hit rate:
56.8% – 57.6%

HiQCache cache hit rate:
72.7% – 73.4%
```

Mean TTFT:

```text
BF16:
1,193 – 2,558 ms

HiQCache:
1,202 – 1,259 ms
```

Three of the four paired runs showed substantial TTFT improvement; one was near
parity/slightly worse.

HiQCache therefore showed substantially lower run-to-run TTFT variance in these
measurements, but the project does **not** claim a universal latency speedup.

Full experiment and all repeated measurements:
[docs/experiment-b-results.md](docs/experiment-b-results.md)

---

## How it works

For Qwen3-8B at TP=1:

```text
36 layers
8 KV heads
128 dimensions/head
BF16 L1 KV
```

A normal BF16 K or V row contains:

```text
8 heads × 128 values × 2 bytes
= 2048 bytes
```

HiQCache stores that row in a fixed **1152-byte encoded record**:

```text
L1:  GPU, BF16                     L2:  CPU, INT8 + BF16 scales, 82,944 B/token

┌──────────────────────┐             ┌────────────────────────────────────┐
│ K,V bf16             │ ──encode──► │   0 .. 1023  1024 B INT8 (8 x 128) │
│ 2048 B per row       │             │ 1024 .. 1039  16 B: 8 BF16 scales  │
│ 147,456 B per token  │ ◄─decode─── │ 1040 .. 1151  112 B padding        │
└──────────────────────┘             └────────────────────────────────────┘
```

---

## Quantization

Quantization is symmetric and per KV head.

For each 128-element head:

```text
s = max(|x|) / 127
q = round(x / s)
```

with values clamped to signed INT8.

Restoration is:

```text
x̂ = q · s
```

and the result is emitted as BF16 because the normal L1 attention path expects
BF16 KV.

The 1152-byte record size is deliberate:

```text
1152 = 9 × 128
```

SGLang's existing HiCache CUDA mover is a byte-copy kernel whose transfer element
must satisfy the required alignment. Packing the INT8 payload, BF16 scales, and
padding into one aligned record lets HiQCache continue using the existing mover
without modifying its CUDA source.

Detailed layout derivation, transfer geometry, and stream-safety constraints:
[docs/design.md](docs/design.md)

---

## Transfer path

HiQCache compresses only the host-cache representation.

### L1 → L2

```text
BF16 device KV
      ↓
quantize + pack on GPU
      ↓
encoded GPU staging
      ↓
existing HiCache CUDA byte mover
      ↓
compressed pinned CPU L2
```

### L2 → L1

```text
compressed pinned CPU L2
      ↓
existing HiCache CUDA byte mover
      ↓
encoded GPU staging
      ↓
unpack + dequantize
      ↓
ordinary BF16 device KV
      ↓
attention
```

The existing SGLang scheduler, radix cache, L2 transfer engine, and layer-loading
event semantics remain responsible for cache policy and synchronization.

HiQCache changes the representation stored in L2 rather than introducing a new
cache policy.

---

## Codec numerical correctness

The encoded format has a verified elementwise error bound:

```text
|x̂ − x| ≤ (0.5 + 2⁻⁸) · s + 2⁻⁸ · |x̂|
```

The terms correspond to:

| Term | Source |
| --- | --- |
| `0.5 · s` | round-to-nearest INT8 quantization |
| `2⁻⁸ · s` | BF16 representation of the scale |
| `2⁻⁸ · |x̂|` | BF16 decode into the L1 representation |

`codec.verified_error_bound()` checks the implementation elementwise against an
exact float64 reference quantizer.

All tested elements satisfied the bound on:

- CPU
- CUDA
- Apple MPS

For a Qwen3-8B-shaped 512-token batch, the largest observed normalized error was:

```text
|x̂ − x| / s = 0.934
```

Example measurements:

| Input | max `|x̂−x|/s` | Mean absolute error | p99 |
| --- | ---: | ---: | ---: |
| `random_unit` | 0.934 | 0.0054 | 0.0156 |
| `qwen3_8b_shaped` | 0.934 | 0.0054 | 0.0156 |
| `random_large` | 0.928 | 5.36 | 16.0 |
| `all_zeros` | 0.000 | 0 | 0 |
| `exact_grid` | 0.000 | 0 | 0 |

---

## Validation

The implementation was tested at several levels.

### Representation

The host-cache byte accounting matches the layout exactly:

```text
BF16:
147,456.00 measured bytes/token

HiQCache:
82,944.00 measured bytes/token
```

These values were observed from real HiCache backup traffic rather than inferred
only from configuration.

### Real L2 restoration

The compressed path was exercised end-to-end on an RTX A6000.

One smoke test observed:

```text
1,797 tokens loaded back from L2
tree state:
evicted=True
backuped=True
host_value=1797
```

The INT8 host pool reported:

```text
82,944 B/token
```

while the BF16 control reported:

```text
147,456 B/token
```

The restored request completed through the normal BF16 attention path.

### Fixed-budget serving

Repeated Experiment B runs reproduced the capacity and cache-reuse effect:

```text
BF16:
~57% cache-hit rate

HiQCache:
~73% cache-hit rate
```

under the same 8 GB host-memory budget.

---

## Limitations

### End-to-end model quality

End-to-end generation-quality impact after a lossy L2 restore has **not** been
established.

`scripts/quality_compare.py` was built to compare:

- first-token agreement
- generated-token sequence agreement
- output-logprob deltas

between BF16 and INT8 restoration.

However, the recorded quality experiment did not successfully exercise the L2
read path:

```text
bf16_load_back_tokens = 0
int8_load_back_tokens = 0
```

so those requests were cold-prefilled rather than restored through the codec.

The invalid result is preserved as:

```text
results/quality_INVALID_no_l2_restores.json
```

and the harness now refuses to report quality results when no L2 restoration is
observed.

What **is** established:

- real compressed D2H/H2D cache restoration executes
- the encoded representation is measured at 82,944 B/token
- the elementwise codec error bound is verified

What is **not** established:

> how that numerical error propagates through Qwen3-8B's 36 transformer layers
> into final token choices or downstream benchmark accuracy.

---

### Serving latency

Serving latency is not presented as a stable universal improvement.

Across four paired Experiment B measurements:

```text
INT8 mean TTFT:
1,202 – 1,259 ms

BF16 mean TTFT:
1,193 – 2,558 ms
```

Three paired runs favored HiQCache substantially, while one was near
parity/slightly worse.

The stable result is the increased host-cache capacity and cache reuse, not a
specific latency percentage.

The workload used:

```text
32 shared-prefix groups
~2048-token configured shared-prefix target
concurrency 8
L1 capped at 16,384 tokens
```

The workload crossover point at which extra capacity ceases to compensate for
codec overhead has not been characterized.

---

### Codec phase timing

Separate encode/decode phase timing was not successfully collected.

The benchmark therefore does not make claims such as:

- codec overhead is negligible
- quantization is faster than PCIe transfer
- Triton fusion would or would not improve performance

The codec remains implemented with PyTorch GPU operations.

A fused Triton implementation was intentionally not added without profiling
evidence that the codec itself is the dominant bottleneck.

---

### Scope

The current encoded format targets:

```text
Qwen3-8B
TP = 1
8 local KV heads
head_dim = 128
page_size = 1
BF16 L1
MHA/GQA
```

The fixed record contains:

```text
1024 B INT8 payload
16 B BF16 scales
112 B alignment padding
```

At TP=2, only four KV heads would be local, so the payload, scale block,
alignment, storage arithmetic, and transfer geometry would all change.

Unsupported local-head geometries therefore fail at construction rather than
silently using the TP=1 format.

HiQCache does not currently target:

- TP > 1
- MLA
- SSM / Mamba
- page sizes > 1
- L3 / remote HiCache storage
- P/D disaggregation
- speculative-draft cache pools
- ROCm / NPU-specific integration

---

## Reproducing

Full GPU setup and experiment instructions:

[docs/reproducing-on-a-pod.md](docs/reproducing-on-a-pod.md)

The main experiments are reproducible from the scripts in this repository and
the accompanying SGLang fork.

For local tests without a GPU:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/capacity_table.py --hicache-size 8
.venv/bin/python scripts/conformance.py generate --device cpu
```

Tests locate the SGLang fork through a sibling `sglang/` checkout by default.

Override it with:

```bash
export HIQCACHE_SGLANG_ROOT=/path/to/sglang
```

Enable the experimental host representation with:

```bash
export SGLANG_EXPERIMENTAL_HICACHE_INT8=1
```

Integration details and fail-fast restrictions:

[docs/sglang-integration.md](docs/sglang-integration.md)

SGLang HiCache source trace:

[docs/hicache-internals.md](docs/hicache-internals.md)

Image/environment compatibility:

[docs/image-compatibility.md](docs/image-compatibility.md)

---

## Experimental provenance

Serving experiments used **Qwen3-8B at TP=1 on one NVIDIA RTX A6000 48 GB**.

Experiment A and Experiment B were measured from pinned SGLang fork revisions.
Some commit identifiers were rewritten later while cleaning the development
history; the raw experiment artifacts were preserved unchanged.

The exact old/new commit mapping and software/hardware provenance are recorded in:

- [docs/experiment-a-results.md](docs/experiment-a-results.md)
- [docs/experiment-b-results.md](docs/experiment-b-results.md)
- [`results/`](results/)

The raw result files are the source of truth for reported measurements.

---

## Licence

Apache-2.0. See [LICENSE](LICENSE).
