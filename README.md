# HiQCache

**A quantized hierarchical KV cache for SGLang HiCache.**

SGLang's HiCache adds a CPU host tier (L2) behind the GPU device pool (L1), so
cached prefixes can outlive device memory instead of being recomputed. That tier
is bounded by host RAM, and every byte moved across PCIe is paid twice — once on
eviction and once on restore. HiQCache changes what an L2 slot *is*: active KV
stays BF16 in L1 so attention reads exactly the dtype it expects, but KV is
compressed to INT8 when it is evicted and decoded back to BF16 on an L2 hit. The
encoded record is 1152 bytes per `(layer, K|V, token)` row — a 1024-byte INT8
payload plus 8 BF16 per-head scales — against 2048 bytes for the BF16 row. On
Qwen3-8B the same host memory holds **1.78× more cached tokens** (54,254 →
96,451 at an 8 GB budget), or the same 54,254 tokens in **56.25% of the bytes**
(8.000 → 4.500 GB). SGLang's caching policy, radix tree, transfer engine and CUDA
kernels are unchanged; the fork adds three files and edits two lines.

```
L1 (GPU, BF16)             L2 (CPU, INT8 + BF16 scales)
┌──────────────┐           ┌───────────────────────────┐
│ K,V bf16     │  encode   │ 1152 B row per (layer,K|V)│
│ 144 KiB/token│ ────────► │  81 KiB/token             │
│              │ ◄──────── │  scale + payload + pad    │
└──────────────┘  decode   └───────────────────────────┘
      attention stays BF16        host memory −43.75%
```

Measured on Qwen3-8B at TP=1 on one NVIDIA RTX A6000 48 GB. Experiment B ran at
SGLang `3bb2ef6602` (pre-rewrite `9a7ac797`, preserved as the fork tag
`measured-exp-b`); Experiment A ran two instrumentation-only commits later at
`26e6d78dba` (pre-rewrite `38e366694d`). Full provenance, the commit-identity
rewrite note and the raw artifacts are in `docs/experiment-a-results.md`,
`docs/experiment-b-results.md` and `results/`.

## Results

### Experiment A — equal logical capacity

Both tiers pinned to the same **54,254** L2 tokens, so the comparison is bytes
rather than capacity:

| | BF16 | **INT8** |
| --- | --- | --- |
| L2 capacity (tokens) | 54,254 | **54,254** |
| L2 used (tokens) | 53,710 | 53,514 |
| measured bytes/token in L2 | 147,456 | **82,944** |
| **L2 bytes for that capacity** | **8,000,077,824 (8.000 GB)** | **4,500,043,776 (4.500 GB)** |
| tokens backed up (D2H) | 216,144 | 222,628 |
| tokens restored (H2D) | 122,678 | 126,979 |
| benchmark cache hit rate | 54.6% | **58.2%** |
| L2 evictions | 0 | 0 |

**The same logical cache, held in 56.25% of the host memory — 1.7778× denser.**
Both configs ran without L2 eviction, and each verified `target == achieved ==
54,254` against the pool's own gauge. The memory result is arithmetic; the
hit-rate difference (54.6% vs 58.2%) is a single observation, not an established
effect. Full record: `docs/experiment-a-results.md`.

### Experiment B — equal physical budget

Same `--hicache-size 8` for every configuration, so the result is how many tokens
fit and what that does to serving:

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
| output throughput | 80.0 tok/s | 105.1 tok/s | **127.8 tok/s** |

**Against BF16 at the same host memory: +16.6 cache-hit points, +32% tokens
served from L2, −25.9% mean TTFT, +21.6% output throughput.**

The capacity and bytes/token figures are exact. `hicache_host_total_tokens`
reports 54,254 and 96,451 — the two values the layout predicts for 8 GB — and
backup bytes divided by backup tokens is 147,456.00 and **82,944.00**, the layout
value to the byte over 31 GiB and 6.24 GiB of real traffic. The latency and
throughput columns are **single runs**; read the limitations in
`docs/experiment-b-results.md` before quoting them.

> **Which hit rate is which.** The figures above are the benchmark's own
> `Cache Hit Details` report. `sglang:cache_hit_rate` read 97.03% / 96.94% in the
> same Experiment A runs and is a *different quantity* — quoting it would
> overstate the hit rate and understate the difference. `scripts/analyse.py`
> reads serving figures from the benchmark report and uses the Prometheus
> counters only for byte arithmetic.

## Accuracy

The codec's accuracy **is** established, elementwise, against an exact float64
quantizer with zero bound violations on CPU and MPS. The bound the format
guarantees:

```
|x̂ − x|  ≤  (0.5 + 2⁻⁸) · s  +  2⁻⁸ · |x̂|
```

Each term is traceable to one design decision and asserted separately, so the
decomposition is proven rather than asserted as a whole:

| Term | Source |
| --- | --- |
| `0.5 · s` | round-to-nearest to INT8 |
| `2⁻⁸ · s` | the scale is BF16; relative error amplified by `\|x/s\| ≤ 127` |
| `2⁻⁸ · \|x̂\|` | the decoder emits BF16 (L1 attention consumes BF16) |

The third term dominates at large magnitudes. The naive `error ≤ scale/2` claim
holds only for the exact product `q · s`; see `docs/design.md` for the derivation
and `verified_error_bound()` for the implementation.

Measured on a Qwen3-8B-shaped 512-token batch:

| vector | max \|x̂−x\|/s | mean \|x̂−x\| | p99 |
| --- | --- | --- | --- |
| `random_unit` | 0.934 | 0.0054 | 0.0156 |
| `qwen3_8b_shaped` | 0.934 | 0.0054 | 0.0156 |
| `random_large` | 0.928 | 5.36 | 16.0 |
| `all_zeros` | 0.000 | 0 | 0 |
| `exact_grid` | 0.000 | 0 | 0 |

## Limitations

* **End-to-end generation quality is not established.** The quality harness
  (`scripts/quality_compare.py`) is written to compare BF16 and INT8 generation,
  first-token and sequence agreement, and logprob deltas — but it could not force
  the L2 restore path. In the recorded run (`results/quality.json`)
  `bf16_load_back_tokens` and `int8_load_back_tokens` are both **0**, so the
  re-sent prompts were re-prefilled rather than decoded from the codec, and the
  agreement figures describe two cold prefills. The harness now refuses to report
  in that state (`no L2 restores for <tag>`), because a vacuous comparison is
  worse than none. What *is* established at the mechanism level is that the
  compressed path executes: a smoke run observed 1,797 tokens loaded back from L2
  with the tree node in state `evicted=True backuped=True host_value=1797`, and
  82,944 B/token stored against the BF16 control's 147,456 B/token. The
  quantizer's per-element error is bounded as above; how that error propagates
  through 36 layers of attention into token choice is **not** measured here.
* **Latency and throughput are single runs.** Capacity and bytes/token are
  arithmetic identities and deterministic; TTFT, TPOT and throughput are one
  sample per configuration. TTFT/ITL at p99 are spiky on this hardware, so the
  latency deltas need the repeat sweep before they are quoted as established.
* **Backup/restore durations read 0.00 s.** The per-ack timing flag was not set,
  so the duration histograms stayed empty. Byte and token volumes are unaffected,
  but no codec-overhead figure is claimed; per-phase codec timing exists and is
  opt-in via `SGLANG_HICACHE_INT8_TIMING`.
* **One workload shape.** 32 shared-prefix groups × 2048 tokens at concurrency 8,
  L1 capped at 16,384 tokens. The crossover point where INT8's extra capacity
  stops helping is not characterised.
* **TP=1, 8 KV heads only.** V1 is a fixed 1024-byte payload plus a 16-byte scale
  block, sized for Qwen3-8B at TP=1. At TP=2 (4 local heads = 512 + 8 bytes) the
  padding, alignment proof, capacity arithmetic and JIT geometry all change
  together, so any other local head count is rejected at construction with a
  message that names the restriction. TP-sharded record formats are future work.
* **No Triton kernels.** The codec is pure `torch`. Fused kernels are gated on
  the codec-timing result; writing them before knowing whether codec compute
  matters next to PCIe transfer would be unjustified.

## Reproducing

* **On a GPU pod:** `docs/reproducing-on-a-pod.md` — the exact image, clone,
  install, verification and measurement commands in order, with expected runtimes
  and output for each step, plus a failure/diagnosis table.
* **Design and rationale:** `docs/design.md` — record layout, quantisation
  scheme, error-bound derivation, capacity arithmetic, the JIT-mover geometry
  argument and the stream-safety rules.
* **Upstream internals:** `docs/hicache-internals.md` — a trace of how SGLang's
  HiCache device pool, host pool, eviction/restore path and stream semantics work,
  with file and line references.
* **Fork diff:** `docs/sglang-integration.md`; **image/version matrix:**
  `docs/image-compatibility.md`.

### Locally, with no GPU

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch pytest numpy

.venv/bin/python -m pytest tests/ -q          # 332 tests, no GPU
.venv/bin/python scripts/capacity_table.py --hicache-size 8
.venv/bin/python scripts/conformance.py generate --device cpu
```

Tests that target the fork's own modules find the checkout automatically from the
sibling `sglang/` directory; override with `HIQCACHE_SGLANG_ROOT=/path/to/sglang`.
They import the **real** fork sources, so there is no copy to keep in sync — and
`tests/test_codec_drift.py` additionally pins the fork's self-contained codec to
this repo's reference implementation, bit for bit.

The codec is pure `torch` with **no SGLang imports and no CUDA-only ops**, so the
same bytes are produced on a laptop and on a rented GPU, and the harness proves
it: the CPU digest is committed under `results/`, and the pod run checks the CUDA
digest against it before any SGLang work starts. A mismatch localises the bug to
the codec or a backend op in seconds instead of after a failed multi-hour run.

## The format

One aligned record per `(layer, K|V, token)` row:

```
  0 ─────────────────────────────── 1023   1024 INT8 payload  (8 heads × 128)
1024 ─────────────────────────────── 1039   8 BF16 per-head scales
1040 ─────────────────────────────── 1151   112 bytes padding
```

`1152 = 9 × 128` is deliberate. SGLang's CUDA HiCache mover is a pure element-wise
byte copier requiring `element_size % 128 == 0`, so a packed row rides the existing
JIT kernel with **zero CUDA changes**.

| | bytes/token | vs baseline |
| --- | --- | --- |
| BF16 baseline | 147,456 | 1.00× |
| HiQCache | 82,944 | **1.78×** |

Because the kernels are byte movers rather than quantizers, a BF16 device row
cannot be written directly into a smaller INT8 host row: `transfer_hicache_all_layer`
takes separate source/destination strides but a single `element_size`, so both
sides of a move must have the same byte width. Compression therefore runs through
persistent GPU staging buffers, one set per transfer direction, with the codec
enqueued on the existing `device_to_host_stream` / `host_to_device_stream` — no
second synchronization system, no host syncs, and decode + scatter completed
before `on_layer_done` fires. `docs/design.md` gives the full argument.

## SGLang integration

Three new files and two edited lines in the fork. Nothing else changes — no
`l2_transfer.py`, no `cache_controller.py`, no radix tree, no CUDA sources.

| Path (in the fork) | Purpose |
| --- | --- |
| `mem_cache/pool_host/int8_codec.py` | INT8 record quantise / pack / decode |
| `mem_cache/pool_host/int8_staging.py` | device staging buffers + pointer tables |
| `mem_cache/pool_host/mha_int8.py` | `MHATokenToKVPoolHostINT8` |
| `mem_cache/pool_host/mha.py` | one dispatch branch in `get_mha_host_pool_cls` |
| `srt/environ.py` | `SGLANG_EXPERIMENTAL_HICACHE_INT8` and staging size |

```bash
SGLANG_EXPERIMENTAL_HICACHE_INT8=1 \
python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
  --enable-hierarchical-cache --hicache-mem-layout layer_first \
  --hicache-size 8 --page-size 1 --tp-size 1
```

Unsupported configurations are refused at construction rather than failing later
as an opaque payload/geometry mismatch — the full fail-fast matrix is in
`docs/sglang-integration.md`.

## Repository layout

```
src/hiqcache/
  layout.py     byte arithmetic, alignment proofs, compression math
  codec.py      quantise / pack / unpack / dequantise + error accounting
tests/          332 tests: layout, codec, capacity, device parity,
                fork codec, fork staging buffers, fork/reference drift guard
scripts/        capacity table, conformance harness, pod bootstrap, smoke test,
                experiment driver, analysis, quality comparison
docs/           design, internals, integration, pod reproduction, results
results/        conformance manifests, benchmark JSON, server logs, provenance
```
