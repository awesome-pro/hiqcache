# HiQCache

**Quantized hierarchical KV cache for SGLang HiCache.**

Keep active L1 KV in normal BF16 for attention, but compress KV when it is evicted
to SGLang's CPU L2 cache, then restore it to BF16 on an L2 hit. Measure the
capacity, transfer, latency and quality trade-off on Qwen3-8B.

```
L1 (GPU, BF16)             L2 (CPU, INT8 + BF16 scales)
┌──────────────┐           ┌───────────────────────────┐
│ K,V bf16     │  encode   │ 1152 B row per (layer,K|V)│
│ 144 KiB/token│ ────────► │  81 KiB/token             │
│              │ ◄──────── │  scale + payload + pad    │
└──────────────┘  decode   └───────────────────────────┘
      attention stays BF16        host memory −43.75%
```

## Status

| Phase | State |
| --- | --- |
| 0–2 Encoded representation, codec, capacity maths | **done** — 311 tests, CPU + MPS bit-identical |
| 3–9 `MHATokenToKVPoolHostINT8`, staging, dispatch, fail-fast | **done** — 36/36 pool tests pass on GPU |
| 10 Kernel-level validation | **done** — JIT at 1152 B, `cudaHostRegister`, pointer-table move, H2D byte copy |
| 11 Integration smoke | **done** — compressed L2 write *and* L2→L1 restore verified |
| 12–13 Benchmarks | **Experiment B done**; Experiment A (equal logical capacity) outstanding |
| 14 Metrics | bytes/capacity exact; codec phase timing instrumented, not yet collected |
| 15 Quality | harness written, **not yet run** |
| 16–17 Torch codec / Triton | torch throughout; Triton gated on the timing result |
| 18–20 Analysis, repo, demo | in progress |

Measured on an RTX A6000 at SGLang `9a7ac7978f49`; full provenance and
limitations in `docs/experiment-b-results.md`.

**Read that document before quoting any number.** The capacity, byte and
cache-hit figures are exact and deterministic. The latency and throughput
figures are single runs and still need the repeats described there.

## The format## The format

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

Under a fixed `--hicache-size=8` (decimal GB), that is **54,254 → 96,451 tokens**
of L2 capacity — +42,197 tokens for the same memory.

**V1 intentionally targets Qwen3-8B at TP=1.** The record is a *fixed* 1024-byte
payload plus 16 scale bytes, sized for 8 local KV heads. At TP=2 there are 4 local
heads (512 + 8 bytes), so the padding, the alignment proof, the capacity maths and
the JIT geometry all change with it. Any other local head count is rejected at
construction with a message that names the restriction, rather than failing later
as an opaque payload/geometry mismatch. TP-sharded record formats are future work.

## Results

Qwen3-8B, 1x RTX A6000 48 GB, SGLang `9a7ac7978f49`, same physical CPU budget
(`--hicache-size 8`) for both cache tiers. Workload: 32 shared-prefix groups x
2048 tokens = 65,536 reusable prefix tokens.

| metric | baseline (no L2) | BF16 HiCache | **HiQCache INT8** |
| --- | --- | --- | --- |
| L2 capacity (tokens) | 0 | 54,254 | **96,451** |
| measured B/token in L2 | – | 147,456 | **82,944** |
| cache hit rate | 12.2% | 56.1% | **72.7%** |
| tokens served from L2 | 0 | 126,997 | **167,899** |
| TTFT mean | 1,643 ms | 1,244 ms | **922 ms** |
| output throughput | 80.0 tok/s | 105.1 tok/s | **127.8 tok/s** |

vs BF16 at the same host memory: **+16.6 cache-hit points, +32% tokens served
from L2, -25.9% mean TTFT, +21.6% throughput.**

The capacity and bytes/token figures are exact: `hicache_host_total_tokens`
reports 54,254 and 96,451 (the values the layout predicts), and backup bytes
divided by backup tokens is 147,456.00 and **82,944.00** respectively. The
latency and throughput deltas are **single runs and still need repeats** --
see `docs/experiment-b-results.md` for the full table and the limitations.

## Accuracy

The bound the format actually guarantees, verified elementwise with zero
violations:

```
|x̂ − x|  ≤  (0.5 + 2⁻⁸) · s  +  2⁻⁸ · |x̂|
```

The three terms are each traceable to one decision, and each is asserted
separately against an exact float64 quantizer:

| Term | Source |
| --- | --- |
| `0.5 · s` | round-to-nearest to INT8 |
| `2⁻⁸ · s` | the scale is BF16; relative error amplified by `\|x/s\| ≤ 127` |
| `2⁻⁸ · \|x̂\|` | the decoder emits BF16 (L1 attention consumes BF16) |

The third term dominates at large magnitudes. The naive `error ≤ scale/2` claim
holds only for the exact product `q · s`; see `PROJECT.md` Phase 2 for the
correction and `verified_error_bound()` for the derivation.

Measured, on a Qwen3-8B-shaped 512-token batch:

| vector | max \|x̂−x\|/s | mean \|x̂−x\| | p99 |
| --- | --- | --- | --- |
| `random_unit` | 0.934 | 0.0054 | 0.0156 |
| `qwen3_8b_shaped` | 0.934 | 0.0054 | 0.0156 |
| `random_large` | 0.928 | 5.36 | 16.0 |
| `all_zeros` | 0.000 | 0 | 0 |
| `exact_grid` | 0.000 | 0 | 0 |

## Layout

```
src/hiqcache/
  layout.py     byte arithmetic, alignment proofs, compression math
  codec.py      quantise / pack / unpack / dequantise + error accounting
tests/          311 tests: layout, codec, capacity, device parity,
                fork codec, fork staging buffers, fork/reference drift guard
scripts/
  conformance.py     cross-device digest harness (Mac ↔ pod)
  capacity_table.py  exact reproduction of SGLang's host-pool sizing
docs/               verification notes, source maps, pod runbook
results/            conformance manifests, benchmark output
```

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

Full source map, data flow and fail-fast matrix: `docs/sglang-integration.md`.

## Reproduce (no GPU needed)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch pytest numpy

.venv/bin/python -m pytest tests/ -q          # 311 tests, no GPU
.venv/bin/python scripts/capacity_table.py --hicache-size 8
.venv/bin/python scripts/conformance.py generate --device cpu
```

Tests that target the fork's own modules find the checkout automatically from
the sibling `sglang/` directory; override with `HIQCACHE_SGLANG_ROOT=/path/to/sglang`.
They import the **real** fork sources, so there is no copy to keep in sync — and
`tests/test_codec_drift.py` additionally pins the fork's self-contained codec to
this repo's reference implementation, bit for bit.

## The Mac ↔ pod contract

The codec is pure `torch` with **no SGLang imports and no CUDA-only ops**, so the
same bytes are produced on a laptop and on a rented GPU. The harness proves it:

```bash
# On the Mac (already committed):
python scripts/conformance.py generate --device cpu --json results/conformance_cpu.json

# On the pod, before touching SGLang:
python scripts/conformance.py generate --device cuda --expect results/conformance_cpu.json
```

A digest mismatch means the bug is in the codec or a backend op — find it in
seconds instead of after a failed multi-hour SGLang run. See `docs/pod-guide.md` (step by step) and `docs/pod-runbook.md` (reference).

## Reproduce (GPU)

See `docs/pod-guide.md` (step by step) and `docs/pod-runbook.md` (reference). Requires the SGLang fork at
`515f5be77e74761c269e007ac41a5895191a1b7d`.

## References

- `PROJECT.md` — the full 20-phase project guide, with Phase 2 corrections
- `sglang-hicache-trace.md` — HiCache internals trace at commit `434c2e3a`
- `docs/verification-notes.md` — facts verified against the fork at `515f5be77e`
