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

| Phase | Where | State |
| --- | --- | --- |
| Encoded representation (format, alignment) | Mac | **done** |
| Standalone codec + tests | Mac (CPU + MPS) | **done** — 172 tests |
| Cross-device conformance harness | Mac | **done** — CPU ≡ MPS bit-identical |
| `MHATokenToKVPoolHostINT8` + SGLang integration | GPU pod | not started |
| Benchmarks, quality, analysis | GPU pod + Mac | not started |

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

Under a fixed `--hicache-size=8` (decimal GB), that is **54,254 → 96,451 tokens**
of L2 capacity — +42,197 tokens for the same host memory.

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
tests/          172 tests: layout, codec, capacity, device parity
scripts/
  conformance.py     cross-device digest harness (Mac ↔ pod)
  capacity_table.py  exact reproduction of SGLang's host-pool sizing
docs/               verification notes, source maps
results/            conformance manifests, benchmark output
```

## Reproduce (no GPU needed)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch pytest numpy

.venv/bin/python -m pytest tests/ -q          # 172 tests
.venv/bin/python scripts/capacity_table.py --hicache-size 8
.venv/bin/python scripts/conformance.py generate --device cpu
```

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
seconds instead of after a failed multi-hour SGLang run. See `docs/pod-runbook.md`.

## Reproduce (GPU)

See `docs/pod-runbook.md`. Requires the SGLang fork at
`515f5be77e74761c269e007ac41a5895191a1b7d`.

## References

- `PROJECT.md` — the full 20-phase project guide, with Phase 2 corrections
- `sglang-hicache-trace.md` — HiCache internals trace at commit `434c2e3a`
- `docs/verification-notes.md` — facts verified against the fork at `515f5be77e`
