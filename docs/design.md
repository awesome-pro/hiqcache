# HiQCache design

HiQCache is an INT8-quantised L2 tier for SGLang's HiCache. Active KV stays BF16
in the GPU device pool (L1) so attention reads exactly the dtype it expects; KV is
compressed only when it is evicted to the CPU host pool (L2), and is restored to
BF16 when L2 is read back. The caching policy, the radix tree, the eviction logic
and the CUDA kernels are all unchanged — only the *representation* of an L2 slot
changes.

This document records the design decisions and the reasoning behind them. For
the upstream internals the design builds on, see `hicache-internals.md`; for the
exact fork diff, see `sglang-integration.md`.

---

## 1. Interception point

Every L1↔L2 byte in HiCache funnels through two host-pool methods
(`l2_transfer.py:136` and `:169`):

```
L2TransferEngine.submit_device_to_host -> host_pool.backup_from_device_all_layer_physical
L2TransferEngine.submit_host_to_device -> host_pool.load_to_device_per_layer_physical
```

`HostKVCache` forwards the `_physical` variants to the plain methods
(`base.py:371`, `:378`), so a subclass that overrides those two intercepts 100% of
the traffic. The radix tree, the eviction policy and the scheduler never inspect
KV bytes — they allocate and free *slots* — so reducing bytes per slot is
invisible to them. That is why the fork adds three files and touches two existing
lines rather than reworking the transfer engine.

The transform is **row-local**: an encoded record depends only on the BF16 values
of one `(token, layer, K|V)` row. It needs no sequence metadata, no cross-token
state and no calibration pass, which is what makes it usable as a drop-in L2
representation.

---

## 2. Encoded record layout

One aligned record per `(layer, K|V, token)` row:

```
   0 ─────────────────────────────── 1023    1024 INT8 payload  (8 heads × 128 dims)
1024 ─────────────────────────────── 1039    8 BF16 per-head scales (16 B)
1040 ─────────────────────────────── 1151    112 bytes padding
```

| quantity | value |
| --- | --- |
| row bytes | 1152 |
| BF16 row for the same data | 2048 |
| K + V per layer per token | 2304 encoded, 4096 baseline |
| storage reduction | 43.75% |
| compression ratio | 1.7778× |
| Qwen3-8B (36 layers) | 82,944 B/token encoded, 147,456 B/token baseline |

The 1024-byte payload follows from the geometry: `head_num × head_dim × 1 byte =
8 × 128 = 1024`. The 16-byte scale block holds one BF16 scale per KV head.

**The padding is intentional and load-bearing.** `1152 = 9 × 128` is chosen so the
record is a whole number of 128-byte groups, which is exactly what SGLang's CUDA
HiCache mover requires (section 7). The 112 unused bytes are not waste to be
recovered later: shrinking the row below a 128-byte multiple would take the
encoded arena off the existing JIT kernel and force a CUDA change. The layout
also reserves a zero-byte header region inside the padding for future per-row
metadata, so adding it would not change `ROW_BYTES` or the capacity arithmetic.

---

## 3. Quantisation scheme

Symmetric, per-head, absmax INT8, computed independently for every
`(token, layer, K|V, KV head)`:

```
s = max(|x|) / 127
q = clamp(round(x / s), -127, 127)
x̂ = q · s
```

Design choices and why:

* **Per-head absmax, not per-row or per-tensor.** One scale per head costs 16
  bytes per row and keeps the transform row-local. A per-row scale would save 14
  bytes and coarsen the quantiser for heads with small dynamic range; a
  per-tensor or per-channel scale would need cross-token state, i.e. calibration
  data the eviction path does not have.
* **`q ∈ [-127, 127]`, never -128.** The range stays symmetric, so sign handling
  cannot drift and the clamp is one comparison. The unused code costs 0.8% of the
  codebook.
* **The quotient is formed in float32.** BF16 division near full scale has ~0.25
  of a scale step of granularity (8 mantissa bits at a magnitude of 127), which
  pushes the normalised reconstruction error from `≤ 0.5` to `~0.75`. Widening
  the division restores the round-to-nearest guarantee.
* **The clamp to 127 is mathematically required, not defensive.** Because `s`
  itself is rounded to BF16, `|x/s|` can reach `127·(1 + 2⁻⁸) < 128`; without the
  clamp the codec could emit -128 from a positive input.
* **Safe nonzero scale for an all-zero head.** `AMAX_FLOOR = 2⁻¹¹²` is applied to
  the absmax (not the scale) before dividing, giving `SCALE_FLOOR = 2⁻¹¹²/127`.
  A power of two is chosen so the clamp is *exact* and the same bits are produced
  on CPU, MPS and CUDA — a clamp against a non-representable value would round
  differently per backend and make cross-device conformance testing meaningless.
  The floor is deliberately tiny: `SCALE_FLOOR` is still a normal BF16 (normals
  start at `2⁻¹²⁶`), and since `2⁻¹¹² / s_floor = 127` exactly, a head whose
  absmax is below the floor quantises to all zeros and decodes back to exact
  zero — the correct answer for negligible input.

---

## 4. Error bound

The naive claim `error ≤ s/2` holds only for the *exact* product `q · s`. Two
representation choices the format forces break it, and both were confirmed by
measurement:

1. **The scale is BF16.** Its relative error is up to `2⁻⁸`, amplified by
   `|x/s| ≤ 127`, contributing up to `127 · 2⁻⁸ · s ≈ 0.5 · s`.
2. **The decoder emits BF16**, because L1 attention consumes BF16, so the final
   product is rounded again. This term, `2⁻⁸ · |x̂|`, **dominates at large
   magnitudes** and reaches ~0.93 · s in units of the scale.

The bound the format actually guarantees:

```
|x̂ − x|  ≤  (0.5 + 2⁻⁸) · s  +  2⁻⁸ · |x̂|
```

| term | source |
| --- | --- |
| `0.5 · s` | round-to-nearest to INT8 |
| `2⁻⁸ · s` | the scale is BF16; relative error amplified by `\|x/s\| ≤ 127` |
| `2⁻⁸ · \|x̂\|` | the decoder emits BF16 |

The bound is implemented as `codec.verified_error_bound()`, and the numerical
tests assert **each term separately against an exact float64 quantiser**, so the
decomposition is proven rather than asserted as a whole. It has been verified
elementwise on CPU and MPS with zero violations; on a Qwen3-8B-shaped batch the
observed `max |x̂ − x| / s` is 0.934, consistent with the `0.5 + 2⁻⁸` coefficient
plus BF16 output rounding.

---

## 5. Capacity arithmetic

`HostKVCache` sizes the host pool from a per-token byte cost that the pool class
reports. Overriding that one number is what converts compression into capacity:

```
get_size_per_token() = 2 × layer_num × row_bytes
                     = 2 × 36 × 1152
                     = 82,944 B/token          (BF16 pool: 147,456 B/token)
```

SGLang then computes (`pool_host/base.py:193-208`, mirrored byte for byte by
`scripts/capacity_table.py`):

```python
if host_size > 0:                                   # --hicache-size <decimal GB>
    size = int(host_size * 1e9 // size_per_token)
else:                                               # --hicache-ratio <float>
    size = int(device_capacity * host_to_device_ratio)
page_num = size // page_size + 1                    # round up one page
size = page_num * page_size
```

Two consequences matter for reproducing results:

* `--hicache-size` is an **integer** number of decimal GB. A fractional value is
  rejected by argparse and the server exits with code 2 before loading the model.
* The `+1` page rounding is why the numbers are not the raw quotients:

| budget | pool | `size_per_token` | floor | +1 page | capacity | allocated |
| --- | --- | --- | --- | --- | --- | --- |
| `--hicache-size 8` | BF16 | 147,456 | 54,253 | 54,254 | **54,254 tokens** | 8,000,077,824 B (8.000 GB) |
| `--hicache-size 8` | INT8 | 82,944 | 96,450 | 96,451 | **96,451 tokens** | 8,000,031,744 B (8.000 GB) |

Both pools are handed the same 8 GB; the INT8 pool holds 1.7778× the tokens.

Experiment A instead holds the *tokens* constant at 54,254 and asks what each
representation costs:

```
BF16: 54,254 × 147,456 = 8,000,077,824 B = 8.000 GB
INT8: 54,254 ×  82,944 = 4,500,043,776 B = 4.500 GB      -> 56.25% of the memory
```

Equal token capacity needs sub-GB precision, which `--hicache-size` cannot
express, so Experiment A sizes the pool with `--hicache-ratio`, a float *token*
ratio. With `--page-size 1` the host size lands on
`int(device_capacity × ratio) + 1` tokens, so the harness targets
`target - 1`; with the device pool pinned to 16,384 tokens the shared ratio is
`54253.5 / 16384 = 3.311370849609` for both configs. The achieved capacity is
never trusted to that arithmetic alone: the runner reads the pool's own gauge
after startup and refuses to continue if it disagrees with the target.

---

## 6. Why GPU staging is mandatory

The existing HiCache kernels are **byte movers, not quantisers**.
`transfer_hicache_all_layer` accepts separate source and destination *strides* but
only **one** `element_size`:

```python
def transfer_hicache_all_layer(
    ..., *, kv_cache_src_stride_bytes, kv_cache_dst_stride_bytes,
    element_size=None, ...
):
    if element_size is None:
        assert kv_cache_dst_stride_bytes == kv_cache_src_stride_bytes
        element_size = kv_cache_dst_stride_bytes
```

A 2048-byte BF16 device row therefore cannot be written directly into a
1152-byte encoded host row: source and destination must have the same byte width,
and there is no dtype conversion in the kernel at all. Staging is a structural
requirement, not an optimisation:

```
D2H:  L1 BF16 device KV
        → index_select(device_indices) → per-head absmax → INT8 payload + BF16 scales
        → GPU encoded staging [n, 1152] uint8
        → existing JIT byte mover (element_size = 1152)
        → pinned uint8 L2 arena

H2D:  pinned uint8 L2 arena
        → existing JIT byte mover (element_size = 1152)
        → GPU encoded staging
        → unpack + dequantise to BF16
        → scatter into device KV at device_indices
```

The two directions keep **separate persistent staging buffers**. The D2H and H2D
transfer streams can be in flight concurrently, so sharing one buffer would let a
restore overwrite rows an unsynchronised backup had not yet copied. Buffers grow
geometrically and never shrink (`required ≤ capacity → reuse`), their CUDA
pointer tables are rebuilt only on growth, and the staging row indices
(`arange(n)`) are persistent rather than reallocated per call.

---

## 7. JIT-mover geometry

`element_size = 1152` is admissible with **no CUDA changes** because it lands on
the existing kernel's fast path:

| step | value |
| --- | --- |
| CUDA constants (`kernels/ops/kvcache/hicache.py:26,31`) | `COPY_GROUP_THREADS = 32`, `GROUP_BYTES = (128,)` |
| unroll selection | `_default_unroll(1152) = 1` (1152 > 1024) |
| derived | `lanes_per_worker = 32`, `group = 128`, `package = 128/32 = 4` ∈ {4, 8, 16} |
| alignment check | `1152 % 128 == 0` ✓ |

The TMA variant is off by default on CUDA (`use_hicache_tma_kernel` requires HIP),
so this is the code path actually executed. `layout.RecordLayout` encodes the
requirement directly: `check_against()` raises if `row_bytes % 128 != 0`, so a
future format change that breaks the geometry fails at construction rather than
producing a kernel error on a rented GPU.

The kernel body never reads the cache dtype (`hicache.cuh:271-307`); `run_all`
validates only the pointer tables and index dtypes. The H2D call views *both*
source and destination as `uint8` with `element_dim = 1152`, because `run_one`
uses a single `SymbolicDType` across all four cache tensors — an earlier revision
viewed only the destination as `bf16` (576 elements) while leaving the source
`uint8`. The byte width was right but the dtypes did not match. Either view both
sides or neither; neither is simpler and needs no reinterpretation at all.
`scripts/preflight.py` now asserts statically that no one-sided
`.view(torch.bfloat16)` remains in the pool.

---

## 8. Stream safety

`_submission` (`l2_transfer.py:112`) wraps each transfer in
`with device_module.stream(stream)`, so "the current stream" inside the pool is
already the correct transfer stream. The codec therefore issues plain torch ops
with no explicit stream argument and no event management of its own: it rides the
existing `start_event` / `ack_finish` / `on_layer_done` chain instead of creating
a second synchronisation system.

Ordering follows from the property that all work is enqueued on one stream:

```
quantise layer 0 → copy layer 0 → quantise layer 1 → copy layer 1 → ...
```

The stream guarantees the copy cannot observe a half-encoded staging buffer, and
one mover call can cover all layers because `num_layers` is taken from the
pointer-table length.

The H2D contract is stricter, because `on_layer_done(layer_id)` fires
**immediately after** `load_to_device_per_layer` returns (`l2_transfer.py:177-178`)
and the model's forward stream waits on exactly that per-layer event
(`memory_pool.py:1532` → `LayerDoneCounter.wait_until`):

```
load encoded bytes → dequantise → scatter BF16 → return → THEN on_layer_done
```

Never `load → return → on_layer_done → dequantise later`: that is a race against
the forward pass, and it manifests as intermittently wrong tokens rather than a
crash. Both the decode and the scatter are enqueued before the function returns.

The hot path contains no `torch.cuda.synchronize()`, no `.cpu()` and no `.item()`.
Persistent staging buffers make `record_stream()` unnecessary, because nothing is
allocated per call and no temporary can be freed while the transfer stream still
has work queued against it.

---

## 9. Configuration scope and fail-fast

V1 targets **Qwen3-8B at TP=1**, which is 8 KV heads of 128 dimensions. The
record is a *fixed* 1024-byte payload plus a 16-byte scale block sized for exactly
that geometry, and `layout.check_against(head_num, head_dim, itemsize)` enforces
the correspondence:

* `head_num × head_dim` must equal `payload_bytes` exactly,
* `scale_bytes` must equal `2 × head_num` (one BF16 per head),
* `row_bytes` must be a multiple of 128.

At TP=2 there are 4 local heads, so the payload is 512 bytes and the scale block
8 bytes: the padding, the alignment proof, the capacity arithmetic and the JIT
geometry all change together. A different head count is therefore rejected at
construction, with an error message that names the restriction, rather than
failing later as an opaque payload/geometry mismatch inside a kernel.
TP-sharded record formats are future work.

Everything else unsupported is refused at startup so a bad configuration cannot
serve a single request: non-`layer_first` layout (SGLang's default is
`page_first`), `page_size != 1`, TP ≠ 1, MLA, quantised device KV, mismatched
K/V head dims, `use_hnd`, MTP/draft pools, layer sharding, and L3 storage
backends. The full matrix is in `sglang-integration.md`; it follows the
`mha_mxfp8.py:245-266` precedent for L3, where an encoded arena cannot supply the
BF16 `kv_buffer` pointers a zero-copy backend derives per-page pointers from.

The activation path is deliberately an experimental environment flag
(`SGLANG_EXPERIMENTAL_HICACHE_INT8=1`) rather than a public codec API. No codec
registry, factory, plugin surface, position metadata or variable-sized pages were
added: those are already active upstream design areas, and duplicating them here
would create a merge conflict rather than a contribution.

The codec is implemented entirely in `torch`. Fused Triton kernels were
considered for the D2H and H2D paths and deliberately not written: the per-phase
codec timing (CUDA events around encode, decode and each mover call, opt-in via
`SGLANG_HICACHE_INT8_TIMING`) is what decides whether codec compute is worth
optimising at all. If encode + decode is negligible next to the PCIe copy, fused
kernels would add a maintenance surface for no measured gain.

---

## 10. Experiment design

Three configurations share every non-codec setting — within an experiment, the
same SGLang SHA, GPU, model, prompts, L1 size, generation settings and attention
backend:

| config | L2 representation |
| --- | --- |
| `baseline` | HiCache disabled — prefill recomputation |
| `bf16` | stock SGLang HiCache, BF16 L2 |
| `int8` | HiQCache, 1152-byte encoded L2 |

Two sizings bracket the same claim from opposite sides:

* **Experiment B — equal physical budget.** Both tiers get `--hicache-size 8`.
  The question is how many tokens fit, so the capacity advantage is the result:
  54,254 → 96,451 tokens.
* **Experiment A — equal logical capacity.** Both tiers are pinned to 54,254
  tokens and the question is how many bytes that costs: 8.000 → 4.500 GB. This
  removes the capacity advantage entirely, so a difference in measured bytes per
  token is attributable to the codec rather than to capacity.

A is the stronger half of the memory claim, because B's hit-rate gain could be
attributed to capacity, whereas A's byte ratio is arithmetic.

Two measurement hazards are handled explicitly. First, SGLang's HiCache Prometheus
counters are cumulative process-lifetime values, so a naive comparison includes
server startup and warmup; the driver snapshots `/metrics` around the measured
phase and reports deltas. Second, `sglang:cache_hit_rate` is **not** the serving
hit rate quoted in the results — it is a different quantity (it reads ~97% in
runs whose benchmark hit rate is ~55%). The results documents quote the
benchmark's own `Cache Hit Details` block, and `scripts/analyse.py` reads serving
figures from that report while using the counters only for byte arithmetic.

---

## 11. Reference implementation and the cross-device contract

`src/hiqcache/` is the reference implementation of the format: byte arithmetic,
alignment proofs and compression maths (`layout.py`), and quantise / pack /
unpack / dequantise with the error accounting (`codec.py`). It is pure `torch`
with **no SGLang imports and no CUDA-only ops**, so the identical module runs on
CPU, MPS and CUDA.

That property is what makes the pod runs cheap to debug. `scripts/conformance.py`
generates a digest of the codec's output; the CPU digest is committed and the
CUDA run is checked against it before any SGLang work starts. A digest mismatch
means the bug is in the codec or a backend op, and it is localised in seconds
instead of after a failed multi-hour server run. The fork carries its own
self-contained copy of the codec (it cannot import this package), and
`tests/test_codec_drift.py` pins the two implementations to each other bit for
bit so they cannot diverge silently.
