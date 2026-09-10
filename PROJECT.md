# HiQCache
**Quantized hierarchical KV cache for SGLang HiCache**

The exact name can change later.

The project thesis is:

> **Keep active L1 KV in normal BF16 for attention, but compress KV when it is evicted to SGLang's CPU L2 cache, then restore it to BF16 on an L2 hit. Measure the capacity, transfer, latency, and quality trade-off on Qwen3-8B.**

---

# 1. Repository structure

Use the same successful pattern as RolloutCore.

```text
awesome-pro/

├── sglang/
│   └── fork of sgl-project/sglang
│
│   branches:
│   hiqcache/base
│   hiqcache/int8-l2
│   hiqcache/triton          # only if needed later
│
└── hiqcache/
    ├── README.md
    ├── benchmarks/
    ├── scripts/
    ├── analysis/
    ├── results/
    ├── docs/
    └── plots/
```

Pin the SGLang starting point:

```text
515f5be77e74761c269e007ac41a5895191a1b7d
```

The standalone repo contains the story, experiments and results.

The SGLang fork contains the actual runtime modification.

---

## Phase status

| Phase | Description | Where | Status |
| --- | --- | --- | --- |
| 0 | Repo scaffold + cross-device conformance harness | Mac | **done** |
| 1 | Encoded representation (`layout.py`) | Mac | **done** |
| 2 | Standalone codec + tests | Mac | **done** (CPU + MPS) |
| 3 | Compressed host pool `mha_int8.py` | Pod | not started |
| 4 | GPU staging architecture | Pod | not started |
| 5 | D2H compression integration | Pod | not started |
| 6 | H2D restoration integration | Pod | not started |
| 7 | Stream semantics | Pod | not started |
| 8 | Fail-fast configuration checks | Pod | not started |
| 9 | Experimental activation path | Pod | not started |
| 10 | Unit + kernel-level validation | both | partially (codec tests done) |
| 11 | Real SGLang integration smoke | Pod | not started |
| 12–20 | Benchmarks, quality, analysis, demo | Pod + Mac analysis | not started |

Verified facts, kernel constraints and the local/pod capability split are recorded in
`docs/verification-notes.md`. Do not re-derive them from the trace doc alone.

---

# Phase 1 — Define the encoded representation

Do this before touching HiCache.

For Qwen3-8B at TP=1:

```text
36 layers
8 KV heads
128 dimensions/head

BF16 K row:
8 × 128 × 2 bytes
= 2048 B

BF16 V row:
= 2048 B

K + V / layer / token:
4096 B

all 36 layers:
147,456 B/token
= 144 KiB/token
```

## V1 quantization

For every:

```text
token
× layer
× K/V
× KV head
```

calculate:

\[
s = \max(|x|)/127
\]

and:

\[
q = \operatorname{round}(x/s)
\]

clamped to signed INT8.

Restore with:

\[
\hat{x} = q \cdot s
\]

For an all-zero head, use a safe nonzero scale and encode all zeros.

---

## Encoded record

Use **one aligned record per K or V row**:

```text
0 ─────────────────────────────── 1023
|       1024 INT8 KV bytes          |

1024 ──────────────────────────── 1039
|   8 BF16 per-head scales          |

1040 ──────────────────────────── 1151
|       112 bytes padding           |
```

Total:

```text
1152 B / K row
1152 B / V row

2304 B / layer / token
```

Thus:

```text
baseline:
4096 B

HiQCache:
2304 B

reduction:
43.75%

compression:
1.78×
```

For Qwen3-8B:

```text
baseline:
147,456 B/token

HiQCache:
82,944 B/token
```

The padding is intentional.

1152 bytes is:

```text
9 × 128 B
```

which fits SGLang's existing HiCache CUDA byte-transfer machinery cleanly.
---

# Phase 2 — Build and validate the standalone codec

Before SGLang integration, implement:

```text
BF16 tensor
     ↓
quantize + pack
     ↓
1152-byte record
     ↓
unpack + dequantize
     ↓
BF16 tensor
```

Test:

```text
random BF16 values
all zeros
small values
large values
mixed magnitudes
realistic KV-shaped tensors
```

Record:

```text
mean absolute error
max absolute error
p50 / p95 / p99 error
relative error
compression ratio
encode time
decode time
```

Also prove mathematically/test-wise that for an exact BF16 scale:

```text
quantization error ≲ scale / 2
```

subject to scale rounding itself.

> **Updated after Phase 2 implementation.** The `scale / 2` claim is correct only
> for the *exact* product `q · s`. It does not survive two representation choices
> that the format forces, and both were confirmed by measurement:
>
> 1. **The scale is BF16.** Its relative error is up to `2**-8`, amplified by
>    `|x / s| <= 127`, contributing up to `127 · 2**-8 · s ≈ 0.5 · s`. This makes
>    the clamp to 127 mathematically required, not defensive, and it is why the
>    quotient is formed in **float32** — forming it in BF16 has ~0.25 of a scale
>    step of granularity near full scale and pushes the normalized error to ~0.75.
> 2. **The decoder emits BF16**, because L1 attention consumes BF16, so the final
>    product is rounded to BF16. This term, `2**-8 · |x̂|`, **dominates at large
>    magnitudes** and reaches ~0.93 · s in units of the scale.
>
> The bound the format actually guarantees, verified elementwise on CPU and MPS
> with zero violations:
>
> ```text
> |x̂ − x|  <=  (0.5 + 2**-8) · s  +  2**-8 · |x̂|
> ```
>
> with the `0.5 · s` term asserted separately against an exact float64 quantizer
> so the decomposition is proven rather than asserted as a whole. See
> `src/hiqcache/codec.py::verified_error_bound` and the numerical tests in
> `tests/test_codec.py`.


### Phase 2 completion

You should have:

```text
codec correctness tests ✓
packed layout tests ✓
43.75% storage reduction ✓
GPU encode/decode ✓
```

No SGLang required yet.

---

# Phase 3 — Implement the compressed HiCache host pool

Create:

```text
python/sglang/srt/mem_cache/pool_host/mha_int8.py
```

with something like:

```python
class MHATokenToKVPoolHostINT8(MHATokenToKVPoolHost):
    ...
```

Do **not** modify `L2TransferEngine`, scheduler, radix cache or tree logic.

The whole point is:

```text
SGLang caching policy
      remains unchanged

only L2 representation changes
```

## Host representation

For `layer_first`:

```text
K:
[layer_num, host_slots, 1152] uint8

V:
[layer_num, host_slots, 1152] uint8
```

Pinned CPU memory.

Then:

```python
get_size_per_token()
```

returns:

```text
2 × layer_num × 1152

for Qwen3-8B:
82,944 bytes
```

This is crucial because `HostKVCache` uses `size_per_token` when `--hicache-size` specifies a fixed host-memory budget.

---

# Phase 4 — GPU staging architecture

This is the important correction to the original agent plan.

The existing HiCache kernels are **byte movers**, not quantizers.

Therefore:

```text
BF16 GPU
→ directly into smaller INT8 CPU buffer
```

is impossible.

Use GPU staging.

## D2H path

```text
L1 BF16 device KV
        │
        │ selected device_indices
        ▼
quantize + pack
        │
        ▼
GPU encoded staging
        │
        │ existing HiCache byte mover
        ▼
pinned compressed L2
```

## H2D path

```text
pinned compressed L2
        │
        │ existing HiCache byte mover
        ▼
GPU encoded staging
        │
        │ unpack + dequantize
        ▼
BF16 device KV
```

---

## Staging buffers

Use separate persistent staging for:

```text
D2H
and
H2D
```

because the two HiCache transfer streams can operate independently.

Conceptually:

```python
d2h_k_staging
d2h_v_staging

h2d_k_staging
h2d_v_staging
```

Shape:

```text
[capacity, 1152] uint8
```

Use a grow-only strategy:

```text
required <= capacity
→ reuse

required > capacity
→ grow to next sensible capacity
```

Also maintain persistent:

```text
staging_indices = [0, 1, 2, ...]
```

rather than repeatedly constructing temporary index tensors.

---

# Phase 5 — D2H compression integration

Implement inside:

```python
backup_from_device_all_layer(...)
```

For each layer:

```text
device K/V
   ↓
gather device_indices
   ↓
shape:
[N, 8, 128]

   ↓
amax across head_dim

scales:
[N, 8]

   ↓
INT8 quantization

payload:
[N, 1024]

   ↓
pack
payload + scale + padding

   ↓
[N, 1152] uint8

   ↓
existing one-layer HiCache mover

GPU staging indices:
0...N-1

→

real host_indices
```

Then reuse the same staging buffer for the next layer.

Because all operations are queued on the same:

```text
device_to_host_stream
```

this is safe:

```text
quantize layer 0
copy layer 0
quantize layer 1
copy layer 1
...
```

The stream guarantees ordering.

---

# Phase 6 — H2D restoration integration

Implement in:

```python
load_to_device_per_layer(...)
```

For the requested layer:

```text
compressed L2 row
      ↓
existing HiCache mover
      ↓
GPU staging
      ↓
decode payload
      ↓
read BF16 scales
      ↓
int8 × scale
      ↓
BF16
      ↓
scatter → device_indices
```

This must all be **enqueued before the function returns**.

Why?

Immediately afterward SGLang invokes:

```text
on_layer_done(layer_id)
```

and the model forward stream waits on that layer-completion event.

So the contract is:

```text
load encoded bytes
+
dequantize
+
scatter BF16
+
return

THEN

on_layer_done
```

Never:

```text
load
return
on_layer_done
dequantize later
```

That would create a correctness race.

---

# Phase 7 — Preserve HiCache stream semantics

This deserves explicit tests.

Codec operations must run inside the existing:

```text
device_to_host_stream
host_to_device_stream
```

Do not use:

```python
torch.cuda.synchronize()
tensor.cpu()
tensor.item()
```

inside the hot path.

No CPU synchronization.

Also make sure temporary CUDA state survives until work completes.

Persistent staging buffers largely solve this.

The existing SGLang event chain remains:

```text
producer work
   ↓
start_event
   ↓
transfer stream
   ↓
codec work
   ↓
ack_finish / layer completion
   ↓
consumer forward
```

Do not create a second independent synchronization system.

---

# Phase 8 — Fail-fast configuration checks

The experimental pool should refuse unsupported configurations at startup.

Reject:

```text
page_size != 1

TP != 1          # V1 restriction

MLA

device quantized KV

K head dim != V head dim

layout != layer_first

io_backend != kernel

MTP/draft pools

L3 storage backend

decode retraction backup=host_pool
```

Also reject any malformed encoded-row/alignment condition.

This is much better than:

```text
"probably works"
```

and then silently corrupting KV.

---

# Phase 9 — Experimental activation path

Do not introduce a generic public codec API.

Use something explicitly experimental, for example:

```bash
SGLANG_EXPERIMENTAL_HICACHE_INT8=1
```

Then the MHA host-pool selection does:

```text
normal MHA
→ MHATokenToKVPoolHost

experimental flag
→ MHATokenToKVPoolHostINT8
```

Small source change.

Do not add:

```text
codec registry
codec factory
generic plugins
position metadata
variable-sized pages
```

Those are already active upstream design areas.

---

# Phase 10 — Unit and kernel-level validation

Before starting a full model, add tests.

### Codec tests

```text
BF16 → INT8 → BF16
zero input
error bounds
packed record correctness
scale encoding
alignment
```

### Host-pool tests

```text
encoded size_per_token

allocation/free

page alignment

double-free behavior

host capacity calculation
```

### Transfer roundtrip

Construct a device KV pool:

```text
known BF16 KV
   ↓
D2H compressed
   ↓
clear device destination
   ↓
H2D restore
   ↓
compare
```

Measure reconstruction error.

### Stream ordering

Adapt SGLang's existing HiCache ordering tests.

Ensure:

```text
producer modifies indices/data on another stream
        ↓
codec waits correctly
        ↓
consumer sees final values
```

And ensure layer completion doesn't fire before dequantization.

---

# Phase 11 — Real SGLang integration smoke

Now run actual SGLang.

Prefer final testing on:

```text
Qwen3-8B
1 GPU
48 GB GPU if available
64+ GB CPU RAM
```

A 24 GB GPU may work, but a 48 GB A6000/L40S gives you more room to debug the cache rather than memory packing.

First prove only:

```text
server boots

HiCache enabled

request populates L1

cache entry moves to L2

compressed L2 bytes exist

later request hits L2

compressed KV moves back

generation completes correctly
```

Instrument enough state to prove:

```text
L1 → L2 write happened
L2 → L1 restore happened
```

Do not rely solely on “server didn't crash.”

---

# Phase 12 — Benchmark design

There should be three configurations.

### A — No reusable L2 / recompute baseline

```text
L1 miss
→ prefill recomputation
```

### B — Standard SGLang HiCache

```text
BF16 L2
```

### C — HiQCache

```text
compressed INT8 L2
```

Every comparison should use:

```text
same SGLang SHA
same GPU
same CPU
same model
same prompts
same L1 size
same generation settings
same attention backend
```

---

# Phase 13 — Run two different benchmark modes

This is important because SGLang has both `--hicache-ratio` and `--hicache-size`.

## Experiment A — same logical capacity

Give baseline and HiQCache approximately the same number of cached tokens.

This isolates the codec's performance cost/benefit.

Measure:

```text
D2H time
encode time

H2D time
decode time

cache-hit TTFT
throughput
```

This answers:

> Is the compressed transfer path faster/slower than raw BF16 for the same cache workload?

---

## Experiment B — same physical CPU-memory budget

Use:

```text
--hicache-size=<same GB>
```

for both.

This is essential because `HostKVCache` calculates:

```text
token capacity
=
host bytes / size_per_token
```

when fixed `--hicache-size` is used.

With the current layout, theoretical capacity is approximately:

```text
BF16:
1.00×

HiQCache:
1.78×
```

under the same L2 memory budget.

This is probably the most valuable experiment.

Construct a workload whose aggregate reusable prefixes:

```text
do not fit in baseline BF16 L2

but

do fit in HiQCache L2
```

Then revisit those prefixes.

You want to measure:

```text
baseline:
host eviction
→ recomputation

HiQCache:
host hit
→ compressed restore
```

This is the real end-to-end value proposition.

---

# Phase 14 — Performance metrics

Record separately:

### Storage

```text
bytes/token
host memory allocated
host token capacity
compression ratio
```

### D2H

```text
quantization time
copy time
total backup time
bytes transferred
effective bandwidth
```

### H2D

```text
copy time
dequantization time
total restore time
```

### Serving

```text
cache-hit TTFT
p50 / p95 TTFT
throughput
HiCache hit rate
prefill tokens avoided
```

### Resource use

```text
GPU scratch memory
CPU pinned memory
kernel launches
GPU utilization if useful
```

Hardware provenance must also be recorded:

```text
GPU
driver
CUDA
SGLang SHA
Torch
CPU
RAM
```

---

# Phase 15 — Numerical/quality validation

Do not call the codec successful based only on bytes.

You need to quantify the lossiness.

### KV reconstruction

Across realistic KV samples:

```text
mean absolute error
max absolute error
p99 absolute error
relative error
```

### Generation agreement

Run the same prefixes through:

```text
baseline BF16 HiCache
HiQCache
```

with deterministic decoding.

Measure:

```text
first-token agreement
exact 16/32-token sequence agreement
token-level agreement
```

### Logprob comparison

If easily exposed by SGLang:

```text
mean |Δ logprob|
p95
p99
max
```

This is more informative than only comparing generated text.

### Small task-level evaluation

If straightforward, use a small subset of something appropriate to Qwen3-8B.

For example:

```text
GSM8K subset
or
MMLU subset
```

Don't build an entire evaluation framework solely for this.

The goal is simply to establish:

> The L2 compression level provides X memory reduction at Y measured quality impact.

---

# Phase 16 — Torch implementation first

Your first real codec can use PyTorch CUDA operations.

For example:

```text
gather
amax
divide
round
clamp
cast
pack
```

And:

```text
unpack
cast
multiply
scatter
```

Do **not** start with Triton.

First prove that the complete end-to-end system works.

---

# Phase 17 — Triton optimization, only if profiling justifies it

After the torch implementation works, profile it.

If results show:

```text
quantize/dequantize kernels dominate the L2 path
```

then implement fused Triton kernels.

Potential D2H kernel:

```text
gather
+
per-head amax
+
quantize
+
pack 1152-byte row
```

Potential H2D kernel:

```text
unpack
+
load scales
+
dequantize
+
scatter BF16
```

Then benchmark:

```text
Torch codec
vs
Triton codec
```

This can add a very strong GPU-kernel component to the project.

But if codec compute is already negligible relative to PCIe transfer, **skip Triton**.

Do not write kernels merely to say the project has Triton.

---

# Phase 18 — Analyze the result without forcing a win

There are several valid outcomes.

### Best case

```text
host bytes        -44%
L2 capacity       +78%
transfer bytes    -44%
TTFT              improves
quality loss      negligible
```

Excellent.

### Capacity win but neutral latency

```text
host bytes        -44%
L2 capacity       +78%
TTFT              roughly unchanged
quality           acceptable
```

Still a strong project.

### Capacity win but codec overhead

```text
host bytes        -44%
L2 capacity       +78%
cache-hit TTFT    +5%

but

under memory pressure:
higher hit rate
less recomputation
better workload-level TTFT
```

Also interesting.

### Negative result

If:

```text
quality degrades badly
or
codec overhead overwhelms cache benefit
```

report it.

The project still demonstrates:

```text
real SGLang HiCache internals
GPU/CPU memory hierarchy
quantization
CUDA stream correctness
benchmark discipline
```

The project must not depend on a predetermined speedup.

---

# Phase 19 — Final project repository

The standalone README should eventually look something like:

```text
# HiQCache

Quantized hierarchical KV caching for SGLang.

[architecture diagram]

## Results

Qwen3-8B

Metric                 BF16 HiCache     HiQCache
Host bytes/token       147,456          82,944
L2 capacity            X                Y
D2H bytes              X                Y
L2-hit TTFT            X                Y
Output agreement       —                Z%
Logprob delta           —                ...
```

Then:

```text
## Why

## Architecture

## SGLang integration

## Compression format

## Stream correctness

## Experiments

## Quality

## Limitations

## Reproduce
```

Detailed source maps and experimental logs go under:

```text
docs/
results/
```

Do not make the README a diary of Phase 1 → Phase 19.

---

# Phase 20 — Demo

The demo should visually show:

```text
same fixed 8 GB host budget

BF16 HiCache
████████████████████
~X cached tokens

HiQCache
██████████████████████████████████
~1.78X cached tokens
```

Then:

```text
request
 ↓
L1 miss
 ↓
L2 compressed hit
 ↓
82 KB/token restored instead of 144 KB/token
 ↓
BF16 attention continues normally
```

And finally real charts:

```text
L2 capacity
transfer bytes
TTFT
quality
```

---

# Definition of done

I would call HiQCache resume-ready when all of the following are true:

- It modifies **real current SGLang HiCache**, not a mock.
- Qwen3-8B is used in the final experiment.
- BF16 KV remains unchanged in L1.
- L2 actually stores the reduced representation.
- Real L1→L2 and L2→L1 paths execute.
- Host `size_per_token` reflects compressed bytes.
- Fixed-GB L2 capacity increase is demonstrated.
- Actual transfer bytes are lower.
- Codec overhead is measured.
- Cache-hit TTFT is measured.
- Reconstruction error is measured.
- Output/logprob quality impact is measured.
- Unsupported configurations fail fast.
- Stream-ordering tests pass.
- SGLang commit/hardware/software are pinned.
- Reproduction scripts exist.
- README contains the real GPU results.
- At least one upstream SGLang discussion/issue/PR is created from the work.
