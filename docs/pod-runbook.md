# Pod runbook — GPU phases of HiQCache

Everything that needs CUDA. Work on the Mac is already done and committed; this
document is the sequence to run on a rented GPU.

Target hardware: **1× A6000 48 GB**, 64+ GB CPU RAM, 100+ GB disk.

| Repo | Ref | Meaning |
| --- | --- | --- |
| `awesome-pro/sglang` | `hiqcache/int8-l2` | the Phase 3–9 implementation |
| `awesome-pro/sglang` | `515f5be77e74761c269e007ac41a5895191a1b7d` | pinned base commit (branch `main`) |
| `awesome-pro/hiqcache` | `main` | story, reference codec, tests, docs |

---

## 0. Why a GPU is needed at all

| Component | Runs on Mac? | Why |
| --- | --- | --- |
| Codec arithmetic | ✅ | pure `torch` |
| Packed byte layout | ✅ | pure indexing |
| Host-pool sizing math | ✅ | integer arithmetic |
| SGLang JWt/JIT HiCache kernels | ❌ | CUDA compilation at runtime |
| `sgl_kernel` / `sgl_kernel.kvcacheio` | ❌ | CUDA extension |
| `cudaHostRegister` pinned arena | ❌ | CUDA driver |
| CUDA stream / event ordering | ❌ | CUDA |
| `torch.ops.sgl_kernel.transfer_kv_direct` | ❌ | CUDA extension |
| Real model forward, TTFT, throughput | ❌ | CUDA |

`device_to_host_stream`, `host_to_device_stream` and `LayerLoadingEvent` are all
CUDA constructs. Stream-ordering correctness (Phase 7) **cannot** be validated on
the Mac.

---

## 1. Environment bootstrap

```bash
git clone git@github.com:awesome-pro/hiqcache.git
cd hiqcache
git checkout main

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch pytest numpy
.venv/bin/python -m pytest tests/ -q            # expect 223 passed, no GPU needed
```

Clone the SGLang fork with the implementation:

```bash
git clone git@github.com:awesome-pro/sglang.git
cd sglang
git checkout hiqcache/int8-l2
git rev-parse HEAD        # e554c653bc..., on top of the pinned base
git merge-base --is-ancestor 515f5be77e74761c269e007ac41a5895191a1b7d HEAD \
  && echo "base commit is an ancestor: OK"
```

Install SGLang per its own instructions. Do **not** let SGLang and the codec share
a venv unless the version pins agree — the codec only needs `torch`.

### Run the fork's own INT8 tests first

These need no GPU (codec + staging) and give the fastest signal:

```bash
cd sglang
python -m pytest test/registered/unit/mem_cache/test_hicache_int8_codec.py -v
```

The `hiqcache` repo can also run these against the fork checkout directly, which
is the recommended order because `test_codec_drift.py` additionally proves the
fork's codec copy equals the reference implementation:

```bash
cd ../hiqcache
HIQCACHE_SGLANG_ROOT=../sglang .venv/bin/python -m pytest tests/ -q
```

### Record the hardware provenance (required by PROJECT.md Phase 14)

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
lscpu | head -20
free -g
cd sglang && git rev-parse HEAD
```

Write it to `results/provenance.json`. Every benchmark result must carry it.

---

## 2. Gate: codec conformance on CUDA — **do this first**

```bash
cd hiqcache
.venv/bin/python scripts/conformance.py generate --device cuda \
    --expect results/conformance_cpu.json
```

Expected: `PASS: codec is bit-identical to the reference device.`

The Mac already established `CPU ≡ MPS` bit-identical over 9 vectors. If CUDA
differs, **stop**. The bug is in the codec or a backend op, and it is far cheaper
to fix here than inside a running SGLang server.

Also run the parity tests on the GPU:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/capacity_table.py --hicache-size 8
```

The capacity numbers printed here are the ones the pod runs must reproduce:
`54,254` (BF16) and `96,451` (HiQCache) tokens.

---

## 3. Facts already verified — do not re-derive

These were checked against the fork's working tree and are recorded in
`docs/verification-notes.md`:

1. **`element_size = 1152` is on the CUDA JIT fast path.**
   `COPY_GROUP_THREADS = 32`, `GROUP_BYTES = (128,)` on CUDA
   (`python/sglang/kernels/ops/kvcache/hicache.py:26,31`).
   `_default_unroll(1152) = 1` → `lanes_per_worker = 32`, `group = 128`,
   package `128/32 = 4 ∈ {4,8,16}` ✓ and `1152 % 128 == 0` ✓.

2. **GPU staging is mandatory, not optional.**
   `transfer_hicache_all_layer` accepts separate
   `kv_cache_src_stride_bytes` / `kv_cache_dst_stride_bytes` but only **one**
   `element_size`. A 2048-byte BF16 device row therefore *cannot* be written
   directly into a 1152-byte host row. Encode into a `[N, 1152]` device staging
   buffer first, then move bytes with matching strides.

3. **Layout must be forced to `layer_first`.** The SGLang default is
   `page_first` (`arg_groups/fields/memory.py:133-147`); the staging design wants
   per-layer contiguous views.

4. **`--hicache-size` is decimal GB**: `int(host_size * 1e9 // size_per_token)`
   (`pool_host/base.py:201`), then `page_num = size // page_size + 1`.

5. **L3 storage is the only real complication.** Follow the
   `pool_host/mha_mxfp8.py:245-266` `_storage_pages_unsupported()` precedent and
   reject `--hicache-storage-backend` in v1.

6. **TMA path is off** unless `_is_hip` — irrelevant on an A6000, but note it if
   the pod ends up being AMD.

---

## 4. Integration status — already written

Phases 3–9 are implemented on `hiqcache/int8-l2`. The source map, data flow and
fail-fast matrix are in `docs/sglang-integration.md`. **Read that before touching
the pool.** Summary of what exists:

| Path (in the fork) | Purpose |
| --- | --- |
| `mem_cache/pool_host/int8_codec.py` | INT8 record quantise / pack / decode |
| `mem_cache/pool_host/int8_staging.py` | device staging buffers + pointer tables |
| `mem_cache/pool_host/mha_int8.py` | `MHATokenToKVPoolHostINT8` |
| `mem_cache/pool_host/mha.py` | one dispatch branch (`+12` lines) |
| `srt/environ.py` | `SGLANG_EXPERIMENTAL_HICACHE_INT8` (`+10` lines) |

`l2_transfer.py`, `cache_controller.py`, `unified_radix_cache.py`,
`unified_tree_core.py` and all CUDA sources are untouched.

### What still has to be proven on the pod

The pool parses, its codec and staging logic are unit-tested, and the fork's
codec copy is pinned bit-for-bit to the reference implementation. **But none of
the following has ever executed:**

- the JIT HiCache kernel against a 1152-byte `element_size`
- `cudaHostRegister` on the encoded arena
- the D2H all-layer move with a per-layer pointer table
- the H2D `element_dim = 576` bf16 reinterpretation trick
- stream ordering around `on_layer_done`
- `index_select` / advanced-index scatter at device scale

Run the pool unit test first — it exercises all six in one file:

```bash
python -m pytest \
  test/registered/unit/mem_cache/test_hicache_int8_pool_host_unit.py -v
```

If the JIT kernel rejects `element_size = 1152`, the failure is in the
`pick_group_bytes` calculation, and `docs/verification-notes.md` has the
arithmetic (`lanes_per_worker=32`, `group=128`, package `4`).

### The correctness trap that will bite

`on_layer_done(layer_id)` fires **immediately after** `load_to_device_per_layer`
returns (`l2_transfer.py:177-178`), and the model's forward stream waits on that
per-layer event. Therefore:

```
load encoded bytes -> dequantise -> scatter BF16 -> return -> THEN on_layer_done
```

Never `load -> return -> on_layer_done -> dequantise later`. That is a data race
against the forward pass, and it will show up as intermittent wrong tokens rather
than a crash. The implementation already enqueues both steps before returning;
the pod test is what proves it.

Also banned in the hot path: `torch.cuda.synchronize()`, `.cpu()`, `.item()`.
Temporary GPU buffers on the transfer stream need `record_stream()`; the pool
avoids this by keeping staging persistent.

### Fail-fast matrix (Phase 8) — implemented

Rejected at construction: `page_size != 1`, `layout != layer_first`, host/device
page-size mismatch, a device pool not covering all layers, layer sharding, MTP
draft pools, quantized device KV, `store_dtype` not BF16/FP16,
`head_dim != v_head_dim`, HND layout, `head_num * head_dim != 1024`, a row not a
multiple of 128 B, and a missing JIT mover.

Rejected at transfer time: `io_backend != "kernel"`, `is_draft=True`.

Refused rather than implemented: `--hicache-storage-backend` (L3).

### Activation (Phase 9) — implemented

`SGLANG_EXPERIMENTAL_HICACHE_INT8=1`, checked in `get_mha_host_pool_cls` after
the MXFP8 and asymmetric cases. No codec registry, no factory, no plugin API.

---

## 5. Smoke test (Phase 11)

Prove more than "the server didn't crash":

- [ ] server boots with HiCache + the INT8 flag
- [ ] a request populates L1
- [ ] a cache entry moves to L2
- [ ] **compressed** L2 bytes exist (assert host bytes/token == 82,944)
- [ ] a later request hits L2
- [ ] compressed KV moves back
- [ ] generation completes correctly

Instrument `L1→L2 write happened` and `L2→L1 restore happened` explicitly.

---

## 6. Benchmarks (Phases 12–14)

Three configurations, identical SGLang SHA / GPU / CPU / model / prompts / L1 size
/ generation settings / attention backend:

| | Config | Baseline A |
| --- | --- | --- |
| A | HiCache **disabled** — pure prefill recomputation | ← chosen |
| B | Standard SGLang HiCache, BF16 L2 | |
| C | HiQCache, compressed INT8 L2 | |

Two modes:

- **Experiment A — same logical capacity.** Isolates codec cost. Measure D2H,
  encode, H2D, decode, cache-hit TTFT, throughput.
- **Experiment B — same physical CPU-memory budget** (`--hicache-size` equal).
  This is the headline experiment: construct a workload whose reusable prefixes
  do **not** fit in baseline BF16 L2 but **do** fit in HiQCache L2, then revisit
  them. Baseline evicts → recomputes; HiQCache hits → restores.

---

## 7. Quality (Phase 15)

Do not call the codec successful on bytes alone.

- KV reconstruction error across realistic samples
- deterministic-decoding agreement: first token, exact 16/32-token sequences, token-level
- logprob deltas: mean / p95 / p99 / max
- a small GSM8K or MMLU subset

---

## 8. Triton (Phase 17) — only if profiling says so

Start with torch. Profile. If quantize/dequantize dominates the L2 path rather
than PCIe transfer, implement fused Triton kernels. If codec compute is already
negligible relative to transfer, **skip it** and say so in the README. Do not
write kernels merely to have Triton in the project.

---

## Cost notes

Debugging on the pod is the expensive part. The ordering above front-loads every
checkable thing onto the Mac:

- codec arithmetic and bounds → Mac
- byte layout and alignment → Mac
- host-pool sizing → Mac
- backend parity → Mac (MPS) and pod (CUDA), gated in seconds

What remains on the pod is genuinely CUDA-only. Keep the pod for that.
