# SGLang integration — Phase 3 source map

What was changed in the SGLang fork, why, and what the runtime contract is.

Pinned commit: `515f5be77e74761c269e007ac41a5895191a1b7d`
Fork: `git@github.com:awesome-pro/sglang.git`, branch `main`

---

## Diff summary

**Three new files, two edited lines of existing code.**

| Path | Kind | Purpose |
| --- | --- | --- |
| `python/sglang/srt/mem_cache/pool_host/int8_codec.py` | new | INT8 record quantise / pack / decode |
| `python/sglang/srt/mem_cache/pool_host/int8_staging.py` | new | device staging buffers + pointer tables |
| `python/sglang/srt/mem_cache/pool_host/mha_int8.py` | new | `MHATokenToKVPoolHostINT8` |
| `python/sglang/srt/mem_cache/pool_host/mha.py` | edited | one dispatch branch in `get_mha_host_pool_cls` + `envs` import |
| `python/sglang/srt/environ.py` | edited | two experimental env vars |
| `test/registered/unit/mem_cache/test_hicache_int8_codec.py` | new | codec unit tests (no GPU) |
| `test/registered/unit/mem_cache/test_hicache_int8_pool_host_unit.py` | new | pool tests (CUDA) |

Untouched, deliberately: `l2_transfer.py`, `cache_controller.py`,
`unified_radix_cache.py`, `unified_tree_core.py`, `hybrid_cache_controller.py`,
and all CUDA sources.

---

## Why nothing else needed changing

Every L1↔L2 byte in HiCache funnels through exactly two methods
(`l2_transfer.py:136` and `:169`):

```
L2TransferEngine.submit_device_to_host -> host_pool.backup_from_device_all_layer_physical
L2TransferEngine.submit_host_to_device -> host_pool.load_to_device_per_layer_physical
```

`HostKVCache` forwards `_physical` to the plain methods (`base.py:371`, `:378`),
so overriding those two in a subclass intercepts 100% of the traffic. The radix
tree, eviction policy and scheduler never inspect KV bytes — they only allocate
and free *slots* — so reducing bytes-per-slot is invisible to them.

---

## The constraint that shaped the design

`transfer_hicache_all_layer` accepts separate source and destination **strides**
but only **one** `element_size`:

```python
def transfer_hicache_all_layer(
    ..., *, kv_cache_src_stride_bytes, kv_cache_dst_stride_bytes,
    element_size=None, ...
):
    if element_size is None:
        assert kv_cache_dst_stride_bytes == kv_cache_src_stride_bytes
        element_size = kv_cache_dst_stride_bytes
```

A 2048-byte BF16 device row therefore **cannot** be written directly into a
1152-byte encoded host row. Staging is mandatory, not an optimisation.

The kernel body itself is a pure byte mover — it never reads the cache dtype
(`hicache.cuh:271-307`), and `run_all` validates only the pointer tables:

```cpp
TensorMatcher({N}).with_dtype<uint64_t>().with_device<kDLGPU>()  // ptr tables
TensorMatcher({L}).with_dtype<int32_t, int64_t>()                // indices
```

Consequence: **the encoded host representation needs no CUDA change at all.**

---

## Data flow

### D2H (eviction, encode)

```
L2TransferEngine sets current stream = device_to_host_stream
  └─ backup_from_device_all_layer(device_pool, host_indices, device_indices, "kernel")
       for layer in 0..layer_num-1:                 # enqueued, not synchronised
         rows = device_pool.{k,v}_buffer[layer].index_select(0, device_indices)
         codec.write_record(rows -> d2h_staging[layer, :n])
       jit_transfer_hicache_all_layer(               # one call, all layers
         k_ptr_dst = [host k_data_refs[l] for l],    # fixed arena addresses
         v_ptr_dst = [host v_data_refs[l] for l],
         k_ptr_src = [d2h_staging.k[l] for l],       # rebuilt only on growth
         v_ptr_src = [d2h_staging.v[l] for l],
         indices_src = arange(n),                    # staging rows 0..n-1
         indices_dst = host_indices,                 # real L2 slots
         kv_cache_{src,dst}_stride_bytes = 1152,
         element_size = 1152)
```

`num_layers` comes from the pointer-table length, so one mover call covers all
layers. The kernel is issued on the transfer stream, which is strictly ordered
after every encode above it, so the copy cannot observe a half-encoded staging
buffer.

### H2D (restore, decode)

```
L2TransferEngine sets current stream = host_to_device_stream
  └─ load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, "kernel")
       jit_transfer_hicache_one_layer(              # encoded bytes -> staging
         k_cache_src = k_data_refs[host_layer_id],  # host arena, uint8
         k_cache_dst = h2d_staging.k[layer_id],     # device, uint8
         element_dim = 576)                         # 576 bf16 == 1152 bytes
       codec.decode_records(staging) -> bf16
       device_pool.k_buffer[layer_id][device_indices] = k_bf16
       ... same for V ...
```

### Why `element_dim = 576`

`run_one` uses a **single** `SymbolicDType` across all four cache tensors:

```cpp
TensorMatcher({-1, D}).with_strides({N, 1}).with_dtype(cache_dtype)  // src
TensorMatcher({-1, D}).with_strides({M, 1}).with_dtype(cache_dtype)  // dst
```

so a uint8 source and a bf16 destination are rejected. Reinterpreting both uint8
buffers as **576 BF16 elements** satisfies that check while leaving the byte
width the kernel actually moves unchanged: `576 * 2 == 1152`. Verified to work
on CPU, MPS and non-contiguous slices.

---

## The correctness trap

`on_layer_done(layer_id)` fires **immediately after**
`load_to_device_per_layer` returns (`l2_transfer.py:177-178`), and the model's
forward stream waits on exactly that per-layer event
(`memory_pool.py:1532` → `LayerDoneCounter.wait_until`).

```
load encoded bytes -> dequantise -> scatter BF16 -> return -> THEN on_layer_done
```

Never `load -> return -> on_layer_done -> dequantise later`. That is a race
against the forward pass, and it manifests as intermittently wrong tokens, not a
crash. Both the decode and the scatter are enqueued before the function returns.

Also banned in the hot path: `torch.cuda.synchronize()`, `.cpu()`, `.item()`.
The pool contains none of them.

---

## Stream ownership

`_submission` (`l2_transfer.py:112`) wraps each transfer in
`with device_module.stream(stream)`, so "the current stream" inside the pool is
already the correct transfer stream. The pool therefore issues plain torch ops
with no explicit stream argument and no event management of its own — it rides
the existing `start_event` / `ack_finish` / `on_layer_done` chain rather than
creating a second synchronisation system.

Staging buffers are persistent attributes of the pool, so temporaries cannot be
freed while the transfer stream still has work queued against them. No
`record_stream()` is required because nothing is allocated per call.

---

## Fail-fast matrix

Raised at construction, so a bad configuration cannot start a server.

| Condition | Result |
| --- | --- |
| `layout != "layer_first"` | `NotImplementedError` |
| `page_size != 1` | `NotImplementedError` |
| host `page_size` != device `page_size` | `NotImplementedError` |
| `start_layer != 0` or `end_layer != layer_num` | `NotImplementedError` |
| `layer_shard_enabled` | `NotImplementedError` |
| `mtp_draft_device_pools` non-empty | `NotImplementedError` |
| `is_quantized_kv_cache` | `NotImplementedError` |
| `store_dtype not in (bf16, fp16)` | `NotImplementedError` |
| `head_dim != v_head_dim` | `NotImplementedError` |
| `use_hnd` | `NotImplementedError` |
| `head_num * head_dim != 1024` | `ValueError` |
| row not a multiple of 128 B | `ValueError` |
| JIT mover unavailable | `NotImplementedError` |

Raised at transfer time:

| Condition | Result |
| --- | --- |
| `io_backend != "kernel"` | `NotImplementedError` |
| `is_draft=True` | `NotImplementedError` |

Refused rather than implemented: `--hicache-storage-backend` (L3). The
`mha_mxfp8.py:245-266` precedent — an encoded arena is not a BF16 `kv_buffer`,
so zero-copy backends cannot derive per-page pointers from it.

---

## Activation

```bash
SGLANG_EXPERIMENTAL_HICACHE_INT8=1 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --enable-hierarchical-cache \
  --hicache-mem-layout layer_first \
  --hicache-size 8 \
  --page-size 1 \
  --tp-size 1
```

Optional: `SGLANG_HICACHE_INT8_STAGING_TOKENS` (default 2048) sets the initial
device staging rows per direction. Transfers larger than this grow the buffer.

The dispatch branch in `get_mha_host_pool_cls` is checked **after** the MXFP8 and
asymmetric cases, because neither is representable in the INT8 record.
