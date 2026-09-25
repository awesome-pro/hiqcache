# SGLang HiCache internals — main @ `434c2e3adcd77afe42d6604b693d97aef71938e6`

Reference for the SGLang HiCache memory hierarchy, taken from a fresh clone:

```
git clone --depth 1 https://github.com/sgl-project/sglang.git
```

HEAD commit `434c2e3adcd77afe42d6604b693d97aef71938e6`, authored **2026-09-25 15:06:24 +0800**,
subject `[HiCache] fix: Drain pending backups before internal Mamba write-back (#41092)`.

No files in SGLang were modified. Local clone used for the trace: `/tmp/sglang_trace`.

Sections A–H document the upstream code as it stands at that commit: the L1 device pools,
the L2 host pools, the exact eviction and restore paths, the Qwen3-8B page geometry, the
stream and event semantics, the seams where a codec can be inserted, the state of the
upstream KV-compression RFC, and the existing host-cache round-trip tests. Section I
describes the INT8 host-cache codec integration this project builds on top of that code;
its line references also point at `434c2e3a`.

---

## A. Device KV pool representation (L1)

All in `python/sglang/srt/mem_cache/memory_pool.py`.

| Item | Value |
| --- | --- |
| Base class | `KVCache(abc.ABC)` `memory_pool.py:1822` |
| Dense MHA pool (Qwen3-8B) | `MHATokenToKVPool(KVCache)` `memory_pool.py:1971` |
| MLA pool | `MLATokenToKVPool(KVCache)` `memory_pool.py:4419` |
| Siblings | `NoOpMHATokenToKVPool:3151`, `MHATokenToKVPoolFP4:3265`, `PageMajorMHATokenToKVPool:3419`, `MHATokenToKVPoolMXFP8:3491`, `MHATokenToKVPoolMXFP8:3491`, `DSATokenToKVPool:4893` |

Representation: **one Python list of tensors per layer**, K and V separate.

`MHATokenToKVPool._kv_buffer_shapes()` `memory_pool.py:2287-2298`:

```python
rows = self.size + self.page_size
return ((rows, self.head_num, self.head_dim),
        (rows, self.head_num, self.v_head_dim))
```

`_create_buffers_normal()` `memory_pool.py:2300-2351`:

```python
self.k_buffer = [torch.zeros(k_shape, dtype=self.store_dtype, device=self.device)
                 for _ in range(self.layer_num)]   # v_buffer identically, :2348-2351
```

So `k_buffer: list[Tensor]` and `v_buffer: list[Tensor]`, indexed by **local** layer
(`local_layer_id = layer_id - self.start_layer`, `memory_pool.py:2530`, `:2554`).

Scalars on the pool: `size` (= `max_total_num_tokens`), `page_size`, `dtype`,
`store_dtype`, `layer_num`, `start_layer`, `end_layer`, `device`, `head_num`
(already TP/DCP-divided), `head_dim`, `v_head_dim`, `row_dim`, `v_row_dim`,
`kv_cache_layout`, `use_hnd`, `num_pages`, `quant_method`, `is_quantized_kv_cache`.
Accessors: `get_kv_buffer_shape():1903`, `get_kv_size_bytes():2446`, `get_contiguous_buf_infos():2468`,
`get_key_buffer():2545`, `get_value_buffer():2569`.

Dtype decision: `KVCache.__init__` `memory_pool.py:1850-1854`
(fp8 family ⇒ `store_dtype = torch.uint8`). Quantized recipes override it in
`_create_quantized_buffers` `memory_pool.py:2187-2195` via
`quant_method.create_buffers(...)` `memory_pool.py:2180-2186`. The `--kv-cache-dtype`
string is mapped in `configure_kv_cache_dtype()` `python/sglang/srt/mem_cache/kv_cache_dtype.py:23`.

Construction: `model_executor/model_runner.py:596-627` → `KVCacheConfigurator.configure()`
`kv_cache_configurator.py:340` → `_build_token_to_kv_pool:1190` → `_build_mha_kv_pool:1950`.
`mem_cache/kv_cache_builder.py:262 build_kv_cache` does **not** build the device pool; it
reads `tp_worker.model_runner.token_to_kv_pool` (`kv_cache_builder.py:288`).

---

## B. Host KV pool representation (L2)

Base: `python/sglang/srt/mem_cache/pool_host/base.py:156 class HostKVCache(abc.ABC)`.
Class attrs: `dcp_size = 1`, `dcp_rank = 0`, `shared_allocation_domain = None`
(`base.py:157-159`). **`wants_positions` does not exist** (0 grep hits repo-wide).

Abstract surface (`base.py`):
- `get_size_per_token()` `:281`
- `init_kv_buffer()` `:319`
- `load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, io_backend, *, is_draft=False)` `:323-337`
- `backup_from_device_all_layer(device_pool, host_indices, device_indices, io_backend)` `:339-346`
- `get_data_page(index, flat=True)` `:397`
- `get_dummy_flat_data_page()` `:404`
- `set_from_flat_data_page(index, data_page)` `:412`

Concrete: `MHATokenToKVPoolHost(HostKVCache)` `pool_host/mha.py:123`; MLA sibling
`MLATokenToKVPoolHost` `pool_host/mla.py:66`; MXFP8 sibling `pool_host/mha_mxfp8.py:38`;
asymmetric `pool_host/mha.py:1172`. Selection: `get_mha_host_pool_cls` `mha.py:1510-1531`,
called from `build_kv_host_pool` `hybrid_cache/hybrid_pool_assembler.py:140-175`.

**Host layout** (`mha.py:222-263`, layout default `page_first`, `arg_groups/fields/memory.py:133-147`):

| `--hicache-mem-layout` | `kv_buffer` shape |
| --- | --- |
| `layer_first` | `(2, layer_num, size, head_num, head_dim)` |
| `page_first` | `(2, size, layer_num, head_num, head_dim)` |
| `page_first_direct` | `(2, page_num, layer_num, page_size, head_num, head_dim)` |
| `page_head` | `(2, page_num, head_num, page_size, layer_num, head_dim)` |

`k_buffer`/`v_buffer` are properties over `kv_buffer[0]`/`[1]` (`mha.py:290-296`).
`token_stride_size = head_num*head_dim*itemsize` (`mha.py:247`),
`layout_dim = token_stride_size * layer_num` (`mha.py:248`).
`get_size_per_token()` `mha.py:208-213` = `head_dim*head_num*layer_num*itemsize*2`.

Allocation (`mha.py:250-262`) goes through `ALLOC_MEMORY_FUNCS` `pool_host/common.py:325-332`;
default is `alloc_with_host_register` `common.py:236-251` (`cudaHostRegister` over an mmap'd
arena — `storage/mmap/mmap_allocator.py:112-168`, optional `MAP_HUGETLB`), so **the host pool
is pinned by construction**. `pin_memory=True` default (`mha.py:134`).

Sizing (`base.py:194-208`): `size_per_token = get_size_per_token()`,
`size = int(device_capacity * host_to_device_ratio)` or from `--hicache-size`,
then `page_num = size // page_size + 1; size = page_num * page_size`.

Index space: `host_indices` are **token slots** in `[0, logical_size)`,
`logical_size = size * dcp_size` (`base.py:462-465`); allocation is page-granular
(`alloc` asserts `need_size % logical_page_size == 0`, `base.py:472-492`).
`get_data_page(index)` maps a page-aligned slot to `kv_buffer[:, index:index+page_size]`
(`mha.py:586-598`); `set_from_flat_data_page` is the inverse (`mha.py:608-640`).

`io_backend` is one of exactly `kernel | direct | kernel_ascend`
(`arg_groups/fields/memory.py:126-132`, **default `kernel`**), dispatched by an
`if io_backend == ...` chain inside each pool: H2D `mha.py:317/383/412`, D2H `mha.py:480/547/566`.
Tensor placement per backend: `managers/cache_controller.py:869-909 move_indices`.

`memory_pool_host.py` is **not** a shim: it holds the DeepSeek-V4 byte-row pools
`LogicalHostPool:59`, `DeepSeekV4PagedHostPool:206`, `DeepSeekV4StateHostPool:714`.

---

## C. Exact L1→L2 eviction and L2→L1 restore functions

### Naming reality check

- **`write_backup`, `load_backup`, `_evict_backup` do not exist at HEAD** (repo-wide grep: 0 hits).
  The hierarchical path is `UnifiedRadixCache` (`unified_radix_cache.py:162`).
  `UnifiedRadixCache.write_backup_storage` `:1878` is L2→**L3** (host→storage), not L1→L2.
- `RadixCache.evict()` `radix_cache.py:555` is **L1-only** (heap over `evictable_leaves`
  via `eviction_strategy.get_priority` `:562-565`, then
  `token_to_kv_pool_allocator.free_segment(x.value, start_pos=0)` `:572`, `_delete_leaf` `:574`).
  No copy. `RadixCache._inc_hit_count` `:692` only feeds eviction priority.
- **There is no L1↔L2 thread.** `write_queue`/`load_queue` are plain lists
  (`cache_controller.py:358-359`; the `PriorityQueue` variant is commented out at `:357`) and are
  drained **inline on the scheduler thread** onto the dedicated streams (`:366`), so
  `CacheOperation.__lt__` `:172` is vestigial. Only L3 has threads:
  `prefetch_thread_func:1226`, `prefetch_io_aux_func:1160`, `prefetch_sync_thread_func:1329`,
  `backup_thread_func:1312`, plus `hybrid_cache_controller.py:1094`.

### L1→L2 (write-back eviction)

```
scheduler asks for device tokens
  UnifiedRadixCache.evict_for_alloc(params)                     unified_radix_cache.py:601
    → UnifiedRadixCache._evict(params, available_size_targets)  unified_radix_cache.py:696
        (write_back flush gate :715-719 writing_check(write_back=True))
      → UnifiedRadixCache._evict_components(...)                unified_radix_cache.py:798
          walk loop :848-882
        → UnifiedRadixCache._evict_device_next_node(ct, tracker) unified_radix_cache.py:750
            → UnifiedTreeCore.evict_device_next_node(...)        unified_cache/unified_tree_core.py:1554
                → components/base.py:520
                → FullComponent._evict_device_next_node          unified_cache/components/full.py:204
                  (cursor heap built in _evict_device_start       components/full.py:194)
        → UnifiedRadixCache._evict_device_leaf(node_id, tracker) unified_radix_cache.py:764
            → UnifiedTreeCore.evict_device_leaf(node_id, is_write_back)
                                                               unified_cache/unified_tree_core.py:1582
              (unbacked + write_back ⇒ returns result.backup_kv, :1593-1598;
               already-backed-up node ⇒ _demote, :1607)
        → UnifiedRadixCache._execute_and_commit_kv_backup(backup_kv, write_back=True)
                                                               unified_radix_cache.py:1576
            → UnifiedTreeCore.build_backup_spec(node_id)         unified_tree_core.py:2203
            → UnifiedRadixCache._build_backup_sidecar(...)       unified_radix_cache.py:1625
            → UnifiedRadixCache._execute_kv_backup(node_id, device_value, comp_xfers, sidecar_xfers)
                                                               unified_radix_cache.py:1632
                → CacheController.write(device_value, node_id=..., extra_pools=..., flush=False)
                                                               unified_radix_cache.py:1648
                    → HybridCacheController.write             hybrid_cache/hybrid_cache_controller.py:513
                        → allocate_host_transfers(...)         hybrid_cache_controller.py:477
                            → HostPoolGroup.alloc(...)          pool_host/group.py:165/174
                                → HostKVCache.alloc(need_size)  pool_host/base.py:472
                        → CacheOperation(host_indices, device_indices, ...)  managers/cache_controller.py:100
                        → write_queue.append(...)              hybrid_cache_controller.py:526
            → UnifiedRadixCache.flush_pending_backups()          unified_radix_cache.py:3489
                → CacheController.start_writing()               managers/cache_controller.py:822
                    → CacheOperation.merge_ops(write_queue)     managers/cache_controller.py:826/:152
                    → _move_write_operation(op)                 cache_controller.py:827 / :911
                    → L2TransferEngine.submit_device_to_host(transfers)
                                                               mem_cache/l2_transfer.py:131
                        → host_pool.backup_from_device_all_layer_physical(device_pool,
                              host_indices, device_indices, io_backend)   l2_transfer.py:136
                            → base.py:371 → MHATokenToKVPoolHost.backup_from_device_all_layer(...)
                                                               pool_host/mha.py:465-584
                                kernel: jit_transfer_hicache_all_layer(...)   mha.py:483
                                        (JIT `HiCacheKernel::run_all`, stream = device_to_host_stream)
        → UnifiedRadixCache._demote(node_id, tracker)            unified_radix_cache.py:779 → :857
  completion observed via HiCacheAck.finish_event (ack_write_queue)
      writing_check(write_back=True) unified_radix_cache.py:3266 → finish_event.synchronize() :3279
      → _finish_write_through_ack :1691 → _demote :779 → UnifiedTreeCore.demote utc.py:1836
```

Byte movement: `python/sglang/kernels/jit/csrc/kvcacheio/hicache.cuh`
(`hicache_transfer_all_layer`, `HiCacheKernel::run_all:398`), JIT-compiled by
`python/sglang/kernels/ops/kvcache/hicache.py:_jit_hicache_module:32` and launched at
`kernels/ops/kvcache/hicache.py:271` → `module.launch_all:313`.
Non-JIT fallbacks: `transfer_kv_all_layer` / `transfer_kv_all_layer_lf_pf` /
`transfer_kv_all_layer_lf_ph` from `sgl_kernel.kvcacheio` (`mha.py:496-546`);
`direct` backend `transfer_kv_direct` / `transfer_kv_all_layer_direct_lf_pf` (`mha.py:549-562`)
→ `kernels/aot/python/sgl_kernel/kvcacheio.py:206` → `torch.ops.sgl_kernel.transfer_kv_direct`.
There is **no Python-level `copy_`/`cudaMemcpy` on the GPU hot path**; the only CPU fallback is
`pool_host/unified.py:871` (`dst.index_copy_(...)`, CPU-device tests).

### L2→L1 (restore / load-back)

```
schedule_policy checks req.needs_host_load_back()          managers/schedule_policy.py:1323
  → UnifiedRadixCache.init_load_back(params)               unified_radix_cache.py:3380
      → UnifiedRadixCache.load_back(node_id, mem_quota, req)  unified_radix_cache.py:1708
          → UnifiedRadixCache._load_back_transfers(...)     unified_radix_cache.py:1745
              → UnifiedTreeCore.build_load_back_spec(node_id, req=req)  :1756
                                       unified_cache/unified_tree_core.py:2276
              → CacheController.load(host_indices, node_id=..., extra_pools=...)
                                                           unified_radix_cache.py:1801
                    → HybridCacheController.load       hybrid_cache/hybrid_cache_controller.py:803
                        (device alloc hybrid_cache_controller.py:817 / cache_controller.py:861)
                        → load_queue.append(CacheOperation(...))  hybrid_cache_controller.py:836
              → UnifiedTreeCore.commit_load_back(node_id, device_indices, ...)  :1811
  → UnifiedRadixCache.ready_to_load_host_cache()            unified_radix_cache.py:3495
      → CacheController.start_loading()                     managers/cache_controller.py:952
          (invoked once per batch from managers/scheduler.py:4084)
          → update_producer() :956, merge_ops :957, producer_event.start_event.record() :961
          → L2TransferEngine.submit_host_to_device(transfers, start_event=...,
                on_layer_done=producer_event.complete,
                transfer_layer_id_max=self.transfer_layer_id_max)   cache_controller.py:971
              → for layer_id in range(transfer_layer_id_max):
                    transfer.layer_mapper(layer_id)                     l2_transfer.py:158
                    host_pool.load_to_device_per_layer_physical(
                        device_pool, host_indices, device_indices,
                        local_layer_id, io_backend, is_draft=...)      l2_transfer.py:169
                      → pool_host/base.py:378
                      → MHATokenToKVPoolHost.load_to_device_per_layer(...)
                                                               pool_host/mha.py:298-435
                          kernel: jit_transfer_hicache_one_layer   mha.py:320/345
                                  → kernels/ops/kvcache/hicache.py:228
                          else transfer_kv_per_layer*             mha.py:331/:356/:368/:385/:399
                          (JIT `HiCacheKernel::run_one`, stream = host_to_device_stream)
                    on_layer_done(layer_id)                             l2_transfer.py:177-178
  forward pass per layer: MHATokenToKVPool.get_key_buffer(layer_id)
      → self.layer_transfer_counter.wait_until(layer_id - start_layer)
                                                           memory_pool.py:1532/:1540, :2545-2549, :2569-2571
          → LayerDoneCounter.wait_until → LayerLoadingEvent.wait
              → device_module.current_stream().wait_event(load_events[i])
                                                           managers/cache_controller.py:63-64, :85-90
  request-level completion: UnifiedRadixCache.loading_check()  unified_radix_cache.py:3324
      → ack.finish_event.synchronize() :3349 → ongoing_load_back.pop :3356
      → UnifiedTreeCore.finish_load_back(node) :3360
      (event-only gate: is_load_back_event_done :3503 → finish_event.query())
```

`CacheOperation` `managers/cache_controller.py:100-173`: fields
`host_indices`, `device_indices`, `node_ids` `:113`, `data` `:114`,
`pool_transfers` `:115`, `id` `:117-118`, `priority` `:120`;
`merge_ops` `:151-170` concatenates indices, unions `node_ids` in op order, takes
`min(priority)`, merged node id `-1`; `_merge_pool_transfers` `:122-149` groups by
`(transfer.name, indices_from_pool)`. **FIFO, no sorting heap.**
Consequence: one merged op = one copy call + one `HiCacheAck` carrying many `node_ids`
(`:838`, `:982`).
Producer/consumer events: `LayerLoadingEvent` `:53-68`, `LayerDoneCounter` `:71-98`
(3 rotating counters, `num_counters = 3`; rollover guard `update_producer` `:80-85`).
Counter registration: `memory_pool.py:1429 register_layer_transfer_counter`
(`tp_worker.py:529`, `hybrid_pool_assembler.py:2213`).
Host eviction from L2: `CacheController.evict_host` `:991-996` → `mem_pool_host.free(host_indices)`;
`UnifiedRadixCache.evict_host` `:1346`; `UnifiedTreeCore.drive_host_eviction` `utc.py:1703`.

Policy vs. data movement are cleanly separated: policy is
`eviction_strategy` / `RadixCache.evict` `radix_cache.py:555` / `EvictPolicy`
`mem_cache/evict_policy.py:10-100` (+ factory `mem_cache/utils.py:72`); decision to back up vs
drop is `unified_tree_core.py:1582 evict_device_leaf(is_write_back)` and
`unified_radix_cache.py:865 _drop_subtree_no_host`; data movement is everything above
`L2TransferEngine`.

---

## D. Page/block shape for Qwen3-8B

Config (verified in-repo at `tools/sglang-simulator/test/assets/qwen3-8b/config.json`):
`num_hidden_layers=36`, `num_attention_heads=32`, `num_key_value_heads=8`,
`head_dim=128`, `hidden_size=4096`, `torch_dtype=bfloat16`, `use_sliding_window=false`
⇒ all 36 layers are full attention, so the host pool has `layer_num = 36`.
Corroborated by `test/registered/kernels/benchmark/attention/bench_fused_qknorm_rope.py:48`
(`(128, 32, 8, 8),  # typical (e.g. Qwen3-8B)`).

`page_size`: default resolved to **1** by `_page_size_default`
`arg_groups/overrides.py:1275-1297` (64 only for HIP + `SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`,
or MUSA). Field: `arg_groups/fields/schedule.py:139-141`.
`head_num = num_key_value_heads / attn_tp_size` (`kv_cache_configurator.py:1970-1972`,
`configs/model_config.py:1507-1518`); `v_head_dim == head_dim` for Qwen3
(`model_config.py:1162-1165`), so K and V are symmetric.

**Device (NHD, the default; `SGLANG_USE_HND_KVCACHE` default False, `environ.py:572`):**

| TP | `head_num` | k_buffer per layer | v_buffer per layer | bytes/token/layer | bytes/token (36 layers) |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | `(size+1, 8, 128)` bf16 | `(size+1, 8, 128)` bf16 | 8·128·2·2 = **4096** | **147,456 B = 144 KiB** |
| 2 | 4 | `(size+1, 4, 128)` | `(size+1, 4, 128)` | 2048 | 73,728 B = 72 KiB |
| 4 | 2 | `(size+1, 2, 128)` | `(size+1, 2, 128)` | 1024 | 36,864 B = 36 KiB |

The `+1` extra row is the padded page that absorbs dummy padded-token writes
(`memory_pool.py:2307`, MLA `:4480`). With NHD there is **no page dimension**: dim 0 is a
flat token slot, `slot = page_index*page_size + offset`
(`memory_pool.py:2365-2372`). Pages only become dim 0 under HND or `vectorized_5d`.

**Host** (`page_first`, TP=1): `kv_buffer` shape `(2, host_size, 36, 8, 128)` bf16,
`size_per_token = 147,456 B` (`mha.py:208-213`) — identical to the device bytes/token, as
required. `k_buffer = kv_buffer[0]`, `v_buffer = kv_buffer[1]`.
`host_size = page_num * 1` with `page_num = int(device_capacity * hicache_ratio) + 1`
(`base.py:200-208`).

Useful derived constants for a codec at TP=1, page_size=1:

- K row per (layer, token) = `8*128*2 = 2048 B`; V row identically.
- One raw page (1 token, all 36 layers) = `2 * 36 * 2048 = 147,456 B`.
- `element_dim = head_num*head_dim = 1024`; JIT `element_size = element_dim*itemsize = 2048 B`
  (`mha.py:154,159-162`). 2048 % 128 == 0, so the CUDA JIT path is taken.

---

## E. Synchronization / stream semantics

`python/sglang/srt/mem_cache/l2_transfer.py`:

- `device_module = get_device_module()` `:14`; `device_to_host_stream` `:56`,
  `host_to_device_stream` `:57`, both `device_module.Stream()`.
- `_submission(...)` `:102-129` is the only synchronization surface:
  - `:104` `_start_event(start_event)` (`:181-186`) — records a fresh event on the
    caller's current stream when none is supplied;
  - `:112` `with device_module.stream(stream):`;
  - `:114` `start_event.wait(stream)` — the transfer stream waits for the producer;
  - `:115` `_prepare_transfers(...)` — index translation runs on the transfer stream;
  - `:116/:118` `ack_start.record()` / `ack_finish.record()` on the transfer stream;
  - `:120` `_record_stream(...)` `:188-197` — `indices.record_stream(stream)` for all
    CUDA index tensors **and temporaries**;
  - `:122` `stream.synchronize()` only in the exception path;
  - `:125-129` shared-layout lease released gated on `ack_finish`.

- Backup (D2H) runs on `device_to_host_stream` (`:131-142`). Host data may only be read
  after the `HiCacheAck.finish_event` in `ack_write_queue` is ready
  (`unified_radix_cache.py:3201`).
- Load (H2D) runs on `host_to_device_stream` (`:144-179`); `on_layer_done(layer_id)` fires
  immediately after that layer's copies are **enqueued** (`:177-178`).

`managers/cache_controller.py`:
- `start_loading()` `:952-989`: `update_producer()` `:956`, `producer_event.start_event.record()` `:961`;
- **WAR fence** `:963-969`: `host_to_device_stream.wait_stream(self.load_fence_stream)`,
  where `load_fence_stream = model_runner.forward_stream` (`scheduler.py:595-597`,
  `model_runner.py:429`) — protects reclaimed pages still being written by the forward thread;
- submit with `start_event=producer_event.start_event`, `on_layer_done=producer_event.complete`,
  `transfer_layer_id_max = mem_pool_device.layer_num` `:971-976`;
- `LayerLoadingEvent.complete(i)` records event `i` on the transfer stream (`:59-61`);
  the forward stream does `wait_event` on exactly the layer it is about to consume.

**Contract a codec must satisfy**

1. Any codec work for a load must be enqueued on `host_to_device_stream` **before** that
   layer's `on_layer_done(layer_id)` call. That per-layer event is the only thing the model's
   forward stream waits on; work enqueued after `ack_finish.record()` is unsynchronized.
2. No host-side synchronization inside the codec (no `.cpu()`, no `.item()`, no
   `synchronize()`), or the L1↔L2 pipeline serializes.
3. Temporary GPU buffers touched by the transfer stream need `record_stream()`,
   mirroring `_record_stream` `:188-197`.
4. Host staging must be pinned (already guaranteed by `pin_memory=True` +
   `alloc_with_host_register`).

NPU/Ascend uses the same `L2TransferEngine` stream pair (`get_device_module()` returns
`torch.npu`); `--hicache-io-backend kernel_ascend` uses `transfer_kv_dim_exchange`
(`pool_host/mha.py:568-585`) and `to_device_no_sync` (`pool_host/npu_memfabric.py:124-158`).

---

## F. Where to insert compression/decompression with the smallest patch

There is **no codec abstraction in main** — 0 hits for `codec`, `wants_positions`,
`HostKVCacheFactory`, `codec_config`, `--hicache-kvcache-codec`.

**Single narrowest seam:** `pool_host/mha.py:465 backup_from_device_all_layer` (encode) and
`pool_host/mha.py:298 load_to_device_per_layer` (decode). Every L1↔L2 byte in HiCache funnels
through those two methods (`l2_transfer.py:136` and `:169`); nothing else needs touching for a
per-layer codec. A second, slightly higher seam is
`l2_transfer.py:131 submit_device_to_host` / `:144 submit_host_to_device`, which is where
PR #40551 hooked in (see §G) — but that requires `L2Transfer`/`CacheOperation` plumbing.

Three shapes are viable, ranked by patch size. The integration described in §I takes the
first; the other two are alternatives it does not use:

1. **Subclass of `MHATokenToKVPoolHost` holding an encoded arena.** New file
   `pool_host/mha_int8.py` modelled directly on the existing `pool_host/mha_mxfp8.py`
   precedent, plus a ~10-line dispatch added to `get_mha_host_pool_cls`
   (`mha.py:1510-1531`). Overrides: `get_size_per_token`, `init_kv_buffer`,
   `backup_from_device_all_layer`, `load_to_device_per_layer`, `get_data_page`,
   `set_from_flat_data_page`, `get_dummy_flat_data_page`, `get_page_buffer_meta`,
   `get_hybrid_pool_buffer`. **No other SGLang subsystem changes.** This is the smallest
   shape that actually reduces host bytes; it is the one this project implements, together
   with two self-contained helper modules alongside it (§I.3).
2. Decorator/wrapper around an existing host pool instance, injected at
   `build_kv_host_pool` (`hybrid_pool_assembler.py:140-175`). Smaller diff, but it must
   proxy ~20 attributes and breaks `isinstance` checks
   (`mooncake_store.py:793-809`, `tensorcast_store.py:449-456`,
   `hybrid_cache_controller.py:1414`), so it is more fragile in practice.
3. Add `wants_positions`/streaming hooks to `HostKVCache` (the RFC's route). Correct
   long-term, much larger patch, and unnecessary for a position-independent scalar codec.

**The kernel already supports an arbitrary per-layer byte layout, so no CUDA change is
needed.** `HiCacheKernel::run_all` (`hicache.cuh:398`) takes `k_ptr_dst`/`v_ptr_dst` as
`uint64` arrays with `N = num_layers` and reads element `pos` at
`ptr[layer] + pos * kv_cache_dst_stride_bytes`; `run_one` (`:329`) does the same per layer via
`TensorMatcher({-1, D}).with_strides({M, 1})`. `kElementSize` only has to satisfy
`group_fits(element_size, lanes_per_worker, 128)` on CUDA (`hicache.cuh:73-83`) — i.e.
**`element_size % 128 == 0`**.

Constraints a codec must respect:
- `get_size_per_token()` is called at `pool_host/base.py:194`, i.e. **before**
  `init_kv_buffer()` at `:254`; the codec's override must compute encoded bytes/token from
  `self.device_pool` directly (the MHA override at `mha.py:208-213` already does exactly this
  to seed `head_num`/`head_dim`/`layer_num`).
- The per-slot byte size is **fixed at construction time** and assumed everywhere:
  `size_per_token` (`base.py:194`, `mha.py:810`, `pool_host/unified.py:669-670`),
  `CacheController._transfer_num_bytes` `cache_controller.py:846`
  (`= len(op.device_indices) * mem_pool_host.size_per_token`), and the `element_size`
  reporting in `get_page_buffer_element_size` `base.py:348` /
  `get_page_buffer_meta` `pool_host/unified.py:1015/1028`. A codec that changes the slot size
  must keep all of these coherent.
- The encoded row per (layer, K|V) must be a multiple of 128 B to keep the JIT path.
- 4-D-per-token shape (`head_num, head_dim`) remains interpretable: a bit-exact / symmetric
  codec can encode *within* a token's row and keep the row addressable exactly as today.
- `can_use_write_back_jit` / staging buffers (`_init_write_back_staging_buffers`
  `mha.py:265-288`) must be disabled for the codec pool — the staged path
  (`jit_transfer_hicache_all_layer_staged_lf_pf`) writes raw bf16 into `k_buffer`.
- Host index space and page alignment must be preserved (`alloc` asserts
  `need_size % logical_page_size == 0`, `base.py:472-492`; `get_data_page` uses
  `index // page_size` for the page-direct layouts, `mha.py:592`).
- **One logical transfer fans out to several host pools.** `HybridCacheController._l2_transfers`
  `hybrid_cache_controller.py:686-720` emits one `L2Transfer` per `PoolEntry` (anchor +
  sidecars + packed draft pools), each with its own `host_pool`/`device_pool`/`layer_mapper`,
  so `l2_transfer.py:157-176` issues `layers × transfers` per-pool copy calls and
  `L2Transfer.host_pool` is the *individual* pool, never the `HostPoolGroup`. A codec is
  therefore applied per pool, and the anchor/sidecar split must be decided explicitly.
- The `direct` io_backend sorts `host_indices` and applies the permutation to
  `device_indices` (`cache_controller.py:877-883`); the codec must treat
  `(host_indices, device_indices)` as an aligned pair, exactly as the kernels do.
- L3 storage registers `get_hybrid_pool_buffer()` for zero-copy I/O
  (`storage/nixl/hicache_nixl.py:417,458-461`, `storage/umbp/umbp_store.py:1045`) and
  derives per-page pointers/sizes from `get_page_buffer_meta(...)`. An encoded arena
  changes both, so v1 rejects `--hicache-storage-backend` (the
  `mha_mxfp8.py:245-266` `_storage_pages_unsupported()` precedent).
- Retraction (`--disaggregation-decode-retraction-backup=host_pool`) bypasses the queues:
  `UnifiedRadixCache.backup_kv_cache` `:1465` / `restore_kv_cache` `:1505` call
  `l2_transfer_engine` directly and `finish_event.synchronize()` immediately. That path uses
  `allocate_host_transfers` `:1470`, so a codec slot-size change reaches it — but each
  retraction re-encodes reconstructed KV, i.e. lossy error can accumulate there
  (RFC §"Lossy recompression").

---

## G. RFC #36522 — interfaces and what has / hasn't landed

Source: [sgl-project/sglang#36522 — "[RFC] KV Cache Compression API (KVTC, nvCOMP, ...)"](https://github.com/sgl-project/sglang/issues/36522),
opened 2026-08-26 by `alancucki`, **state: open**, based on SGLang v0.5.18.
One comment so far (2026-09-25, `ch-wan`): a codec-per-`(codec, cache-type)` `HostKVCache`
subclass "does not seem very scalable"; he proposes decoupling codec from `HostKVCache`.

Proposed but **NOT present in main** (all verified by grep on the clone):

| Proposed symbol | Status in `434c2e3a` |
| --- | --- |
| `--hicache-kvcache-codec` flag | absent |
| `HostKVCacheFactory.register/resolve` | absent |
| `codec_config` constructor arg | absent |
| `HostKVCache.wants_positions: ClassVar[bool]` | absent (0 hits) |
| `positions` kwarg on `load_to_device_per_layer` / `backup_from_device_all_layer` | absent (0 hits in `cache_controller.py`) |
| `CacheOperation.positions` + merge/permutation rules | absent |
| `load_to_device_layers_streaming()` generator | absent |
| `create_host_kv_pool()` shared dispatch helper | absent |
| Per-page compression-state metadata | absent |

Proposed and **already true in main** (so the RFC's target state partly exists):
`_submission`/`submit_host_to_device` already provide the documented ordering and
`on_layer_done` semantics; `L2TransferEngine` already walks layers and supports
`layer_mapper`/`is_draft`; `HostKVCache` already has the four operations plus
`get_data_page`/`set_from_flat_data_page`; `merge_ops` already exists.

Related work found (not merged into main):
- [#40551 "[Draft Feature] Add a shared KV compression framework for P/D transfer and HiCache L2"](https://github.com/sgl-project/sglang/pull/40551)
  — **open draft**, 5 commits, 46 files, +6733/−71, branch `bytedance/pd-hicache-compression`
  against main, last updated 2026-09-21. Introduces `sglang/srt/kv_compression/{backend,host_io,layout,provider,runtime,store,types,verification}.py`,
  `mem_cache/hicache_compression.py`, `mem_cache/hicache_lifecycle.py`, `mem_cache/l2_completion.py`,
  a `CompressedHostKVCache` byte-arena store, and an `async_state` parameter threaded into
  `L2TransferEngine.submit_device_to_host`. Reference codec is nvCOMP LZ4 (lossless).
  Its `KVLayoutAdapter` hard-restricts to **BF16 MHA/GQA with `page_size=1`** and one token per
  page. It is a parallel design to #36522, not an implementation of it.
- [#39740 "[RFC] Pluggable KV Compression for Disaggregated Serving"](https://github.com/sgl-project/sglang/issues/39740) — open.
- [#30419 "[RFC] KVTC KV-cache compression method for SGLang"](https://github.com/sgl-project/sglang/issues/30419) — closed.
- Codec kernels that exist but are unrelated to HiCache L2: MXFP4 DSV4
  ([#37136](https://github.com/sgl-project/sglang/pull/37136)), MXFP8 MHA host pool
  (`pool_host/mha_mxfp8.py`, merged), KVarN ([#31967](https://github.com/sgl-project/sglang/pull/31967), open),
  TurboQuant ([#23133](https://github.com/sgl-project/sglang/pull/23133), closed).

---

## H. Tests for host-cache round trips

Kernel/pool-level round trips (the closest thing to a codec conformance test):

- `test/registered/kernels/ops/kvcache/test_hicache.py` — the primary file.
  - `_run_transfer_roundtrip_mha(layout, element_dim)` `:167`, `_run_transfer_roundtrip_mla` `:261`
  - `test_hicache_transfer_mha` `:546`, `test_hicache_transfer_mla` `:552`
    (parametrized over `LAYOUTS = ["layer_first", "page_first"]` and
    `MHA_ELEMENT_DIMS = [128, 256, 512, 1024]`, `MLA_ELEMENT_DIMS = [576]`; `:34-39`)
  - `_run_page_first_staged_write_back_mha` `:337`, `..._mla` `:452`,
    `test_hicache_page_first_staged_write_back_*` `:559-578`
    (page counts `[1, 63, 64, 65, 67, 128, 129]`)
  - `test_unified_l2_waits_for_index_producer[d2h|h2d][direct|kernel]` `:46` — the
    stream-ordering test: a delayed producer stream mutates `device_indices` and the
    transfer must observe the post-mutation value.
- `test/registered/kernels/ops/kvcache/test_hicache_page_first_write_back.py` —
  `_run_mha:116`, `_run_mla:199`, `test_page_first_staged_write_back_*:266,272`, plus
  registered-mmap pointer-domain tests `:280`, `:309`, `:366`.
- `test/registered/kernels/ops/kvcache/test_kvcacheio_asymmetric.py` —
  `test_asymmetric_mha_kernel_page_first_roundtrip:138`,
  `test_asymmetric_mha_direct_page_first_direct_roundtrip:173`.
- `test/registered/unit/mem_cache/test_hicache_copy_rounds.py` — JIT copy-round screening
  (`_screen` `:14`, 6 tests) mirroring `pick_group_bytes()` in `hicache.cuh`.

Pool bookkeeping / lifecycle:

- `test/registered/unit/mem_cache/test_mem_pool_host.py` — `TestHostKVCache:26`
  (`test_multiple_attention_rows_per_token:51`, double-alloc/free `:92,:105,:116,:124`,
  `test_shm_allocator:133`), `TestLazyHostPoolRelease:157`, `TestHostMemoryBudget:278`,
  `TestHostPoolGroup:315` (`test_host_reclamation_is_independent_of_transfer_order:371`).
- `test/registered/unit/mem_cache/test_hicache_load_back_timing.py` — `TestLoadBackDurationMetric:17`.
- `test/registered/unit/mem_cache/test_hicache_host_register.py`, `test_hicache_dcp_host_pool.py`,
  `test_hicache_staged_write_back_dispatch.py`, `test_hicache_pp_sync_drain.py`.
- `test/registered/unit/mem_cache/test_unified_hicache_regressions.py` —
  `TestTransferStreamOrdering.test_load_translation_follows_supplied_start_event:204`,
  `TestDirectBackendTranslation:271`, `TestHiCacheIndexDomains:25`.
- `test/registered/unit/mem_cache/test_mxfp8_mha_pool_host_unit.py`,
  `test_asymmetric_mha_pool_host_unit.py`, `test_minimax_sparse_pool_host_unit.py`,
  `test_dsa_pool_host_unit.py` — per-variant host-pool units (pattern to copy).
- `test/registered/unit/mem_cache/test_decode_retraction_backup.py` —
  `test_restores_target_and_draft_kv:233`, `test_host_receive_restores_target_and_draft_kv:413`.

End-to-end:

- `test/registered/hicache/test_hicache_storage.py`, `test_hicache_variants.py`
  (`TestHiCacheStandard:59`, `TestHiCacheMLA:76`, `TestHiCacheEagle:94`, `TestHiCachePage:127`),
  `test_pp_with_hicache.py`, `test_qwen35_hicache.py`, `test_hicache_storage_runtime_attach_detach.py`.
- `test/registered/e2e/hicache/test_hicache_unified_memory.py`.
- NPU: `test/registered/npu/basic_function/HiCache/test_npu_hicache_{mha,mla,mamba}.py`.

---

## I. INT8 host-cache codec integration design

This section is a design description, not part of the upstream trace: it specifies the INT8
L2 host-cache codec this project integrates into the HiCache paths documented above. All
upstream line references in it still point at `434c2e3a`. The integration lands in this
project's SGLang fork as three new files under `python/sglang/srt/mem_cache/pool_host/` —
`int8_codec.py` (record quantise / pack / decode), `int8_staging.py` (device staging buffers
and pointer tables), `mha_int8.py` (`MHATokenToKVPoolHostINT8`) — plus one dispatch branch in
`get_mha_host_pool_cls` and the `SGLANG_EXPERIMENTAL_HICACHE_INT8` /
`SGLANG_HICACHE_INT8_STAGING_TOKENS` env knobs. Full source map: `docs/sglang-integration.md`.

### I.1 Blocking facts

- Host ARENA bytes/token at TP=1, page_size=1 is `147,456` for a raw page.
- Qwen3-8B K and V are symmetric (`v_head_dim == head_dim == 128`), `head_num = 8`,
  128 elements per head per token per K/V → 256 B bf16 per (K|V, head, token).
- The JIT transfer kernel is a **pure element-wise byte mover** with a configurable
  per-layer destination pointer and destination stride, and requires
  `element_size % 128 == 0` on CUDA.
- A page is `page_size` consecutive token slots; the codec touches whole pages only.

### I.2 Codec v1: symmetric per-(layer, K|V, head) INT8

Quantization granularity: one scale per `(layer, k|v, head, token)` group of **128 elements**.
This is exactly one `head_dim` row, so the transform is a pure element-wise/row-wise map
and both the encode and the decode can be expressed as `torch` ops on the transfer stream
(no custom CUDA kernel in v1).

**Encoded record.** v1 packs the whole row into one aligned record per
`(token, layer, K|V)`: `1024 B` INT8 payload (`8 heads * 128 B`) plus `16 B` bf16 scales
plus `112 B` padding = **1152 B**, and `1152 = 9 * 128`. One row width therefore serves both
the host arena and the staging buffers, so every copy stays on the existing JIT kernel.
Per token per (layer, K|V) that is `1152 B` versus 2048 B raw ⇒ **1.78×**; one Qwen3-8B page
(1 token, 36 layers) is `147,456 B → 82,944 B`.

Two further encodings were considered and are not the one v1 packs:

- **Payload-only rows plus a separate scale arena.** Storage per token per (layer, K|V):
  `8 heads * 128 B = 1024 B` int8 payload plus `8 * 2 B = 16 B` bf16 scales → **1040 B**,
  versus 2048 B raw ⇒ **1.97×**; total for one Qwen3-8B page (1 token, 36 layers):
  `147,456 B → 74,880 B`. **Alignment for the JIT kernel:** the payload is the kernel
  destination and the scales move in a second pass. Payload (`int8`, `uint8` view) per token
  per layer is `2 * 1024 = 2048 B` for K+V, which is `% 128 == 0`; with the arena packing
  K rows contiguously then V rows contiguously, `element_size = 2048` is admissible and
  `kv_cache_dst_stride_bytes = 2048`. Scales: a second, tiny registered arena with
  `element_size = 32` would break the 128 B round, so instead the scales move with plain
  `torch` index copies on the same stream
  (`scale_arena[page_slot, layer, k|v, head, :] = ...`), which is cheap
  (`8*2*2 = 32 B` per token per layer) and needs no kernel work. This keeps the fast path
  entirely on the existing JIT kernel.
- **Per-(layer, K|V) fixed scale, no side buffer.** Folding the scale into the payload row
  with a **per-(layer, K|V) fixed scale** (calibrated once, or `amax` over the whole page)
  makes the encoded row exactly `1024 B` (`% 128 == 0`), `element_size = 1024`,
  `kv_cache_dst_stride_bytes = 1024`, compression **2.0×**, and removes the side buffer —
  at the cost of per-token adaptivity.

All three need zero kernel changes. v1 takes the packed `1152 B` record because it keeps
per-token adaptivity, satisfies the 128 B round with a single row width, and needs no second
arena; the packed row also makes the D2H and H2D movers symmetric, at `element_size = 1152`
on both sides.

Arena layout (all uint8, one pinned allocation, page-major):

```
arena[page, K_part(2 * layer_num * enc_kv_row_bytes) | V_part(2 * layer_num * enc_kv_row_bytes)]
```

Per-page byte count `encoded_size_per_token * page_size`, exposed as
`get_size_per_token()`'s encoded analogue so `HostKVCache.__init__`
(`base.py:194-208`) sizes the arena automatically.

### I.3 File and function map

New file `python/sglang/srt/mem_cache/pool_host/mha_int8.py`:

```python
class MHATokenToKVPoolHostINT8(MHATokenToKVPoolHost):
    """BF16/FP16 MHA host pool that stores int8 payload + bf16 row scales."""
```

Alongside it, `pool_host/int8_codec.py` holds the row quantise/pack/decode and
`pool_host/int8_staging.py` the staging buffers and pointer tables.

Overrides, with the reason each is required:

| Override | Base / reference | Why |
| --- | --- | --- |
| `get_size_per_token()` | `mha.py:208-213` | Return `encoded_bytes_per_token`. Called at `base.py:194` before `init_kv_buffer`, so it must read `self.device_pool` directly and set `head_num/head_dim/layer_num` as the parent does. |
| `init_kv_buffer()` | `mha.py:222-263` | Allocate the uint8 arena through `ALLOC_MEMORY_FUNCS` (`common.py:325`) with `pin_memory=True`; build per-layer `k_data_ptrs`/`v_data_ptrs` via `make_kernel_ptr_table` (`common.py:296`); allocate the directional staging buffer pair here as well. Do not allocate bf16 `k_buffer`/`v_buffer`. |
| No use of the inherited staged `page_first` write-back path | `base.py:191`, `mha.py:265-288` | That staged path writes raw bf16 into `k_buffer`, so it cannot carry an encoded row. The encoded pool is restricted to `layer_first` at construction (§I.4) and owns its own staging buffers instead. |
| `backup_from_device_all_layer(device_pool, host_indices, device_indices, io_backend)` | `mha.py:465-584` | **Encode.** Encode each layer's rows into the device staging buffer with the row-wise absmax/scale + `torch` int8 cast, then issue one `jit_transfer_hicache_all_layer(..., k_ptr_dst=<arena K ptrs>, v_ptr_dst=<arena V ptrs>, k_ptr_src=<staging K ptrs>, v_ptr_src=<staging V ptrs>, indices_src=arange(n), indices_dst=host_indices, kv_cache_dst_stride_bytes=enc_row_bytes, element_size=enc_row_bytes)` (`mha.py:482-495`). The mover takes a single `element_size` and requires it to equal the destination stride, so a 2048 B bf16 device row cannot be written straight into a 1152 B encoded row: staging is mandatory. Everything is enqueued on the current (D2H) stream, so the existing `_submission` event records after the encode. |
| `load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, io_backend, *, is_draft=False)` | `mha.py:298-435` | **Decode.** Move this layer's encoded rows from the arena into the H2D staging buffer with `jit_transfer_hicache_one_layer` (`mha.py:320/345`), then enqueue the dequant (`int8.to(bf16) * scale`) into `device_pool.k_buffer[device_layer_id]` / `v_buffer[...]` at `device_indices`, on `host_to_device_stream`. Reuse the layer-ownership guard at `mha.py:308-315` verbatim. Because `on_layer_done(layer_id)` fires right after this returns (`l2_transfer.py:177`), the dequant must be enqueued here, not later. |
| `get_data_page(index, flat=True)` | `mha.py:586-598` | Return the encoded page blob (+ optional self-describing header) instead of raw kv. |
| `set_from_flat_data_page(index, data_page)` | `mha.py:608-640` | Inverse. |
| `get_dummy_flat_data_page()` | `mha.py:600-606` | Encoded-size dummy for prefetch/init. |
| `get_page_buffer_meta(indices)` / `get_split_heads_page_buffer_meta(...)` | `mha.py:690`, `mha.py:672` | Report encoded pointers/sizes so L3 zero-copy stays coherent. |
| `get_hybrid_pool_buffer()` | `mha.py:215-217` | Return the arena so registration (`hicache_nixl.py:417,458`, `umbp_store.py:1045`) sees real buffers. |
| `is_stride_page_aligned()` | `base.py:419-430` | Declare O_DIRECT alignment honestly for the encoded stride. |

Dispatch — the only edit to an existing file (`pool_host/mha.py:1510-1531`):

```python
def get_mha_host_pool_cls(device_pool: MHATokenToKVPool) -> type:
    if isinstance(device_pool, MHATokenToKVPoolMXFP8):
        ...
    if envs.SGLANG_EXPERIMENTAL_HICACHE_INT8.get():
        from sglang.srt.mem_cache.pool_host.mha_int8 import MHATokenToKVPoolHostINT8
        return MHATokenToKVPoolHostINT8
    if device_pool.head_dim != device_pool.v_head_dim:
        return AsymmetricMHATokenToKVPoolHost
    return MHATokenToKVPoolHost
```

The landed branch gates on `envs.SGLANG_EXPERIMENTAL_HICACHE_INT8.get()`, sizes the device
staging rows from `SGLANG_HICACHE_INT8_STAGING_TOKENS` (default `2048`), and is checked
**after** the MXFP8 and asymmetric cases, because neither is representable in the INT8
record.

That is **one new pool file + one dispatch branch**, plus two self-contained helper modules
and the env knobs. No changes to `l2_transfer.py`, `cache_controller.py`,
`unified_radix_cache.py`, or any CUDA source.

### I.4 Validation the codec performs at construction

Reject (raising during init, per the RFC's "fail fast" requirement):

- `device_pool.page_size != 1` for v1 (mirrors `mha_mxfp8.py:84-89` and PR #40551's
  `KVLayoutAdapter` restriction).
- `storage_dtype not in (torch.bfloat16, torch.float16)`; `device_pool.is_quantized_kv_cache`.
- `layout != "layer_first"` (or supply the layout-specific stride formula for `page_first`).
- `device_pool.head_dim != device_pool.v_head_dim` (asymmetric K/V out of scope).
- Any HiCache storage backend configured (`--hicache-storage-backend`): an encoded arena
  needs `get_data_page`/`get_page_buffer_meta` fully implemented for the encoded blob before
  L3 I/O can be allowed, and the integration refuses L3 storage outright
  (`mha_mxfp8.py:245-266`).
- `device_pool.row_dim * device_pool.dtype.itemsize % 128 != 0` ⇒ codec not admissible on the
  CUDA JIT path (on ROCm the 64/32/16 B rounds widen this).
- Warn that repeated `--disaggregation-decode-retraction-backup=host_pool` cycles re-encode
  reconstructed KV, i.e. lossy error can accumulate on that path (RFC §"Lossy recompression").

### I.5 Tests for the codec

1. `test/registered/kernels/ops/kvcache/test_hicache_int8.py` — mirror
   `test_hicache.py::_run_transfer_roundtrip_mha:167` and
   `test_hicache_page_first_write_back.py::_run_mha:116`: fill device pool, D2H, H2D,
   assert `max |x_restored − x_original|` within the int8 tolerance bound and that
   `host_pool.size_per_token` is the encoded value.
2. A host-pool unit in the style of
   `test/registered/unit/mem_cache/test_mxfp8_mha_pool_host_unit.py` covering
   `alloc`/`free` page alignment, `get_data_page`/`set_from_flat_data_page` round trip,
   and double-free detection through the encoded arena.
3. An e2e accuracy test in the style of
   `test/registered/hicache/test_hicache_variants.py` (MMLU/GSM8K with
   `--hicache-ratio` large enough to force L2 hits) to bound quality loss.
4. A stream-ordering regression modelled on
   `test_unified_hicache_regressions.py::TestTransferStreamOrdering:203`, asserting the
   decode is *enqueued* before `on_layer_done` and that no host sync appears on the path.
