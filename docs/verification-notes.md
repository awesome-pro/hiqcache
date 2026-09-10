# Environment & code verification notes

Verified against working tree of `sglang/` fork, HEAD `515f5be77e74761c269e007ac41a5895191a1b7d`.

## Repos
| Item | Value |
| --- | --- |
| sglang fork | `git@github.com:awesome-pro/sglang.git`, branch `main`, HEAD `515f5be77e` |
| hiqcache repo | `git@github.com:awesome-pro/hiqcache.git`, **empty** (no commits yet) |
| Trace doc base commit | `434c2e3a` (one HiCache commit behind HEAD) |

Trace line numbers still resolve at `515f5be77e` — spot-checked
`pool_host/mha.py:465` `backup_from_device_all_layer`, `:298` `load_to_device_per_layer`,
`:1510` `get_mha_host_pool_cls`, `base.py:194` `size_per_token` call, `l2_transfer.py:136/:169`.

## Verified facts that constrain the design

1. **`element_size = 1152` IS on the CUDA JIT fast path.**
   `COPY_GROUP_THREADS = 32`, `GROUP_BYTES = (128,)` on CUDA (`kernels/ops/kvcache/hicache.py:26,31`).
   `_default_unroll(1152) = 1` (1152 > 1024) → `lanes_per_worker = 32`, `group = 128`,
   `package = 128/32 = 4` ∈ {4,8,16} ✓, and `1152 % 128 == 0` ✓.
   TMA path is off by default on CUDA (`use_hicache_tma_kernel` requires `_is_hip`).

2. **JIT kernel requires `src_stride`, `dst_stride`, and `element_size` to be mutually
   consistent.** `transfer_hicache_all_layer` takes both strides but a *single*
   `element_size`, so `src` and `dst` rows must be the same byte width.
   ⇒ BF16 device row (2048 B) cannot be written into a 1152 B host row by the mover.
   **This independently confirms PROJECT.md Phase 4: GPU staging is mandatory, not optional.**

3. **Defaults at HEAD**: `--hicache-io-backend kernel` (default), `--hicache-mem-layout
   page_first` (default) ⇒ INT8 pool must force/require `layer_first`.
   `page_size` defaults to `1` on CUDA (`arg_groups/overrides.py:1275`).

4. **`--hicache-size` is decimal GB**: `int(host_size * 1e9 // size_per_token)`
   (`pool_host/base.py:201`), then `page_num = size // page_size + 1; size = page_num * page_size`.

5. **Storage (L3) hooks are the only real complication** — `mha_mxfp8.py:245-266`
   `_storage_pages_unsupported()` is the precedent for rejecting them in v1.

## Mac capability probe

| Item | Result |
| --- | --- |
| Chip / RAM | Apple M4, 24 GiB (25,769,803,776 B), 10 cores |
| macOS | 26.6.2 |
| System Python | 3.14.5 — **no torch** (and no torch wheels for 3.14) |
| Available Python | `/opt/homebrew/bin/python3.12` ✓ |
| `uv` | 0.11.26 ✓ |
| Free disk | 347 GiB |

**Conclusion:** a `uv`-managed **Python 3.12** venv with CPU/MPS torch is required for local
codec work. CUDA-specific paths (`sgl_kernel`, JIT kernels, `cudaHostRegister`,
stream semantics) cannot run on this device and must be validated on the pod.
