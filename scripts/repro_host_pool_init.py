"""Minimal reproduction of the INT8 host-pool construction, with full traceback.

The unit test reports failures through pytest's short traceback, which truncates
the interesting part. This script does exactly what the pool tests' first
fixture does -- build a device pool, then build the INT8 host pool with
pin_memory=True -- and prints the complete traceback, the registered address
ranges, and the CUDA registration state around each step.

Run on the pod::

    python scripts/repro_host_pool_init.py
"""

from __future__ import annotations

import sys
import traceback

import torch

LAYER_NUM = 2
HEAD_NUM = 8
HEAD_DIM = 128
POOL_SIZE = 64
PAGE_SIZE = 1


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n=== {text}\n{'=' * 72}")


def registration_state(label: str, buffer) -> None:
    attr = getattr(buffer, "_sglang_cuda_host_registered_ranges", "MISSING")
    if attr == "MISSING":
        print(f"  {label}: no registration metadata")
    else:
        total = sum(size for _, size in attr)
        print(f"  {label}: {len(attr)} registered range(s), {total} B total")


def main() -> int:
    banner("environment")
    print(f"torch  : {torch.__version__}  cuda {torch.version.cuda}")
    print(f"device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 2

    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.pool_host.mha_int8 import MHATokenToKVPoolHostINT8

    banner("step 1: device pool")
    try:
        device_pool = MHATokenToKVPool(
            size=POOL_SIZE,
            page_size=PAGE_SIZE,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
            layer_num=LAYER_NUM,
            device="cuda",
            enable_memory_saver=False,
        )
        print(f"  ok: layer_num={device_pool.layer_num} "
              f"start_layer={device_pool.start_layer} end_layer={device_pool.end_layer}")
        print(f"  k_buffer: {len(device_pool.k_buffer)} buffers, "
              f"per-layer shape {tuple(device_pool.k_buffer[0].shape)}")
        print(f"  row_dim={device_pool.row_dim} head_dim={device_pool.head_dim} "
              f"v_head_dim={device_pool.v_head_dim}")
    except Exception:
        print("  FAILED:")
        traceback.print_exc()
        return 1

    banner("step 2: host pool, pin_memory=True (the failing case)")
    host_pool = None
    try:
        host_pool = MHATokenToKVPoolHostINT8(
            device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PAGE_SIZE,
            layout="layer_first",
            pin_memory=True,
            device="cpu",
            allocator_type="default",
        )
        print("  ok")
        print(f"  size_per_token={host_pool.size_per_token} size={host_pool.size}")
        registration_state("kv_buffer", host_pool.kv_buffer)
    except Exception:
        print("  FAILED -- full traceback follows:")
        traceback.print_exc()

    banner("step 3: host pool, pin_memory=False (isolates registration)")
    try:
        host_pool_unpinned = MHATokenToKVPoolHostINT8(
            device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PAGE_SIZE,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )
        print("  ok -- so the failure is specific to registration, not to the arena")
        host_pool_unpinned.destroy()
    except Exception:
        print("  FAILED too -- the problem is upstream of registration:")
        traceback.print_exc()

    banner("step 4: destroy and rebuild in sequence")
    # The pool tests build many pools in one process; this reproduces that.
    for i in range(3):
        try:
            pool = MHATokenToKVPoolHostINT8(
                device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=PAGE_SIZE,
                layout="layer_first",
                pin_memory=True,
                device="cpu",
                allocator_type="default",
            )
            registration_state(f"  build {i} kv_buffer", pool.kv_buffer)
            pool.destroy()
        except Exception as exc:  # noqa: BLE001
            print(f"  build {i} FAILED: {type(exc).__name__}: {str(exc).splitlines()[0]}")
            traceback.print_exc()
            break
    else:
        print("  three sequential build/destroy cycles succeeded")

    banner("step 5: registered ranges still live after destroy (cascade check)")
    if host_pool is None:
        print("  step 2 never built a pool, so there is nothing to inspect")
    elif host_pool.kv_buffer is None:
        print("  kv_buffer released by destroy(); its ranges should be unregistered")
    else:
        registration_state("kv_buffer after destroy", host_pool.kv_buffer)

    banner("verdict")
    print("Compare which of steps 2/3/4 failed. If only step 2 failed, the issue is")
    print("registration specifically. If all failed, it is the arena allocation.")
    print("If none failed, the pool tests' failure is elsewhere and their traceback")
    print("is needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
