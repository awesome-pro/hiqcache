"""Diagnose the cudaHostRegister call signature on this machine.

SGLang's ``_cuda_host_register`` calls ``torch.cuda.cudart().cudaHostRegister(
ptr, size, flags)`` with plain ints. On some torch builds that binding wants
different argument types, and the failure surfaces as::

    TypeError: cudart.cudaHostRegister(): incompatible function arguments.
    Invoked with: 71

This script tries each plausible calling convention against a small registered
buffer, so the working one is identified rather than guessed. It registers,
verifies, and unregisters, so it leaves no state behind.

Run on the pod::

    python scripts/diagnose_host_register.py
"""

from __future__ import annotations

import ctypes
import sys

import torch


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def main() -> int:
    banner("environment")
    print(f"python  : {sys.version.split()[0]}")
    print(f"torch   : {torch.__version__}")
    print(f"cuda    : {torch.version.cuda}")
    print(f"device  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    if not torch.cuda.is_available():
        print("\nNo CUDA device; nothing to diagnose.")
        return 2

    banner("cudart object")
    cudart = torch.cuda.cudart()
    print(f"type              : {type(cudart)}")
    print(f"has cudaHostRegister   : {hasattr(cudart, 'cudaHostRegister')}")
    print(f"has cudaHostUnregister : {hasattr(cudart, 'cudaHostUnregister')}")
    fn = getattr(cudart, "cudaHostRegister", None)
    if fn is None:
        print("\ncudaHostRegister is absent from this torch build.")
        return 1
    print(f"doc               : {(fn.__doc__ or '(no docstring)')[:400]}")

    banner("candidate call signatures")
    # A plain page-aligned host allocation, so registration has something real.
    buf = torch.empty(1 << 20, dtype=torch.uint8, pin_memory=False)
    base = buf.data_ptr()
    size = buf.numel() * buf.element_size()
    print(f"buffer  : ptr=0x{base:x} size={size}")

    results: list[tuple[str, bool, str]] = []

    def attempt(label: str, fn_):
        try:
            rc = fn_()
            rc_i = int(rc) if not isinstance(rc, int) else rc
            if rc_i != 0:
                results.append((label, False, f"returned cudaError {rc_i}"))
                return False
            # Verify and immediately release, so the next attempt starts clean.
            unregistered = False
            for unreg in (
                lambda: cudart.cudaHostUnregister(ctypes.c_void_p(base)),
                lambda: cudart.cudaHostUnregister(base),
            ):
                try:
                    if int(unreg()) == 0:
                        unregistered = True
                        break
                except Exception:  # noqa: BLE001
                    continue
            results.append((label, True, f"OK (unregister {'ok' if unregistered else 'failed'})"))
            return True
        except TypeError as exc:
            results.append((label, False, f"TypeError: {str(exc).splitlines()[0][:120]}"))
            return False
        except Exception as exc:  # noqa: BLE001
            results.append((label, False, f"{type(exc).__name__}: {str(exc)[:120]}"))
            return False

    attempt("ints (what SGLang does)", lambda: cudart.cudaHostRegister(base, size, 0))
    attempt("c_void_p, c_size_t, c_uint",
            lambda: cudart.cudaHostRegister(ctypes.c_void_p(base), ctypes.c_size_t(size), ctypes.c_uint(0)))
    attempt("tensor, flags", lambda: cudart.cudaHostRegister(buf, 0))
    attempt("tensor", lambda: cudart.cudaHostRegister(buf))

    for label, ok, detail in results:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label:<28} {detail}")

    banner("the exact arena path the pool uses (mmap-backed, not torch.empty)")
    # The pool's arena comes from alloc_mmap -> torch.frombuffer(mmap), not from
    # torch.empty. That is the one difference between this diagnostic's first
    # attempt and the real construction path, so exercise it explicitly.
    try:
        from sglang.srt.mem_cache.storage.mmap.mmap_allocator import alloc_mmap
        from sglang.srt.mem_cache.pool_host.common import (
            ALLOC_MEMORY_FUNCS,
            _cuda_host_register,
            _cuda_host_unregister,
        )

        dims = (2, 36, 64, 1152)  # same shape family as the pool's arena
        arena = alloc_mmap(dims, torch.uint8)
        print(f"alloc_mmap  : shape={tuple(arena.shape)} dtype={arena.dtype} "
              f"ptr=0x{arena.data_ptr():x} contig={arena.is_contiguous()}")
        print(f"element_size={arena.element_size()} numel={arena.numel()}")

        def probe(label, fn_):
            try:
                fn_()
                results.append((label, True, "OK"))
                print(f"  [OK  ] {label}")
            except Exception as exc:  # noqa: BLE001
                first = str(exc).splitlines()[0][:160]
                results.append((label, False, f"{type(exc).__name__}: {first}"))
                print(f"  [FAIL] {label}\n         {type(exc).__name__}: {first}")

        probe("_cuda_host_register(arena, None)",
              lambda: _cuda_host_register(arena, None))
        probe("_cuda_host_register(arena, layout_dim)",
              lambda: _cuda_host_register(arena, 2 * 1152 * 36))
        # Check the metadata the unregister path depends on.
        attr = getattr(arena, "_sglang_cuda_host_registered_ranges", "MISSING")
        print(f"  registration metadata: {attr if attr == 'MISSING' else len(attr)}")
        probe("_cuda_host_unregister(arena)", lambda: _cuda_host_unregister(arena))
        probe("re-register after unregister (same range must be reusable)",
              lambda: _cuda_host_register(arena, None))
        try:
            _cuda_host_unregister(arena)
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        print(f"  could not exercise the mmap path: {type(exc).__name__}: {exc}")

    banner("verdict")
    working = [label for label, ok, _ in results if ok]
    if "ints (what SGLang does)" in working:
        print("SGLang's call convention works here.")
        print("If the pool tests still fail with a TypeError, the failing call is")
        print("somewhere else -- check the full traceback for the argument value.")
        return 0
    if working:
        print(f"SGLang's convention FAILS, but this one works: {working[0]}")
        print("\nThat means SGLang's _cuda_host_register needs adapting for this")
        print("torch build. Report the working signature and it can be patched")
        print("in pool_host/common.py, or worked around in the INT8 pool.")
        return 1
    print("No calling convention worked. The CUDA driver may refuse host")
    print("registration entirely on this pod (for example inside an unprivileged")
    print("container without the right capabilities).")
    print("Workaround: construct the pool with pin_memory=False, which skips")
    print("registration but keeps the arena in ordinary pageable memory.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
