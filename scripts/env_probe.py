"""Independent build/runtime probe: can this box do the HiCache JIT path at all?

This is deliberately **standalone** -- it runs before either repo exists and
imports nothing from them. Its only job is to answer one question early:

    can this machine compile and launch CUDA extensions from Python?

The HiCache transfer kernels are JIT-compiled at first use by
``sglang.kernels.ops.kvcache.hicache`` via ``load_jit``. When that fails, SGLang
does not crash -- it logs a warning and falls back or returns ``False`` from
``can_use_hicache_jit_kernel``, and the INT8 host pool then raises a generic
"needs the JIT HiCache kernel" error. That indirection makes a missing build tool
look like a HiCache bug, so it is worth establishing the answer up front.

Usage::

    python scripts/env_probe.py            # human readable
    python scripts/env_probe.py --quiet    # exit code only
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: Capability report, filled in by the individual probes.
RESULTS: list[tuple[str, bool, str, bool]] = []  # (name, ok, detail, fatal)


def add(name: str, ok: bool, detail: str = "", *, fatal: bool = True) -> bool:
    RESULTS.append((name, bool(ok), detail, fatal))
    return bool(ok)


def probe_python() -> None:
    version = sys.version_info
    add(
        "python >= 3.10",
        version >= (3, 10),
        f"{version.major}.{version.minor}.{version.micro} at {sys.executable}",
    )


def probe_torch() -> object | None:
    try:
        import torch
    except ImportError as exc:
        add("torch importable", False, str(exc))
        return None
    add("torch importable", True, f"torch {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    add(
        "torch sees a CUDA device",
        cuda_available,
        f"{torch.cuda.get_device_name(0)}"
        + (f", capability {torch.cuda.get_device_capability(0)}" if cuda_available else "")
        if cuda_available
        else "torch.cuda.is_available() is False",
    )
    add(
        "torch reports a CUDA runtime version",
        torch.version.cuda is not None,
        f"torch.version.cuda={torch.version.cuda}",
    )
    if cuda_available:
        cap = torch.cuda.get_device_capability(0)
        add(
            "GPU is compute capability >= 8.0 (Ampere or newer)",
            cap[0] >= 8,
            f"sm_{cap[0]}{cap[1]}",
            fatal=False,
        )
        props = torch.cuda.get_device_properties(0)
        add(
            "GPU has >= 20 GiB memory",
            props.total_memory / 2**30 >= 20,
            f"{props.total_memory / 2**30:.1f} GiB",
            fatal=False,
        )
        add(
            "host RAM >= 48 GiB",
            _total_ram_gib() >= 48,
            f"{_total_ram_gib():.1f} GiB",
            fatal=False,
        )
        add(
            "disk free >= 80 GiB at this path",
            _free_gib(Path.cwd()) >= 80,
            f"{_free_gib(Path.cwd()):.1f} GiB free at {Path.cwd()}",
            fatal=False,
        )
    return torch


def _total_ram_gib() -> float:
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            return (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
            )
    except (ValueError, OSError):
        pass
    return 0.0


def _free_gib(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 2**30
    except OSError:
        return 0.0


def probe_cuda_toolkit() -> None:
    nvcc = shutil.which("nvcc")
    add(
        "nvcc on PATH",
        nvcc is not None,
        nvcc or "nvcc not found; a JIT compile cannot work without it",
    )
    if nvcc:
        try:
            out = subprocess.run(
                [nvcc, "--version"], capture_output=True, text=True, timeout=60
            ).stdout
            line = next(
                (l for l in out.splitlines() if "release" in l.lower()), ""
            ).strip()
            add("nvcc --version works", bool(line), line or out.strip()[:120])
        except Exception as exc:  # noqa: BLE001
            add("nvcc --version works", False, f"{type(exc).__name__}: {exc}")

    from torch.utils.cpp_extension import CUDA_HOME  # type: ignore

    add("CUDA_HOME resolves", CUDA_HOME is not None, str(CUDA_HOME))
    if CUDA_HOME:
        for header in ("cuda_runtime.h", "cuda_fp16.h", "cuda_bf16.h"):
            found = list((Path(CUDA_HOME) / "include").glob(header))
            add(
                f"CUDA header {header} present",
                bool(found),
                str(found[0]) if found else f"missing under {CUDA_HOME}/include",
            )


def probe_build_tools() -> None:
    """``load_inline``/``load_jit`` shell out to ninja; its absence is common."""
    ninja_bin = shutil.which("ninja")
    ninja_mod = importlib.util.find_spec("ninja") is not None
    add(
        "ninja available",
        ninja_bin is not None or ninja_mod,
        (ninja_bin or "python -m ninja")
        if (ninja_bin or ninja_mod)
        else "no ninja binary and no ninja module -- `pip install ninja`",
    )
    add(
        "C++ compiler available",
        any(shutil.which(c) for c in ("g++", "c++", "clang++")),
        next(
            (shutil.which(c) for c in ("g++", "c++", "clang++") if shutil.which(c)),
            "no C++ compiler on PATH",
        ),
    )


def probe_jit_compile() -> None:
    """The real test: compile and load a trivial CUDA extension."""
    import torch
    from torch.utils.cpp_extension import load_inline

    if not torch.cuda.is_available():
        add("JIT compile + load a CUDA extension", False, "no CUDA device")
        return

    source = 'extern "C" __global__ void hq_probe_kernel() {}\n'
    with tempfile.TemporaryDirectory() as tmp:
        try:
            module = load_inline(
                name="hiqcache_env_probe",
                cpp_sources="",
                cuda_sources=source,
                functions=[],
                extra_cuda_cflags=["-O1"],
                build_directory=tmp,
                verbose=False,
            )
            add(
                "JIT compile + load a CUDA extension",
                module is not None,
                "compiled and dlopen'd a trivial CUDA extension",
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).splitlines()
            tail = " | ".join(m.strip() for m in message if m.strip())[-400:]
            hint = ""
            if "ninja" in tail.lower():
                hint = "  -> fix: pip install ninja"
            elif "cuda" in tail.lower() and "not found" in tail.lower():
                hint = "  -> fix: CUDA toolkit missing or CUDA_HOME wrong"
            add(
                "JIT compile + load a CUDA extension",
                False,
                f"{type(exc).__name__}: {tail}{hint}",
            )


def probe_git_auth() -> None:
    """Is GitHub SSH available? Informational only -- nothing requires it.

    Both repos are public and everything is cloned over HTTPS by default, so a
    missing key is not a problem. This check exists only to tell you whether you
    *could* push results back to the repo, which is one of two ways to get data
    off the pod. Reported as a warning, never as a blocker, because "no key on
    the pod" is the normal state and rsync works without it.

    Note the common confusion: ``git@github.com:`` failing with "Permission
    denied (publickey)" says nothing about whether a repo is public. SSH always
    requires a key; public repos cloned over HTTPS need none.
    """
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
             "-T", "git@github.com"],
            capture_output=True, text=True, timeout=45,
        )
        combined = (proc.stdout + proc.stderr).strip()
        ok = "successfully authenticated" in combined
        add(
            "GitHub SSH key present (optional: only needed to push results)",
            ok,
            "authenticated" if ok
            else "no key on this pod -- harmless; repos clone over HTTPS and "
                 "results come back via rsync",
            fatal=False,
        )
    except Exception as exc:  # noqa: BLE001
        add(
            "GitHub SSH key present (optional: only needed to push results)",
            False,
            f"{type(exc).__name__}: {exc}",
            fatal=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="exit code only")
    parser.add_argument("--no-network", action="store_true", help="skip the SSH check")
    args = parser.parse_args()

    probe_python()
    torch = probe_torch()
    if torch is not None:
        probe_cuda_toolkit()
    probe_build_tools()
    if torch is not None:
        probe_jit_compile()
    if not args.no_network:
        probe_git_auth()

    if not args.quiet:
        print("HiQCache environment probe")
        print()
        for name, ok, detail, fatal in RESULTS:
            mark = "PASS" if ok else ("FAIL" if fatal else "WARN")
            print(f"[{mark}] {name}")
            if detail:
                print(f"       {detail}")
        blocking = [n for n, ok, _, fatal in RESULTS if not ok and fatal]
        warnings = [n for n, ok, _, fatal in RESULTS if not ok and not fatal]
        print()
        if blocking:
            print(f"RESULT: {len(blocking)} blocking problem(s):")
            for name in blocking:
                print(f"  - {name}")
            print("\nDo not continue until these are fixed.")
        else:
            print("RESULT: this box can run the HiCache JIT path.")
        if warnings:
            print(f"({len(warnings)} warning(s): {', '.join(warnings)})")

    return 0 if all(ok for _, ok, _, fatal in RESULTS if fatal) else 1


if __name__ == "__main__":
    raise SystemExit(main())
