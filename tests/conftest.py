"""Shared test setup.

Two responsibilities:

1. Make ``hiqcache`` importable from ``src/`` without an editable install.
2. Let the fork-local INT8 HiCache tests run against the **real** fork sources.
   The codec and staging modules under
   ``sglang/python/sglang/srt/mem_cache/pool_host/`` import only ``torch``, so
   they can be exercised anywhere -- no CUDA, no ``sgl_kernel``, no SGLang
   runtime. The tests import the fork files directly rather than a copy, so
   there is nothing to keep in sync.

The fork is located from ``HIQCACHE_SGLANG_ROOT`` when set, otherwise from the
sibling ``sglang/`` directory of this repo. Pool-level tests that need a real
device pool, allocator or JIT kernel live in the fork and run on the pod.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Fork source discovery
# ---------------------------------------------------------------------------


def _find_fork_root() -> Path | None:
    """Locate the SGLang fork checkout.

    ``__file__`` is ``<repo>/tests/conftest.py``, so ``parents[1]`` is the repo
    root and ``parents[2]`` is the workspace directory that holds both repos.
    """
    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    env = os.environ.get("HIQCACHE_SGLANG_ROOT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(repo_root.parent / "sglang")  # <workspace>/sglang
    candidates.append(repo_root / "sglang")  # vendored checkout
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "python" / "sglang" / "srt").is_dir():
            return root
    return None


FORK_ROOT = _find_fork_root()
POOL_HOST_DIR = (
    FORK_ROOT / "python" / "sglang" / "srt" / "mem_cache" / "pool_host"
    if FORK_ROOT
    else None
)

#: Set by the fixtures below; ``None`` means "fork not available, skip".
FORK_AVAILABLE = POOL_HOST_DIR is not None and POOL_HOST_DIR.is_dir()


def _install_stubs() -> None:
    """Register the minimum fake package structure the imports need.

    The codec module imports nothing from SGLang, but the test files use
    ``sglang.test.test_utils.CustomTestCase`` for consistency with the rest of
    the fork's suite. Stubbing it (rather than teaching the tests about pytest)
    keeps a single test source that runs both here and in SGLang CI.
    """
    if "sglang.test.test_utils" in sys.modules:
        return
    sglang = types.ModuleType("sglang")
    sglang.__path__ = [str(FORK_ROOT / "python" / "sglang")] if FORK_ROOT else []
    sglang_test = types.ModuleType("sglang.test")
    test_utils = types.ModuleType("sglang.test.test_utils")
    test_utils.CustomTestCase = unittest.TestCase
    sys.modules.setdefault("sglang", sglang)
    sys.modules.setdefault("sglang.test", sglang_test)
    sys.modules.setdefault("sglang.test.test_utils", test_utils)


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pytest_configure(config):
    if not FORK_AVAILABLE:
        return
    _install_stubs()
    _load_module("int8_codec", POOL_HOST_DIR / "int8_codec.py")
    _load_module("int8_staging", POOL_HOST_DIR / "int8_staging.py")


@pytest.fixture(scope="session")
def fork_root() -> Path:
    if not FORK_AVAILABLE:
        pytest.skip("SGLang fork not found; set HIQCACHE_SGLANG_ROOT")
    return FORK_ROOT


@pytest.fixture(scope="session")
def pool_host_dir() -> Path:
    if not FORK_AVAILABLE:
        pytest.skip("SGLang fork not found; set HIQCACHE_SGLANG_ROOT")
    return POOL_HOST_DIR

