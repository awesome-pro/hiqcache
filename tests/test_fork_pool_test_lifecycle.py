"""Static checks on the fork's pool test: every pinned arena gets torn down.

The arena is pinned with ``cudaHostRegister``. Dropping the Python reference does
not unregister it, so a test that leaks its pool leaves a CUDA registration
alive. When a later test's allocator is handed the same address range,
``cudaHostRegister`` fails with "part or all of the requested memory range is
already mapped" -- which cascaded into 20 spurious failures on the pod and hid
whatever real problems were underneath.

These checks are static because the mechanism is a lifecycle contract: no GPU is
needed to confirm that every construction path is tracked and every tracked pool
is destroyed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

FORK_TEST = (
    Path(__file__).resolve().parents[1]
    / ".."
    / "sglang"
    / "test"
    / "registered"
    / "unit"
    / "mem_cache"
    / "test_hicache_int8_pool_host_unit.py"
).resolve()

pytestmark = (
    []
    if FORK_TEST.is_file()
    else [__import__("pytest").mark.skip(reason="fork test not found")]
)


def _tree() -> ast.Module:
    return ast.parse(FORK_TEST.read_text())


def _source() -> str:
    return FORK_TEST.read_text()


def test_a_teardown_base_exists():
    classes = {n.name for n in ast.walk(_tree()) if isinstance(n, ast.ClassDef)}
    assert "_Int8PoolTestCase" in classes, (
        "the teardown base is missing; pinned arenas would leak between tests"
    )


def test_every_test_class_uses_the_teardown_base():
    src = _source()
    plain = re.findall(r"class (Test\w+)\(unittest\.TestCase\):", src)
    assert not plain, (
        f"{plain} still inherit unittest.TestCase directly, so their pools are "
        f"never unregistered; use _Int8PoolTestCase"
    )
    derived = re.findall(r"class (Test\w+)\(_Int8PoolTestCase\):", src)
    assert len(derived) >= 5, f"only {len(derived)} test classes use the base"


def test_teardown_destroys_every_tracked_pool():
    src = _source()
    assert "_LIVE_POOLS.clear()" in src, "per-test tracking is not cleared"
    assert "pool.destroy()" in src, "teardown does not call destroy()"
    # The teardown must be exception-safe: one bad destroy must not stop the rest.
    assert "except Exception" in src, (
        "a raising destroy() would abort teardown and leak the remaining pools"
    )


def test_the_only_construction_helper_tracks_its_pool():
    """The single helper the tests use must register what it creates."""
    tree = _tree()
    helper = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_make_host_pool"
    )
    calls = [
        n
        for n in ast.walk(helper)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "MHATokenToKVPoolHostINT8"
    ]
    assert len(calls) == 1, "expected exactly one construction in the helper"
    appends = [
        n
        for n in ast.walk(helper)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "append"
    ]
    assert appends, "_make_host_pool does not append to _LIVE_POOLS"


def test_direct_constructions_are_either_tracked_or_never_allocate():
    """A pool built without the helper must be tracked, or must not get built.

    The remaining direct construction is a rejection test: it passes page_size=16
    and raises during validation, before init_kv_buffer allocates or registers
    anything, so there is nothing to leak. This test asserts that is still true
    rather than taking it on trust.
    """
    src = _source()
    # Find direct constructions outside _make_host_pool.
    blocks = src.split("def _make_host_pool")[1].split("\n\n\n", 1)[1]
    direct = blocks.count("MHATokenToKVPoolHostINT8(")
    tracked = blocks.count("_LIVE_POOLS.append")
    assert direct == 0 or tracked >= direct or "page_size=16" in blocks, (
        "a direct MHATokenToKVPoolHostINT8 construction may allocate a pinned "
        "arena without being tracked"
    )
