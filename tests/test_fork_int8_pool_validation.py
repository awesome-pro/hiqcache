"""Validate the INT8 pool against a stubbed device pool, on CPU.

Every failure that reached the pod so far has been an assumption about SGLang's
API rather than about the codec: the coverage check demanded
``end_layer == layer_num`` when ``end_layer`` is inclusive, for instance, which
rejected every ordinary pool and produced 22 identical failures.

``_validate_configuration`` is a pure predicate over device-pool attributes. It
needs no GPU, no allocator and no kernel, so it can be exercised here by loading
the real module with its three heavy imports stubbed out.

What is real: the module under test, and every attribute name and value this
reads off the device pool. What is stubbed: the JIT availability checks, which
are genuinely GPU-dependent, and the base host pool, which is not needed to
validate a configuration.

The stub device pool documents exactly which SGLang attributes this pool
depends on. If SGLang changes one, this test is where it should break.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
POOL_HOST = REPO_ROOT.parent / "sglang" / "python" / "sglang" / "srt" / "mem_cache" / "pool_host"


# ---------------------------------------------------------------------------
# Load the real module with heavy imports stubbed
# ---------------------------------------------------------------------------


def _install_stubs(jit_ok: bool = True):
    """Stub the three imports mha_int8 needs but that require a GPU or SGLang.

    Always replaces the stubs: a previous call may have installed the opposite
    ``jit_ok``, and ``sys.modules.setdefault`` would silently keep it.
    """
    base = "sglang.srt.mem_cache.pool_host"
    for name in (f"{base}.mha_int8", f"{base}.int8_codec", f"{base}.int8_staging"):
        sys.modules.pop(name, None)

    hicache = types.ModuleType("sglang.kernels.ops.kvcache.hicache")
    hicache.can_use_hicache_jit_kernel = lambda **kw: jit_ok
    hicache.can_use_write_back_jit_kernel = lambda **kw: jit_ok
    hicache.transfer_hicache_all_layer = lambda **kw: None
    hicache.transfer_hicache_one_layer = lambda **kw: None

    mha = types.ModuleType("sglang.srt.mem_cache.pool_host.mha")

    class MHATokenToKVPoolHost:  # noqa: D401 - marker base for isinstance only
        """Stands in for the real base pool; validation never calls into it."""

    mha.MHATokenToKVPoolHost = MHATokenToKVPoolHost

    common = types.ModuleType("sglang.srt.mem_cache.pool_host.common")
    common.ALLOC_MEMORY_FUNCS = {}
    common.get_allocator_from_storage = lambda *a, **k: None
    common.make_kernel_ptr_table = lambda *a, **k: None

    environ = types.ModuleType("sglang.srt.environ")
    envs = types.SimpleNamespace(
        SGLANG_HICACHE_INT8_STAGING_TOKENS=types.SimpleNamespace(get=lambda: 2048)
    )
    environ.envs = envs

    for name, mod in (
        ("sglang", types.ModuleType("sglang")),
        ("sglang.kernels", types.ModuleType("sglang.kernels")),
        ("sglang.kernels.ops", types.ModuleType("sglang.kernels.ops")),
        ("sglang.kernels.ops.kvcache", types.ModuleType("sglang.kernels.ops.kvcache")),
        ("sglang.kernels.ops.kvcache.hicache", hicache),
        ("sglang.srt", types.ModuleType("sglang.srt")),
        ("sglang.srt.environ", environ),
        ("sglang.srt.mem_cache", types.ModuleType("sglang.srt.mem_cache")),
        ("sglang.srt.mem_cache.pool_host", types.ModuleType("sglang.srt.mem_cache.pool_host")),
        ("sglang.srt.mem_cache.pool_host.mha", mha),
        ("sglang.srt.mem_cache.pool_host.common", common),
    ):
        sys.modules[name] = mod


def _load_int8_module(jit_ok: bool = True):
    """Import mha_int8 plus its codec/staging siblings from the real fork files."""
    _install_stubs(jit_ok)
    base = "sglang.srt.mem_cache.pool_host"
    for name in ("int8_codec", "int8_staging"):
        spec = importlib.util.spec_from_file_location(
            f"{base}.{name}", POOL_HOST / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{base}.{name}"] = module
        spec.loader.exec_module(module)

    spec = importlib.util.spec_from_file_location(
        f"{base}.mha_int8", POOL_HOST / "mha_int8.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{base}.mha_int8"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Stub device pool: documents the SGLang attributes this pool depends on
# ---------------------------------------------------------------------------


class StubDevicePool:
    """A device pool exposing only what ``_validate_configuration`` reads.

    ``end_layer`` is inclusive, matching ``KVCache``::

        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
    """

    def __init__(
        self,
        *,
        layer_num: int = 36,
        head_num: int = 8,
        head_dim: int = 128,
        v_head_dim: int | None = None,
        page_size: int = 1,
        store_dtype: torch.dtype = torch.bfloat16,
        start_layer: int = 0,
        end_layer: int | None = None,
        is_quantized_kv_cache: bool = False,
        use_hnd: bool = False,
        layer_shard_enabled: bool = False,
        k_buffer_layers: int | None = None,
    ):
        self.layer_num = layer_num
        self.head_dim = head_dim
        self.v_head_dim = head_dim if v_head_dim is None else v_head_dim
        self.row_dim = head_num * head_dim
        self.page_size = page_size
        self.store_dtype = store_dtype
        self.start_layer = start_layer
        self.end_layer = layer_num - 1 if end_layer is None else end_layer
        self.is_quantized_kv_cache = is_quantized_kv_cache
        self.use_hnd = use_hnd
        self.layer_shard_enabled = layer_shard_enabled
        self.k_buffer = [None] * (layer_num if k_buffer_layers is None else k_buffer_layers)


@pytest.fixture(scope="module")
def int8_module():
    return _load_int8_module()


def _validate(module, pool, *, page_size: int = 1, layout: str = "layer_first", mtp=()):
    return module.MHATokenToKVPoolHostINT8._validate_configuration(
        pool, page_size, layout, mtp
    )


# ---------------------------------------------------------------------------
# The regression that cost a pod run
# ---------------------------------------------------------------------------


def test_accepts_an_ordinary_device_pool(int8_module):
    """Regression: end_layer is INCLUSIVE, so a full pool reports layer_num - 1.

    An earlier check demanded ``end_layer == layer_num`` and rejected every
    ordinary pool, which surfaced on the pod as 22 identical failures.
    """
    pool = StubDevicePool(layer_num=36)
    assert pool.end_layer == 35
    _validate(int8_module, pool)  # must not raise


def test_accepts_a_two_layer_pool(int8_module):
    """The unit test's own geometry: this is the case that failed on the pod."""
    _validate(int8_module, StubDevicePool(layer_num=2))


def test_rejects_a_pool_that_does_not_start_at_zero(int8_module):
    with pytest.raises(NotImplementedError, match="covering every layer"):
        _validate(int8_module, StubDevicePool(start_layer=2))


def test_rejects_a_pool_that_stops_short(int8_module):
    with pytest.raises(NotImplementedError, match="covering every layer"):
        _validate(int8_module, StubDevicePool(layer_num=36, end_layer=20))


def test_rejects_a_mismatched_buffer_count(int8_module):
    with pytest.raises(NotImplementedError, match="one device K buffer per layer"):
        _validate(int8_module, StubDevicePool(layer_num=36, k_buffer_layers=10))


def test_tolerates_an_uninitialised_buffer_list(int8_module):
    """k_buffer is None until the device pool creates its buffers."""
    pool = StubDevicePool(layer_num=36)
    pool.k_buffer = None
    _validate(int8_module, pool)


# ---------------------------------------------------------------------------
# Every other rejection path
# ---------------------------------------------------------------------------


def test_rejects_page_size_above_one(int8_module):
    with pytest.raises(NotImplementedError, match="page-size"):
        _validate(int8_module, StubDevicePool(page_size=16), page_size=16)


def test_rejects_non_layer_first_layout(int8_module):
    with pytest.raises(NotImplementedError, match="layer_first"):
        _validate(int8_module, StubDevicePool(), layout="page_first")


def test_rejects_mtp_draft_pools(int8_module):
    with pytest.raises(NotImplementedError, match="MTP"):
        _validate(int8_module, StubDevicePool(), mtp=(object(),))


def test_rejects_quantized_device_pool(int8_module):
    with pytest.raises(NotImplementedError, match="quantized"):
        _validate(int8_module, StubDevicePool(is_quantized_kv_cache=True))


def test_rejects_asymmetric_head_dims(int8_module):
    with pytest.raises(NotImplementedError, match="symmetric"):
        _validate(int8_module, StubDevicePool(v_head_dim=64))


def test_rejects_hnd_layout(int8_module):
    with pytest.raises(NotImplementedError, match="NHD"):
        _validate(int8_module, StubDevicePool(use_hnd=True))


def test_rejects_layer_sharding(int8_module):
    with pytest.raises(NotImplementedError, match="layer-sharded"):
        _validate(int8_module, StubDevicePool(layer_shard_enabled=True))


def test_rejects_non_bf16_dtype(int8_module):
    with pytest.raises(NotImplementedError, match="BF16 or FP16"):
        _validate(int8_module, StubDevicePool(store_dtype=torch.float32))


def test_rejects_tp_greater_than_one(int8_module):
    """4 local KV heads is TP=2 on Qwen3-8B; the fixed record cannot express it."""
    with pytest.raises(NotImplementedError, match="TP=1"):
        _validate(int8_module, StubDevicePool(head_num=4))


def test_rejects_wrong_head_dim(int8_module):
    """The geometry check lives in __init__, not in _validate_configuration.

    ``MHATokenToKVPoolHostINT8.__init__`` calls ``codec.check_layout`` separately
    from ``_validate_configuration``, so this exercises both the way the
    constructor does. An earlier version of this test called only the validator
    and wrongly expected it to catch a geometry error.
    """
    pool = StubDevicePool(head_dim=64)
    _validate(int8_module, pool)  # configuration itself is fine
    codec = sys.modules["sglang.srt.mem_cache.pool_host.int8_codec"]
    with pytest.raises(ValueError, match="payload"):
        codec.check_layout(pool.row_dim // pool.head_dim, pool.head_dim, 2)


def test_geometry_check_accepts_the_real_geometry(int8_module):
    codec = sys.modules["sglang.srt.mem_cache.pool_host.int8_codec"]
    pool = StubDevicePool()
    codec.check_layout(pool.row_dim // pool.head_dim, pool.head_dim, 2)  # no raise


def test_rejects_host_device_page_size_mismatch(int8_module):
    pool = StubDevicePool(page_size=2)
    with pytest.raises(NotImplementedError, match="page_size"):
        _validate(int8_module, pool, page_size=1)


def test_rejects_missing_jit_kernel(int8_module):
    """When the JIT mover is unavailable the pool must refuse at construction."""
    module = _load_int8_module(jit_ok=False)
    with pytest.raises(NotImplementedError, match="JIT HiCache kernel"):
        module.MHATokenToKVPoolHostINT8._validate_configuration(
            StubDevicePool(), 1, "layer_first", ()
        )


def test_a_healthy_pool_passes_every_check(int8_module):
    """The positive control: 36 layers, 8 heads, 128 dims, bf16, NHD, TP=1."""
    _validate(int8_module, StubDevicePool())
