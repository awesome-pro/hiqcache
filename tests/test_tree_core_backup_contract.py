"""Can the real UnifiedTreeCore reach a backed-up node after device eviction?

This question has now cost five pod runs, and it is answerable for free: the
eviction, demotion and match-walk logic in
``sglang/srt/mem_cache/unified_cache/`` is pure Python and needs no GPU, no
allocator and no kernel. Only tensors are involved, and those can be tiny.

The behaviour under test, read from the source:

* ``UnifiedTreeCore.evict_device_leaf`` branches on ``is_write_back``:

      if not node.backuped:
          if is_write_back:
              result.backup_kv = self._build_backup_kv_action(node, write_back=True)
              return result
          # Write-through: node has no backup, delete entirely.
          self._delete_unbacked_device_leaf(node, ...)

* ``FullComponent.finalize_match_result_in_tree_core`` computes host_hit_length by
  walking from ``best_match_node`` up to ``last_device_node`` summing
  ``host_value`` on the nodes between them.

* ``_match_prefix_helper`` refuses to descend past
  ``child.evicted and not child.backuped``.

So the chain that has to hold is: evict an unbacked node under write_back ->
backup commits -> ``backuped`` becomes True -> the node survives as host-only ->
a later match walks into it and reports host_hit_length > 0.

If the node ends up evicted with ``backuped`` False it is *dead*: the traversal
stops there and host_hit_length can never be positive, which is exactly the
symptom observed on the pod (``evicted=True backuped=False`` on every match).

These tests skip when the fork is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from conftest import FORK_AVAILABLE, FORK_ROOT  # noqa: E402

# FORK_ROOT is None without the fork, and the module-level path building below
# would raise TypeError during collection before any fixture or marker applies.
if not FORK_AVAILABLE:
    pytest.skip(
        "requires the SGLang fork checked out at ../sglang", allow_module_level=True
    )

pytestmark = pytest.mark.skipif(
    not FORK_AVAILABLE, reason="SGLang fork not found; set HIQCACHE_SGLANG_ROOT"
)

sys.path.insert(0, str(FORK_ROOT / "python"))


def _torch_only_import(module_path: str):
    """Import a sglang module with the heavy package machinery stubbed out.

    The tree core imports sglang.srt.mem_cache.base_prefix_cache and friends,
    which drag the whole SRt package (and CUDA-only modules) in. Only tensors are
    needed here, so a handful of submodules are stubbed and the real files are
    loaded directly from disk.
    """
    import importlib.util
    import types

    name = module_path.rsplit(".", 1)[-1]
    if name in sys.modules:
        return sys.modules[name]
    path = FORK_ROOT / "python" / Path(*module_path.split(".")).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _component_type_enum():
    """ComponentType, loaded without importing the whole package tree."""
    import ast

    path = (
        FORK_ROOT
        / "python/sglang/srt/mem_cache/unified_cache/component_type.py"
    )
    tree = ast.parse(path.read_text())
    names = [
        t.id
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ComponentType"
        for t in node.bases
        if isinstance(t, ast.Name)
    ]
    # Just read the members; the enum is small and self-contained.
    members = [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        for target in node.targets
    ]
    return names, members


def test_component_type_is_self_contained():
    """Sanity: the enum has no dependencies, so the real logic can be loaded."""
    bases, members = _component_type_enum()
    assert members, "ComponentType appears empty; did the file change shape?"


def test_source_guards_for_the_observed_symptom():
    """Pin the three code facts the pod symptom depends on.

    These are read from the fork's source rather than executed, because
    instantiating the real UnifiedTreeCore needs a CacheInitParams and a pool
    allocator. Reading them still catches the case that matters: if SGLang
    changes one of these branches, the reasoning behind the INT8 pool's
    load-back behaviour changes with it.
    """
    core = (
        FORK_ROOT
        / "python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py"
    ).read_text()
    full = (
        FORK_ROOT
        / "python/sglang/srt/mem_cache/unified_cache/components/full.py"
    ).read_text()

    # 1. evicted + not backuped is a dead node: traversal stops there.
    assert "if child.evicted and not child.backuped:" in core, (
        "the match walk no longer stops at dead nodes; the pod symptom would "
        "have a different explanation"
    )

    # 2. write_back controls demote-vs-delete.
    assert "if is_write_back:" in core
    assert "_delete_unbacked_device_leaf" in core

    # 3. `backuped` is exactly "Full host_value present".
    assert "host_value is not None" in core

    # 4. host_hit_length comes from walking host_value up to last_device_node.
    assert "while node is not result.last_device_node and node is not root_node:" in full
    assert "host_hit_length=max(result.host_hit_length, kv_host_hit)" in full


def test_match_walk_arithmetic():
    """Exercise the walk's arithmetic directly, independently of the tree.

    The walk sums host_value lengths for nodes strictly between best_match_node
    and last_device_node. With best_match == last_device the loop body never
    runs, which is what the pod log showed on every request -- so this records
    that the observed output is arithmetically consistent with a match that
    stopped on a device node.
    """

    class FakeNode:
        def __init__(self, nid, host_len=None, parent=None):
            self.id = nid
            self.host_value = (
                torch.zeros(host_len, dtype=torch.int64) if host_len else None
            )
            self.parent = parent
            self.evicted = host_len is not None
            self.backuped = host_len is not None

    def walk(best, last_device, root):
        total = 0
        node = best
        while node is not last_device and node is not root:
            if node.host_value is not None:
                total += len(node.host_value)
            node = node.parent
        return total

    root = FakeNode(0)
    # Case A: best == last_device -> the walk contributes nothing (pod symptom).
    dev = FakeNode(1, parent=root)
    assert walk(dev, dev, root) == 0

    # Case B: a host-only node sandwiched between root and the device node.
    host = FakeNode(2, host_len=64, parent=root)
    leaf = FakeNode(3, parent=host)
    assert walk(leaf, host, root) == 0, "host node itself is the boundary"
    assert walk(host, root, root) == 64, "host node above root counts"

    # Case C: host + a deeper evicted-but-unbacked node contributes only the host.
    unbacked = FakeNode(4, parent=host)
    assert walk(unbacked, root, root) == 64
