"""Tests for the trace exporter and the trace-derived oracle.

These are the two pieces that turn "we measured 72.7%" into "72.7% is the
ceiling". A bug here would silently change the project's headline claim, so the
prefix/capacity arithmetic is pinned with small hand-checkable cases.

No torch, no tokenizer, no GPU: both modules are importable with the standard
library.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = _load("hiqcache_oracle", "scripts/oracle.py")
trace = _load("hiqcache_export_trace", "scripts/export_trace.py")


def test_trie_counts_the_shared_prefix_once():
    sequences = [[1, 2, 3, 4], [1, 2, 9], [7, 8]]
    assert oracle.trie_hits(sequences) == [0, 2, 0]


def test_trie_counts_a_repeated_sequence_in_full():
    # The oracle asks "what reuse exists", not "what SGLang will serve". A
    # repeated prompt is fully reusable here; a real server still has to prefill
    # the last token to produce logits. This is one of the documented omissions.
    assert oracle.trie_hits([[1, 2, 3], [1, 2, 3]]) == [0, 3]


def test_trie_does_not_confuse_sibling_suffixes():
    # Same first token, different second: the second request reuses one token.
    assert oracle.trie_hits([[1, 2], [1, 3]]) == [0, 1]


def test_published_capacities_reproduce_from_the_sizing_rule():
    assert oracle.capacity_tokens(8, oracle.BF16_BYTES_PER_TOKEN) == 54_254
    assert oracle.capacity_tokens(8, oracle.INT8_BYTES_PER_TOKEN) == 96_451


def test_fit_groups_is_smallest_prefix_first_and_stops_at_capacity():
    prefix_lens = {0: 100, 1: 100, 2: 100, 3: 100}
    fitted, used = oracle.fit_groups(250, prefix_lens)
    assert len(fitted) == 2
    assert used == 200


def test_group_prefix_is_the_minimum_pairwise_lcp():
    sequences = [[1, 2, 3, 9], [1, 2, 4, 8], [5, 6]]
    groups = [0, 0, 1]
    assert oracle.group_prefix_lengths(sequences, groups) == {0: 2, 1: 0}


def test_exporter_groups_by_shared_system_prompt():
    a = list(range(100))
    b = a[:70] + [999]
    c = list(range(500, 600))
    groups = trace.derive_groups([a, b, c], key_len=64)
    assert groups[0] == groups[1] != groups[2]


def test_exporter_lcp_and_prefix_minimum():
    assert trace._lcp([1, 2, 3], [1, 2, 4]) == 2
    assert trace.group_prefix_lengths(
        [[1, 2, 3, 9], [1, 2, 4, 8]], [0, 0]
    ) == {0: 2}
