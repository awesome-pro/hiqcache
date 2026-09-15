"""Regression tests for the cross-device conformance harness.

The harness compares codec output across devices. Its weakest link was input
generation: ``torch.randn(generator=...)`` is not guaranteed bit-identical across
backends, so generating vectors on the device under test compares the codec on
two different problems. That produced a real false alarm on the pod -- identical
error statistics, different digests, on one vector only.

These tests pin the fix: vectors are built from a CPU-only generator and
persisted, and the comparison reports an input mismatch separately from a codec
mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from conformance import (  # noqa: E402
    _digest,
    compare,
    load_or_make_vectors,
    make_vectors,
)

VECTORS_PATH = Path(__file__).resolve().parents[1] / "results" / "conformance_vectors.pt"


def test_make_vectors_is_deterministic():
    """Two calls must produce identical bits, or nothing downstream is stable."""
    a = make_vectors()
    b = make_vectors()
    assert set(a) == set(b)
    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} is not reproducible"
        assert a[name].dtype == torch.bfloat16


def test_vector_generation_does_not_depend_on_the_requested_device():
    """Requesting a device must only move the tensors, never regenerate them.

    This is the property that makes a CPU reference and a CUDA run comparable.
    """
    cpu = make_vectors("cpu")
    moved = make_vectors("cpu")  # same seed, independent call
    for name in cpu:
        assert torch.equal(cpu[name], moved[name])


def test_persisted_vectors_round_trip(tmp_path):
    path = tmp_path / "vectors.pt"
    created, was_created = load_or_make_vectors("cpu", path)
    assert was_created is True
    assert path.is_file()

    loaded, was_created2 = load_or_make_vectors("cpu", path)
    assert was_created2 is False
    assert set(created) == set(loaded)
    for name in created:
        assert torch.equal(created[name], loaded[name]), (
            f"{name} changed across persistence; a machine reading the file would "
            f"encode different inputs"
        )


def test_committed_vector_file_is_stable():
    """The committed set must still match a fresh generation.

    If this fails, the vector set drifted and every recorded manifest digest is
    stale -- regenerate the manifests, do not just update this test.
    """
    if not VECTORS_PATH.is_file():
        pytest.skip("no committed vector set yet")
    stored, _ = load_or_make_vectors("cpu", VECTORS_PATH)
    fresh = make_vectors()
    assert set(stored) == set(fresh)
    for name in fresh:
        assert torch.equal(stored[name], fresh[name]), (
            f"{name} differs from the committed vector set; regenerate "
            f"results/conformance_*.json if the change was intentional"
        )


def test_vectors_are_reproducible_across_processes():
    """A fresh interpreter must derive identical digests.

    This is the property that broke on the pod: the *same* torch version in a
    different environment produced different ``torch.randn(generator=...)``
    values, so the committed reference and a fresh generation disagreed. Vectors
    are now derived from integer arithmetic, so a separate process must agree
    exactly. Run in subprocesses because an in-process comparison can share
    cached state that masks the problem.
    """
    import subprocess

    script = (
        "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from conformance import make_vectors, _digest;"
        "v = make_vectors();"
        "print('\\n'.join(f'{k} {_digest(v[k])}' for k in sorted(v)))"
    )
    repo = Path(__file__).resolve().parents[1]
    outputs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo, capture_output=True, text=True, check=True,
        )
        outputs.append(proc.stdout.strip())
    assert outputs[0] == outputs[1], (
        "vector generation is not reproducible across processes; do not use a "
        "generator-based RNG here"
    )
    assert outputs[0], "subprocess produced no digests"


def test_no_generator_based_rng_in_the_harness():
    """Guard the root cause directly: no torch.randn/rand/randint in generation.

    Generator-based torch RNG is not guaranteed reproducible across torch builds,
    and a harness whose inputs vary cannot distinguish a codec difference from an
    input difference -- the one question it exists to answer.

    Inspects the AST rather than the text: the docstrings deliberately *name* the
    banned calls to explain why they are banned, and a text scan would flag its
    own explanation.
    """
    import ast

    path = Path(__file__).resolve().parents[1] / "scripts" / "conformance.py"
    tree = ast.parse(path.read_text())
    generation = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_pseudo_uniform", "_pseudo_normal", "make_vectors")
    ]
    assert generation, "generation functions not found; did they get renamed?"

    banned = {"randn", "rand", "randint", "rand_like", "randn_like", "normal_"}
    offenders = []
    for node in generation:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                func = sub.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if name in banned:
                    offenders.append(f"{node.name}: {name}()")
    assert not offenders, (
        f"generator-based RNG in vector generation: {offenders}. It is not "
        f"reproducible across torch builds; derive values arithmetically instead."
    )


def test_input_digest_distinguishes_input_from_codec_mismatch():
    """A perturbed input must be reported as an input mismatch, not a codec bug."""
    vectors = make_vectors()
    name = "qwen3_8b_shaped"
    digest = _digest(vectors[name])

    reference = {
        "device": "cpu",
        "layout": {"row_bytes": 1152},
        "vectors": {
            name: {
                "input_digest": digest,
                "records_digest": "aaaa",
                "payload_digest": "bbbb",
                "scales_digest": "cccc",
                "restored_digest": "dddd",
                "bound_violations": 0,
            }
        },
    }
    other = json.loads(json.dumps(reference))
    other["device"] = "cuda"

    # Identical inputs and outputs -> agreement.
    ok, _ = compare(reference, other)
    assert ok

    # Same outputs, different input -> must be called out as an input mismatch.
    other["vectors"][name]["input_digest"] = "0000"
    ok, problems = compare(reference, other)
    assert not ok
    assert any("INPUT MISMATCH" in p for p in problems), problems

    # Same input, different output -> a genuine codec difference.
    other = json.loads(json.dumps(reference))
    other["vectors"][name]["records_digest"] = "eeee"
    ok, problems = compare(reference, other)
    assert not ok
    assert not any("INPUT MISMATCH" in p for p in problems), problems
    assert any("records_digest" in p for p in problems), problems


def test_committed_cpu_manifest_matches_a_fresh_cpu_run():
    """The reference manifest must be reproducible on this machine.

    Guards against a stale manifest silently becoming the comparison baseline.
    """
    manifest_path = Path(__file__).resolve().parents[1] / "results" / "conformance_cpu.json"
    if not manifest_path.is_file():
        pytest.skip("no committed CPU manifest yet")
    manifest = json.loads(manifest_path.read_text())
    vectors = make_vectors()
    for name, recorded in manifest["vectors"].items():
        assert name in vectors, f"{name} is in the manifest but not generated"
        assert recorded["input_digest"] == _digest(vectors[name]), (
            f"{name}: recorded input digest does not match this machine's "
            f"vectors; regenerate results/conformance_cpu.json"
        )
