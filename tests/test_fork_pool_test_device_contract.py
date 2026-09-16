"""Pin the device contract that the pool tests must respect.

Two consecutive pod failures came from the same mistake in the *test*, not the
pool: folding a CPU-resident tensor together with a CUDA one.

  RuntimeError: Expected all tensors to be on the same device, but found at
  least two devices, cuda:0 and cpu!

The host arena is on CPU by construction, while the device pool and the index
tensors the mover needs are on the GPU. Any assertion touching both sides has to
bridge them explicitly. That is easy to get wrong and cheap to state, so this
module states it and checks the fork's test against it.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FORK_TEST = (
    REPO_ROOT.parent
    / "sglang"
    / "test"
    / "registered"
    / "unit"
    / "mem_cache"
    / "test_hicache_int8_pool_host_unit.py"
)

pytestmark = pytest.mark.skipif(
    not FORK_TEST.is_file(), reason="fork pool test not found"
)

HEAD_NUM = 8
HEAD_DIM = 128
ROW_BYTES = 1152
IDENT = re.compile(r"[A-Za-z_]\w*")

HOST_MARKERS = ("k_data_refs", "v_data_refs", "get_data_page")
DEVICE_MARKERS = (".k_buffer", ".v_buffer")


# ---------------------------------------------------------------------------
# The device contract
# ---------------------------------------------------------------------------


def test_host_arena_is_cpu_and_a_device_row_is_2048_bytes():
    """The two sides of a transfer live on different devices."""
    host_side = torch.zeros((2, 36, 4, ROW_BYTES), dtype=torch.uint8)
    assert host_side.device.type == "cpu"

    device_row = torch.zeros((HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16)
    assert device_row.numel() * device_row.element_size() == 2048


def test_decoding_a_host_record_yields_a_cpu_tensor():
    """decode_records is a pure op, so the output device follows the input."""
    record = torch.zeros((3, ROW_BYTES), dtype=torch.uint8)
    payload = record[..., :1024].contiguous().view(torch.int8)
    scales = record[..., 1024:1040].contiguous().view(torch.bfloat16)
    decoded = (
        payload.reshape(3, HEAD_NUM, HEAD_DIM).to(torch.bfloat16)
        * scales.reshape(3, HEAD_NUM).unsqueeze(-1)
    )
    assert decoded.device.type == "cpu"
    assert decoded.shape == (3, HEAD_NUM, HEAD_DIM)


def test_encoded_bytes_are_smaller_than_raw_bytes_not_fewer_elements():
    """Bytes must be compared as bytes. This was a real pod failure.

    ``stored.numel()`` on a uint8 arena is a byte count, while ``raw.numel()`` on
    a bf16 device row is an element count -- half its byte count. Comparing the
    two counts directly made the encoded arena look LARGER than the raw one
    (3456 vs 3072) when it is 1.78x smaller.
    """
    stored = torch.zeros((3, ROW_BYTES), dtype=torch.uint8)
    raw = torch.zeros((3, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16)

    stored_bytes = stored.numel() * stored.element_size()
    raw_bytes = raw.numel() * raw.element_size()

    assert stored_bytes == 3456
    assert raw_bytes == 6144
    assert stored_bytes < raw_bytes
    assert raw_bytes / stored_bytes == pytest.approx(2048 / 1152, rel=1e-12)


def test_combining_a_cpu_record_with_a_gpu_scale_needs_an_explicit_move():
    """The shape of the failure, stated on CPU-only tensors."""
    decoded = torch.zeros((3, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16)
    scales = torch.full((3, HEAD_NUM), 1e-3, dtype=torch.bfloat16)

    s = scales.float().unsqueeze(-1)
    bound = (0.5 + 2**-8) * s + 2**-8 * decoded.float().abs()
    assert bound.shape == decoded.shape
    assert bound.device.type == decoded.device.type == "cpu"


# ---------------------------------------------------------------------------
# Device-flow analysis of the fork's test
# ---------------------------------------------------------------------------


def _assignments(func: ast.FunctionDef, source: str) -> dict[str, str]:
    """``{name: right-hand-side source}`` for simple single-target assignments."""
    out: dict[str, str] = {}
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            segment = ast.get_source_segment(source, node.value)
            if segment is not None:
                out[node.targets[0].id] = segment
    return out


def classify_origin(name: str, assigns: dict[str, str], _depth: int = 0) -> str:
    """Trace a name to whichever side of the transfer produced it.

    Returns HOST, DEVICE or OTHER, following every identifier on the right-hand
    side so ``codec.compute_scales(raw)`` is traced through to ``raw``. An
    earlier version stopped at the first dotted segment -- ``codec`` -- found
    nothing, returned OTHER, and so passed silently while the bug was present.
    """
    if _depth > 12 or name not in assigns:
        return "OTHER"
    rhs = assigns[name]
    if any(marker in rhs for marker in HOST_MARKERS):
        return "HOST"
    if any(marker in rhs for marker in DEVICE_MARKERS):
        return "DEVICE"
    if ".cpu()" in rhs:
        return "HOST"
    for ident in IDENT.findall(rhs):
        if ident in assigns and ident != name:
            kind = classify_origin(ident, assigns, _depth + 1)
            if kind != "OTHER":
                return kind
    return "OTHER"


def error_bound_argument_kinds(func: ast.FunctionDef, source: str) -> list[str]:
    """Classifications of the positional names passed to ``error_bound``."""
    assigns = _assignments(func, source)
    for call in [n for n in ast.walk(func) if isinstance(n, ast.Call)]:
        if getattr(call.func, "id", None) == "error_bound":
            return [
                classify_origin(a.id, assigns)
                for a in call.args
                if isinstance(a, ast.Name)
            ]
    return []


def test_error_bound_arguments_do_not_span_devices():
    """The exact bug that reached the pod, checked against the fork's test."""
    source = FORK_TEST.read_text()
    offenders = []
    for func in [
        n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)
    ]:
        if "error_bound(" not in (ast.get_source_segment(source, func) or ""):
            continue
        kinds = error_bound_argument_kinds(func, source)
        if "HOST" in kinds and "DEVICE" in kinds:
            offenders.append(f"{func.name}: {kinds}")
    assert not offenders, (
        "error_bound arguments span devices (one from the CPU host arena, one from "
        f"the GPU device pool): {offenders}"
    )


def test_no_host_arena_is_indexed_with_a_gpu_tensor():
    """Every subscript of k_data_refs / v_data_refs must be CPU-safe.

    These are views of the CPU host arena. Indexing one with a CUDA tensor raises
    "indices should be either on cpu or on the same device as the indexed tensor
    (cpu)" -- an earlier pod failure.
    """
    source = FORK_TEST.read_text()
    offenders = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for arena in ("k_data_refs", "v_data_refs"):
            if arena not in stripped:
                continue
            after = stripped.split(arena, 1)[1]
            if "[" not in after:
                continue
            subscript = after.split("[", 1)[1].split("]", 1)[0]
            if not subscript or subscript == ":":
                continue
            if subscript.isdigit() or ".cpu()" in subscript:
                continue
            if subscript.replace(":", "").replace(" ", "").isdigit():
                continue
            if subscript in ("i", "layer", "layer_id", "token"):
                continue
            offenders.append((lineno, stripped[:100]))
    assert not offenders, (
        "a host arena is indexed with something that may not be a CPU tensor: "
        f"{offenders}"
    )


def test_the_device_flow_check_actually_catches_a_span():
    """Self-test: the classifier must flag the bug and clear the fix.

    Without this, the check above could silently degrade to always passing --
    which is precisely what its first version did. A guard that cannot fail is
    not a guard.
    """
    source = textwrap.dedent(
        """\
        def with_bug(host_pool, device_pool):
            stored = host_pool.k_data_refs[0][idx.cpu()]
            raw = device_pool.k_buffer[0][src]
            decoded = codec.decode_records(stored, head_num=8, head_dim=128)
            scales = codec.compute_scales(raw)
            bound = error_bound(decoded, scales)
            return bound

        def without_bug(host_pool, device_pool):
            stored = host_pool.k_data_refs[0][idx.cpu()]
            raw = device_pool.k_buffer[0][src]
            raw_cpu = raw.cpu()
            decoded = codec.decode_records(stored, head_num=8, head_dim=128)
            scales = codec.compute_scales(raw_cpu)
            bound = error_bound(decoded, scales)
            return bound
        """
    )
    funcs = {
        n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
    }

    buggy = error_bound_argument_kinds(funcs["with_bug"], source)
    fixed = error_bound_argument_kinds(funcs["without_bug"], source)

    assert buggy == ["HOST", "DEVICE"], (
        f"the classifier must report HOST vs DEVICE for the buggy pair, got {buggy}"
    )
    assert "DEVICE" not in fixed, (
        f"the classifier must not report a span for the fixed pair, got {fixed}"
    )
