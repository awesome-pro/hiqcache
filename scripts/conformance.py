"""Cross-device conformance harness.

Generates a deterministic set of codec test vectors, encodes them with the
reference codec on whatever device is available, and writes a compact JSON
manifest of digests plus error statistics.

The point is the Mac/pod handoff:

    Mac  --generate-->  results/conformance_<device>.json
    Pod  --compare -->   must match the Mac's digests exactly

If the pod's digest differs, the bug is in the codec or a backend kernel, and
you find out in seconds instead of after a failed SGLang run. If the digests
match, the codec is bit-identical across devices and any remaining failure is in
the HiCache integration.

Usage::

    python scripts/conformance.py generate                # this device
    python scripts/conformance.py generate --device mps
    python scripts/conformance.py compare a.json b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hiqcache.codec import (  # noqa: E402
    decode_records,
    encode_records,
    error_stats,
    unpack_records,
    verified_error_bound,
)
from hiqcache.layout import V1_LAYOUT, qwen3_8b_compression  # noqa: E402

HEAD_NUM = 8
HEAD_DIM = 128
LAYER_NUM = 36


# ---------------------------------------------------------------------------
# Deterministic test vectors
# ---------------------------------------------------------------------------


def _pseudo_uniform(shape, *, seed: int) -> torch.Tensor:
    """Deterministic, bit-exact ``[0, 1)`` values with no RNG and no transcendentals.

    Built from integer arithmetic and exact float32 division by ``2**24``, so the
    result is identical on every torch version, every CPU and every accelerator.
    That property is the entire point: the harness compares codec output across
    machines, so the *inputs* must be reproducible even when everything else is
    not.

    An earlier revision used ``torch.randn(generator=...)`` and two failed
    attempts followed from it:

    1. generating on the device under test meant the pod encoded a different
       tensor from the Mac's reference;
    2. even pinning generation to CPU was not enough -- a fresh generation on the
       pod differed from the committed reference despite both reporting torch
       2.14.0, so the stream is not stable across builds.

    Deriving values from a counter removes the dependency completely.
    """
    n = 1
    for dim in shape:
        n *= dim
    idx = torch.arange(n, dtype=torch.int64)
    # splitmix64: integer-only, well-defined in int64.
    # The seed is reduced to 32 bits: it is only a stream selector, and a
    # larger value would overflow int64 in the addition below.
    z = (idx + (seed & 0xFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    # 24 bits keeps every value exact in float32, so the cast below is exact too.
    vals = ((z >> 40) & 0xFFFFFF).to(torch.float32).div_(float(1 << 24))
    # Map to [-1, 1): exact float32 arithmetic, still no transcendentals.
    vals = (vals - 0.5) * 2.0
    return vals.reshape(shape)


def _pseudo_normal(shape, *, seed: int) -> torch.Tensor:
    """Deterministic normal-ish values: sum of 12 uniforms, centred and scaled.

    A sum of uniforms is bounded and has no tails, which is fine here -- these
    vectors exercise the quantiser's *range handling*, not its tail behaviour.
    Every step is exact float32, so the result is reproducible everywhere.
    """
    acc = None
    for k in range(12):
        # Seeds stay well inside 32 bits so the int64 addition cannot overflow.
        part = _pseudo_uniform(shape, seed=seed * 1_000_003 + k * 7919)
        acc = part if acc is None else acc + part
    return acc - 6.0


def make_vectors(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """The vector set every device must agree on.

    Reproducible by construction: integer bit manipulation plus exact float32
    arithmetic, with no RNG and no transcendentals. The same code therefore
    produces the same bytes under any torch version and on any device.

    Do not reintroduce ``torch.randn`` here. It is not reproducible across torch
    builds, and a harness whose inputs vary cannot distinguish a codec difference
    from an input difference -- which is precisely the question it exists to
    answer.
    """
    vectors: dict[str, torch.Tensor] = {}

    vectors["random_unit"] = _pseudo_normal((128, HEAD_NUM, HEAD_DIM), seed=1).to(
        torch.bfloat16
    )
    vectors["random_small"] = (
        _pseudo_normal((64, HEAD_NUM, HEAD_DIM), seed=2) * 1e-3
    ).to(torch.bfloat16)
    vectors["random_large"] = (
        _pseudo_normal((64, HEAD_NUM, HEAD_DIM), seed=3) * 1e3
    ).to(torch.bfloat16)
    vectors["all_zeros"] = torch.zeros((32, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16)
    vectors["tiny_values"] = (
        _pseudo_normal((32, HEAD_NUM, HEAD_DIM), seed=4) * 1e-30
    ).to(torch.bfloat16)

    mixed = _pseudo_normal((64, HEAD_NUM, HEAD_DIM), seed=5).to(torch.bfloat16)
    mixed[0, :, :] = 0.0
    mixed[1, 0, :] = 1e-30
    mixed[2, 1, :] = 1e4
    mixed[3] *= 1e-8
    vectors["mixed_magnitudes"] = mixed

    # Exactly representable quotients: every value is k/127 for integer k.
    steps = torch.arange(-127, 128, dtype=torch.float32) / 127.0
    row = steps.repeat(HEAD_DIM // steps.numel() + 1)[:HEAD_DIM]
    vectors["exact_grid"] = (
        row.unsqueeze(0).unsqueeze(0).expand(16, HEAD_NUM, HEAD_DIM).contiguous().to(
            torch.bfloat16
        )
    )

    # A Qwen3-8B-shaped batch: 512 tokens is a realistic HiCache transfer size.
    vectors["qwen3_8b_shaped"] = _pseudo_normal(
        (512, HEAD_NUM, HEAD_DIM), seed=6
    ).to(torch.bfloat16)

    # Powers of two as absmax, so the scale needs no bf16 rounding and any drift
    # is a genuine arithmetic difference rather than scale quantisation.
    pow2 = torch.zeros((24, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16)
    for i in range(24):
        mag = 2.0 ** (i - 12)
        pow2[i, :, 0] = mag
        pow2[i, :, 1] = -mag
        pow2[i, :, 2:] = (
            _pseudo_normal((HEAD_NUM, HEAD_DIM - 2), seed=100 + i) * mag
        ).to(torch.bfloat16)
    vectors["power_of_two_absmax"] = pow2

    if str(device) != "cpu":
        return {k: v.to(device) for k, v in vectors.items()}
    return vectors


def _vectors_fingerprint(vectors: dict[str, torch.Tensor]) -> str:
    """Order-independent digest of every vector's value bits."""
    parts = [f"{name}:{_digest(vectors[name])}" for name in sorted(vectors)]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def load_or_make_vectors(
    device: torch.device | str, path: Path
) -> tuple[dict[str, torch.Tensor], bool]:
    """Load the shared vector set, creating or repairing it as needed.

    Generation is deterministic, so the file is a cross-machine guarantee rather
    than a necessity. That makes a *stale* file the real hazard: it would pin both
    sides to an outdated vector set while the code produces different values. The
    loader therefore verifies the file against a fresh generation and rewrites it
    on mismatch, so the two cannot diverge unnoticed.

    Returns ``(vectors, regenerated)``.
    """
    fresh = make_vectors()
    expected = _vectors_fingerprint(fresh)

    if path.is_file():
        stored = None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            stored = dict(payload["vectors"])
        except Exception as exc:  # noqa: BLE001 - an unreadable file is just stale
            print(f"WARNING: could not read {path.name} ({exc}); regenerating")
        if stored is not None and _vectors_fingerprint(stored) == expected:
            if str(device) != "cpu":
                stored = {k: v.to(device) for k, v in stored.items()}
            return stored, False
        print(
            f"WARNING: {path.name} does not match this build's vector set; "
            f"regenerating. Any previously recorded manifest digests are stale "
            f"and must be regenerated too."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vectors": {k: v.cpu() for k, v in fresh.items()},
            "torch": torch.__version__,
            "fingerprint": expected,
        },
        path,
    )
    if str(device) != "cpu":
        fresh = {k: v.to(device) for k, v in fresh.items()}
    return fresh, True


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _digest(tensor: torch.Tensor) -> str:
    """Stable digest of a tensor's *value bits*, independent of device."""
    t = tensor.detach().to("cpu").contiguous()
    if t.dtype == torch.bfloat16:
        # Compare the bf16 bit pattern, not a float expansion, so NaN payloads
        # and signed zeros are handled exactly.
        raw = t.view(torch.uint16).numpy().tobytes()
    elif t.dtype == torch.int8:
        raw = t.view(torch.uint8).numpy().tobytes()
    elif t.dtype == torch.uint8:
        raw = t.numpy().tobytes()
    else:
        raise TypeError(f"unsupported dtype for digest: {t.dtype}")
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(raw)
    return h.hexdigest()[:32]


def run(device: torch.device, vectors_path: Path) -> dict:
    """Encode every vector on ``device`` and report digests plus error stats."""
    vectors, created = load_or_make_vectors(device, vectors_path)
    if created:
        print(f"created the shared vector set at {vectors_path}")
        print("NOTE: commit it, so both machines encode identical inputs.\n")
    else:
        print(f"using the shared vector set at {vectors_path}\n")
    results: dict[str, dict] = {}

    for name, x_dev in sorted(vectors.items()):
        x_cpu = x_dev.to("cpu")
        records = encode_records(x_dev)
        payload, scales = unpack_records(records, head_num=HEAD_NUM, head_dim=HEAD_DIM)
        restored = decode_records(records, head_num=HEAD_NUM, head_dim=HEAD_DIM)

        # Statistics need float64 reductions; do them on CPU but keep the
        # accelerator's exact bytes in the digests below.
        stats = error_stats(x_cpu, restored.to("cpu"), scales.to("cpu"))
        bound = verified_error_bound(restored.to("cpu"), scales.to("cpu"))
        err = (restored.to("cpu").float() - x_cpu.float()).abs()
        worst_excess = float((err - bound).max())

        results[name] = {
            "input_shape": list(x_cpu.shape),
            # Digest of the INPUT as the accelerator sees it. When this matches
            # across devices but an output digest does not, the difference is
            # genuinely in the codec -- which is the question this whole harness
            # exists to answer. Without it, a generator difference and a codec
            # difference are indistinguishable.
            "input_digest": _digest(x_cpu),
            "records_digest": _digest(records),
            "payload_digest": _digest(payload),
            "scales_digest": _digest(scales),
            "restored_digest": _digest(restored),
            "max_abs_over_scale": stats.max_abs_over_scale,
            "mean_abs": stats.mean_abs,
            "max_abs": stats.max_abs,
            "p99_abs": stats.p99_abs,
            "p50_rel": stats.p50_rel,
            "bound_violations": int((err > bound).sum()),
            "worst_bound_excess": worst_excess,
        }

    return {
        "device": str(device),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cwru": _cuda_runtime(),
        "vectors_file": vectors_path.name,
        "layout": {
            "row_bytes": V1_LAYOUT.row_bytes,
            "payload_bytes": V1_LAYOUT.payload_bytes,
            "scale_bytes": V1_LAYOUT.scale_bytes,
            "padding_bytes": V1_LAYOUT.padding_bytes,
        },
        "qwen3_8b_compression": qwen3_8b_compression(),
        "vectors": results,
    }


def _cuda_runtime() -> dict | None:
    if not torch.cuda.is_available():
        return None
    return {
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "cuda_version": torch.version.cuda,
    }


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def compare(a: dict, b: dict) -> tuple[bool, list[str]]:
    """Return ``(ok, messages)`` comparing two conformance manifests."""
    problems: list[str] = []

    la, lb = a.get("layout"), b.get("layout")
    if la != lb:
        problems.append(f"layout differs: {la} != {lb}")

    va, vb = a.get("vectors", {}), b.get("vectors", {})
    if set(va) != set(vb):
        problems.append(
            f"vector sets differ: only in A {sorted(set(va) - set(vb))}, "
            f"only in B {sorted(set(vb) - set(va))}"
        )
        return False, problems

    # Input digests are checked first and reported distinctly. If inputs differ,
    # the two runs solved different problems and the output digests say nothing
    # about the codec.
    input_mismatches = [
        name
        for name in sorted(va)
        if va[name].get("input_digest") != vb[name].get("input_digest")
    ]
    if input_mismatches:
        problems.append(
            "INPUT MISMATCH on "
            + ", ".join(input_mismatches)
            + " -- the two runs encoded different tensors, so output digests are "
            "not comparable. Both sides must use the same --vectors file."
        )

    digest_keys = (
        "records_digest",
        "payload_digest",
        "scales_digest",
        "restored_digest",
    )
    mismatched = 0
    for name in sorted(va):
        for key in digest_keys:
            if va[name][key] != vb[name][key]:
                mismatched += 1
                problems.append(
                    f"{name}.{key}: {va[name][key][:12]} != {vb[name][key][:12]}"
                )
        if va[name]["bound_violations"] or vb[name]["bound_violations"]:
            problems.append(
                f"{name}: bound violations A={va[name]['bound_violations']} "
                f"B={vb[name]['bound_violations']}"
            )

    if mismatched == 0 and not input_mismatches:
        problems.insert(
            0,
            f"OK: all {len(va)} vectors bit-identical across "
            f"{a.get('device')} and {b.get('device')}",
        )
    return mismatched == 0 and not input_mismatches, problems


def _print_summary(manifest: dict) -> None:
    print(f"device    : {manifest['device']}")
    print(f"platform  : {manifest['platform']} ({manifest['machine']})")
    print(f"torch     : {manifest['torch']}")
    if manifest.get("cwru") or manifest.get("cuda_runtime"):
        print(f"cuda      : {manifest.get('cwru')}")
    print(
        f"layout    : row={manifest['layout']['row_bytes']} B "
        f"payload={manifest['layout']['payload_bytes']} "
        f"scale={manifest['layout']['scale_bytes']} "
        f"padding={manifest['layout']['padding_bytes']}"
    )
    print()
    print(f"{'vector':<24}{'max|x-x̂|/s':>12}{'mean|x-x̂|':>14}{'p99':>12}{'viol':>6}")
    print("-" * 70)
    for name, r in sorted(manifest["vectors"].items()):
        print(
            f"{name:<24}{r['max_abs_over_scale']:>12.5f}{r['mean_abs']:>14.6g}"
            f"{r['p99_abs']:>12.6g}{r['bound_violations']:>6}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="generate a manifest on this device")
    gen.add_argument(
        "--device",
        default="auto",
        help="torch device: cpu, mps, cuda, or auto (default: best available)",
    )
    gen.add_argument("--json", default=None, help="output path")
    gen.add_argument(
        "--vectors",
        type=Path,
        default=None,
        help=(
            "shared vector set (.pt). Created if absent. Both machines MUST use "
            "the same file, otherwise they encode different inputs and the "
            "comparison is meaningless."
        ),
    )
    gen.add_argument(
        "--expect",
        type=Path,
        default=None,
        help=(
            "compare against a reference manifest (e.g. the Mac's "
            "results/conformance_cpu.json) and exit non-zero on any difference"
        ),
    )

    cmp_ = sub.add_parser("compare", help="compare two manifests")
    cmp_.add_argument("a", type=Path)
    cmp_.add_argument("b", type=Path)

    args = parser.parse_args()

    if args.cmd == "compare":
        a = json.loads(args.a.read_text())
        b = json.loads(args.b.read_text())
        ok, messages = compare(a, b)
        for line in messages:
            print(line)
        return 0 if ok else 1

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    vectors_path = args.vectors or (REPO_ROOT / "results" / "conformance_vectors.pt")
    manifest = run(device, vectors_path)
    _print_summary(manifest)

    out = args.json
    if out is None:
        tag = str(device).replace(":", "")
        out = f"results/conformance_{tag}.json"
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {path}")

    if args.expect is not None:
        if not args.expect.exists():
            print(f"\nERROR: reference manifest not found: {args.expect}")
            return 2
        reference = json.loads(args.expect.read_text())
        print(f"\ncomparing against {args.expect} ({reference.get('device')}):")
        ok, messages = compare(reference, manifest)
        for line in messages:
            print("  " + line)
        if not ok:
            print(
                "\nFAIL: this device does not reproduce the reference codec bytes.\n"
                "Do not proceed to SGLang integration until this passes -- the bug is "
                "in the codec or a backend op, not in HiCache."
            )
            return 1
        print("\nPASS: codec is bit-identical to the reference device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
