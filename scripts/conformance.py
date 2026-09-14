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


def make_vectors(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """The vector set every device must agree on.

    All values are built from an explicit **CPU** generator, because
    ``torch.randn(generator=...)`` is not guaranteed to be bit-identical across
    backends. Generating the inputs on the device under test would silently
    compare the codec on two different problems -- which is exactly the failure
    that produced "identical statistics, different digest" on the pod.

    The generator is one shared stream consumed in a fixed order, so reordering
    this function changes every downstream digest and requires regenerating the
    reference manifest.

    Prefer :func:`load_or_make_vectors`, which also persists the exact tensors so
    both machines read identical bytes from disk.
    """
    g = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
    vectors: dict[str, torch.Tensor] = {}

    vectors["random_unit"] = torch.randn((128, HEAD_NUM, HEAD_DIM), generator=g).to(
        torch.bfloat16
    )
    vectors["random_small"] = (
        torch.randn((64, HEAD_NUM, HEAD_DIM), generator=g) * 1e-3
    ).to(torch.bfloat16)
    vectors["random_large"] = (
        torch.randn((64, HEAD_NUM, HEAD_DIM), generator=g) * 1e3
    ).to(torch.bfloat16)
    vectors["all_zeros"] = torch.zeros((32, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16)
    vectors["tiny_values"] = (
        torch.randn((32, HEAD_NUM, HEAD_DIM), generator=g) * 1e-30
    ).to(torch.bfloat16)

    mixed = torch.randn((64, HEAD_NUM, HEAD_DIM), generator=g).to(torch.bfloat16)
    mixed[0, :, :] = 0.0
    mixed[1, 0, :] = 1e-30
    mixed[2, 1, :] = 1e4
    mixed[3] *= 1e-8
    vectors["mixed_magnitudes"] = mixed

    steps = torch.arange(-127, 128, dtype=torch.float32) / 127.0
    row = steps.repeat(HEAD_DIM // steps.numel() + 1)[:HEAD_DIM]
    vectors["exact_grid"] = (
        row.unsqueeze(0).unsqueeze(0).expand(16, HEAD_NUM, HEAD_DIM).contiguous().to(
            torch.bfloat16
        )
    )

    # A Qwen3-8B-shaped batch: 512 tokens is a realistic HiCache transfer size.
    vectors["qwen3_8b_shaped"] = torch.randn(
        (512, HEAD_NUM, HEAD_DIM), generator=g
    ).to(torch.bfloat16)

    # Powers of two as absmax: exercises every exponent without bf16 scale
    # rounding, so any drift here is a real arithmetic difference.
    pow2 = torch.zeros((24, HEAD_NUM, HEAD_DIM), dtype=torch.bfloat16)
    for i in range(24):
        mag = 2.0 ** (i - 12)
        pow2[i, :, 0] = mag
        pow2[i, :, 1] = -mag
        pow2[i, :, 2:] = (
            torch.randn((HEAD_NUM, HEAD_DIM - 2), generator=g) * mag
        ).to(torch.bfloat16)
    vectors["power_of_two_absmax"] = pow2

    if str(device) != "cpu":
        return {k: v.to(device) for k, v in vectors.items()}
    return vectors


def load_or_make_vectors(
    device: torch.device | str, path: Path
) -> tuple[dict[str, torch.Tensor], bool]:
    """Load the shared vector set, creating it if absent.

    Returns ``(vectors, created)``. Persisting the tensors rather than trusting
    the generator alone means both machines read the same bytes off disk, which
    removes the entire class of "same statistics, different digest" confusion:
    with a matching input digest, any output difference really is the codec.
    """
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        vectors = dict(payload["vectors"])
        if str(device) != "cpu":
            vectors = {k: v.to(device) for k, v in vectors.items()}
        return vectors, False

    vectors = make_vectors()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vectors": {k: v.cpu() for k, v in vectors.items()},
            "torch": torch.__version__,
        },
        path,
    )
    if str(device) != "cpu":
        vectors = {k: v.to(device) for k, v in vectors.items()}
    return vectors, True


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
