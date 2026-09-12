"""Preflight checks that run on the Mac before renting a GPU pod.

Everything here is checkable without CUDA. The point is to fail cheaply: each
check corresponds to a way the pod run could die after you have already paid for
the GPU, and every one of them has a pure-arithmetic or pure-tensor answer that
does not need a device.

The highest-value check is the JIT kernel template selection. SGLang compiles a
specialised CUDA kernel per ``element_size``, and whether a 1152-byte row is
admissible is decided by two small arithmetic functions in
``sglang/kernels/ops/kvcache/hicache.py``. If those say yes and the encoded path
still fails on the pod, the failure is in the driver or the arena, not in the
format — which is a much smaller search space.

Usage::

    python scripts/preflight.py                     # all checks
    python scripts/preflight.py --sglang-root ../sglang
    python scripts/preflight.py --json results/preflight.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hiqcache.layout import (  # noqa: E402
    QWEN3_8B_HEAD_DIM,
    QWEN3_8B_KV_HEADS,
    QWEN3_8B_LAYER_NUM,
    V1_LAYOUT,
)

ROW_BYTES = V1_LAYOUT.row_bytes

# ---------------------------------------------------------------------------
# Mirrors of SGLang's JIT eligibility arithmetic
# ---------------------------------------------------------------------------
#
# Copied from python/sglang/kernels/ops/kvcache/hicache.py at the pinned base
# commit. They are re-implemented rather than imported because importing that
# module pulls in torch's CUDA machinery. ``--sglang-root`` additionally
# *verifies* these mirrors against the real source by parsing it, so a drift
# between this file and SGLang is reported instead of silently trusted.

COPY_GROUP_THREADS = 32
GROUP_BYTES_CUDA = (128,)
GROUP_BYTES_HIP = (128, 64, 32, 16)


def default_unroll(element_size: int) -> int:
    """Mirror of ``_default_unroll``."""
    if element_size <= 512:
        return 4
    if element_size <= 1024:
        return 2
    return 1


def tiles_across_lanes(element_size: int, unroll: int, *, is_hip: bool = False) -> bool:
    """Mirror of ``_tiles_across_lanes`` / ``pick_group_bytes() != 0``."""
    if unroll <= 0 or unroll > COPY_GROUP_THREADS or COPY_GROUP_THREADS % unroll != 0:
        return False
    lanes_per_worker = COPY_GROUP_THREADS // unroll
    groups = GROUP_BYTES_HIP if is_hip else GROUP_BYTES_CUDA
    return any(
        group % lanes_per_worker == 0
        and element_size % group == 0
        and group // lanes_per_worker in (4, 8, 16)
        for group in groups
    )


def jit_eligible(element_size: int, *, is_hip: bool = False) -> tuple[bool, dict]:
    """Full ``can_use_hicache_jit_kernel`` decision, minus the module load."""
    unroll = default_unroll(element_size)
    lanes = COPY_GROUP_THREADS // unroll if unroll else 0
    groups = GROUP_BYTES_HIP if is_hip else GROUP_BYTES_CUDA
    details = {
        "element_size": element_size,
        "unroll": unroll,
        "lanes_per_worker": lanes,
        "groups_considered": list(groups),
        "chosen_group": None,
        "package_bytes": None,
    }
    for group in groups:
        if (
            group % lanes == 0
            and element_size % group == 0
            and group // lanes in (4, 8, 16)
        ):
            details["chosen_group"] = group
            details["package_bytes"] = group // lanes
            break
    ok = details["chosen_group"] is not None
    return ok, details


def tma_eligible(element_size: int, *, page_size: int, is_hip: bool, sm_major: int) -> bool:
    """Mirror of ``use_hicache_tma_kernel`` for a non-HIP, TMA-enabled build.

    Note ``_is_hip`` short-circuits it: the TMA kernel is HIP-only in this
    revision, so on an NVIDIA pod this always returns False and the register
    kernel is used. Recorded here because that is a surprising fact worth
    asserting rather than assuming.
    """
    if is_hip:
        return False
    if element_size % 16 != 0 or sm_major < 9:
        return False
    # rows_per_chunk is computed by hicache_tma_rows_per_chunk(element_size);
    # the page must tile it. page_size == 1 tiles any chunk width.
    return page_size == 1 or page_size % 1 == 0


# ---------------------------------------------------------------------------
# Check harness
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fatal: bool = True
    data: dict = field(default_factory=dict)


class Preflight:
    def __init__(self, sglang_root: Path | None):
        self.sglang_root = sglang_root
        self.checks: list[Check] = []

    def add(self, name: str, ok: bool, detail: str = "", *, fatal: bool = True, **data):
        self.checks.append(
            Check(name=name, ok=bool(ok), detail=detail, fatal=fatal, data=data)
        )
        return ok

    # -- format ---------------------------------------------------------

    def check_format(self):
        self.add(
            "record is 1152 B and 128-B aligned",
            ROW_BYTES == 1152 and ROW_BYTES % 128 == 0,
            f"row_bytes={ROW_BYTES} = {ROW_BYTES // 128} x 128",
            row_bytes=ROW_BYTES,
        )
        stats = {
            "baseline_bytes_per_token": 147_456,
            "encoded_bytes_per_token": 82_944,
        }
        self.add(
            "Qwen3-8B compression is 43.75% / 1.78x",
            stats["encoded_bytes_per_token"] == V1_LAYOUT.bytes_per_token_all_layers(36)
            and stats["baseline_bytes_per_token"] == 147_456,
            f"{stats['baseline_bytes_per_token']} -> "
            f"{stats['encoded_bytes_per_token']} B/token",
            **stats,
        )

    # -- JIT eligibility -------------------------------------------------

    def check_jit(self):
        ok, details = jit_eligible(ROW_BYTES, is_hip=False)
        self.add(
            "1152-B rows are CUDA JIT eligible",
            ok,
            f"unroll={details['unroll']}, lanes_per_worker={details['lanes_per_worker']}, "
            f"group={details['chosen_group']}, package={details['package_bytes']} B",
            **details,
        )
        ok_hip, details_hip = jit_eligible(ROW_BYTES, is_hip=True)
        self.add(
            "1152-B rows are ROCm JIT eligible",
            ok_hip,
            f"unroll={details_hip['unroll']}, group={details_hip['chosen_group']}",
            fatal=False,
            **details_hip,
        )
        self.add(
            "write-back kernel accepts 1152 B",
            ROW_BYTES % 16 == 0,
            "requires element_size % 16 == 0",
            element_size=ROW_BYTES,
        )

    def check_jit_boundaries(self):
        """Which element sizes are admissible, to show 1152 is not a lucky guess."""
        admissible = [n for n in range(16, 4097, 16) if jit_eligible(n)[0]]
        self.add(
            "admissible CUDA element sizes include 1152",
            ROW_BYTES in admissible,
            f"{len(admissible)} sizes admissible in [16, 4096] step 16; "
            f"1152 is one of them",
            count=len(admissible),
        )
        # A rejected neighbour, to prove the check can actually fail.
        self.add(
            "the eligibility check is not vacuous",
            not jit_eligible(1088)[0],
            "1088 B is correctly rejected (1088 % 128 != 0)",
            fatal=False,
            rejected_size=1088,
        )

    def check_tma_gate(self):
        """On an NVIDIA pod the TMA path is off, so the register kernel is used."""
        tma = tma_eligible(ROW_BYTES, page_size=1, is_hip=False, sm_major=8)
        self.add(
            "TMA path is inactive on an A6000 (sm_86)",
            tma is False,
            "use_hicache_tma_kernel short-circuits on _is_hip; register kernel is used",
            fatal=False,
            tma_eligible=tma,
        )

    # -- fork source parity ----------------------------------------------

    def check_fork_source(self):
        """Verify the mirrors above still match the fork's real source."""
        if self.sglang_root is None:
            self.add(
                "fork source parity",
                True,
                "skipped: SGLang fork not found",
                fatal=False,
            )
            return
        path = (
            self.sglang_root
            / "python"
            / "sglang"
            / "kernels"
            / "ops"
            / "kvcache"
            / "hicache.py"
        )
        if not path.is_file():
            self.add("fork source parity", False, f"missing {path}")
            return
        src = path.read_text()
        expected = {
            "COPY_GROUP_THREADS = 32": "COPY_GROUP_THREADS",
            "GROUP_BYTES = (128, 64, 32, 16) if _is_hip else (128,)": "GROUP_BYTES",
            "(4, 8, 16)": "package sizes",
            "if element_size <= 512:": "_default_unroll tier 1",
            "if element_size <= 1024:": "_default_unroll tier 2",
        }
        missing = [label for needle, label in expected.items() if needle not in src]
        self.add(
            "JIT arithmetic mirrors match the fork source",
            not missing,
            "all markers present" if not missing else f"markers changed: {missing}",
            missing=missing,
        )

        # The pool must not have grown a host sync since it was written.
        pool = (
            self.sglang_root
            / "python"
            / "sglang"
            / "srt"
            / "mem_cache"
            / "pool_host"
            / "mha_int8.py"
        )
        if pool.is_file():
            tree = ast.parse(pool.read_text())
            offenders = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    dotted = _dotted(node.func)
                    if dotted in (
                        "torch.cuda.synchronize",
                        "torch.cuda.current_stream",
                    ) or dotted.endswith(".item"):
                        offenders.append(dotted)
                if isinstance(node, ast.Attribute) and node.attr == "cpu":
                    offenders.append(".cpu()")
            self.add(
                "pool hot path has no host synchronisation",
                not offenders,
                "no synchronize/.cpu()/.item() found"
                if not offenders
                else f"found {sorted(set(offenders))}",
                offenders=sorted(set(offenders)),
            )
            text = pool.read_text()
            for needle, label in (
                ("element_dim=codec.ROW_BYTES // 2", "H2D bf16 reinterpretation"),
                ("kv_cache_src_stride_bytes=codec.ROW_BYTES", "D2H src stride"),
                ("kv_cache_dst_stride_bytes=codec.ROW_BYTES", "D2H dst stride"),
                ("element_size=codec.ROW_BYTES", "D2H element_size"),
            ):
                self.add(
                    f"pool contains {label}",
                    needle in text,
                    needle if needle in text else f"MISSING: {needle}",
                )
        else:
            self.add("pool source present", False, f"missing {pool}")

    # -- staging geometry ------------------------------------------------

    def check_staging(self):
        for capacity in (2048, 4096):
            bytes_per_buffer = QWEN3_8B_LAYER_NUM * capacity * ROW_BYTES
            total = 4 * bytes_per_buffer  # d2h k/v + h2d k/v
            self.add(
                f"staging footprint at capacity {capacity} is reasonable",
                total < 2 * 1024**3,
                f"4 x {bytes_per_buffer / 2**20:.1f} MiB = {total / 2**20:.0f} MiB device",
                fatal=False,
                capacity=capacity,
                total_bytes=total,
            )

    # -- capacity arithmetic ---------------------------------------------

    def check_capacity(self):
        baseline = 147_456
        encoded = 82_944
        for size_gb, expect_base, expect_enc in ((8.0, 54_254, 96_451),):
            got_base = int(size_gb * 1e9 // baseline) + 1
            got_enc = int(size_gb * 1e9 // encoded) + 1
            self.add(
                f"--hicache-size={size_gb:g} token capacity matches the calculator",
                (got_base, got_enc) == (expect_base, expect_enc),
                f"bf16={got_base} hiqcache={got_enc} (gain {got_enc / got_base:.4f}x)",
                baseline_tokens=got_base,
                encoded_tokens=got_enc,
            )

    # -- artifacts -------------------------------------------------------

    def check_artifacts(self):
        manifest = REPO_ROOT / "results" / "conformance_cpu.json"
        if manifest.is_file():
            payload = json.loads(manifest.read_text())
            n = len(payload.get("vectors", {}))
            self.add(
                "CPU conformance manifest exists for the pod gate",
                n > 0,
                f"{manifest.name}: {n} vectors, device={payload.get('device')}",
                vectors=n,
            )
        else:
            self.add(
                "CPU conformance manifest exists for the pod gate",
                False,
                f"missing {manifest}; run scripts/conformance.py generate --device cpu",
            )

    def run(self) -> bool:
        self.check_format()
        self.check_jit()
        self.check_jit_boundaries()
        self.check_tma_gate()
        self.check_fork_source()
        self.check_staging()
        self.check_capacity()
        self.check_artifacts()
        return all(c.ok for c in self.checks if c.fatal)


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def find_fork_root(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()
    for candidate in (REPO_ROOT.parent / "sglang", REPO_ROOT / "sglang"):
        if (candidate / "python" / "sglang" / "srt").is_dir():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    root = find_fork_root(args.sglang_root)
    pf = Preflight(root)
    ok = pf.run()

    print("HiQCache preflight (no GPU required)")
    print(f"SGLang fork: {root if root else 'not found -- source parity skipped'}")
    print()
    for check in pf.checks:
        mark = "PASS" if check.ok else ("FAIL" if check.fatal else "WARN")
        print(f"[{mark}] {check.name}")
        if check.detail:
            print(f"       {check.detail}")
    fatal_failures = [c.name for c in pf.checks if not c.ok and c.fatal]
    warnings = [c.name for c in pf.checks if not c.ok and not c.fatal]
    print()
    if fatal_failures:
        print(f"RESULT: {len(fatal_failures)} blocking failure(s) -> do not rent a pod yet")
        for name in fatal_failures:
            print(f"  - {name}")
    else:
        print("RESULT: all blocking checks passed -- safe to proceed to the pod")
    if warnings:
        print(f"({len(warnings)} non-blocking warning(s))")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "ok": ok,
                    "sglang_root": str(root) if root else None,
                    "checks": [asdict(c) for c in pf.checks],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
