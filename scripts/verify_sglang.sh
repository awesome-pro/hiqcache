#!/usr/bin/env bash
# Post-SGLang-install verification, in one command.
#
# pod_bootstrap.sh covers everything that does NOT need SGLang installed: the
# environment probe, the codec suite, the preflight and CUDA conformance. This
# script covers what does, and is the second half of a fresh-pod setup:
#
#   1. pod_bootstrap.sh   gates 0-3  (no SGLang needed)
#   2. install SGLang     (the heavy step)
#   3. verify_sglang.sh   this file  (pool tests + reproduction suite)
#
# Usage:
#   bash scripts/verify_sglang.sh              # pool tests, then everything
#   bash scripts/verify_sglang.sh --tests-only # just the pool tests
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"
SGLANG_ROOT="${SGLANG_ROOT:-$REPO_ROOT/../sglang}"
TESTS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --tests-only) TESTS_ONLY=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[ -d "$SGLANG_ROOT/python/sglang" ] || die "no SGLang at $SGLANG_ROOT"

# ------------------------------------------------------------ what we skipped
# The install deliberately omits the [all] and [test] extras (diffusion stack,
# lm-eval, ...), so pytest is not present. Installing it here keeps the extra
# small and makes the failure mode obvious rather than "No module named pytest".
if ! $PY -c "import pytest" >/dev/null 2>&1; then
  say "Installing pytest (it lives in SGLang's [test] extra, which we skip)"
  $PY -m pip install --quiet pytest || die "could not install pytest"
fi

# ---------------------------------------------------------------- importable
say "SGLang is importable, and the INT8 flag exists"
$PY - <<'PY' || die "SGLang import failed -- did the install complete?"
import sys
try:
    from sglang.srt.environ import envs
except Exception as exc:                                  # noqa: BLE001
    print(f"  cannot import sglang.srt.environ: {type(exc).__name__}: {exc}")
    sys.exit(1)
print("  sglang imports OK")
print("  SGLANG_EXPERIMENTAL_HICACHE_INT8 default:",
      envs.SGLANG_EXPERIMENTAL_HICACHE_INT8.get())
assert envs.SGLANG_EXPERIMENTAL_HICACHE_INT8.get() is False, "flag should default off"
PY

# The JIT path needs a working toolchain *with SGLang's environment active*,
# which is a different question from the venv GATE 0 checked.
say "JIT compile works in the SGLang environment"
$PY scripts/env_probe.py --no-network || die "environment probe failed under SGLang"

# ---------------------------------------------------------------- pool tests
# Codec and staging units need no GPU and are fast; run them first so a failure
# there is not buried under pool-test output.
say "INT8 codec unit tests (no GPU)"
(cd "$SGLANG_ROOT" && $PY -m pytest \
    test/registered/unit/mem_cache/test_hicache_int8_codec.py -q) \
  || die "codec unit tests failed"

say "INT8 host pool tests (GPU: JIT kernel, pinned arena, transfer round trip)"
(cd "$SGLANG_ROOT" && $PY -m pytest -q \
    test/registered/unit/mem_cache/test_hicache_int8_pool_host_unit.py) \
  || die "host pool tests failed -- the integration regressed"

say "POOL TESTS PASSED"
if [ "$TESTS_ONLY" -eq 1 ]; then
  echo "  (--tests-only: stopping before the measurement suite)"
  exit 0
fi

# ------------------------------------------------------------- measurements
say "Measurement suite (see docs/reproducing-on-a-pod.md for what each step proves)"
exec bash scripts/reproduce_all.sh
