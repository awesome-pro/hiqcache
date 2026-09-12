#!/usr/bin/env bash
# Bootstrap a fresh GPU pod for the HiQCache pod run.
#
# Idempotent and safe to re-run. Does NOT install SGLang itself -- that is a
# separate, heavier step (see docs/pod-guide.md step 3) because the right command
# depends on the CUDA version the pod image ships.
#
# Usage:
#   bash scripts/pod_bootstrap.sh /workspace
set -euo pipefail

WORKSPACE="${1:-/workspace}"
HICACHE_REF="${HICACHE_REF:-main}"
SGLANG_REF="${SGLANG_REF:-hiqcache/int8-l2}"
BASE_COMMIT="515f5be77e74761c269e007ac41a5895191a1b7d"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

say "GPU and driver"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this is not a GPU pod"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see a GPU'" \
  2>/dev/null || die "torch is missing or cannot see the GPU"

mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# ---------------------------------------------------------------- provenance
say "Recording provenance (required by PROJECT.md Phase 14)"
mkdir -p provenance
{
  echo "=== captured $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  nvidia-smi | head -12
  echo "--- cpu ---"; lscpu | head -20 || true
  echo "--- memory ---"; free -h || true
  echo "--- nvcc ---"; nvcc --version 2>/dev/null | tail -3 || echo "no nvcc"
} > provenance/hardware.txt
cat provenance/hardware.txt

# ------------------------------------------------------------------- repos
say "hiqcache ($HICACHE_REF)"
if [ -d hiqcache/.git ]; then
  git -C hiqcache fetch --quiet origin && git -C hiqcache checkout --quiet "$HICACHE_REF"
  git -C hiqcache pull --quiet --ff-only
else
  git clone git@github.com:awesome-pro/hiqcache.git
  git -C hiqcache checkout "$HICACHE_REF"
fi
echo "hiqcache @ $(git -C hiqcache rev-parse --short HEAD)"

say "sglang ($SGLANG_REF)"
if [ -d sglang/.git ]; then
  git -C sglang fetch --quiet origin && git -C sglang checkout --quiet "$SGLANG_REF"
  git -C sglang pull --quiet --ff-only
else
  git clone git@github.com:awesome-pro/sglang.git
  git -C sglang checkout "$SGLANG_REF"
fi
echo "sglang @ $(git -C sglang rev-parse --short HEAD)"
git -C sglang merge-base --is-ancestor "$BASE_COMMIT" HEAD \
  || die "the pinned base commit $BASE_COMMIT is not an ancestor of $SGLANG_REF"
echo "pinned base $BASE_COMMIT is an ancestor: OK"

git -C sglang rev-parse HEAD > provenance/sglang_sha.txt
git -C hiqcache rev-parse HEAD > provenance/hiqcache_sha.txt

# --------------------------------------------------------------- codec venv
say "Codec venv (torch + pytest only; independent of the SGLang install)"
if [ ! -d hiqcache/.venv ]; then
  (cd hiqcache && python -m venv .venv)
fi
hiqcache/.venv/bin/pip install --quiet --upgrade pip
hiqcache/.venv/bin/pip install --quiet torch pytest numpy
hiqcache/.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ------------------------------------------------------------------- gates
say "GATE 1/3: local test suite must be green"
(cd hiqcache && HIQCACHE_SGLANG_ROOT="../sglang" .venv/bin/python -m pytest tests/ -q)

say "GATE 2/3: preflight"
(cd hiqcache && .venv/bin/python scripts/preflight.py --sglang-root ../sglang)

say "GATE 3/3: codec conformance on CUDA vs the Mac's CPU manifest"
(cd hiqcache && .venv/bin/python scripts/conformance.py generate --device cuda \
    --json results/conformance_cuda.json \
    --expect results/conformance_cpu.json)

say "GATES PASSED"
cat <<'EOF'
Next: install SGLang, then run the pool tests.

  cd sglang
  pip install -e "python[all]"        # see docs/pod-guide.md step 3 for the
                                      # CUDA-version-specific command
  python -m pytest test/registered/unit/mem_cache/test_hicache_int8_codec.py -v
  python -m pytest test/registered/unit/mem_cache/test_hicache_int8_pool_host_unit.py -v
EOF
