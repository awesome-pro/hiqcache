#!/usr/bin/env bash
# Bootstrap a fresh GPU pod for the HiQCache pod run.
#
# Idempotent and safe to re-run. Does NOT install SGLang itself -- that is a
# separate, heavier step (see docs/pod-guide.md step 3) because the right command
# depends on the CUDA version the pod image ships.
#
# Usage:
#   bash scripts/pod_bootstrap.sh /workspace
#
# Overrides (defaults work on a pod with no GitHub credentials at all):
#   HICACHE_URL   repo to clone (default: HTTPS; public repo, needs no key)
#   SGLANG_URL    repo to clone
#   HICACHE_REF   branch/tag (default: main)
#   SGLANG_REF    branch/tag (default: hiqcache/int8-l2)
set -euo pipefail

WORKSPACE="${1:-/workspace}"
HICACHE_REF="${HICACHE_REF:-main}"
SGLANG_REF="${SGLANG_REF:-hiqcache/int8-l2}"

# HTTPS by default, deliberately. Both repos are public, and an SSH clone needs a
# private key that a fresh pod does not have -- `git@github.com:` fails with
# "Permission denied (publickey)" no matter how public the repo is. Override with
# an SSH URL only if you have actually put a key on the pod.
HICACHE_URL="${HICACHE_URL:-https://github.com/awesome-pro/hiqcache.git}"
SGLANG_URL="${SGLANG_URL:-https://github.com/awesome-pro/sglang.git}"
BASE_COMMIT="515f5be77e74761c269e007ac41a5895191a1b7d"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

EXPECTED_IMAGE="${EXPECTED_IMAGE:-runpod/pytorch:1.4.0-rc.164-cu1300-torch2130-ubuntu2404}"

say "Expected image"
cat <<EOF
  This runbook targets: ${EXPECTED_IMAGE}
  Requirements it satisfies, all verified against the registry and PyPI:
    - CUDA 13.0 toolkit (nvcc present) -- SGLang's base deps need CUDA 13
    - torch 2.13.0+cu130 preinstalled -- matches SGLang's pin exactly
    - Ubuntu 24.04, python 3.12, amd64
  To override for a different image: EXPECTED_IMAGE=... bash $0
EOF

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
echo "  writing provenance/hardware.txt (a few seconds; plain nvidia-smi can pause briefly)"
{
  echo "=== captured $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  echo "--- driver ---"; nvidia-smi --query-gpu=driver_version --format=csv,noheader
  echo "--- cpu ---"; lscpu 2>/dev/null | head -20 || echo "lscpu unavailable"
  echo "--- memory ---"; free -h 2>/dev/null || echo "free unavailable"
  echo "--- python / torch ---"
  python -c "import sys, torch; print('python', sys.version.split()[0]);
print('torch', torch.__version__); print('torch cuda', torch.version.cuda);
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" \
    2>/dev/null || echo "torch not importable"
  echo "--- nvcc ---"; nvcc --version 2>/dev/null | tail -3 || echo "no nvcc"
} > provenance/hardware.txt
cat provenance/hardware.txt

# ------------------------------------------------------------------- repos
say "hiqcache ($HICACHE_REF)"
if [ -d hiqcache/.git ]; then
  git -C hiqcache fetch --quiet origin && git -C hiqcache checkout --quiet "$HICACHE_REF"
  git -C hiqcache pull --quiet --ff-only
else
  git clone "$HICACHE_URL"
  git -C hiqcache checkout "$HICACHE_REF"
fi
echo "hiqcache @ $(git -C hiqcache rev-parse --short HEAD)"

say "sglang ($SGLANG_REF)"
if [ -d sglang/.git ]; then
  git -C sglang fetch --quiet origin && git -C sglang checkout --quiet "$SGLANG_REF"
  git -C sglang pull --quiet --ff-only
else
  git clone "$SGLANG_URL"
  git -C sglang checkout "$SGLANG_REF"
fi
echo "sglang @ $(git -C sglang rev-parse --short HEAD)"
git -C sglang merge-base --is-ancestor "$BASE_COMMIT" HEAD \
  || die "the pinned base commit $BASE_COMMIT is not an ancestor of $SGLANG_REF"
echo "pinned base $BASE_COMMIT is an ancestor: OK"

git -C sglang rev-parse HEAD > provenance/sglang_sha.txt
git -C hiqcache rev-parse HEAD > provenance/hiqcache_sha.txt

# ------------------------------------------------------- GATE 0: env probe
# Runs before any pip install. Answers the only question that decides whether
# the compiled HiCache kernels can work at all: can this box JIT-compile a CUDA
# extension? A missing ninja or CUDA header otherwise surfaces much later as a
# generic "needs the JIT HiCache kernel" error that looks like a HiCache bug.
say "GATE 0/4: environment probe (JIT compile capability)"
# Resolve the probe by absolute path. This script clones into $WORKSPACE and
# therefore cds there, so a relative "scripts/env_probe.py" would not resolve --
# the probe lives in the hiqcache checkout, not in the workspace root.
PROBE="$WORKSPACE/hiqcache/scripts/env_probe.py"
if [ ! -f "$PROBE" ]; then
  die "environment probe not found at $PROBE (is the hiqcache clone complete?)"
fi
if ! python "$PROBE" --no-network; then
  cat <<'NOTE' >&2

GATE 0 failed. The probe output above names the blocking item(s).

Most common case, and a one-line fix:

    pip install ninja

(`ninja` is a declared SGLang dependency, so it would arrive with SGLang -- but
without it the JIT path fails in a way that is hard to attribute.)

If nvcc or the CUDA headers are missing, the compiled HiCache kernels cannot
work on this image at all. Switch to an image with a full CUDA toolkit rather
than debugging this one; see docs/pod-guide.md Appendix A.

NOTE
  die "environment probe failed (gate 0)"
fi

# --------------------------------------------------------------- codec venv
say "Codec venv (torch + pytest only; independent of the SGLang install)"
if [ ! -d hiqcache/.venv ]; then
  (cd hiqcache && python -m venv .venv)
fi
hiqcache/.venv/bin/pip install --quiet --upgrade pip
hiqcache/.venv/bin/pip install --quiet torch pytest numpy
hiqcache/.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ------------------------------------------------------- torch pin alignment
# The whole reason an image has to be chosen carefully: SGLang pins a torch
# version, and torch wheels only exist for certain CUDA versions. If the
# container's torch already satisfies the pin, `pip install -e python` replaces
# nothing and the image's nvcc (which compiles the JIT kernels) keeps matching
# torch's bundled CUDA. If it does not, pip swaps torch and the two can diverge.
say "torch pin alignment (SGLang requires a specific torch)"
# Parsed with sed rather than `grep -oP`: the latter is a GNU extension that BSD
# grep (macOS) rejects, and this script should stay runnable anywhere.
PINNED_TORCH="$(sed -n 's/^[[:space:]]*"torch==\([0-9][^"]*\)".*/\1/p' \
  sglang/python/pyproject.toml | head -1)"
INSTALLED_TORCH="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)"
INSTALLED_CUDA="$(python -c 'import torch; print(torch.version.cuda or "cpu")' 2>/dev/null || echo none)"
NVCC_RELEASE="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
NVCC_RELEASE="${NVCC_RELEASE:-none}"
printf '  SGLang pins torch : %s\n  container has     : %s (cuda %s)\n  nvcc release      : %s\n' \
  "${PINNED_TORCH:-unknown}" "$INSTALLED_TORCH" "$INSTALLED_CUDA" "$NVCC_RELEASE"
if [ -z "$PINNED_TORCH" ]; then
  echo "  -> could not parse the torch pin; skipping the alignment check."
elif [ "${INSTALLED_TORCH#"$PINNED_TORCH"}" != "$INSTALLED_TORCH" ]; then
  echo "  -> container torch already satisfies the pin; pip will not replace it."
  echo "     nvcc $NVCC_RELEASE and torch cuda $INSTALLED_CUDA are the operative pair."
else
  cat <<EOF
  -> WARNING: container torch does not satisfy the pin.
     'pip install -e python' will replace it, and torch's bundled CUDA runtime
     may then differ from this image's nvcc ($NVCC_RELEASE). Pure-torch work
     (codec, conformance) is unaffected; the JIT-compiled HiCache kernels may
     fail to build. See docs/image-compatibility.md.
     An image whose torch already matches avoids this, e.g.
     runpod/pytorch:1.3.3-rc.169-cu1300-torch2130-ubuntu2404
EOF
fi

# ------------------------------------------ SGLang dependency resolution
say "SGLang dependency resolution (does this image support SGLang's CUDA?)"
# SGLang's base dependencies pin CUDA 13 packages (cuda-python>=13.0,
# flashinfer_python[cu13], humming-kernels[cu13], nvidia-cutlass-dsl[cu13],
# nvshmem4py-cu13) and docker/Dockerfile supports only CUDA_VERSION=13.0.3. An
# image with an older toolkit cannot resolve them, and the failure normally
# appears deep inside an install. Resolve without downloading so it surfaces here.
CUINDEX="$(printf '%s' "$NVCC_RELEASE" | tr -d '.')"
if [ -n "$CUINDEX" ] && [ "$CUINDEX" != "none" ]; then
  echo "  probing with --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX}"
  echo "  (--dry-run resolves only; nothing is downloaded)"
  if python -m pip install --dry-run --quiet \
      --extra-index-url "https://download.pytorch.org/whl/cu${CUINDEX}" \
      -e sglang/python >/tmp/sglang_resolve.log 2>&1; then
    echo "  -> dependency set resolves on this image."
  else
    echo "  -> WARNING: could not resolve SGLang's dependencies on this image."
    grep -iE "conflict|no matching distribution|cannot install|requires" \
      /tmp/sglang_resolve.log | head -8 | sed 's/^/     /'
    cat <<EOF

     SGLang at the pinned commit needs CUDA 13 (docs/image-compatibility.md).
     On a CUDA 12.x image, switch to:
       runpod/pytorch:1.3.3-rc.169-cu1300-torch2130-ubuntu2404
     The codec gates below stay valid on any image -- they are pure torch and
     never touch the compiled kernels -- but the pool tests cannot pass here.
EOF
  fi
else
  echo "  -> skipped: could not determine the CUDA toolkit version."
fi

# ------------------------------------------------------------------- gates
say "GATE 1/4: local test suite must be green"
(cd hiqcache && HIQCACHE_SGLANG_ROOT="../sglang" .venv/bin/python -m pytest tests/ -q)

say "GATE 2/4: preflight"
(cd hiqcache && .venv/bin/python scripts/preflight.py --sglang-root ../sglang)

say "GATE 3/4: codec conformance on CUDA vs the Mac's CPU manifest"
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
