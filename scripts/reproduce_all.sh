#!/usr/bin/env bash
# Run the full HiQCache measurement suite on a GPU pod, in one command.
#
# The point is repeatability: the results in docs/experiment-b-results.md came
# from an ad-hoc sequence of commands, and a claim that cannot be re-run is a
# claim that cannot be checked. This script is that sequence, with the ordering
# and sizing decisions that were learned the hard way encoded as defaults.
#
# Usage:
#   bash scripts/reproduce_all.sh                 # everything
#   bash scripts/reproduce_all.sh --skip-repeats  # faster, no repeat sweep
#   HOST_SIZE=8 TAG=exp-b bash scripts/reproduce_all.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"
HOST_SIZE="${HOST_SIZE:-8}"
SMOKE_HOST_SIZE="${SMOKE_HOST_SIZE:-2}"
TAG="${TAG:-exp-b}"
REPEATS="${REPEATS:-3}"
SKIP_REPEATS=0
SKIP_SMOKE=0

for arg in "$@"; do
  case "$arg" in
    --skip-repeats) SKIP_REPEATS=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

command -v nvidia-smi >/dev/null || die "no nvidia-smi: this is not a GPU pod"
if [ ! -d "$REPO_ROOT/../sglang/python/sglang" ]; then
  die "SGLang fork not found at $REPO_ROOT/../sglang"
fi
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

say "Provenance"
{
  echo "captured $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/nvcc \1/p' || echo "no nvcc"
  $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  echo "sglang $(git -C ../sglang rev-parse HEAD)"
  echo "hiqcache $(git rev-parse HEAD)"
} | tee results/provenance.txt

# ---------------------------------------------------------------- smoke tests
# These are REGRESSION checks, not new measurements. Both passed before; if
# either fails now, stop -- something in the integration or the harness regressed
# and no benchmark below would mean anything.
if [ "$SKIP_SMOKE" -eq 0 ]; then
  say "Phase 11 smoke: INT8 compressed L2 (expect 82,944 B/token, restore > 0)"
  $PY scripts/smoke_test.py --config int8 --host-size "$SMOKE_HOST_SIZE"

  say "Phase 11 control: BF16 L2 (expect 147,456 B/token)"
  $PY scripts/smoke_test.py --config bf16 --host-size "$SMOKE_HOST_SIZE"
fi

# ------------------------------------------------------- Experiment B, main
say "Experiment B (same physical budget, ${HOST_SIZE} GB)"
for cfg in baseline bf16 int8; do
  $PY scripts/run_experiment.py --config "$cfg" --tag "$TAG" \
      --host-size "$HOST_SIZE" --workload reusable-prefixes
done

# ------------------------------------------------------- Experiment B, repeats
# Alternating order, not all-BF16 then all-INT8: a monotonic drift in pod
# conditions (thermal, neighbours, page cache) would otherwise be attributed to
# the config. This is what makes the latency deltas quotable.
if [ "$SKIP_REPEATS" -eq 0 ]; then
  say "Experiment B repeats (${REPEATS}x, alternating order)"
  for i in $(seq 1 "$REPEATS"); do
    for cfg in bf16 int8; do
      $PY scripts/run_experiment.py --config "$cfg" --tag "${TAG}-r${i}" \
          --host-size "$HOST_SIZE" --workload reusable-prefixes
    done
  done
fi

# --------------------------------------------- Experiment A: equal tokens
# Equal LOGICAL capacity. The BF16 capacity at this budget is the target, and
# each config gets the host bytes that buys it -- INT8 needs ~56% of BF16's, so
# this deliberately gives INT8 LESS memory. That is the point: it isolates codec
# cost by removing the capacity advantage.
# HOST_SIZE is a plain shell variable, so it is passed by interpolation
# rather than read from the environment, where it would silently fall back.
BF16_TOKENS=$($PY -c "
import sys
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from run_experiment import capacity_budget
print(capacity_budget(float('$HOST_SIZE'), 36)['bf16']['token_capacity'])
")
BF16_TOKENS=${BF16_TOKENS:-54254}
say "Experiment A (equal logical capacity: ${BF16_TOKENS} tokens in both tiers)"
for cfg in bf16 int8; do
  $PY scripts/run_experiment.py --config "$cfg" --tag "${TAG}-exp-a" \
      --workload reusable-prefixes \
      --sizing equal-tokens --target-l2-tokens "$BF16_TOKENS"
done

# ------------------------------------------------------------- quality
say "Phase 15 quality: identical deterministic prompts, BF16 vs INT8"
$PY scripts/quality_compare.py --host-size "$SMOKE_HOST_SIZE" --prompts 20

say "DONE -- pull results/ before terminating the pod"
cat <<EOF

  From your Mac:
    rsync -avz -e "ssh -p <PORT>" root@<HOST>:/workspace/hiqcache/results/ ./pod-results/

  Then analyse locally (no GPU needed):
    python scripts/analyse.py pod-results/ --markdown

  Reminder: this pod has no volume, so anything not copied out is lost.
EOF
