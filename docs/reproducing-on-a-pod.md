# Reproducing HiQCache on a RunPod pod

HiQCache adds an INT8-quantized host (L2) tier to SGLang's HiCache. This is the complete GPU sequence, in order, with the expected output and pass criteria for each step. Budget **90-120 minutes**, of which ~60 is waiting; the first server start downloads Qwen3-8B (~16 GB).

## Target GPU and image

| Item | Value |
| --- | --- |
| GPU / RAM | 1x NVIDIA RTX A6000, 48 GB (46,068-49,140 MiB reported, sm_86); 64+ GB host RAM |
| Image | `runpod/pytorch:1.4.0-rc.164-cu1300-torch2130-ubuntu2404` |
| Toolkit / torch | CUDA 13.0 (`nvcc` release 13.0, V13.0.98); torch 2.13.0+cu130 |
| OS / disk | Ubuntu 24.04, python 3.12, amd64; 150 GB container |

The image is not interchangeable: SGLang's dependencies hard-require CUDA 13 (`cuda-python>=13.0`, `flashinfer_python[cu13]`, `humming-kernels[cu13]`, `nvidia-cutlass-dsl[cu13]`, `nvshmem4py-cu13`), and no CUDA 12.8 build of torch 2.13.0 exists, so on a mismatched image pip replaces torch while the image's `nvcc` still compiles the JIT kernels (see [`docs/image-compatibility.md`](image-compatibility.md)). No volume is needed, so copy results out before terminating.

Verify the pod before building anything (~1 minute); the expected output is in the comments, and anything else means the wrong image, so recreate rather than debug:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version | tail -2
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# expected:
#   NVIDIA RTX A6000, <driver>, 46,068-49,140 MiB   (driver and reported MiB vary by host)
#   Cuda compilation tools, release 13.0, V13.0.98
#   2.13.0+cu130 13.0
```

## Clone and bootstrap gates (~8 minutes)

```bash
cd /workspace
git clone https://github.com/awesome-pro/hiqcache.git
cd hiqcache
bash scripts/pod_bootstrap.sh /workspace
```

Clone over **HTTPS, not SSH**: both repos are public and a fresh pod has no key, so `git@github.com:` fails with `Permission denied (publickey)` however public the repo is. The script is idempotent (`HICACHE_URL`, `SGLANG_URL`, `HICACHE_REF`, `SGLANG_REF` override the defaults). It records provenance under `/workspace/provenance/`, clones `hiqcache` at `main` and the SGLang fork at `hiqcache/int8-l2`, asserts the pinned base `515f5be77e74761c269e007ac41a5895191a1b7d` is an ancestor of the fork's `HEAD`, builds a codec venv (`torch`, `pytest`, `numpy`), installs `ninja` if missing, and runs four gates:

| Gate | Proves | Expected output |
| --- | --- | --- |
| 0 | the box can JIT-compile a CUDA extension | `RESULT: this box can run the HiCache JIT path.` |
| 1 | the codec suite is green on the pod | `332 passed` |
| 2 | preflight (layout, alignment, pool sizing) | `RESULT: all blocking checks passed -- safe to proceed to the pod` |
| 3 | the CUDA codec is bit-identical to the CPU reference | `PASS: codec is bit-identical to the reference device.` |

The published results were measured on `hiqcache/int8-l2`: Experiment B at SGLang `3bb2ef6602c12c102699f39830c80fa6cc2768d8` (pre-rewrite `9a7ac797`, preserved as the fork tag `measured-exp-b`), Experiment A two instrumentation-only commits later at `26e6d78dba`. The tip advances, so record `git -C ../sglang rev-parse HEAD` per run (the suite writes it to `results/provenance.txt`; provenance tables in [`docs/experiment-a-results.md`](experiment-a-results.md) and [`docs/experiment-b-results.md`](experiment-b-results.md)).

### The codec conformance gate

Gate 3 is decisive and runs before SGLang is installed: the codec is pure `torch` with no CUDA-only ops, so identical inputs must produce identical bytes on a laptop and on the GPU. The laptop generates the reference manifest from integer-only vectors (no RNG, 9 vectors) and commits it as `results/conformance_cpu.json`; `results/conformance_vectors.pt` is re-verified against a fresh generation on load, so a stale file cannot pin both sides to old inputs.

```bash
# On the laptop, before renting (the manifest is already committed):
python scripts/conformance.py generate --device cpu --json results/conformance_cpu.json

# Gate 3, or by hand on the pod:
.venv/bin/python scripts/conformance.py generate --device cuda \
    --json results/conformance_cuda.json --expect results/conformance_cpu.json
```

A pass prints `OK: all 9 vectors bit-identical across cpu and cuda` then `PASS: codec is bit-identical to the reference device.` Each vector carries an `input_digest`; if those differ the harness prints `INPUT MISMATCH` rather than blame the codec, because the runs encoded different tensors. If CUDA digests differ, stop: the bug is in the codec or a backend op, not in HiCache.

## Install SGLang (~15-25 minutes)

```bash
cd /workspace/sglang

# 1. Resolve only, and confirm it plans to KEEP torch 2.13.0+cu130.
SGLANG_BUILD_RUST_EXTS=none python -m pip install --dry-run \
    --extra-index-url https://download.pytorch.org/whl/cu130 -e python | tail -20

# 2. Install.
SGLANG_BUILD_RUST_EXTS=none python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu130 -e python

# 3. Confirm torch survived and still matches the image's nvcc.
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version | tail -2
```

Step 3 must still print `2.13.0+cu130`; if pip replaced torch, stop. Why the flags:

- `--extra-index-url https://download.pytorch.org/whl/cu130` — SGLang pins `torchaudio==2.11.0` beside `torch==2.13.0` and no torchaudio 2.13 build exists, so the pair resolves only through PyTorch's own wheel index (what its Dockerfile does).
- `SGLANG_BUILD_RUST_EXTS=none` — the Rust radix-tree core is optional (`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND` defaults to `"python"`) and the grpc/multimodal/server extensions are imported only by the features that need them; nothing on the HiCache path uses them, and building them needs `cargo`, which the image lacks, so `setup.py` fails at metadata time without this flag.
- No `[all]` extra — it pulls the diffusion stack (opencv, diffusers, moviepy), which nothing here uses.
- `ninja` and `pytest` are needed but absent by default: `ninja` drives the JIT compile and is installed by `pod_bootstrap.sh`; `pytest` lives in the skipped `[test]` extra and is installed by `scripts/verify_sglang.sh`.

```bash
# The INT8 flag must exist and default to off:
python -c "from sglang.srt.environ import envs; print('INT8 flag default:', envs.SGLANG_EXPERIMENTAL_HICACHE_INT8.get())"
# expect: INT8 flag default: False
```

## Verify the install

```bash
cd /workspace/hiqcache
bash scripts/verify_sglang.sh               # verification, then the measurements
bash scripts/verify_sglang.sh --tests-only  # stop after the pool tests
```

`verify_sglang.sh` stops on first failure and: installs `pytest` if missing; confirms SGLang imports with the INT8 flag `False`; re-runs the environment probe **inside the SGLang environment** (a different question from gate 0); runs the INT8 codec unit tests (**25 passed**, no GPU); runs the host-pool tests (**36 tests** on the GPU: the JIT kernel at `element_size = 1152`, `cudaHostRegister` on the encoded arena, the D2H all-layer move with a per-layer pointer table, the H2D byte copy at `element_dim = 1152` with uint8 on both sides, `index_select`/advanced-index scatter, and the fail-fast guards); then hands off to the measurement suite unless `--tests-only` was given. A failure in the last two is an integration regression, not a tuning problem — the traceback names the file. To run the codec suite against a fork checkout other than the sibling `../sglang`:

```bash
HIQCACHE_SGLANG_ROOT=../sglang .venv/bin/python -m pytest tests/ -q   # 332 passed
```

## Measure (~50-70 minutes)

```bash
cd /workspace/hiqcache
bash scripts/reproduce_all.sh                 # everything
bash scripts/reproduce_all.sh --skip-repeats  # faster first pass
HOST_SIZE=8 TAG=exp-b bash scripts/reproduce_all.sh   # the defaults, for reference
```

| Step | Duration | Expected |
| --- | --- | --- |
| Smoke INT8 + BF16 | ~6 min | `verdict: PASS`; 82,944 vs 147,456 B/token |
| Experiment B, 3 configs | ~12 min | identical GPU/SHA/workload, different L2 |
| Experiment B repeats x3, alternating | ~30 min | whether latency deltas survive repetition |
| Experiment A, equal logical capacity | ~12 min | both tiers at 54,254 L2 tokens |
| Quality, BF16 vs INT8 | ~15 min | agreement and logprob deltas, valid only if the re-send restored from L2 (see below) |

Manual equivalents, for re-running one piece:

```bash
python scripts/smoke_test.py --config int8 --host-size 2
python scripts/smoke_test.py --config bf16 --host-size 2
for cfg in baseline bf16 int8; do
  python scripts/run_experiment.py --config "$cfg" --tag exp-b-8gb --host-size 8 --workload reusable-prefixes
done
for cfg in bf16 int8; do
  python scripts/run_experiment.py --config "$cfg" --tag exp-b-exp-a --workload reusable-prefixes --sizing equal-tokens --target-l2-tokens 54254
done
python scripts/quality_compare.py --host-size 2 --prompts 20
```

`run_experiment.py` builds all three configs: `baseline` (HiCache disabled), `bf16` (standard BF16 L2) and `int8` (HiQCache, `SGLANG_EXPERIMENTAL_HICACHE_INT8=1`). Each server boots with `--hicache-io-backend kernel`, `--hicache-mem-layout layer_first`, `--hicache-write-policy write_back`, a capped L1 (`--max-total-tokens`, default 16,384) and a flushed prefix cache, then `/metrics` is snapshotted before and after. **Pass criteria:**

- **Smoke.** Every recorded check must pass, and the run prints `verdict: PASS`. The decisive check compares measured L2 bytes/token against the codec layout: `int8` must report **82,944 B/token** (36 layers x 2 x 1152-byte records) and the `bf16` control **147,456 B/token** — exact identities, so they prove the INT8 pool is live; anything between means padding or the wrong pool class. `sglang:load_back_tokens_total` must also rise on the re-send, and the regenerated text must match the first generation.
- **Experiment B.** Before a server starts, check both printed lines: `=== L2 budget @ 8 GB: bf16 54,254 tokens | int8 96,451 tokens (1.7778x)` and `DISCRIMINATING` (not `TOO SMALL` or `TOO LARGE`). The `int8` run prints measured encoded bytes/token against the layout as `PASS`/`FAIL`; `bf16` prints measured bytes/token against 147,456. Expected: baseline and BF16 evict and recompute where INT8 hits.
- **Experiment A.** Sized by `--hicache-ratio`, never a fractional `--hicache-size`, and checked against the pool's gauge: `L2 capacity verified against the pool's gauge: 54,254 tokens`. Both tiers hold the same token count, so the difference is the host memory each needs.
- **Quality.** Valid only if the re-send restored from L2; the harness raises `no L2 restores for <tag>` rather than report agreement from two cold prefills. On success it prints first-token, full-sequence and token-level agreement plus `|dlogprob|` mean/p95/p99/max.

Each run writes `results/exp_<config>_<tag>.json` (server args, SGLang SHA, metrics deltas, log tail), `results/exp_<config>_<tag>.server.log`, and for INT8 `results/codec_timing_int8_<tag>.json` with encode/decode timings written at teardown. Published numbers: [`docs/experiment-a-results.md`](experiment-a-results.md), [`docs/experiment-b-results.md`](experiment-b-results.md).

## Retrieve results

Copy results out **before terminating** — there is no volume, so nothing survives. From the laptop (take the exact SSH command from RunPod's Connect panel):

```bash
rsync -avz -e "ssh -p <PORT>" root@<POD_HOST>:/workspace/hiqcache/results/ ./pod-results/
rsync -avz -e "ssh -p <PORT>" root@<POD_HOST>:/workspace/provenance/ ./pod-provenance/
```

Or archive on the pod first with `cd /workspace && tar czf hiqcache-results.tgz hiqcache/results provenance`. Analyse on the laptop, not on GPU time:

```bash
cd hiqcache
python scripts/analyse.py ../pod-results/ --markdown
```

## Failure and diagnosis

| Symptom | Cause | Action |
| --- | --- | --- |
| `JIT COMPILE: FAILED -> RuntimeError: Ninja is required to load C++ extensions` | `ninja` missing; `load_inline`/`load_jit` shell out to it, and SGLang then disables the JIT path with a warning instead of crashing | `pip install ninja`, re-run `python scripts/env_probe.py`; do not replace the pod |
| Gate 0 reports missing `nvcc` or CUDA headers, `pip install -e python` replaces torch, or the JIT compile hits header errors | wrong image, or image/torch mismatch: the toolkit compiling the kernels disagrees with torch's bundled CUDA runtime | recreate with the cu1300 CUDA 13 image rather than debug this one; the resolve in `pod_bootstrap.sh` reports whether SGLang's CUDA 13 pins are satisfiable |
| `No module named pytest` | the `[test]` extra is skipped by design | run `bash scripts/verify_sglang.sh`; it installs pytest |
| Server exits with code 2 before loading the model, on a fractional `--hicache-size` | `hicache_size` is an `int` in decimal GB, so `8.000078` is rejected by argparse | size with `--hicache-ratio` (a float token ratio), as `--sizing equal-tokens` does |
| Server runs and L2 fills, but `host_hit_length` stays 0 and every revisit re-prefills | `write_through`/`write_through_selective` sets `is_write_back=False`, so evicted unbacked leaves are deleted instead of demoted and L2 holds unreferenced KV (measured once as 21.8 GB backed up, 0 tokens restored) | pass `--hicache-write-policy write_back`; without it L2 is unreachable |
| Zero restores after a `/flush_cache` | `/flush_cache` clears the radix cache, and when that reaches the host tier it deletes the L2 state being measured | do not flush between populate and re-send; let filler traffic evict L1 by LRU, as `scripts/quality_compare.py` does |
| Dtype or shape/stride assertion in the JIT mover during the H2D step | H2D dtype mismatch: one side reinterpreted as BF16 while the other is uint8 | keep both sides uint8 at `element_dim = 1152` (`mha_int8.py::load_to_device_per_layer`); the kernel binds one symbolic dtype to all four tensors |
| `cudaHostRegister` error | huge-page or granularity mismatch | check `registration_granularity_bytes` in `init_kv_buffer`; isolate the call with `python scripts/diagnose_host_register.py` |
| Server rejected at pool construction, or on `--hicache-storage-backend` | fail-fast guards (`--page-size != 1`, layout not `layer_first`, host/device page-size mismatch, quantized device KV, `store_dtype` not BF16/FP16, a row not a multiple of 128 B, missing JIT mover); L3 storage is out of scope | use `--page-size 1 --hicache-mem-layout layer_first`; drop `--hicache-storage-backend` |
| Smoke `verdict: FAIL` on the restore step | the prefix never left L1, so nothing needed restoring | lower `--max-total-tokens` (the smoke test defaults to 8192) |
| `verdict: TOO SMALL` before an Experiment B run | the reusable working set also fits the BF16 L2, so both configs hold it | raise `--num-groups` or `--gsp-prefix-len`; lower them for `TOO LARGE` |

Source map, data flow and the full fail-fast matrix: [`docs/sglang-integration.md`](sglang-integration.md). Design rationale — record layout, quantisation scheme, error bound, capacity arithmetic, stream rules: [`docs/design.md`](design.md). Upstream HiCache internals with file and line references: [`docs/hicache-internals.md`](hicache-internals.md).
