# Fresh pod: start to finish

For a **completely empty pod** — no repos, no venv, no SGLang, no model. Every
command is copy-pasteable in order. Nothing assumes anything already exists.

Total: **~90–120 minutes**, of which ~60 is waiting. Cost at A6000 rates: **under $1**.

---

## Before you start: create the pod with the right image

```
Image: runpod/pytorch:1.4.0-rc.164-cu1300-torch2130-ubuntu2404
GPU:   1x RTX A6000 (48 GB)
Disk:  150 GB container
```

The image matters and is not interchangeable — SGLang's base dependencies require
CUDA 13 and this image ships `torch 2.13.0+cu130` matching SGLang's pin exactly.
See `docs/image-compatibility.md` for the matrix.

**No volume is needed**, but that means the container disk is disposable: nothing
survives termination, so results must be copied out (step 7).

---

## Step 1 — Verify the image before building anything (~1 min)

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version | tail -2
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

**Expect exactly:**

```
NVIDIA RTX A6000, 580.xx, 46068 MiB
Cuda compilation tools, release 13.0, V13.0.98
2.13.0+cu130 13.0
```

If the CUDA release is **not 13.x**, or torch is not **2.13.0+cu130**, stop and
recreate the pod — the instructions below will fail in confusing ways.

---

## Step 2 — Clone and run the bootstrap gates (~8 min)

```bash
cd /workspace
git clone https://github.com/awesome-pro/hiqcache.git
cd hiqcache
bash scripts/pod_bootstrap.sh /workspace
```

One command. It clones both repos at the right refs, installs `ninja`, builds an
isolated codec venv, records provenance, and runs four gates:

| Gate | Proves | Expected |
| --- | --- | --- |
| 0 | the box can JIT-compile CUDA | `this box can run the HiCache JIT path` |
| 1 | local suite green on the pod | `315 passed` |
| 2 | preflight | `all blocking checks passed` |
| 3 | **CUDA codec ≡ the Mac's CPU reference, bit for bit** | `PASS: bit-identical` |

It is idempotent — re-run it any time.

**If GATE 0 fails**, the output names the missing thing. If it is `nvcc` or the
CUDA headers, the image is wrong; recreate rather than debug. If it is `ninja`,
the auto-install could not reach PyPI.

**If GATE 3 fails**, stop. The codec disagrees with the CPU reference; nothing
downstream is meaningful. Send me the digest diff.

---

## Step 3 — Install SGLang (~15–25 min)

This is the heavy step. Two things the bare `pip install -e python` gets wrong,
both verified:

```bash
cd /workspace/sglang

# 1. Resolve only. Confirm it plans to KEEP torch 2.13.0+cu130.
SGLANG_BUILD_RUST_EXTS=none python -m pip install --dry-run \
    --extra-index-url https://download.pytorch.org/whl/cu130 -e python | tail -20

# 2. Install.
SGLANG_BUILD_RUST_EXTS=none python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu130 -e python

# 3. Confirm torch survived and still matches nvcc.
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version | tail -2
```

Why each flag:

- **`--extra-index-url`** — SGLang pins `torchaudio==2.11.0` beside
  `torch==2.13.0`. No torchaudio 2.13 build exists, so the pair only resolves
  through PyTorch's own wheel index. This is what SGLang's Dockerfile does.
- **`SGLANG_BUILD_RUST_EXTS=none`** — the Rust radix-tree core is *optional*
  (`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND` defaults to `"python"`) and building
  it needs `cargo`, which the image lacks. Skipping it avoids a toolchain install
  and a multi-minute compile.
- **No `[all]`** — that extra pulls the diffusion stack (opencv, diffusers,
  moviepy), which nothing here needs.

Step 3 must still print `2.13.0+cu130`. If pip replaced torch, stop — the JIT
kernels will not build against a mismatched toolchain.

---

## Step 4 — Verification and the full measurement suite (~60–80 min)

```bash
cd /workspace/hiqcache
bash scripts/verify_sglang.sh
```

This does, in order, and stops on the first failure:

1. Confirms SGLang imports and the INT8 flag defaults to off
2. Re-runs the environment probe **under the SGLang environment** (a different
   question from the codec venv GATE 0 checked)
3. Runs the INT8 codec unit tests (no GPU)
4. Runs the **pool tests** — 36 tests covering the JIT kernel at 1152 bytes,
   `cudaHostRegister`, the pointer-table move, the H2D byte copy and all the
   fail-fast guards
5. Only if those pass, runs the measurement suite

The measurement suite (`scripts/reproduce_all.sh`) then does:

| Step | Duration | What it proves |
| --- | --- | --- |
| Smoke INT8 + BF16 | ~6 min | compressed write path, L2→L1 restore, 82,944 vs 147,456 B/token |
| Experiment B, 3 configs | ~12 min | the headline comparison at an 8 GB host budget |
| Experiment B repeats ×3, alternating | ~30 min | whether the latency deltas survive repetition |
| Experiment A, equal logical capacity | ~12 min | what the codec costs once its capacity advantage is removed |
| Quality, BF16 vs INT8 | ~15 min | logprob deltas and generation agreement |

**The first run downloads Qwen3-8B (~16 GB)** during the smoke test. Startup
prints progress every 15 s, so it will not look like a hang.

Run less if you want: `--skip-repeats` drops ~30 min, `--tests-only` stops after
the pool tests.

---

## Step 5 — What to watch

- **`verdict: DISCRIMINATING`** before each Experiment B/B run. If it says
  `TOO SMALL` or `TOO LARGE` the working set does not fit between the two L2
  capacities and that sweep is uninformative. The script tells you before
  starting a server.
- **`verdict: PASS`** on the smoke tests — six checks each.
- **`measured 82,944 B/token`** for INT8, **`147,456`** for BF16. These are exact
  identities, not estimates; a different number means padding or the wrong pool.

---

## Step 6 — Expect the quality harness to be the likely failure

`scripts/quality_compare.py` has never run against a real server. Its *arithmetic*
is unit-tested, but the `return_logprob` response shape and `/flush_cache`
behaviour are inferred from source. If it errors, that is the most likely cause —
paste the output and it is a small fix.

A failure there does **not** invalidate the rest: the smoke tests and Experiments
A/B write their results independently.

---

## Step 7 — Get the results off the pod (~1 min)

**Do this before terminating. There is no volume; nothing survives.**

```bash
# from your Mac — copy the exact ssh command from RunPod's Connect panel
cd /Users/abhinandan/Desktop/kvcodec
rsync -avz -e "ssh -p <PORT>" root@<POD_HOST>:/workspace/hiqcache/results/ ./pod-results/
rsync -avz -e "ssh -p <PORT>" root@<POD_HOST>:/workspace/provenance/ ./pod-provenance/ 2>/dev/null || true
```

Then **terminate the pod**. Everything else is analysis and can be done on the
Mac, so GPU hours stop there.

---

## Step 8 — Analyse on the Mac (no GPU)

```bash
cd hiqcache
python scripts/analyse.py ../pod-results/ --markdown
```

Prints the comparison table for the README and states the Experiment B verdict
explicitly: whether INT8 avoids recomputation that BF16 cannot.

Send me `pod-results/` and I will fold it into
`docs/experiment-b-results.md`, or write it up as a negative result if the
latency deltas do not survive repetition. PROJECT.md Phase 18 explicitly allows
that outcome — the capacity result stands either way.

---

## Quick reference: what each failure means

| Symptom | Meaning | Action |
| --- | --- | --- |
| GATE 0: nvcc / headers missing | wrong image | recreate with the cu1300 image |
| GATE 0: ninja missing | auto-install could not reach PyPI | `pip install ninja`, re-run |
| GATE 1 fails | codec/harness regression | send me the failure; do not proceed |
| GATE 3 fails | codec not device-independent | stop, send the digest diff |
| `No module named pytest` | you skipped `verify_sglang.sh` | run it; it installs pytest |
| `pip install` fights torch | image/torch mismatch | stop; check step 3 output |
| pool tests fail | integration regressed | send the traceback |
| smoke `verdict: FAIL` step 5 | prefix never left L1 | lower `--max-total-tokens` |
| `verdict: TOO SMALL` | working set fits both L2s | raise `--num-groups` |
