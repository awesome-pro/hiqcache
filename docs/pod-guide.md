# Pod guide — step by step

Everything to do on a rented GPU, in order. Each step says what it proves, what
you should see, and what to do if it fails.

Target: **1× A6000 48 GB**, 64+ GB RAM, 150+ GB disk.

**The image matters more than the hardware.** SGLang pins `torch==2.13.0`, which
has no CUDA 12.8 build, so a mismatched image wastes a run. Use
`runpod/pytorch:1.3.3-rc.169-cu1300-torch2130-ubuntu2404` — see step 1 and
`docs/image-compatibility.md`.

**Budget roughly 3–5 hours of GPU time** for steps 1–7, plus however long the
benchmark sweeps take. Steps 1–2 are the failure-prone ones; after step 1 passes
the rest is mostly waiting.

---

## Before you rent: the Mac is already done

Run this locally. If it fails, do not rent anything yet.

```bash
cd hiqcache
.venv/bin/python scripts/preflight.py
```

You want `RESULT: all blocking checks passed`. That confirms the format is
128-byte aligned, 1152-byte rows are JIT-eligible
(`unroll=1, lanes_per_worker=32, group=128, package=4`), the pool has no host
synchronisation in its hot path, and the CPU conformance manifest exists for the
pod to check against.

What is **not** proven before the pod: the CUDA JIT compile, `cudaHostRegister`,
the transfer kernels, and stream ordering. Those are steps 3–5 below.

---

## Step 1 — Rent the pod and get the code on it

### Pick a matching image first

This is worth two minutes and saves a wasted run. SGLang pins **`torch==2.13.0`**,
and PyTorch publishes **no CUDA 12.8 build of it** — the cu128 line stops at
2.11.0. Full matrix and reasoning: `docs/image-compatibility.md`.

Use one of these (torch 2.13.0, exact match for the pin):

```text
runpod/pytorch:1.3.3-rc.169-cu1300-torch2130-ubuntu2404   <- use this
runpod/pytorch:1.3.3-rc.169-cu1290-torch2130-ubuntu2404
```

Use **cu1300**. The torch pin is satisfied by both cu129 and cu130 images, but
SGLang's base dependencies also hard-require CUDA 13 (`cuda-python>=13.0`,
`flashinfer_python[cu13]`, `humming-kernels[cu13]`, `nvidia-cutlass-dsl[cu13]`,
`nvshmem4py-cu13`), and its Dockerfile supports only CUDA 13.0.3. A cu129 pod
cannot resolve that set. Two constraints, and the stricter one wins.

**A CUDA 12.8 image cannot work.** There is no torch 2.13.0 build for it, so pip
substitutes a torch built for a different CUDA. The runtime is bundled *inside
the torch wheel*, while the *toolkit* (`nvcc`) that compiles the JIT kernels
comes from the image — both must agree. `pod_bootstrap.sh` prints this pair and
warns if they diverge. Details: `docs/image-compatibility.md`.

Everything else about the pod — A6000 48 GB, 62 GB RAM, 150 GB disk — is correct.

### Get the code on it

```bash
# On the pod
cd /workspace
git clone https://github.com/awesome-pro/hiqcache.git   # HTTPS: public repo,
cd hiqcache                                             # no key needed
bash scripts/pod_bootstrap.sh /workspace
```

`pod_bootstrap.sh` is idempotent. It records hardware provenance, clones both
repos at the right refs, asserts the pinned base commit is an ancestor, builds an
isolated codec venv, and runs **four gates**:

| Gate | Proves | Expected |
| --- | --- | --- |
| 0 | environment can JIT-compile CUDA (`scripts/env_probe.py`) | `this box can run the HiCache JIT path` |
| 1 | local suite green on the pod | `223 passed` |
| 2 | preflight | `all blocking checks passed` |
| 3 | **codec conformance on CUDA** | `PASS: codec is bit-identical to the reference device` |

**Gate 0 is the fastest to fail and the cheapest to fix.** If it reports
`RuntimeError: Ninja is required to load C++ extensions`, that is
`pip install ninja` and re-run — see Appendix A. Do not replace the pod for it.

**Gate 3 is the important one.** The Mac already proved CPU ≡ MPS over 9 vectors.
Its vectors are derived from **integer arithmetic, with no RNG**, so the same code
produces the same bytes under any torch version and on any device. A shared copy
lives at `results/conformance_vectors.pt` and is verified against a fresh
generation on load, so a stale file cannot silently pin both sides to old inputs.

Each manifest records an `input_digest` per vector: if those differ, the
comparison reports `INPUT MISMATCH` rather than blaming the codec.
If CUDA disagrees, stop — the bug is in the codec or a backend op, not in
HiCache, and it is far cheaper to fix here.

**If gate 3 fails:** the manifest diff names the vector and the digest. Check
`torch.round` behaviour on CUDA first (`tests/test_fork_int8_codec.py` has the
float64 differential test), then bf16 division. Do not proceed to step 4.

---

## Step 2 — Install SGLang

This is the heaviest step and the one most likely to need iteration.

**Two things the bare `pip install -e python` gets wrong**, both verified:

1. SGLang pins `torchaudio==2.11.0` alongside `torch==2.13.0`. No torchaudio 2.13
   build exists, so the pair only resolves through PyTorch's own wheel index —
   which is exactly what SGLang's Dockerfile passes (`--extra-index-url
   https://download.pytorch.org/whl/cu${CUINDEX}`).
2. The base dependencies hard-require **CUDA 13** (`cuda-python>=13.0`,
   `flashinfer_python[cu13]`, `humming-kernels[cu13]`,
   `nvidia-cutlass-dsl[cu13]`, `nvshmem4py-cu13`). `pod_bootstrap.sh` now
   `--dry-run` resolves this and warns if the image cannot satisfy it.

So install with the extra index, and **dry-run first** to confirm torch is not
replaced:

```bash
cd /workspace/sglang
CU=cu1300                     # match the image: cu1300 or cu1290
CUINDEX="${CU#cu}"            # -> 1300 (unused below; kept for clarity)

# 1. Resolve only. Confirm it plans to keep torch 2.13.0+cu130.
SGLANG_BUILD_RUST_EXTS=none python -m pip install --dry-run \
    --extra-index-url https://download.pytorch.org/whl/${CU} \
    -e python | tail -20

# 2. Install. Two flags matter:
#    - SGLANG_BUILD_RUST_EXTS=none: the Rust radix-tree core is OPTIONAL
#      (SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND defaults to "python"), and the
#      grpc/multimodal/server extensions are only imported inside the feature
#      that needs them. Nothing on the HiCache path requires them, and building
#      them needs cargo, which the image does not ship -- setup.py fails at
#      metadata time without it.
#    - omit the [all] extra: it pulls the diffusion stack (opencv, diffusers,
#      moviepy) which none of these tests need.
SGLANG_BUILD_RUST_EXTS=none python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/${CU} \
    -e python

# 3. Confirm torch survived, and nvcc still matches it.
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version | tail -2
```

Verify the import and that the new flag exists:

```bash
python -c "
from sglang.srt.environ import envs
print('INT8 flag default:', envs.SGLANG_EXPERIMENTAL_HICACHE_INT8.get())
"
# expect: INT8 flag default: False
```

Then run the codec tests **inside the SGLang environment**:

```bash
python -m pytest test/registered/unit/mem_cache/test_hicache_int8_codec.py -v
```

Expect 25 passed. These need no GPU, so a failure here is an environment problem,
not a CUDA problem.

---

## Step 3 — The pool tests (the real unknown)

```bash
python -m pytest \
  test/registered/unit/mem_cache/test_hicache_int8_pool_host_unit.py -v
```

This is where six previously-unexecuted things get exercised for the first time:

1. the JIT kernel compiled against `element_size = 1152`
2. `cudaHostRegister` on the encoded arena
3. the D2H all-layer move with a per-layer pointer table
4. the H2D byte copy at `element_dim = 1152` (uint8 on both sides)
5. `index_select` / advanced-index scatter at device scale
6. staging growth without cross-direction reallocation

**Failure triage:**

| Symptom | Cause | Where to look |
| --- | --- | --- |
| `Failed to load JIT HiCache kernel` | template reject for 1152 | `_tiles_across_lanes`; preflight already proved the arithmetic, so suspect the toolkit |
| `Unsupported element_size` warning | same | confirm `preflight.py` still passes on the pod |
| shape/stride or dtype assertion in the kernel | H2D dtype mismatch | `mha_int8.py::load_to_device_per_layer` |
| wrong values but no error | pointer table staleness | `_rebuild_d2h_staging_tables` |
| `cudaHostRegister` error | huge-page or granularity | `registration_granularity_bytes` in `init_kv_buffer` |

If the JIT path cannot be made to work quickly, there is a fallback: set
`SGLANG_HICACHE_TMA_TRANSFER=0` (it is already inactive on sm_86) and confirm the
register kernel is the one being selected. Record whatever you find — a failed
JIT integration is a legitimate result.

---

## Step 4 — Phase 11 smoke test

Prove the path executes end to end and that L2 really holds compressed bytes.

```bash
cd /workspace/hiqcache
python scripts/smoke_test.py --config int8 --host-size 2
```

Six checks run in sequence. The decisive one is **"L2 stores the compressed
representation"**: it divides `sglang:hicache_backup_bytes_total` by
`hicache_backup_tokens_total` and compares against the codec layout.

| Config | Measured B/token must be | Meaning |
| --- | --- | --- |
| `int8` | **82,944** | compression is on the wire |
| `bf16` | 147,456 | baseline sanity |

A number between the two means records are being padded or the wrong pool class
was selected. Run the baseline too, as a control:

```bash
python scripts/smoke_test.py --config bf16 --host-size 2
```

**Passing this is the milestone that matters most.** It means the entire
experimental path works and the remaining work is measurement, not engineering.

---

## Step 5 — Experiment B (the headline result)

Same physical CPU budget, workloads sized to sit *between* the two L2
capacities. At 8 GB: **BF16 holds 54,254 tokens, HiQCache holds 96,451**.

```bash
cd /workspace/hiqcache

# Baseline A: no reusable L2 at all.
python scripts/run_experiment.py --config baseline --tag exp-b-8gb \
    --workload reusable-prefixes

# B: standard BF16 HiCache.
python scripts/run_experiment.py --config bf16 --tag exp-b-8gb \
    --workload reusable-prefixes

# C: HiQCache.
python scripts/run_experiment.py --config int8 --tag exp-b-8gb \
    --workload reusable-prefixes
```

Each run boots its own server, flushes the prefix cache, snapshots `/metrics`
before and after, and writes `results/exp_<config>_<tag>.json`.

Tune the workload with `--num-groups`, `--prompts-per-group`,
`--gsp-question-len` (the shared prefix length) so the aggregate reusable
prefixes exceed ~54k tokens but stay under ~96k. The driver prints the budget so
you can see where you are:

```
=== L2 budget @ 8 GB: bf16 54,254 tokens | int8 96,451 tokens (1.7778x)
```

You are looking for: **baseline and bf16 evicting and recomputing, int8 hitting.**
That is the end-to-end value proposition.

---

## Step 6 — Experiment A (codec cost at equal logical capacity)

Same *number* of cached tokens, so this isolates codec overhead.

```bash
for cfg in bf16 int8; do
  python scripts/run_experiment.py --config $cfg --tag exp-a \
      --workload small --num-groups 16 --prompts-per-group 8
done
```

Compare `hicache_backup_duration_seconds`, `load_back_duration_seconds` and
cache-hit TTFT. Answer: *is the compressed path faster or slower than raw BF16
for the same cache workload?*

---

## Step 7 — Quality (Phase 15)

Bytes are not enough; quantify the loss.

```bash
# Deterministic generation agreement, logprobs, then a small task subset.
python -m sglang.benchmark.serving --backend sglang \
    --dataset-name random --num-prompts 200 \
    --random-input-len 1024 --random-output-len 64 \
    --return-logprob --temperature 0
```

Run the identical command against `bf16` and `int8`, then compare logprob deltas
and token agreement. For a task-level number use a small GSM8K or MMLU subset
through SGLang's own eval harness — do not build an evaluation framework.

---

## Step 8 — Bring the results home

```bash
cd /workspace/hiqcache
tar czf /workspace/hiqcache-results.tgz results/ provenance/
```

Copy that down, then analyse it **on the Mac** — do not burn GPU hours doing
analysis. The JSON files carry hardware provenance, SGLang SHA and the exact
server arguments, so results stay interpretable after the pod is gone.

Then, locally:

```bash
cd hiqcache
python scripts/analyse.py results/       # once written
```

---

## Rules that will save you money

1. **Steps 1–3 before any benchmark.** A benchmark on a broken transfer path
   produces numbers that look plausible and mean nothing.
2. **Never compare across GPU models.** All three configs must come from one pod,
   one session, one SGLang SHA.
3. **Save the server log on every run.** `run_experiment.py` already tails it into
   the JSON; keep `results/exp_*.server.log` too.
4. **Snapshot the pod** after step 4 passes, so a later mistake costs a restart
   rather than a reinstall.
5. **Do not tune for a win.** PROJECT.md Phase 18 lists four acceptable outcomes.
   A capacity win with neutral latency is a strong project; a negative result,
   properly measured, still demonstrates the internals.

---

## What "done" looks like

- [ ] gate 3 passes: CUDA codec is bit-identical to the Mac CPU manifest
- [ ] pool tests pass on GPU
- [ ] smoke test PASS with `int8` reporting **82,944 B/token** in L2
- [ ] baseline `bf16` control reports 147,456 B/token
- [ ] Experiment B: int8 avoids recomputation where bf16 cannot
- [ ] Experiment A: codec overhead quantified at equal logical capacity
- [ ] quality: logprob deltas and generation agreement measured
- [ ] all results JSON + provenance committed to the `hiqcache` repo

---

## Appendix A — Checking a fresh pod before you commit to it

`pod_bootstrap.sh` runs this automatically as **GATE 0/4**, before any pip
install. You can also run it by hand the moment the pod is up:

```bash
cd hiqcache
python scripts/env_probe.py
```

It answers the only question that decides whether the compiled HiCache kernels
can work at all: **can this box JIT-compile and load a CUDA extension?** It
checks the interpreter, torch, GPU visibility, `nvcc`, the CUDA headers
(`cuda_runtime.h`, `cuda_fp16.h`, `cuda_bf16.h`), `ninja`, a C++ compiler, an
actual compile-and-load, and GitHub SSH auth.

### `JIT COMPILE: FAILED -> RuntimeError: Ninja is required to load C++ extensions`

This is the most common first failure and it is **not** a reason to replace the
pod. `load_inline` and `load_jit` shell out to ninja to drive the build:

```bash
pip install ninja
python scripts/env_probe.py        # re-run; should now report the JIT compile OK
```

`ninja` is a declared SGLang dependency (`python/pyproject.toml:52`), so it
would arrive with SGLang anyway — but without it the failure is easy to
misattribute. SGLang does **not** crash when a JIT kernel fails to build:
`can_use_hicache_jit_kernel` logs a warning and returns `False`, and the INT8
host pool then raises a generic "needs the JIT HiCache kernel" error. A missing
build tool would look exactly like a HiCache bug. Establish the answer up front.

### Version compatibility

SGLang at this commit pins `torch==2.13.0`, and its own Dockerfile supports
**CUDA 13.0** only. A pod image shipping an older torch (2.8.x, CUDA 12.8) will
have torch replaced by pip during install, so:

- the pip-installed torch brings its **own** bundled CUDA runtime, and only the
  **driver** has to be new enough — an older system CUDA toolkit is not fatal;
- but `nvcc` from the image is still what compiles the JIT kernels, so the
  image's CUDA 12.8 toolkit *is* what you will be compiling with.

If `pip install -e python` fights the preinstalled torch, or the JIT compile
fails against a mismatched toolkit, **switch to a CUDA 13 image** rather than
debugging the image. It is faster and cheaper than fighting version skew, and
GATE 0 tells you immediately whether the new image is better.

## Appendix B — Getting results back, and surviving a pod restart

**A pod without a mounted volume loses everything when it is terminated or
recreated.** Model weights (~16 GB), the SGLang install and all results
regress to nothing. Two independent mitigations — use both:

1. **Pull results to the Mac as soon as each step passes.** From the Mac:

   ```bash
   # RunPod SSH uses a non-standard port; copy the exact command from the
   # pod's "Connect" panel, then:
   rsync -avz -e "ssh -p <PORT>" \
       root@<POD_HOST>:/workspace/hiqcache/results/ ./pod-results/
   rsync -avz -e "ssh -p <PORT>" \
       root@<POD_HOST>:/workspace/provenance/ ./pod-provenance/
   ```

   Git is the second line of defence: commit the JSON into the `hiqcache` repo
   and push. Then a dead pod costs GPU time, never data. This is the only step
   that needs a GitHub credential on the pod, and it is optional — rsync alone is
   enough.

2. **Attach a network volume** if you expect to iterate across sessions, and set
   `HF_HOME` to it so the Qwen3-8B download happens once:

   ```bash
   export HF_HOME=/workspace/hf
   ```

### Cloning: use HTTPS, not SSH

Both repos are **public**, so a fresh pod needs **no credentials at all**:

```bash
git clone https://github.com/awesome-pro/hiqcache.git
git clone https://github.com/awesome-pro/sglang.git
```

`pod_bootstrap.sh` uses these HTTPS URLs by default and accepts overrides:

```bash
HICACHE_URL=... SGLANG_URL=... bash scripts/pod_bootstrap.sh /workspace
```

**Do not reach for `git@github.com:` URLs.** SSH always requires a private key,
and a fresh pod has none:

```
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.
```

That error says nothing about whether the repo is public or whether it exists —
it only means no key is present. Two ways out, in order of preference:

1. **Use the HTTPS URL** (above). No key, no token, works immediately.
2. If you specifically want SSH (e.g. to push results), add a deploy key in the
   repo settings and mount it on the pod, then verify with
   `ssh -T git@github.com`. Note that a key belongs to a *user or deploy key*,
   not to the pod, so this has to be set up per pod template.

`scripts/env_probe.py` reports whether an SSH key is present, as a **warning
only**. A pod with no key is the normal case and passes GATE 0.

**Keep the pod alive through step 4.** It contains a one-off, already-paid-for
setup: pip environment, JIT kernel cache under `~/.cache/sglang`, and the model
weights. Stopping it to save $0.55/hr and rebuilding later costs far more than
the idle time.
