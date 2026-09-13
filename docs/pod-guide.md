# Pod guide — step by step

Everything to do on a rented GPU, in order. Each step says what it proves, what
you should see, and what to do if it fails.

Target: **1× A6000 48 GB**, 64+ GB RAM, 150+ GB disk.

**The image matters more than the hardware.** SGLang pins `torch==2.13.0`, which
has no CUDA 12.8 build, so a mismatched image wastes a run. Use
`runpod/pytorch:1.3.3-rc.169-cu1290-torch2130-ubuntu2404` — see step 1 and
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
runpod/pytorch:1.3.3-rc.169-cu1290-torch2130-ubuntu2404   <- preferred
runpod/pytorch:1.3.3-rc.169-cu1300-torch2130-ubuntu2404
```

Prefer **cu1290**: it needs a less recent driver than cu1300 and matches what
SGLang can actually be built against. A CUDA 12.8 image will install a torch
built for a different CUDA version, and no `pip install` fixes that — the CUDA
runtime is bundled inside the torch wheel, so only the *driver* needs to be new
enough; the *toolkit* (which compiles the JIT kernels) has to match.

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
If CUDA disagrees, stop — the bug is in the codec or a backend op, not in
HiCache, and it is far cheaper to fix here.

**If gate 3 fails:** the manifest diff names the vector and the digest. Check
`torch.round` behaviour on CUDA first (`tests/test_fork_int8_codec.py` has the
float64 differential test), then bf16 division. Do not proceed to step 4.

---

## Step 2 — Install SGLang

This is the heaviest step and the one most likely to need iteration.

```bash
cd /workspace/sglang
pip install -e "python"          # or the CUDA-version-specific command from
                                 # SGLang's own README
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
4. the H2D `element_dim = 576` bf16 reinterpretation
5. `index_select` / advanced-index scatter at device scale
6. staging growth without cross-direction reallocation

**Failure triage:**

| Symptom | Cause | Where to look |
| --- | --- | --- |
| `Failed to load JIT HiCache kernel` | template reject for 1152 | `_tiles_across_lanes`; preflight already proved the arithmetic, so suspect the toolkit |
| `Unsupported element_size` warning | same | confirm `preflight.py` still passes on the pod |
| shape/stride assertion in the kernel | the `576` bf16 view | `mha_int8.py::load_to_device_per_layer` |
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
