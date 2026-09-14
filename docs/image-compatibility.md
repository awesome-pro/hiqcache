# Image and version compatibility

Why the pod image matters, and the exact matrix, so this is not rediscovered by
trial and error on a rented GPU.

Verified against `https://download.pytorch.org/whl/<cuda>/torch/` and the
`runpod/pytorch` tag list.

---

## The constraint

SGLang at the pinned commit requires **`torch==2.13.0`** (`python/pyproject.toml:6`).
PyTorch publishes no CUDA 12.8 build of that version:

| torch | cu128 | cu129 | cu130 |
| --- | --- | --- | --- |
| 2.10.0 | ✓ | ✓ | ✓ |
| 2.11.0 | ✓ | ✓ | ✓ |
| 2.12.x | – | ✓ | ✓ |
| **2.13.0** | **–** | **✓** | **✓** |
| 2.14.0 | – | – | ✓ |

So on a CUDA 12.8 image, installing SGLang either:

- replaces the image's torch 2.8.0 with 2.13.0 built for a **different** CUDA
  (pip's default), leaving the image's CUDA 12.8 toolkit to compile JIT kernels
  against a torch built for 12.9/13.0; or
- fails to satisfy the pin.

**There is no torch 2.13.0 + CUDA 12.8 combination.** No amount of `pip install`
resolves it.

---

## Why "just upgrade CUDA on the pod" does not help

Two separate things are usually conflated:

1. **The CUDA runtime torch links against.** This is *bundled inside the wheel*
   (`torch/lib/libcudart.so`). It is not provided by the image.
2. **The CUDA toolkit (`nvcc`, headers).** Provided by the image. This is what
   compiles SGLang's JIT kernels.

The GPU **driver** (580.x here) is backwards-compatible and already supports CUDA
13.x, so only the driver has to be new enough — that is not the problem.

Installing a CUDA 13 toolkit *inside the running container* gives you cu13
headers with a cu128 torch, which is a **worse** mismatch than the one you were
trying to fix. A consistent set has to come from the image.

---

## Use a matching image

`runpod/pytorch` publishes a torch 2.13.0 + CUDA 12.9 image, which is an exact
match for SGLang's pin:

```text
runpod/pytorch:1.3.3-rc.169-cu1290-torch2130-ubuntu2404
```

| Tag | CUDA | torch |
| --- | --- | --- |
| `1.3.3-rc.169-cu1290-torch2130-ubuntu2404` | 12.9 | **2.13.0** |
| `1.3.3-rc.169-cu1300-torch2130-ubuntu2404` | 13.0 | 2.13.0 |
| `1.4.0-rc.164-cu1290-torch2130-ubuntu2404` | 12.9 | 2.13.0 |
| `1.4.0-rc.164-cu1300-torch2130-ubuntu2404` | 13.0 | 2.13.0 |

`torch==2.13.0` is published for **cu129 and cu130 only** — there is no cu128 and
no cu131 build. Both images above are therefore valid; the pin is what matters.

### Why cu1290 and not cu1300

CUDA 12.9 is not a downgrade — it is the *other* valid answer, and it is the
conservative one:

- **It satisfies the pin exactly.** The image's preinstalled torch is already
  `2.13.0+cu129`, so `pip install -e python` has nothing to replace. No swap, no
  risk of the bundled runtime diverging from the image's `nvcc`.
- **It needs a less recent driver.** CUDA 12.9 runs on older drivers than 13.0,
  which matters on rented hardware where the driver is not yours to change.
- **The feature difference is irrelevant here.** Nothing HiQCache touches —
  `cudaHostRegister`, `cudaMemcpy`, element-wise kernels — differs between 12.9
  and 13.0. SGLang's JIT kernels compile from the same source either way.

What actually matters is that **two pairs stay consistent**, and neither is the
container's headline CUDA version:

| Pair | Why |
| --- | --- |
| `torch.version.cuda` ↔ image `nvcc` | `nvcc` compiles the JIT kernels against torch's headers |
| torch's bundled runtime ↔ **driver** | the driver must be new enough for the runtime torch ships |

`pod_bootstrap.sh` now prints and checks the first pair explicitly, so a silent
torch replacement during the SGLang install cannot go unnoticed.

### What a healthy pod looks like

```text
SGLang pins torch : 2.13.0
container has     : 2.13.0+cu129 (cuda 12.9)
nvcc release      : 12.9
  -> container torch already satisfies the pin; pip will not replace it.
```

`scripts/env_probe.py` reports the same pair as
`[PASS] torch CUDA version matches the image toolkit`.

Everything else about the pod — A6000 48 GB, 62 GB RAM, 150 GB disk — was
correct and does not need to change.

---

## If you must keep a mismatched image

The pod still works for everything that does not need a compiled kernel. Two
independent layers:

| Layer | Needs nvcc? | Works on a mismatched image? |
| --- | --- | --- |
| `scripts/conformance.py`, codec, staging, capacity | no | **yes** — pure torch ops |
| `test_hicache_int8_codec.py` | no | **yes** |
| `test_hicache_int8_pool_host_unit.py` and any real serving | **yes** | no |

So a mismatched pod can still clear GATE 0–3 and prove the codec is
CUDA-bit-identical. That is genuinely useful evidence and costs a few minutes.
What it cannot do is exercise the HiCache transfer kernels, which is the entire
point of the pod run.

**Decision rule:** if `pip install -e python` completes and GATE 0 passes, try
the pool tests. If the JIT compile fails with header errors (not the ninja
error), stop — switch images rather than fighting it.
