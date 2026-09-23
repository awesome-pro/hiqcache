# Experiment A — equal logical capacity

**Both tiers pinned to the same 54,254 L2 tokens. The measured difference is the
memory each needs to hold them.**

Experiment B held the *bytes* constant and asked how many tokens fit. Experiment A
holds the *tokens* constant and asks how many bytes that costs. Together they
bracket the same claim from both sides, and A is the stronger half: in B the
hit-rate gain could be attributed to capacity, whereas A removes the capacity
advantage entirely and shows the codec costs nothing in cache effectiveness.

---

## Headline

| | bf16 | **HiQCache INT8** |
| --- | --- | --- |
| L2 capacity (tokens) | 54,254 | **54,254** |
| L2 used (tokens) | 53,710 | 53,514 |
| measured bytes/token in L2 | 147,456 | **82,944** |
| **L2 bytes for that capacity** | **8,000,077,824 (8.000 GB)** | **4,500,043,776 (4.500 GB)** |
| tokens backed up (D2H) | 216,144 | 222,628 |
| tokens restored (H2D) | 122,678 | 126,979 |
| cache hit rate | 97.03% | **96.94%** |
| L2 evictions | 0 | 0 |

**The claim: the same logical cache, in 56.25% of the memory — 1.7778x denser —
with a hit-rate difference of −0.09 points.**

The hit rate is *expected* to match, because capacity is equal by construction.
That is the result, not a disappointment: it says the codec's compression is free
at the level of cache behaviour, so the capacity it buys in Experiment B is pure
gain rather than a trade against hit rate.

Zero evictions on both sides confirms the workload sat inside the pinned capacity
rather than spilling through it, so both configs were measured under the same
(non-evicting) regime.

---

## Provenance

| item | value |
| --- | --- |
| GPU | NVIDIA RTX A6000, 49,140 MiB |
| driver | 595.91.07 |
| CUDA / torch | nvcc 13.0; torch 2.13.0+cu130 |
| SGLang SHA | `38e366694d536564f5c607a9061d495ffd2d1938` |
| hiqcache SHA | `4ddea29` (the harness that ran these two configs) |
| results | `results/exp_bf16_exp-b-exp-a.json`, `results/exp_int8_exp-b-exp-a.json` |
| server logs | `results/exp_{bf16,int8}_exp-b-exp-a.server.log` |
| workload | 32 shared-prefix groups x 2048 = 65,536 reusable prefix tokens |
| L1 cap | 16,384 tokens (`--max-total-tokens`) |
| requests | 128 prompts, concurrency 8 |

Two provenance notes, because this run straddled a history rewrite and a pod
restart:

* **`38e366694d` is a pre-rewrite SHA.** The pod's fork clone predates the
  commit-identity rewrite, so it recorded the old hash. It is the same tree as the
  published `26e6d78dba`; the full mapping is in `docs/experiment-b-results.md`.
* **`results/provenance.txt` describes the earlier suite run, not these two
  configs.** It was captured at the top of `reproduce_all.sh`, which aborted
  before Experiment A. These two configs ran afterwards, against the harness at
  `4ddea29`, after the pod's clone was reconciled onto `main`.

---

## Method, and why the sizing is subtle

Equal *token* capacity needs sub-GB precision: 54,254 BF16 tokens is 8.000078 GB.
That cannot be expressed with `--hicache-size`, which SGLang declares as an
`int` in gigabytes (`arg_groups/fields/memory.py`):

```python
hicache_size: A[int, "The size of host KV cache memory pool in gigabytes. ..."] = 0
```

Passing `8.000078` is rejected by argparse and the server exits with **code 2**
before loading the model — which is exactly how the first attempt failed.

So the host pool is sized with `--hicache-ratio`, a float *token* ratio
(`pool_host/base.py`):

```python
self.size = int(device_capacity * host_to_device_ratio)   # tokens
self.page_num = self.size // self.page_size + 1           # rounded up a page
self.size = self.page_num * self.page_size
```

With `--page-size 1` that lands on `int(device_capacity * ratio) + 1` tokens, so
the harness targets `int(device * ratio) = target - 1`. With the device pool
pinned to 16,384 tokens by `--max-total-tokens`, the ratio is
`54253.5 / 16384 = 3.311370849609` for **both** configs — equal capacity means one
shared token ratio, and INT8 then simply needs fewer bytes to hold it.

The achieved capacity is not trusted to that arithmetic. The runner reads the
pool's own gauge after startup and refuses to continue if it disagrees:

```
=== L2 capacity verified against the pool's gauge: 54,254 tokens
```

Both configs reported `target_l2_tokens = achieved_l2_tokens = 54,254`
(`results/exp_*_exp-b-exp-a.json`).

---

## What this does and does not show

**Shows.** At an identical, verified logical capacity, the INT8 L2 holds the same
working set in 56.25% of the bytes, with the same hit rate and no evictions. The
measured 82,944 bytes/token is the codec's own figure, not a model of it.

**Does not show.** Quality. Nothing here says the restored KV is *correct* — that
is what the Phase 15 comparison is for, and it is not yet valid (the first
attempt exercised no L2 restore at all and captured no tokens; see the quality
notes). Nor does it show latency: one run per config, and the backup/restore
duration histograms still read 0.00 s, so no per-phase timing is claimed.

## Limitations

* **One run per config.** The hit-rate delta of −0.09 points is well inside what a
  repeat would move; treat the two hit rates as equal rather than ranked.
* **Equal hit rate is by construction.** This experiment cannot detect a codec
  accuracy problem — that is deliberately out of scope.
* **A single workload shape** (one prefix length, one concurrency), as in
  Experiment B.
* **The capacity figure assumes `--max-total-tokens` pins the device pool to
  exactly that token count**; the gauge check confirms it for this build.
