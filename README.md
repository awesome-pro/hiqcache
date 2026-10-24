# HiQCache

**Quantized hierarchical KV caching for SGLang HiCache.**

HiQCache keeps active KV in BF16 on the GPU, but stores evicted KV in an INT8
representation inside SGLang's CPU HiCache tier. On an L2 hit, the cached KV is
restored to BF16 before attention consumes it.

This changes the representation stored in L2 without changing SGLang's caching
policy, radix tree, transfer engine, or existing CUDA HiCache mover.

## Key Results

| Finding | Evidence |
| --- | --- |
| **43.75% less CPU L2 memory per token** | 147,456 bytes/token (BF16) → **82,944 bytes/token** (INT8 + scales + padding) |
| **1.78× more L2 tokens under an 8 GB host budget** | 54,254 BF16 → **96,451 INT8** tokens |
| **72.72% ideal prefix-reuse ceiling for the benchmark trace** | Exact tokenized 128-request trace, sequential/unlimited-capacity trie oracle: 206,670 / 284,198 prompt tokens |
| **57.57% → 71.97% cache-hit rate in an independent replay** | BF16 vs INT8, same trace, 8 concurrent requests; one run per configuration |
| **Valid end-to-end L2 restore spot-check** | Both variants restored 1,797 tokens; 16 of 16 generated token IDs and the text matched |
| **16.1 ms to restore a 2,152-token prefix** | 7.6 ms PCIe + 8.5 ms decode; 1.2% of the 1,352 ms recompute |

The main result is storage density: at a fixed host budget, the L2 holds 1.78x more tokens, which lets more repeated prefixes survive. Serving latency varies by run; the new replay has one run per configuration.

The trace has 4 requests per shared prefix. The 32 first prompts are compulsory misses, and about 3% of each prompt is a unique question tail, so the ceiling is (4-1)/4 x 97.0% = 72.72%. BF16's L2 holds about 25 of 32 prefixes, so it serves 76 of 96 reuse events. INT8 serves 95.

- [SGLang integration PR](https://github.com/sgl-project/sglang/pull/41365)
- [Validation summary](results/validation-2026-10/summary.md), [Theory and oracle](docs/theory.md), [Overhead and quality](docs/overhead-and-quality.md)
- [Raw artifacts and environment manifest](results/validation-2026-10/manifest.json)

---

## Cache capacity and experimental results

### Geometry and encoded format

For Qwen3-8B at TP=1, the tested geometry is **36 layers × 8 KV heads × 128 dimensions**, with K and V, BF16 active KV on the GPU.

One BF16 K or V row requires `8 × 128 × 2 = 2,048` bytes. HiQCache encodes each corresponding row as:

```
0 … 1023       1,024 B  signed INT8 payload (8 heads × 128 values)
1024 … 1039       16 B  eight BF16 scales (one scale per KV head)
1040 … 1151      112 B  alignment padding
──────────────────────────────────────────────────────────────────
                  1,152 B per K or V row
```

Per token, across all layers and K/V:

```
BF16:    2 × 36 × 2,048 = 147,456 bytes
INT8 L2: 2 × 36 × 1,152 =  82,944 bytes
Saving:  1 − 82,944 / 147,456 = 43.75%
Capacity multiplier: 147,456 / 82,944 = 1.7778×
```

Pure BF16 → INT8 data would suggest 50% savings; the actual **43.75%** accounts for scales and padding. The 1,152-byte row is a multiple of 128 bytes to satisfy the existing HiCache byte-mover geometry. See [the design notes](docs/design.md).

### Experiment A: equal logical L2 capacity

Both variants hold **54,254 L2 tokens**. BF16 requires approximately **8.00 GB** of host storage, while the encoded INT8 representation needs approximately **4.50 GB**. This measures the memory needed for the same token capacity. It is not a latency experiment. The observed hit rates in one run differed (54.6% and 58.2%), and that difference is not an established codec effect.

[Experimental detail](docs/experiment-a-results.md)

### Experiment B: equal host-memory budget

With `--hicache-size 8` (8 decimal GB), the original serving benchmark compared no reusable CPU L2, BF16 HiCache, and HiQCache INT8:

| Metric | No CPU L2 | BF16 L2 | INT8 L2 |
| --- | ---: | ---: | ---: |
| Host L2 capacity | 0 | 54,254 | **96,451** |
| Cache-hit rate, benchmark | 12.2% | 56.1% | **72.7%** |
| Tokens restored from host L2 | 0 | 126,997 | **167,899** |
| Mean TTFT | 1,643 ms | 1,244 ms | 922 ms |
| Mean TPOT | 47.9 ms | 37.0 ms | 33.3 ms |
| Output throughput | 80.0 tok/s | 105.1 tok/s | 127.8 tok/s |

This is a **representative run, not a universal performance guarantee**. Across four earlier BF16/INT8 paired configurations, the storage/capacity effect was deterministic and the cache-hit effect repeated; TTFT improvements were less stable, with one near-parity/slightly worse run. The no-L2 baseline still has L1; its 12.2% is device-side reuse, not a cold cache.

[Original results and repetition details](docs/experiment-b-results.md)

### Independent replay (October 2026)

The same trace was replayed with exact token ids at concurrency 8, one run per configuration:

| Metric | Baseline | BF16 | INT8 |
| --- | ---: | ---: | ---: |
| Hit rate | 12.05% | 57.57% | 71.97% |
| TTFT mean | 2,046 ms | 1,229 ms | 990 ms |
| TPOT mean | 60.1 ms | 42.7 ms | 39.9 ms |
| ITL p99 | 1,043.9 ms | 444.4 ms | 118.8 ms |

On the 20 requests INT8 serves from L2 but BF16 cannot, TTFT is 667 ms lower (95% CI -862 to -477). Requests that both hit are 262 ms lower, from less queueing. Requests that both miss differ by 56 ms (CI -235 to +341). The replay pod was slower than the published runs; compare within this table.

---

## Data flow and quantization

For each `(layer, token, K-or-V, KV head)`, HiQCache uses symmetric absmax quantization, with a BF16 scale for each 128-value head:

```
s = max(abs(x)) / 127
q = clamp(round(float32(x) / float32(s)), -127, 127)
x_restored = BF16(q * s)
```

Special handling avoids division by zero for an all-zero head. Quantization happens when GPU KV is stored in CPU L2 and is reversed before attention consumes restored KV.

**Backup (GPU → CPU L2)**

```
GPU BF16 KV → gather/encode into aligned GPU staging
             → existing HiCache mover → compressed pinned CPU L2
```

**Restore (CPU L2 → GPU)**

```
pinned CPU L2 → existing HiCache mover → GPU encoded staging
              → unpack/dequantize/scatter → normal GPU BF16 KV → attention
```

This is a storage-tier optimization, not a change to attention math, cache eviction policy, or the scheduler.

The codec has a verified *elementwise* representation-error bound described in [the design document](docs/design.md). An elementwise bound does **not** prove model-output equivalence.

---

## Codec numerical correctness

The encoded format has a verified elementwise error bound:

```text
|x̂ − x| ≤ (0.5 + 2⁻⁸) · s + 2⁻⁸ · |x̂|
```

The three terms are the INT8 rounding, the BF16 scale, and the BF16 output
rounding. `codec.verified_error_bound()` checks the implementation elementwise
against an exact float64 reference quantizer.

All tested elements satisfied the bound on:

- CPU
- CUDA
- Apple MPS

For a Qwen3-8B-shaped 512-token batch, the largest observed normalized error was
0.934.

---

## GPU overhead: preliminary standalone measurements

The new validation includes CUDA-event timings on the RTX A6000 for PyTorch encode/decode operations and **standalone bulk `copy_` transfers to/from pinned host buffers**. These are **not timings of the complete SGLang L2 scheduler/mover critical path**. The saved run's BF16 control copied the INT8 staging buffer and is marked invalid in `microbench.json`. The code now allocates a BF16 buffer, but the control has not been rerun.

For an approximately 2,152-token prefix, the standalone INT8 medians (3 warmups, 10 measured repetitions) are: encode 15.11 ms, H2D 7.64 ms, decode and scatter 8.46 ms, and 16.10 ms for H2D plus decode. Backup (encode plus D2H) is 22.8 ms.

The compressed copy moved 178.5 MB at an effective bulk-copy rate near 23.4 GB/s in that standalone test. **16.10 ms is a phase-sum estimate, not an instrumented end-to-end SGLang restore latency.** Recomputing the same prefix takes 1,352 ms, so the restore is 1.2% of that. The saved BF16 baseline is invalid; a rerun is needed.

The measurements show that **encode/decode costs are substantial relative to compressed-transfer time**. Whether a fused kernel pays off depends on the actual serving critical path, overlap, and workload; the current measurements do not justify claiming that codec optimization is unnecessary.

[Overhead experiment, assumptions, and limitations](docs/overhead-and-quality.md), [Raw microbenchmark](results/validation-2026-10/raw/microbench.json)

## Correctness and output-quality spot-check

The codec's numerical tests verify elementwise error bounds on CPU, CUDA, and MPS. The new validation also repaired a quality-test issue: asking SGLang for full prompt logprobs (`logprob_start_len=0`) caused the test to bypass the intended prefix-cache restore. Output-logprob-only capture uses `logprob_start_len=-1` for the relevant tested runtime.

A **forced-L2-restore smoke test** exercised one 5,895-token prompt with controlled filler traffic. Both BF16 and INT8 variants restored **1,797 tokens** from host memory, with 5,894 cached prompt tokens for that request. In this one test:

- All **16 generated greedy token IDs** matched between the BF16-restored and INT8-restored responses, and the generated text matched.
- The maximum absolute *chosen-output-token* logprob difference between restored variants was approximately **0.01117**.
- A cold-versus-BF16-restored comparison showed a maximum difference of 0.01989; those differences should **not** be treated as a statistically established noise floor or as proof that lossy quantization is harmless.

This shows the compressed path was exercised in a **limited generation spot-check**. It does not show that accuracy is preserved across diverse tasks, sampling modes, models, or long outputs. The wider multi-prompt quality experiment did **not** complete successfully and remains open.

[Quality methodology and raw evidence](docs/overhead-and-quality.md)

---

## Known limitations and next work

- **Fixed layout:** The current record targets Qwen3-8B's 8 local KV heads, 128-dimensional heads, TP=1, page size 1, and BF16 GPU KV. Other TP configurations, MLA, hybrid architectures, and additional backends need separate engineering and validation.
- **Quality scope:** One forced-restore prompt with 16 greedy output tokens is insufficient for downstream quality or high-confidence numerical impact claims.
- **Performance scope:** Serving-latency measurements are sensitive to workload and environment. The new replay has one run per configuration; its p95/p99 TTFT results did not improve in that single run.
- **Transfer-method scope:** The microbenchmark uses standalone pinned-memory copies, not the full in-server HiCache mover critical path. Its saved BF16 control is invalid; a rerun is needed.
- **Workload scope:** The only *GPU-measured* workloads are concentrated around one repeated-prefix shape. A working-set, concurrency, context-length, and additional-model sweep would characterize the cross-over region and generality.
- **Missing complete overhead attribution:** Collect in-server encode/transfer/decode critical-path traces and validate overlap before introducing a fused Triton implementation.

The next steps are to rerun the BF16 control with the fixed code, profile actual in-server L2 events, run multiple independent repetitions, and expand the forced-restore generation-quality evaluation. A Triton optimization or model port should wait for those measurements.

## Reproducing

Start with [the pod setup guide](docs/reproducing-on-a-pod.md). The original fixed-budget and equal-capacity results and their raw artifacts are unchanged. October validation data resides under [`results/validation-2026-10/`](results/validation-2026-10/).

For offline checks:

```bash
python -m pytest tests/ -q
python scripts/capacity_table.py --hicache-size 8
python scripts/oracle.py \
  --trace results/validation-2026-10/trace/trace-seed42.jsonl \
  --out results/validation-2026-10/oracle.json \
  --md results/validation-2026-10/oracle.md
```

For the GPU experiments, use the pinned SGLang fork revision and a compatible CUDA/PyTorch environment, then run the [replay](scripts/replay_trace.py), [microbenchmark](scripts/microbench_codec.py), and [forced-restore smoke test](scripts/smoke_test.py). Preserve the exact software revision, hardware, workload, and raw logs for new measurements. The October validation used a newer SGLang/PyTorch environment than the original published results; see the [manifest](results/validation-2026-10/manifest.json) for details.

The INT8 host representation is experimental and disabled by default:

```bash
export SGLANG_EXPERIMENTAL_HICACHE_INT8=1
```

For codec-format details and integration constraints, see [design](docs/design.md), [SGLang integration](docs/sglang-integration.md), and [HiCache internals](docs/hicache-internals.md).

## License

Apache-2.0. See [LICENSE](LICENSE).