# Exact CPU multiplication experiments — 2026-10-04

None of the new multiplication implementations beat the existing Case33 CPU
backend on the tested Pearl panels. Keep them as a standalone experiment, with
no change to the miner or release defaults. This is a CPU result, not a CUDA
Tensor Core comparison.

The subsequent [CMP50 CUDA multiplication experiment](cuda_matmul_experiment.md)
tests exact wider-operand Tensor Core leaves and all 32 GPU milestones. None
of its tested alternatives beats the existing fused CUDA path either.

## Scope and reproducibility

- AMD Ryzen 7 8745HS, Windows 10; GCC 16.2.0 from MSYS2 UCRT64.
- One OpenMP thread, process affinity on logical CPU 2. Runtime detection
  confirmed AVX2 and AVX512-VNNI. Normal laptop boost and background activity
  remain enabled; clocks and power are not fixed.
- Unmodified production `Case33GemmXor` sources are linked into the benchmark.
  `Auto` resolves to AVX512-VNNI. Forced AVX2 and scalar are controls.
- Nine samples per method, after warm-up and adaptive batching to at least
  approximately 60 ms. Method order rotates by round and alternates direction.
  Raw sample durations, batch counts, checksums, compiler commands and build
  fingerprints are saved in [the timing data](benchmarks/cpu-matmul-2026-10-04.json).
- [Final validation](benchmarks/cpu-matmul-validation-2026-10-04.json) was repeated
  after formatting cleanup. The benchmark object SHA-256 remained identical:
  `d0a9a2c9ecc4aac04006a0f2dc8d557734f54c0157b45d7539f945484a368220`.
- No GitHub Actions, pool connection, GPU rental or test-server deployment is
  required. The executable has its own entry point and never calls networking
  entry points. Real pool/cancellation dependencies are linked without fake
  replacements.

Run from the repository root, with Python 3, x86-64 GCC/OpenMP and the existing
vendored BLAKE3 C sources:

```sh
python scripts/bench_cpu_matmul.py --repeats 9 --sample-ms 60 \
  --output build/cpu-matmul.json
python scripts/bench_cpu_matmul.py --verify-only \
  --output build/cpu-matmul-validation.json
```

Use `--cxx` for a specific `g++` path; the runner can locate MSYS2 UCRT64 on
Windows. `--quick` uses two final-GEMM shapes and one full-range Pearl panel.
The build lives under ignored `build/cpu-matmul-experiment`. It does not modify
CMake or the production build. GCC/Clang target intrinsics are used; this harness
does not support MSVC or ARM. Windows affinity is reported when successfully
applied; the runner does not implement Linux affinity.

## Implementations

The standalone kernels use transposed B storage for contiguous dot products.
Root inputs are signed INT8. Intermediate operands use INT16; intermediate
products/recombination use explicitly wrapping UINT32 arithmetic, preserving
the exact result modulo 2^32 and its XOR representation. An independent INT64
reference checks the mathematical result. No floating-point approximation,
rounding or saturating dot instruction is used in the experimental kernels.

| Method | Implementation |
| --- | --- |
| Classical compiler | Ordinary dot loops, optimized by GCC; not forced scalar assembly |
| Blocked AVX2 INT16 | 16x16 output blocking, signed word dot products with `VPMADDWD` |
| Classical VNNI INT8 | Unsigned A offset +128, signed B, non-saturating `VPDPBUSD`, compensation |
| Pairwise Winograd | One multiplication per adjacent pair using `(a0+b1)(a1+b0)`, minus row/column factors; odd K handled separately |
| Strassen | Seven recursive products instead of eight, depths 1, 2 and 3; AVX2 INT16 leaves |
| Strassen–Winograd | Seven recursive products with fewer block additions, depths 1, 2 and 3; AVX2 INT16 leaves |
| Adaptive depth 1 | Strassen / Strassen–Winograd with INT8 VNNI leaves only when every operand fits; otherwise INT16 |

The Strassen–Winograd formulas follow section 2 of
[Boyer, Dumas, Pernet and Zhou](https://arxiv.org/pdf/0707.2347).
The pairwise identity is a different algorithm from recursive
Strassen–Winograd.

Scratch buffers are allocated before timing. Operand sums, range checks,
adaptive packing, leaf products, recombination, pair factors and required XOR
work are inside timing. Root INT8-to-INT16 conversion is outside timing. Case33
A/B packing is outside its compute-only row; a second row measures per-attempt A
packing plus the scan. B preparation is outside both Case33 rows, matching
reuse of B across attempts. Noise generation, nonce generation, jackpot
BLAKE3, proof generation and submission are outside this multiplication test.

The prototypes recompute B transforms/factors during multiplication and have
simple dot-product leaves. They do not integrate transforms into a production
register-tile kernel or persist transformed B across attempts. Consequently,
these numbers describe these implementations, rather than the best possible
implementation of each mathematical algorithm. Practical fast multiplication
also depends on packing and memory traffic; see
[Huang, Rice, Matthews and van de Geijn](https://arxiv.org/abs/1611.01120).

## Correctness and the Pearl workload

The unit suite covers rectangular and odd dimensions, K tails, zeros,
alternating extrema, +127/-128, all -128, random signed 7-bit and full INT8
values. Depths 1–3 are checked against INT64, including paths whose intermediate
operands exceed INT8.

The Pearl suite uses K=4096 and **all 32 cumulative products**, spaced every
128 terms. Every prototype computes a delta for each 128-term slice, adds it
to the accumulated matrix and derives the contiguous 8x16 tile XOR. Every C
element at every milestone is checked against INT64, both with and without
diagnostic statistics enabled. Every production tile XOR is checked against
the same reference. Timed output is checked after every sample and consumed
through a checksum; an optimizer barrier separates repeated calls.

The complete timing run passed 5,778 comparisons covering 204,533,520 UINT32
values. All **benchmarked** outputs matched. Two additional full-range AVX2
failures are explicitly recorded and excluded from speed measurements, as
described below. The final verification-only run passed 4,338 comparisons
covering 141,299,472 values, with the same two range failures.

Inputs include synthetic signed 7-bit/full INT8 data and generated zero-B data.
The zero-B fixtures call the miner's actual `pearl_generate_random_a`,
`pearl_build_noisy_a` and `pearl_build_noisy_b` with deterministic synthetic
seeds and rank 128. They do not contain pool credentials, wallet data or a real
mining job.

M=N=128 and 256 are **panels**, not a complete production mining attempt. In
particular, the prototype materializes panel C, while Case33 keeps tile
accumulators in its microkernel and emits only milestone XORs. That difference
is part of the measured cost. No throughput or share-rate estimate for the
complete miner is inferred from this experiment.

## Results on generated zero-B data

Median milliseconds, including all 32 prefixes and their XORs; lower is faster.

| Method | 128x128x4096 | 256x256x4096 |
| --- | ---: | ---: |
| Current Case33 Auto / AVX512-VNNI | **0.5163** | **2.0559** |
| Current Auto, A pack + scan | 0.5459 | 2.1304 |
| Current Case33 AVX2 | 0.5104 | 2.0439 |
| Current Case33 scalar | 21.5144 | 86.5011 |
| Classical compiler | 2.9155 | 11.7379 |
| Blocked AVX2 INT16 | 2.4792 | 10.2859 |
| Classical VNNI INT8 prototype | 1.9844 | 7.3756 |
| Pairwise Winograd | 3.7910 | 20.5409 |
| Strassen depth 1 | 3.8437 | 14.0320 |
| Strassen–Winograd depth 1 | 3.8514 | 14.7962 |
| Strassen depth 2 | 7.2205 | 24.1704 |
| Strassen–Winograd depth 2 | 6.6171 | 24.1766 |
| Strassen depth 3 | 12.9773 | 43.3066 |
| Strassen–Winograd depth 3 | 11.8387 | 41.0772 |
| Adaptive Strassen depth 1 | 4.0059 | 14.4219 |
| Adaptive Strassen–Winograd depth 1 | 3.9264 | 14.1792 |

The small Auto/AVX2 timing difference does not establish an ISA-dispatch
improvement. Both remain far faster than every prototype.

On the 128 panel, one recursion level saves 12.5% of leaf MACs
(58,720,256 vs 67,108,864); three save about 33.0% (44,957,696).
Nevertheless, additions, intermediate storage and narrower leaf kernels cost
more than the arithmetic saved. Pairwise Winograd needs about half the scalar
multiplications, including its factors, but introduces additions and SIMD
shuffles; it is also slower.

INT8 eligibility is a real restriction. For generated zero-B data, depth-1
Strassen reaches magnitude 232 and Strassen–Winograd 379 on the 128 panel.
The adaptive versions use INT8 for only 64 of 224 leaf products; 160 require
INT16. At depth 3, Strassen–Winograd reaches magnitude 1,401. These are observed
maxima for the fixture, not general bounds. The JSON's `leaf_macs` is an
algorithmic count, not a hardware instruction count; `workspace_bytes` covers
prototype scratch only. Zero-valued production scratch/leaf-range statistics
mean uninstrumented, not zero production memory use.

The separate final-GEMM suite tests 128^3, 256^3, 512^3,
128x128x4096 and 256x256x4096. Some fast algorithms beat the generic compiler
loops on larger final products, but none beat the faster classical SIMD
prototype in the same shape. For example, at 512^3, Strassen depth 1 takes
3.0248 ms vs 3.7585 ms for generic loops, 2.4025 ms for blocked AVX2 and
1.8341 ms for the simple VNNI prototype. Final-product timings cannot substitute
for the required Pearl prefixes.

Sample spread is saved rather than hidden. Some prototype batches vary by up
to approximately 36% between fastest and slowest samples. The several-fold
gap to Case33 is much larger, but close rankings among prototypes should not be
treated as stable universal results.

## Existing AVX2 full-range limitation

For synthetic full-range signed INT8 roots, forced Case33 AVX2 fails all 4,096
milestone tile XORs on the 128 panel and all 16,384 on the 256 panel. The first
128-panel mismatch is 4,294,764,937 vs the expected 4,294,828,507.
AVX512-VNNI and scalar pass the same inputs.

The AVX2 fast path uses `VPMADDUBSW`, which saturates paired products into signed
INT16 before the INT32 sum. For example, unsigned A values 255/255 and signed B
127/127 produce 64,770 before compensation, beyond 32,767. Offset compensation
cannot recover information already lost to saturation. Therefore the algebraic
full-range claim in the Case32 mode comment is insufficient for this AVX2
instruction sequence.

This does **not** demonstrate a failure in the tested zero-B mining path.
Its B noise is the difference of two 6-bit values, hence lies in [-63,63], where
the pair sum fits even for unsigned A up to 255. Forced AVX2 passes both
generated zero-B panels and signed 7-bit inputs. No production correctness fix
is included here. Before reusing this fast path for wider B inputs, add an
exact wider-input path or an enforced range guard with an appropriate fallback
and regression tests. The negative stress records remain in the report so this
finding cannot be mistaken for a successful full-range parity test.

## Decision

Keep the current production multiplication path. Preserve this harness for
future ideas and comparisons. A further fast-multiplication prototype would
need B-transform reuse, packing integrated with the register-tile kernel, and
efficient handling of wider intermediates, while still producing every Pearl
milestone. Validate and measure such a concrete implementation before making
any CPU or GPU speed claim.
