# Exact CUDA multiplication experiments — 2026-10-04

None of the tested alternatives beats the existing fused CUDA Tensor Core path
on the CMP 50HX. On the widest tested Pearl panel, the current 256x128 kernel
takes **35.688 ms**, exact Strassen depth 1 takes **1639.227 ms**, and
Strassen–Winograd depth 1 takes **1780.418 ms**. Keep the current production
path. These prototypes remain a standalone experiment and do not change the
published v0.5-fork.4 miner.

This follows the [CPU multiplication experiment](cpu_matmul_experiment.md).
The GPU experiment uses actual CUDA kernels and the production CUTLASS wrapper,
including every required Pearl milestone and final jackpot hashing.

## Hardware and measurement scope

- NVIDIA CMP 50HX, compute capability 7.5, 20 GiB VRAM; CUDA 12.6.85,
  driver 610.43.03, unchanged 225 W power limit.
- Generated zero-signal-B fixtures use the actual `pearl_generate_random_a`,
  `pearl_build_noisy_a` and `pearl_build_noisy_b` functions, fixed synthetic seeds,
  K=4096 and noise rank 128. The fixtures contain no real mining job or wallet.
- Shapes are 1024x8192x4096, 4096x8192x4096 and 4096x131072x4096. They are
  panels, rather than a complete 131072x131072 mining attempt. The last shape
  covers the normal full N width but only part of M.
- Five warmup invocations, a pilot, then nine CUDA-event samples per method.
  Faster methods are batched to approximately 20 ms per sample, capped at 128
  invocations. CPU wall times and all raw samples are also saved.
- Each timed invocation performs all 32 cumulative 128-term products, scattered
  8x8 tile XORs, the 16-word rotate/XOR state, final keyed BLAKE3 and a target
  comparison. An all-zero target prevents an early successful-share exit.
- Prototype operand additions, limb packing, pair factors, GEMMs, recombination,
  cumulative C updates, global-state initialization, XOR and BLAKE3 are timed.
  Root noise generation, uploads, INT8-to-INT16 conversion, scratch allocation
  and diagnostic downloads are outside timing for all prototype measurements.
  Proof construction, pool traffic and complete miner preparation are excluded.
- Normal boost remains enabled. Methods are measured in sequential groups,
  without randomized or interleaved ordering. One-second GPU telemetry is saved;
  clocks, temperature and power are not fixed. Close rankings are inconclusive.
  The approximately 46–50x gaps are much larger than sample spread.

The original miner was stopped only during bounded experiments and restored
after each run. An existing idle ComfyUI process was left alone. No RTX 3090
was rented, no GitHub Actions workflow was dispatched, and no experimental
mining binary was installed.

Raw measurements, build commands, source/binary hashes, GPU telemetry,
individual run outcomes and restoration checks are in
[the CUDA benchmark data](benchmarks/cuda-matmul-2026-10-04.json).

## Implementations and exact arithmetic

| Method | Implementation |
| --- | --- |
| Current Tensor Core 256 / 128 | Unmodified production fused milestone GEMM; 256x128 or 128x128 threadblock |
| Current DP4A | Unmodified production fused SIMT/DP4A control |
| Classical limb Tensor Core | Standalone CUTLASS INT8 GEMM for each 128-term slice, followed by cumulative C and proof-state kernels |
| Classical INT16 SIMT | Shared-memory 16x16 output tiles, ordinary exact integer arithmetic |
| Pairwise Winograd | SIMT `(a0+b1)*(a1+b0)` with row/column factor subtraction |
| Strassen | Seven recursive block products, depths 1, 2 and 3; exact INT8-limb Tensor Core leaves |
| Strassen–Winograd | Seven block products with fewer block additions, depths 1, 2 and 3; the same exact leaves |

Recursive operands widen beyond INT8. This harness represents an INT16 operand
as `x = low + 256*high`, with both limbs signed INT8. A wide operand requires
two Tensor Core products; two wide operands require four. No FP8, FP16,
rounding or approximate multiplication is used. Native signed-INT8 inputs and
INT32 outputs follow the
[CUTLASS GEMM model](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api.md).
The production DP4A control uses four signed byte products per instruction; see
[NVIDIA's integer intrinsic documentation](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html).

Interval bounds, propagated through block sums and differences, determine
whether a high-limb pass is needed. These bounds are conservative: they can
schedule extra work even when a particular transformed block's actual values
fit INT8. With root values in [-128,127], at most three Strassen–Winograd levels
give operand magnitude at most 8192 and leaf K at most 16. Thus transformed
operands fit INT16 and leaf products fit INT32. Recombination and cumulative C
explicitly use UINT32 modular arithmetic; an independent INT64 calculation
checks their exact bit representation. The harness is restricted to K=4096,
128-term slices, depth <=3 and dimensions divisible by 128.

The recursive formulas follow
[Boyer, Dumas, Pernet and Zhou](https://arxiv.org/pdf/0707.2347), section 2.
Pairwise Winograd is a different identity from recursive Strassen–Winograd.

## Correctness

The full independent reference covers a 256x128x4096 panel for four input
patterns: all zero, all -128, random full-range signed INT8, and actual generated
zero-signal-B noise. For each of nine prototypes it checks **every C element at
every one of the 32 milestones**, every scattered-tile XOR, and every final
keyed BLAKE3 digest against the C BLAKE3 implementation. The three production
controls are checked for every tile XOR at every milestone. These tests pass
**38,687,232 UINT32 value comparisons**.

Before timing each larger method, three spatially separated tiles are checked
against an INT64 CPU reference at all 32 milestones. The two smaller timing
panels add 1152 comparisons each; the wide panel adds 576. In total the original
validation and timing sessions pass **38,690,112 comparisons**. Large-panel
checks are sampled XOR checks, not an exhaustive comparison of their full C
matrices or every final digest. No zero-target hit occurred during timing.

After formatting the harness, the full correctness suite is repeated with the
final source and build fingerprint. Its separate result is retained in the raw
data. Timing source versions differ only in formatting/comments and in the
wide-panel selection described below; the arithmetic kernels are unchanged.

Compute Sanitizer previously could not instrument this CMP50 because GPU
debugging is disabled. This experiment does not claim a passed memcheck.
CPU-reference parity is the correctness evidence available here.

## Results

Median CUDA-event milliseconds per complete panel invocation; lower is faster.

| Method | 1024x8192 | 4096x8192 | 4096x131072 |
| --- | ---: | ---: | ---: |
| Current Tensor Core 256 | **0.597** | **2.279** | **35.688** |
| Current Tensor Core 128 | 0.647 | 2.421 | 38.822 |
| Current DP4A | 1.740 | 6.502 | 105.585 |
| Classical limb Tensor Core | 12.783 | 48.962 | 773.641 |
| Classical INT16 SIMT | 195.572 | 781.003 | — |
| Pairwise Winograd | 196.935 | 782.850 | — |
| Strassen depth 1 | 29.106 | 103.886 | 1639.227 |
| Strassen depth 2 | 66.869 | 262.258 | — |
| Strassen depth 3 | 310.179 | 495.683 | — |
| Strassen–Winograd depth 1 | 31.452 | 112.389 | 1780.418 |
| Strassen–Winograd depth 2 | 66.337 | 262.083 | — |
| Strassen–Winograd depth 3 | 294.195 | 480.475 | — |

The first wide-panel run was intentionally stopped while executing the very
slow SIMT control; its guard restored the miner. Its completed validation and
two smaller panel runs are retained. The wide panel was then rerun to completion
with the three production controls, classical Tensor Core prototype and both
depth-1 recursive prototypes. SIMT and depths 2–3 remain measured on the two
smaller panels; dashes mean not measured at this width. Partial wide-panel
timings from the interrupted run are excluded from the result table and data
records. Both session outcomes and build manifests remain explicit.

On the wide panel, Strassen depth 1 is **45.93x** slower than the production
256x128 kernel; Strassen–Winograd is **49.89x** slower. Even classical separate
slice GEMMs are **21.68x** slower than the fused production implementation.

## Why fewer mathematical products did not help

For the generated noise ranges, depth-1 Strassen schedules 12 signed-INT8
Tensor Core leaf passes per milestone, compared with eight classical quadrant
passes. The seven algebraic products therefore become **1.50x the original
INT8 MAC count**, after wider operands are decomposed. Strassen–Winograd uses
14 passes and **1.75x the original count**. Over all 32 milestones this is 384
and 448 standalone GEMM calls respectively, plus transforms and state kernels;
the current implementation keeps the work in a single fused launch.

At deeper levels, this implementation further increases INT8 MAC counts and
launches thousands of small GEMMs. Pairwise Winograd halves scalar products in
the main dot loop but uses SIMT arithmetic and extra additions; it does not
beat the production INT8 Tensor Core path or even its simple SIMT control.

The production kernel retains accumulators and jackpot state in its fused tile
loop. Prototypes repeatedly materialize C/deltas, intermediate quadrant results
and proof state in global memory, reading C again for every milestone XOR.
B transforms and factors are recomputed rather than cached across attempts.
This design and its simple leaf geometry are part of the observed cost.
No dynamic hardware-counter evidence is collected here, so individual fractions
of time cannot be assigned to memory, launch overhead or multiplication.

The JSON field `int8_gemm_calls` counts standalone INT8 leaf invocations for
Tensor Core prototypes, or one fused production invocation for controls.
`tensor_leaf_macs` counts their scheduled scalar MAC equivalents, not hardware
instructions or measured Tensor Core utilization. For the DP4A control it is
the ordinary product's algorithmic MAC count; zero counts in SIMT prototype rows
mean uninstrumented, rather than zero arithmetic. Scratch frames reserve some
unused leaf buffers; the telemetry is allocation usage, not a minimal memory
requirement for the algorithms.

## Reproduce

On Linux with CUDA and the repository's existing BLAKE3/CUTLASS dependencies:

```sh
python3 scripts/build_cuda_matmul_bench.py --output build/gpu-matmul --arch sm_75
build/gpu-matmul/matmul-bench --verify
build/gpu-matmul/matmul-bench 1024 8192 9
build/gpu-matmul/matmul-bench 4096 8192 9
build/gpu-matmul/matmul-bench 4096 131072 9
```

The standalone executable never calls pool/network entry points. Real common
dependencies are linked with unused sections discarded, without dummy API
implementations. Nothing is added to the production build or CI. `sm_80` and
`sm_86` compilation targets are available but were not tested here.

The widest prototype panel requires substantial VRAM: the observed process
allocation plus existing GPU use reached about 11.8 GiB. Use a smaller panel on
devices without enough available memory. Results apply to this CMP50, not to an
untested RTX 3090 or other GPU family.

## Decision and restoration

Retain the current fused Tensor Core multiplier. There is no release speedup
from these implementations. A future recursive prototype would need efficient
handling of widened operands, much larger/fused leaf work, B-transform reuse and
all 32 proof milestones before another comparison is justified. These results
reject the tested implementations, rather than all possible fast-multiplication
algorithms. Practical packing and memory integration are discussed by
[Huang, Rice, Matthews and van de Geijn](https://arxiv.org/abs/1611.01120).

Protected server-local snapshots preserve the original executable, complete
arguments, environment, working directory and log destinations. Each guard
restores them in `finally`, including the intentional interruption. Final
independent verification checks the running release SHA-256, exact process
state, worker uniqueness, active GPU use and continuing scan-log output.
Credentials and wallet-bearing logs are excluded from committed artifacts.

The subsequent [data-feed/PTX load experiment](cuda_data_feed_experiment.md)
retains this multiplication algorithm and changes how operands are fetched.
The `.cg` candidate gives a confirmed 1.016% full-kernel gain on CMP50; source
reorderings alone produce identical machine code. No production default changes.
