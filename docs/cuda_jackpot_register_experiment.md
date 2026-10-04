# Constant-index CUDA jackpot experiment

This experiment follows the local-memory observation in the
[Pearl optimization suite](pearl_optimization_suite.md). It tests whether keeping
the 16-word jackpot state in registers improves valid Pearl mining work.
The answer is a small, repeatable kernel improvement for predicated constant
indices; removing local loads/stores alone is not sufficient.

Measurements were taken on 2026-10-04 with the existing CMP 50HX, 20 GiB VRAM,
CUDA 12.6.85, sm_75 and the unchanged 225 W power limit. No rented GPU or GitHub
Actions run was used. The published v0.5-fork.4 binary and production kernel
source remain unchanged. Generated variants and the test miner are isolated
under `/w/experiment-jackpot-register` on the test server.

## Implementations

The baseline updates `jackpot_words[step % 16]`. Two equivalent alternatives
replace that dynamic index:

- A 16-case switch, with a literal word index in each case.
- An unrolled 16-word loop that conditionally updates the matching word.

Every required cumulative GEMM milestone, rotate/XOR update, final keyed BLAKE3,
target comparison and hit publication is retained in the full variants. No hash
is reused across changed inputs. This uses registers rather than adding a VRAM
cache. The compiler decides whether the array can be promoted.

`scripts/build_jackpot_register_profile.py` creates complete private copies of
the CUTLASS wrapper headers. Using only a copied helper mixed with original
headers would produce duplicate definitions through different `#pragma once`
paths. Generated headers never replace the checked-in miner headers.

## Correctness

For each of the three implementations, the standalone CUDA test checks 1024
independent streams across 64 update prefixes, including zero/all-one inputs,
nonzero initial states and four complete cycles through the 16 words. It compares
all **1,048,576 state words** against an independent CPU recurrence and all
**65,536 keyed BLAKE3 digests** against the reference C BLAKE3 library. Every
comparison passed. The test checks intermediate states and digests, not only the
final XOR checksum.

Separate GEMM diagnostic builds replace final BLAKE3/target checking with a
checksum store. Three sampled tiles per execution are compared against a direct
CPU calculation of all 32 cumulative INT8 products and folds. All 18 sampled
comparisons passed. The rolling checksum over all 8,388,608 tile outputs was
`1221268778992207394` for every variant and both threadblock shapes. These
diagnostic builds are invalid mining work and never connect to a pool.

The complete predicated miner also passed the three CTest executables, the
CPU/GPU alignment suite at 8192 × 8192 × 4096, and four full-size mock runs with
proof construction and local verification. The mock cases cover 128×128 and
256×128 threadblocks, with both dense and incremental signal-A preparation.

## Compiler and full-kernel measurements

The timed panel is 4096 × 131072 × 4096, with deterministic inputs and an all-zero
target so a hit cannot shorten the computation. Each full-kernel group uses
80 warmup launches and 100 CUDA-event samples. Four permutations of variant
order are measured for each threadblock: 400 samples per variant/shape, 2400
full-kernel samples in total. A 60-attempt full-size miner warmup precedes them.

The table reports the mean of the four run medians. Throughput change is
`baseline_time / variant_time - 1`, not a change in share luck.

| Threadblock | Dynamic | Switch | Predicated | Predicated throughput gain |
|---|---:|---:|---:|---:|
| 256×128 | 35.8110 ms | 36.0641 ms | 35.4965 ms | +0.886% |
| 128×128 | 38.8183 ms | 37.8949 ms | 37.7731 ms | +2.767% |

The switch loses about 0.70% throughput on the production 256×128 shape. Its
better 128×128 result does not make that shape faster overall: 256×128 remains
faster for this panel.

| Full kernel | Registers/thread, dynamic → constant | Local bytes/thread | Shared bytes | Predicted blocks/SM |
|---|---:|---:|---:|---:|
| 256×128 | 216 → 246 | 128 → 0 | 49152 | 1 → 1 |
| 128×128 | 226 → 254 | 128 → 0 | 32768 | 2 → 2 |

Both constant-index versions eliminate every static LDL/STL instruction from
these full kernels, including local-memory accesses in the tensor-core loops.
The state now costs additional registers and control instructions. Both full
shapes remain below the architectural per-thread register limit, but the
128×128 version has little margin. Static full-kernel instruction counts increase
from 3616/3680 to 4064/4048 for predication and 4480/4536 for the switch
(256×128 / 128×128). Static counts are not dynamic instruction or stall counters.

The checksum-only variants produce a different relative ranking on 256×128;
they must not be used to choose a mining implementation. Removing BLAKE3 changes
register allocation and scheduling. The full valid kernel is the deciding test.

Profile GPU snapshots were taken between subprocesses, after their allocations
were freed. Their 1935 MHz clock samples describe the boundary/idle state, not
sustained scan clocks. Active-load samples are recorded separately during the
whole-miner comparisons. Hardware performance counters remain unavailable
because the driver denies access; its policy was not changed.

## Whole-miner comparison

The control is the previously compiled experimental-branch miner in dense mode,
not a re-labelled published executable. Comparing 172 production source files
found exactly one substantive difference: `cp_cutlass_jackpot.cuh`. Two other
files differ only in line endings. CMake options and Release compiler flags
match. Dense and incremental A are measured separately to isolate the jackpot
change from the cache change.

Each attempt performs 131072 × 131072 × 4096 work against a fixed loopback header
and an all-zero target. Overlap is enabled, the threadblock is 256×128, and
incremental mode uses 4096 updates. Fifty full-size control attempts warm the
GPU before timing. Dense mode uses eight 60-attempt blocks in a symmetric order;
incremental mode uses four 40-attempt ABBA blocks. Five initial attempts per
block are discarded. Effective TMAC/s includes preparation plus scan time.
GPU temperature, SM/memory clocks, power and allocated memory are sampled during
the scans. Accepted shares are used only for correctness/connectivity checks.

| Signal-A preparation | Dynamic | Predicated | Throughput gain | Measured attempts |
|---|---:|---:|---:|---:|
| Dense | 60.2716 TMAC/s | 60.8126 TMAC/s | +0.898% | 440 |
| Incremental, 4096 updates | 61.1750 TMAC/s | 61.7393 TMAC/s | +0.922% | 140 |

Aggregates divide total completed MACs by total preparation-plus-scan time;
they do not average share counts. All 640 fixed-work attempts completed, with
580 retained after warmup. Each predicated block exceeds every corresponding
dynamic block: dense controls span 60.238–60.288 versus 60.787–60.858 TMAC/s,
and incremental controls span 61.163–61.187 versus 61.725–61.753 TMAC/s.

During samples above 200 W with the miner allocation present, median SM clocks
were 1500 MHz for the dynamic version and 1530 MHz for predication in both modes.
Median power was approximately 223 W, memory clock was always 7000 MHz, and
temperature was 63–65 °C. The gain is therefore a wall-clock result under the
same 225 W limit with normal GPU Boost; it includes the different achieved clock
and is not evidence of a 0.9% improvement at a locked, identical frequency.
Allocated memory was identical within each A mode: 2863 MiB dense, 2935 MiB
incremental. Additional jackpot registers do not add a VRAM cache.

The bounded live verification trial is recorded in the raw report below. The
original release process is restored with its exact saved
arguments, environment, working directory and log destinations after each GPU
phase; restoration is checked independently.

The predicated miner with incremental A produced **two locally verified,
pool-accepted HeroMiners shares**, with zero rejected shares and no CUDA,
jackpot or proof mismatches. The live run completed 186 attempts and 195
full-reference cached-tree checks (including preparation ahead of a scan).
This short live trial validates the submission path, not expected earnings.

The independent restoration check confirms the original executable SHA-256
`3e5a589350558afc34c5d15297612ef16aa3273d0999c3b5bc3d1e9737fd300e`, exact original
arguments/environment/working directory, one original worker and resumed GPU
mining. The test-miner SHA-256 is
`343bdd72c2446722bd9e7c9f71585ef89a8736da475ba5eb1f4b670d5f9577e2`.

[Raw measurements, source/binary hashes, SASS statistics and restoration checks](benchmarks/jackpot-register-2026-10-04.json)
retain all per-launch timings and active-load GPU samples. Wallet data, saved
environment and live log contents are excluded.

## Reproduction and scope

On a Linux CUDA host with the project's existing vendored dependencies:

```sh
python3 scripts/build_jackpot_register_profile.py --output build/jackpot-profile
build/jackpot-profile/dynamic/fold-test
build/jackpot-profile/switch/fold-test
build/jackpot-profile/predicated/fold-test
build/jackpot-profile/dynamic/full 100 large
build/jackpot-profile/switch/full 100 large
build/jackpot-profile/predicated/full 100 large
# Omit "large" for 128x128. Diagnostic binaries are named "diagnostic".
```

For a complete miner experiment, use a throwaway source copy and replace only
its `src/cuda/cutlass/cp_cutlass_jackpot.cuh` with the generated predicated helper.
Build with CUDA enabled, arch 75, Release, CPU enabled, and OpenCL/cuBLAS/wgpu
disabled. Preserve the original miner and restore it after testing.

This does not establish a benefit on Ampere/RTX 3090 or on other compiler
versions. Promotion and the additional register cost must be measured there
before making the implementation a general default. The experiment does not
modify the published release or leave the test server mining an experimental
binary.
