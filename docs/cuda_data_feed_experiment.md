# CUDA data-feed and PTX load experiment — 2026-10-04

Changing the 16-byte operand loads to **`ld.global.cg.L2::128B.v4.u32`** gives
a small, repeatable improvement in the complete CMP50 kernel. The separate
balanced confirmation measures **35.200956 ms versus 35.558624 ms**, a
**1.016% throughput gain** on the production 256x128 threadblock. Keep this
as a candidate for a complete-miner experiment; no release or production
default is changed here.

Moving global fetches earlier or later in the C++ mainloop produces exactly
the same SASS as the original kernel, including instruction encodings and
scheduling bits. Their timing differences are experimental variation, not
an optimization. Streaming loads, a smaller L2 prefetch hint, and removing
the existing prefetch hint all lose on the tested panels.

This follows the [multiplication experiment](cuda_matmul_experiment.md) and
tests operand delivery while retaining the existing mathematical algorithm.
[Raw data](benchmarks/cuda-data-feed-2026-10-04.json) includes every sample,
build command, source/binary fingerprint, SASS digest, GPU telemetry and
restoration check.

## Scope and implementations

Hardware: NVIDIA CMP 50HX, sm_75, 20 GiB VRAM, CUDA 12.6.85, driver 610.43.03,
unchanged 225 W limit. Normal boost remains enabled. RTX 3090 is not tested.

All variants retain signed-INT8 Tensor Core multiplication, INT32 accumulation,
the continuous two-stage pipeline, all 32 cumulative milestones, scattered
tile XORs, rotate/XOR state, final keyed BLAKE3, target comparison and hit
publication. No synchronization or milestone is deleted.

| Variant | Change |
| --- | --- |
| `baseline` | Original mainloop and CUTLASS 16-byte operand load with a 128-byte L2 prefetch hint |
| `early` | Move the next global fragment fetch before the first shared-memory warp-fragment load |
| `late1` | Fetch after the second warp MMA group, before the final group stores the fragment to shared memory |
| `cg128` | Keep the 128-byte L2 prefetch hint and add `.cg` to the 16-byte operand load |
| `cs128` | Use `.cs` streaming cache behavior, retaining the same prefetch hint |
| `prefetch64` | Reduce the L2 prefetch hint from 128 to 64 bytes |
| `no_prefetch` | Remove the L2 prefetch hint from that operand load |

The existing CUTLASS version already uses inline PTX for both `mma.sync` and
operand loads, and enables the 128-byte L2 hint on sm_75 with this compiler.
The memory variants modify only the 16-byte `global_load` specialization in a
private copy of `cutlass/arch/memory.h`. Pointer operands, predicates, access
width, masked-load initialization and matrix layout are preserved. Wrapper
headers and architecture helpers are copied into one private include tree,
with their original license notices. Shared dependency trees are never edited.

NVIDIA defines `.cg` as caching in L2 and below while bypassing L1, and `.cs`
as streaming loads with an evict-first policy. The L2 prefetch size is a
performance hint, not an additional mathematical operation. See
[PTX cache operators](https://docs.nvidia.com/cuda/archive/12.6.0/parallel-thread-execution/index.html#cache-operators)
and [PTX loads](https://docs.nvidia.com/cuda/archive/12.6.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld).

The scheduling variants are specific to the tested Turing mainloop, which has
four warp MMA groups per K tile. `late1` still fetches before the last group
stores the next fragment and enters the unchanged block synchronization.
It is not a general replacement for every CUTLASS iterator or GPU architecture.

## Correctness

Every variant has a separate verification executable. At 256x256x4096 it tests
four patterns: zeros, all -128, random full signed INT8, and actual generated
zero-signal-B noise. Both 128x128 and 256x128 threadblocks are tested.

The CPU reference accumulates products in INT64 and independently derives each
tile XOR at every 128-term prefix, the 16-word fold state, and keyed BLAKE3 with
the C reference library. Every GPU tile XOR and digest is compared. This covers
**1,835,008 milestone XOR words and 57,344 complete BLAKE3 digests** across the
seven variants, including all four patterns and both threadblock shapes.
These are exhaustive tile-output checks, not direct comparisons of each GPU C
element individually.

Digest capture is compiled only into verification builds, after the real
BLAKE3 compression and before the unchanged target check. Timed builds contain
no digest-capture instructions. Before each timed block, the uninstrumented
full build also checks three spatially separated tile XORs against INT64 at
all 32 milestones. Those diagnostic dumps are disabled for timing.

All checks passed, totaling **2,300,096 UINT32 comparisons** across validation,
conditioning and both timing sessions. No zero-target hit occurred. Compute
Sanitizer could not instrument this hardware in earlier experiments; no passed
memcheck is claimed here. Full mining proofs and live pool submission were not
exercised with these candidates.

## Measurement

The panel is 4096x131072x4096, using the actual noise-generation functions,
rank 128 and deterministic synthetic seeds. It spans full production N width
and part of M. It is not an entire 131072x131072 attempt.

Allocation, noise generation, uploads, CPU reference computation and diagnostic
output are outside timing. Timed kernels do all required arithmetic and hashing
with an all-zero target so a successful share cannot shorten the scan. CUDA
event and CPU wall times are both saved.

The screening run starts with a 500-sample baseline conditioning block, then
measures four permutations of the seven variants. Threadblock-shape order
alternates by round. Each block has 40 warmup launches and 60 event samples:
240 samples per variant/shape, **3360 comparison samples** in total.

The confirmation run uses a 200-sample conditioning block and tests only
`baseline` and `cg128`, on 256x128, in ABBA and BAAB order. Each of eight blocks
has 40 warmups and 100 samples, giving **800 comparison samples**. Conditioning
samples are retained in raw data but excluded from comparison aggregates.

Times below are the mean of each implementation's four block medians.
Throughput change is `baseline_time / candidate_time - 1`.

| Variant | 256x128 time | Throughput change | 128x128 time | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 35.558371 ms | — | 38.433848 ms | — |
| `early` | 35.492497 ms | +0.186%* | 38.435308 ms | -0.004%* |
| `late1` | 35.466292 ms | +0.260%* | 38.434024 ms | -0.0005%* |
| `cg128` | **35.146548 ms** | **+1.172%** | **38.217808 ms** | **+0.565%** |
| `cs128` | 35.900136 ms | -0.952% | 47.706112 ms | -19.436% |
| `prefetch64` | 37.150224 ms | -4.285% | 39.431104 ms | -2.529% |
| `no_prefetch` | 37.153840 ms | -4.294% | 39.422875 ms | -2.509% |

\* The scheduling variants have identical SASS to the baseline. These deltas
therefore measure timing/environment variation and must not be called gains.

Independent confirmation:

| Variant | Mean block median | Block median range |
| --- | ---: | ---: |
| `baseline` | 35.558624 ms | 35.507056–35.621776 ms |
| `cg128` | **35.200956 ms** | **35.154368–35.223904 ms** |

Every confirmation `cg128` block is faster than every baseline block. The
confirmed throughput gain is **1.016%**. It is a complete-kernel wall-clock
result under normal GPU Boost, not a locked-frequency instruction-cycle gain
or a demonstrated increase in whole-miner throughput or accepted shares.

## Machine-code evidence and interpretation

Baseline, `early` and `late1` have identical normalized SASS hashes for each
shape. Cache-policy and prefetch variants have different instruction encodings.
For example, the 256x128 operand load changes from
`LDG.E.LTC128B.128.SYS` to `LDG.E.LTC128B.128.STRONG.GPU` for `cg128` in this
compiler's disassembly. Addresses, destinations and the surrounding load count
are preserved. The PTX `.cg` modifier is the source-level policy definition;
the SASS mnemonic is the compiler's architecture-specific representation.

All variants retain the same measured resources:

| Threadblock | Registers/thread | Local bytes/thread | Shared bytes/block | Predicted blocks/SM |
| --- | ---: | ---: | ---: | ---: |
| 256x128 | 216 | 128 | 49152 | 1 |
| 128x128 | 226 | 128 | 32768 | 2 |

Static IMMA, global-load, shared-store and local-load/store counts remain equal
within each shape. Counts and encodings are saved in the manifest; they do not
measure dynamically executed instructions, cache-hit rates or stall time.
The raw `LDS` counter excludes the separate `LDSM` mnemonic and should not be
interpreted as zero shared-memory reads.

Bypassing L1 for matrix operands may reduce competition with local state spills
and other data, but this is an inference, not measured cache-counter evidence.
GPU performance counters were unavailable under the existing driver policy;
that policy was not changed. Timestamped temperature, power, memory usage and
SM/memory-clock samples are retained. An existing idle GPU process was left
alone, and normal boost/background variation remains a limitation.

## Reproduce and decision

On Linux with CUDA and the repository's existing dependencies:

```sh
python3 scripts/build_cuda_data_feed_profile.py --output build/feed-profile
build/feed-profile/baseline/verify
build/feed-profile/cg128/verify
build/feed-profile/baseline/full 100 large
build/feed-profile/cg128/full 100 large
# Omit "large" to measure the 128x128 threadblock.
# Repeat in balanced order after warming the GPU.
```

Verification binaries ignore timing arguments and run the full correctness
suite. Other variants live in their corresponding output directories. The
standalone executable never connects to a pool. Real common dependencies are
linked, with unused sections discarded and no dummy replacements. CMake,
release packaging and CI are unchanged.

After measurement, the builder gained a stale-manifest invalidation guard.
Its final file hash and the measured builder hash are both recorded; generated
CUDA headers, harness source and compiler flags are unchanged by that guard.

Retain `cg128` as a measured candidate. Reject the two C++ reorderings as
machine-code no-ops and reject the losing cache/prefetch settings. Before adding
the candidate to a release, compare complete miner attempts and proof checks,
including interaction with the earlier
[constant-index jackpot candidate](cuda_jackpot_register_experiment.md).
Those gains cannot be assumed additive, and other GPU families need separate
validation. There is no production flag or default change in this experiment.

Both bounded server runners restore the original v0.5-fork.4 process in
`finally`. Final independent verification confirms its published binary
SHA-256, exact arguments/environment/cwd/log destinations, one original worker,
active GPU use, and 11 newly completed mining attempts in a 12-second observation.
Private snapshots and wallet-bearing logs remain server-local and are excluded
from the committed data. No rented GPU, GitHub Actions run or upstream PR is
required.
