# Milestone XOR scheduling on CMP50HX — 2026-10-04

This report records the experiments before production integration. The selected
`before_loads` variant has since been integrated with R1 and incremental signal-A;
see [integrated build and deployment](cuda_integrated_improvements.md). The
historical experiment builders below expect pre-integration source at commit
`717823f5d6b20953e48738a3c27d85dc1ed01f37` and generate their controls there.

This experiment examines the cumulative XOR callbacks inside the Turing GEMM
mainloop. All experiments run on the authorized CMP50HX test server, at the
existing 225 W power limit. No GitHub Actions, rented GPU, upstream PR or new
release is involved. Experimental kernels operate on synthetic data; the
installed v0.5-fork.4 miner is restored after each bounded GPU run.

## What the Akoya comparison actually establishes

At Akoya commit `0e42b2a50b4cb62cb8995b77420226c2182137ce`, the
[consumer kernel](https://github.com/akoyapool/akoya-miner/blob/0e42b2a50b4cb62cb8995b77420226c2182137ce/native/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu)
issues the next asynchronous copy, reduces the previous milestone, then loads
shared-memory fragments and performs MMA. Its comment explicitly describes
moving the previous snapshot into the shadow of those loads.

That is an Ampere-capable path. CMP50HX is sm75 and has no hardware `cp.async`.
Akoya's separate
[sm75 kernel](https://github.com/akoyapool/akoya-miner/blob/0e42b2a50b4cb62cb8995b77420226c2182137ce/native/pearl-gemm/csrc/portable/transcript_gemm_sm75.cu)
uses ordinary copies, a CTA barrier, `ldmatrix`, GEMM, another barrier, and then
the milestone reduction. Consequently its Ampere scheduling argument cannot
be assumed to apply unchanged to this GPU.

## SASS observations

The installed release is disassembled with an explicit physical architecture
filter: `cuobjdump --dump-sass --gpu-architecture sm_75`. Its binary SHA256 is
`3e5a589350558afc34c5d15297612ef16aa3273d0999c3b5bc3d1e9737fd300e`.
Unfiltered multi-architecture dumps can repeat function names; the parser keys
by function name, so they must not be used to identify the active sm75 code.

For the release's 256×128 sm75 mining loop, the static span has 256 IMMA, six
LDG, 16 LDSM, six STS and one CTA barrier. The rotated pre-MMA callback span has
136 instructions, 14 butterfly shuffles, 50 three-input XOR LOP3s, and two local
loads plus two local stores. These are static spans with conditional control
flow, not executed instruction counts per K tile or stall measurements.

The already validated research control uses `.cg` loads and the predicated,
constant-index transcript fold from R1. Its corresponding pre-MMA span has
246 instructions and no local-memory traffic: 48 three-input XORs, 62 two-input
XORs, 14 butterfly shuffles, 28 selections and 32 funnel shifts. The larger
static span replaces dynamically indexed local state with predicated registers.
Its speed must be judged by measurements, not instruction-count reduction.

Each thread's 16 local partial reductions use eight accumulator values each.
The compiler already implements each eight-value reduction with three XOR3s and
one XOR2. Replacing the same expressions with a serial inline PTX spelling
would not reduce their instruction count. A balanced PTX tree instead tests a
different dependency structure: three independent partials, then one XOR3,
giving two logical dependency levels rather than four.

`scripts/sass_milestone_stats.py` now reports address spans, global/shared
loads/stores, pre-MMA spans and independent butterfly-shuffle spans. The latter
are explicitly heuristic: compiler rotation can put a callback outside the
simple backward-branch interval. The tool does not identify dynamic stalls.

## Exact callback-placement controls

The same R1 research control is used for all three variants. The changes are
generated into private headers; production sources and vendor headers remain
unchanged.

- `baseline`: the existing callback after the milestone's final MMA.
- `before_loads`: defer it to the beginning of the next K tile, before that
  tile's shared/global loads and before its first MMA.
- `after_loads`: defer it until the next tile has issued its fragment/global
  loads, still before the first MMA modifies the accumulator.

The original post-loop callback flushes the final complete milestone, with no
extra K iteration. All 32 prefix XORs retain their original index and order.

Both 128×128 and 256×128 shapes pass independent INT64 accumulation and keyed
CPU BLAKE3 oracles for four input patterns. The test checks every prefix XOR,
every dump-path digest, and every null-dump mining-path digest. Each variant
passes 393,216 comparisons. Timed builds have no digest-capture instructions.

Timing uses a `4096 × 131072 × 4096` panel, rank 128, 40 warmup launches, and
80 CUDA-event observations per block. Four balanced orders produce four block
medians per variant; the table averages those medians. Conditioning uses an
additional 300 control launches. Default 256×128 resources and results:

| Variant | Mean block median | Change in time | Registers/thread | Local bytes | Shared bytes | CTAs/SM |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 34.7373 ms | reference | 246 | 0 | 49,152 | 1 |
| before_loads | 34.5134 ms | −0.644% | 250 | 0 | 49,152 | 1 |
| after_loads | 38.2385 ms | +10.079% | 255 | 0 | 49,152 | 1 |

The attempted overlap placement is slower despite retaining exact results and
not spilling. It also changes register assignment and control flow; unavailable
GPU performance counters prevent attributing the regression to one stall class.
The smaller gain of `before_loads` is confirmed below in the full miner.

## Complete miner confirmation

The control is the previously confirmed R1 full miner, SHA256
`bb5b2ca399544438ab201700ec1baae94338dd3412ba0b72a6373e7442e6f140`.
The candidate changes only callback placement on top of that control, SHA256
`dbf60e957a49d17328f3ee2f0d9e0a24f40b48ff6a3364ba0ae10b169ce3fda6`.
Private checkout headers are restored byte-for-byte after compilation.

Three CTest checks pass, as does the 8192×8192 production alignment test.
Eight verified mock cases cover both binaries, both threadblock shapes, and
both dense/incremental signal-A modes, with incremental CPU checks enabled.

Throughput is measured on a zero-target loopback job with M=N=131072, K=4096,
rank 128, and 4096 signal-A updates. Each mode uses ABBA order, 30 complete
attempts per block, discarding the first four. The four blocks provide 104
measured attempts per mode. An initial 24-attempt conditioning run discards four.
Rates include recorded preparation plus scan time for each complete attempt;
they do not estimate earnings from short-window accepted-share counts.

| Signal-A mode | R1 control | R1 + before_loads | Incremental gain |
|---|---:|---:|---:|
| dense | 61.8626 TMAC/s | 62.2351 TMAC/s | +0.602% |
| incremental | 62.7345 TMAC/s | 63.1295 TMAC/s | +0.630% |

This is an additional small gain over R1, not a comparison with the installed
release. It is not evidence for adding two independent 2.5% improvements.
The test retains all hashes and all milestones. The installed miner remains
v0.5-fork.4; no experimental binary is deployed to the pool.

## Diagnostic decomposition

These kernels are deliberately invalid mining work and never connect to a
pool. All use identical deterministic inputs, the R1 load/fold control, the
same panel, 40 warmups, 60 CUDA-event observations per block, and four balanced
orders. Every diagnostic output is sampled against an independent CPU result
at three tiles per invocation. All four builds retain final-accumulator
dependencies, preventing removal of the actual matrix multiplication.

| Variant | Mean block median | Registers/thread | What remains |
|---|---:|---:|---|
| full | 34.8261 ms | 246 | All prefixes, 16-word fold, BLAKE3 |
| no-final-hash | 34.0500 ms | 238 | All prefixes and 16-word fold; checksum output |
| xor-only | 32.5033 ms | 238 | All prefix reductions; one-word XOR accumulator |
| final-milestone-only | 30.1863 ms | 238 | Final prefix and final fold; checksum output |

All have zero local bytes, 49,152 shared bytes and one CTA/SM. Relative to full
panel time, the diagnostic differences are:

- BLAKE3 removal: 0.7761 ms, or 2.228%.
- Replacing the 16-word rotate/fold with a one-word XOR: 1.5468 ms, or 4.441%.
- Removing intermediate prefix reductions from that simpler diagnostic:
  2.3169 ms, or 6.653%.

These are end-to-end sensitivity measurements, not additive timings of the
original instructions. The latter difference also changes control and output
state; register scheduling changes even when the resource counts match.
No valid miner can omit those prefixes or the transcript fold.

## Balanced inline PTX

A separate balanced four-block run uses the same panel, 40 warmups, 80 samples
per block, and the complete 393,216-comparison oracle for each of three builds:

| Variant | Mean block median | Change in time | Registers/thread |
|---|---:|---:|---:|
| Same R1 baseline | 34.7551 ms | reference | 246 |
| Balanced eight-input PTX XOR | 35.1168 ms | +1.041% | 246 |
| Balanced PTX plus before_loads | 34.5789 ms | −0.507% | 250 |

All retain zero local bytes, 49,152 shared bytes and one CTA/SM. SASS confirms
the intended independent partial XORs followed by the final XOR3, with unchanged
instruction counts. Shortening that dependency tree does not improve this GPU's
measured full-kernel throughput. The combined variant supplies no evidence of
extra benefit over callback placement alone; those two candidates were measured
in separate cohorts, so their small difference is not a paired comparison.
Neither PTX candidate receives full-miner confirmation or production deployment.

The useful outcome is the modest, verified `before_loads` gain. It is available
as a generated research patch and verified binary, while production defaults
remain untouched. Testing the actual asynchronous-copy opportunity on sm80/86
requires a separately authorized Ampere server; no RTX3090 is rented here.

## Evidence and reproduction

- [Panel timings, exact oracles, ablations and GPU telemetry](benchmarks/cuda-xor-milestones-2026-10-04.json).
- [Full-miner ABBA, alignment and proof verification](benchmarks/cuda-xor-miner-2026-10-04.json).
- [Static SASS counts and reduction spans](benchmarks/cuda-xor-sass-2026-10-04.json).
- [Balanced PTX timings and exact oracles](benchmarks/cuda-xor-ptx-2026-10-04.json).
- [Final production restoration audit](benchmarks/cuda-xor-restoration-2026-10-04.json).
- [Previous R1/R4/R6 research](cuda_research_validation.md).
- [Previous BLAKE3 and final-only ablations](pearl_optimization_suite.md).

Build in a private Linux/CUDA checkout with populated dependencies and verified
data-feed common objects:

```sh
python3 scripts/build_cuda_milestone_schedule.py \
  --output build/milestone-schedule --objects build/feed-profile
python3 scripts/build_cuda_milestone_ablation.py \
  --control build/milestone-schedule/baseline --output build/milestone-ablation
python3 scripts/run_cuda_research_profile.py \
  --build build/milestone-schedule --ablation-build build/milestone-ablation \
  --output results/milestones --tile 256x128 --repeats 80 \
  --ablation-repeats 60 --seconds 900
# Alternative build for the separate PTX cohort:
python3 scripts/build_cuda_milestone_schedule.py \
  --output build/milestone-ptx --objects build/feed-profile \
  --variants baseline ptx_balanced ptx_before
```

The guarded runner is specific to the authorized server and installed release.
Private restoration snapshots contain process arguments/environment, stay
server-local with mode 0600, and must never be exported with numerical results.
