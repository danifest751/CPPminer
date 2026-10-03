# Pearl optimization validation

This suite extends the [incremental signal-A experiment](cuda_incremental_a_experiment.md).
It is experimental work on `perf/cuda-incremental-a`, based on the published
v0.5-fork.4 release. The published release and the default dense CUDA path are
unchanged. Tests use the existing CMP 50HX; no rented GPU or GitHub Actions run
is required.

Measured on 2026-10-04, CMP 50HX, 20 GiB VRAM, CUDA 12.6.85, 225 W limit.
Production uses the Turing 256×128 threadblock. The long fixed-work comparison
gave **60.24 TMAC/s for v0.5-fork.4 and 61.10 TMAC/s for incremental mode
(+1.43%)**. The 1030 complete attempts include 1010 after warmup. Active samples
showed 64–65 °C and approximately 1470–1515 MHz SM / 7000 MHz memory clocks.
The extra allocation remains 72 MiB with two signal buffers.

The live ABBA blocks produced five accepted release shares and two accepted
incremental shares, all locally verified, with zero rejected shares. A subsequent
1024-update live test produced one more locally verified, accepted share and
258 full-reference cached-tree checks. The unequal share counts are not evidence
of a throughput reversal: this is a short stochastic observation with changing
pool jobs and targets.

All 860 independent mutation/root/subroot comparisons, eight full-size verified
mock cases, five invalid-option cases, job-change/cancellation checks, the three
CTest executables and the existing CPU loopback protocol suite passed. Two
diagnostic ablations also passed 24 sampled CPU tile-checksum comparisons.

Raw measurements, source hashes, executable hashes and restoration checks:
[optimization suite data](benchmarks/pearl-optimization-suite-2026-10-04.json).

## Scope and controls

Five questions are tested: sustained incremental-cache throughput, mutation
count, fused CUDA stage costs, exact zero-signal-B factorization, and a warm
second pool connection. Independent correctness checks precede timing tests.

Fixed-work measurements use `131072 × 131072 × 4096`, rank 128, a fixed loopback
header and an all-zero target. This prevents shares from shortening a scan.
Production preparation overlaps scanning unless a test explicitly disables it.
The CMP power limit remains 225 W; temperature, SM clock, memory clock and
instantaneous power are recorded. Those power samples do not measure energy
per accepted share. Throughput comes from complete preparation-plus-scan times;
logs round durations to 1 ms, so overlapping preparation can appear as zero.

The long comparison uses four 300-second fixed-work blocks in ABBA order, then
four 300-second HeroMiners blocks in the same order. Five attempts per fixed
block are discarded for warmup. The live comparison uses the original wallet
with a temporary worker and `--verify`. Accepted-share counts are correctness
and connectivity observations, not an earnings benchmark.

## Mutation-count experiment

`CP_CUDA_A_UPDATES` controls sparse/incremental signal writes per attempt.
The default is still 4096. Accepted values are 1 through 1048576, additionally
bounded by the matrix byte count; malformed or overflowing values fail.
Dense mode does not use this setting. Counts at or below 4096 preserve the
previous row/value sequence for each visited column. Larger counts visit
consecutive rows starting at that column's seeded row. Every writer addresses
a distinct byte, including the extra waves.

The standalone test compares the complete mutated signal against a CPU oracle,
roots against CPU BLAKE3, and witness subroots against full GPU hashing. It
also checks distinct writers, re-keying, signed signal bounds, cleared dirty
flags, and repeated commitments in the larger multi-update cases. It exercises
1, 256, 1024, 4096 and 16384 writes, eight tree sizes and the full 512 MiB signal.
A requested write can set the existing value again; write count is not a promise
that exactly that many values differ. A one-write strategy is therefore not
recommended merely because its preparation cost is small.

Performance comparisons use 256, 1024, 4096 and 16384 writes, serial and
overlapping preparation, forward and reverse order, 18 attempts per group and
two warmup attempts discarded. Full-reference hash checks are disabled during
timing. Separate full-size mock runs verify proofs and cached roots with the
checks enabled. Same-ID job changes and active cancellation are also exercised.

| Writes per attempt | Mean serial preparation | Mean overlap throughput |
|---:|---:|---:|
| 256 | 7.97 ms | 61.245 TMAC/s |
| 1024 | 8.00 ms | 61.265 TMAC/s |
| 4096 | 8.97 ms | 61.238 TMAC/s |
| 16384 | 9.00 ms | 61.202 TMAC/s |

The full overlap spread is only about 0.10%; 1024 exceeds 4096 by about 0.04%.
Those differences are comparable to variation between repeats. There is no
convincing reason to change the default count. Initial serial groups warmed from
49 to 62 °C, so their scan-throughput differences should not be attributed to
mutation count; the preparation values and later overlap groups are more useful.
Counts below 4096 visit the first requested number of columns, rather than a
random subset of all columns.

The newly compiled dense control measured 59.75 TMAC/s serial and 60.32 TMAC/s
with overlap, consistent with the earlier release measurements within the
temperature/timing variation. No default behavior was changed.

## Fused CUDA stage experiment

`tests/cp_cuda_stage_profile.cu` and `scripts/build_cuda_stage_profile.py` build
three standalone diagnostic executables. Generated header copies live in the
selected build directory; production kernel source is not modified:

- Full GEMM, all 32 cumulative XOR/fold milestones, final BLAKE3 and target check.
- All GEMM and milestones, with final BLAKE3 replaced by a checksum output.
- All GEMM, with only the final cumulative milestone and checksum output.

The latter two produce invalid mining work and never connect to a pool. Their
outputs keep the GEMM live, and three sampled tiles per execution are checked
against an independent CPU calculation. The benchmark uses the same deterministic
inputs, a `4096 × 131072 × 4096` panel, five warmup launches and 30 CUDA-event
samples. Both 128×128 and 256×128 Turing threadblocks are tested in forward and
reverse order. Register count, local memory, shared memory and predicted active
blocks per SM are recorded alongside static SASS instruction counts.

Differences between ablations indicate optimization headroom. They are not
additive stage timings: removing work changes register allocation, instruction
scheduling and occupancy, and the checksum introduces an output store. Deleting
a required milestone or hash cannot be released as an optimization.

Mean of the two per-run medians (30 samples each):

| Threadblock | Full kernel | No final BLAKE3 | Only final milestone, no BLAKE3 |
|---|---:|---:|---:|
| 128×128 | 38.816 ms | 38.196 ms | 35.355 ms |
| 256×128 | 35.615 ms | 34.825 ms | 31.393 ms |

Removing final BLAKE3 saved 1.6% / 2.2% of full panel time. Removing it together
with the first 31 milestone folds saved 8.9% / 11.9%. This is a measured upper
bound from modified kernels, not a promised valid-miner speedup. Most execution
time remains in matrix multiplication, operand access and the surviving control
work. Final BLAKE3 by itself is a relatively small target.

The 256×128 full kernel uses 216 registers/thread, 128 bytes/thread of local
storage, 49152 bytes of shared memory and one predicted active block/SM. Its
hash-free ablations use 210 registers and the same local/shared allocation.
The 128×128 full kernel uses 226 registers, 128 local bytes, 32768 shared bytes
and two predicted blocks/SM. Static SASS contains local loads/stores inside
tensor-core loops, including loops without diagnostic global stores. Testing
constant-index jackpot state and its register cost is therefore a concrete next
hypothesis; static counts do not establish dynamic stall time.

Nsight Compute returned `ERR_NVGPUCTRPERM`: the target driver denied access to
performance counters. Driver policy was not changed. These results use CUDA
events, CPU diagnostic comparisons, function attributes and static SASS, not
measured tensor utilization or DRAM-stall counters.

Build and run on Linux with CUDA:

```sh
python3 scripts/build_cuda_stage_profile.py --output build/stage-profile
build/stage-profile/full/profile 30
build/stage-profile/no-final-hash/profile 30
build/stage-profile/final-milestone-only/profile 30
# Append "large" to use the 256x128 threadblock.
```

## Exact factorization

With zero signal-B, write the noisy matrices as
`A' = A + E_A P_A^T` and `B' = E_B P_B^T` (B stored transposed). Each row of
`P_B` has one +1 and one -1. At the end of milestone t, with `L = t × 128`,

`C_t = A'[:, :L] P_B[:L, :] E_B^T = D_t E_B^T`.

The Rust example constructs actual keyed signal commitments, derives the
vendored legacy or salted noise seeds, and compares every element of every
cumulative product against the ordinary dot product. It then checks all 16
jackpot words and the final keyed BLAKE3 against the vendored reference helper.
Forty-eight cases cover zero, sparse and dense A, four changing headers/seeds,
two tile anchors, and both seed derivations: 1536 milestones and 98304 cumulative
output values agree exactly.

For an 8×8 tile, direct multiplication uses 262144 MACs. Recomputing all 32
factorized prefix products also uses 262144 MACs, plus 65536 projection additions
(the projection can be shared across column tiles). Computing only the final
product uses fewer MACs but does not reproduce the required earlier milestones.

Observed projection coefficients reached absolute value 1466; 41.68% lay outside
signed INT8. An exact low/high signed-INT8 decomposition reproduced every tested
product, but it requires two INT8 product passes. The observed two-limb range
is not a guarantee for every possible input/job; a general implementation must
check bounds or handle additional limbs. Approximate arithmetic is not suitable
for this equality requirement.

This identity is correct, but the straightforward form does not reduce required
MACs and loses the single-pass INT8 representation. Its remaining attraction is
compact B storage and possible memory-traffic changes, which would require an
end-to-end GPU implementation to measure. No factorized GPU mining path is
enabled by this suite.

```sh
cargo run --offline --release --manifest-path rust/cp-proof-ffi/Cargo.toml \
  --example factorized_jackpot
```

## Two-pool experiment

`tests/cp_pool_failover_experiment.py` measures the current miner against a real
loopback socket, then compares cold and pre-authorized reserve sessions in a
separate, minimal protocol model. It is not multi-pool support in CPPminer.

The current Windows CPU miner resumed a new loopback job about 13–34 ms after
an explicit TCP close. With a valid job already received, an open silent socket
and no pending submissions, it did not reconnect during a 35-second observation.
The production session has authorization, first-job and pending-submit deadlines;
these are not a general idle-job deadline.

Ten model samples per case use synthetic authorization delays of 0, 50 and
200 ms. Median cold time after a close was about 14, 66 and 216 ms respectively;
selecting an already authorized reserve job took under 0.1 ms. These are
protocol-model timings on Windows, excluding GPU cancellation, reserve matrix
preparation, real DNS/network delays and actual mining resumption.

A separate simulated 200 ms silence-detection policy remained necessary with
both cold and warm reserves. Warm reserve eliminated the subsequent handshake,
but did not shorten detection. The checks also cover colliding job IDs on two
pools, the latest reserve update, stale primary proofs, old reserve headers and
target/certificate changes, and old session generations. Proof ownership must
include pool/session identity and the full job snapshot.

A second connection can improve availability. It does not create additional
compute attempts; alternating two jobs on the same GPU divides the available
work. A production feature needs separate connection/session state, explicit
failure detection, cancellation and submission routing, plus measurements on
real mining jobs. This suite leaves that implementation for a separately
reviewed change.

```sh
python tests/cp_pool_failover_experiment.py path/to/cppminer \
  --output build/pool-failover.json
```

## Operational restoration and limitations

Hardware runners save the original process arguments, environment, working
directory and log paths in protected server-local files. Bounded runs restore
v0.5-fork.4 in a `finally` block, including failures. Final verification compares
the executable checksum, process arguments and GPU process state. Credentials
and wallet-bearing logs are not committed.

Compute Sanitizer could not instrument the CMP50 because GPU debugging features
are disabled. Independent CPU/GPU comparisons do not replace a passed memcheck.
RTX 3090 and other GPU families are not tested by this suite. The default mode
should remain unchanged until broader validation and longer stability data are
available.

The original published SHA-256
`3e5a589350558afc34c5d15297612ef16aa3273d0999c3b5bc3d1e9737fd300e`
was independently checked after restoration. Original arguments, environment,
working directory, worker uniqueness and active GPU mining all matched. The
test branches do not replace that production executable.

## Decision

Keep incremental mode opt-in and retain 4096 as its default mutation count.
The sustained cache gain is modest but reproducible on this CMP50; the smaller
counts offer no clear additional throughput. Investigate jackpot-state indexing
and GEMM/operand traffic with exact proof checks before pursuing more cache work.
The simple factorization offers no MAC-count shortcut. Warm reserve connections
are an availability feature whose production implementation needs explicit
failure detection and session-aware submission routing.
