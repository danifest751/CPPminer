# Quantus CUDA (Turing sm_75) carry-chain experiments: results

Branch: `perf/qpow-cuda-turing`. Box: CMP 50HX, sm_75, driver 610.43.03, CUDA 13.3, ~1500 MHz
effective under the 225 W cap. Harness: `tools/cuda-quantus/q2-harness/qtc.cu`, mode `cyc` (SM
cycles per hash via clock64), interleaved A/B, 6 rounds.

## What the open references suggested (and how it held up on sm_75)

The plan (`docs/qpow_cuda_turing_carry_chain_plan.md`) rested on three shape changes that the
official `quantus-miner`, KanQ and the AMD backend converged on. On sm_75 all three are refuted:

| Variant | Switch | Result on sm_75 | Note |
|---|---|---|---|
| signed reduce (KanQ `fold128_lazy32` / official `combined_reduce`) | `QTC_RED_SIGNED=1` | **+35% cycles**, also fails the exact field/hash test | official reported +5% on sm_86 |
| native `__umul64hi` product (official v5) | `QTC_MULHI=1` | **+4% cycles** | official +7% on samsung 3060 Ti (sm_86) |
| split internal product (sum + separate mul/add128) | `QTC_INT_SPLIT=1` | **+5% cycles** | official rejected the fused form on sm_86; on sm_75 the fused form wins |

Conclusion: the carry-chain structure that is optimal on Ampere (sm_86) is **not** optimal on Turing
(sm_75). The two microarchitectures' integer pipes differ; results do not transfer between them.

## What actually helps on sm_75 (exclusive GPU, production miner stopped)

The production `cppminer` on the box had been sharing the CMP, halving every number (~128 MH/s);
with it stopped the miner measures **279 MH/s** at `--mock` (matches the handoff's 278). All numbers
below are on the exclusive GPU, interleaved `cyc 256`, noise < 0.2 cyc.

Baseline = the miner's own shape: `QTC_SUB=1` (nonce line) + `QTC_SQR3=2` (funnel-shift squaring),
`QTC_GFMA2=2` (the `gfma_y` even/odd chains the kernel already uses).

| Variant | cycles/hash | delta |
|---|---|---|
| base (= miner) | 300.7 | — |
| row sum split (`QTC_SUM_SPLIT=1`) | 302.6 | +0.6% |
| fused `gfma_w` (`QTC_GFMA2=0`) | 306.7 | +2.0% |
| native `__umul64hi` (`QTC_MULHI=1`) | 313.2 | +4.2% |
| split internal (`QTC_INT_SPLIT=1`) | 315.0 | +4.8% |
| signed reduction (`QTC_RED_SIGNED=1`) | 406.1 | +35% |

The kernel already sits at the best shape of every option tried; the sm_86-optimal shapes are worse
on sm_75.

## Nonces per thread (ILP) also refuted

`QTC_NPT` computes N independent hashes back-to-back per thread (registers 96/112/114/116):

| NPT | cycles/hash (median) |
|---|---|
| 1 | 300.5 |
| 2 | 390.2 |
| 3 | 472.8 |
| 4 | 544.1 |

More per-thread ILP makes it strictly worse, so the kernel is **issue/throughput-bound, not
latency-bound**: there are no idle multiply-pipe slots to fill. Adding work per thread only lowers
occupancy. Dual-nonce interleaving (the official's -6.5% on sm_86) is likewise not a win on sm_75.

## SASS shape

`qpow_cuda_scan` is 2176 static instructions (both round loops are rolled, `#pragma unroll 1`);
dynamic ~25.6k per hash. Static mix: IMAD 913, IADD3 896, SEL 138, LEA 108, SHF 54, ISETP 16,
LDC 13. IMAD (multiply pipe) and IADD3 (ALU) are roughly balanced and the multiply pipe is saturated;
only 13 constant-bank loads (EPS/MDS are already in `__constant__`).

## Lane-parallel layout also refuted

A second prototype (`tools/cuda-quantus/q2-harness/qlane.cu`) puts one field element per thread: a
warp holds 2 hashes in two 16-lane groups (12 active lanes each), and every coupling (M4 blocks,
external diffusion, internal row sum) goes through `__shfl_sync`. Verified against the host
reference (`qlane test` passes). Fair throughput on the CMP, 256-thread blocks vs warp blocks:

| Layout | Mperm/s | regs | blocks/SM |
|---|---|---|---|
| 1 state / thread | **327.8** | 70 | 3 |
| lane-parallel (thread per lane) | 84.2 | 42 | 16 |

Lane-parallel is **3.9x slower**. The shuffle traffic (7 per external round, 4 per internal round)
plus the idle lanes dominate; the lower register pressure does not compensate. Occupancy is not the
limiter (already known), and the shuffle overhead is the same every round, so this direction is dead
on Turing.

## Where this leaves the 1.34x / 1.38x-per-clock gap to RGminer

Everything tried is now refuted on the real card: every arithmetic shape (signed/mulhi/split/fused/
sum-split), per-thread ILP (`QTC_NPT`), and the lane-parallel layout. The kernel is issue-bound with
a saturated multiply pipe and the 1-state/thread layout is the fastest of all of them.

Closing the gap needs **~28% fewer instructions per hash** (our ~25.6k vs PeakMiner ~17.7k) with the
same layout. Nothing in the open references (official quantus-miner, KanQ, sppark) or the shapes we
tried delivers that on sm_75. The remaining plausible levers are all research-scale and none is
validated:

- a cheaper 64x64->128 product + reduction than the 3-mul.wide + 6-op `red128` we and KanQ use;
- a *partially reduced* state carried across rounds (`sppark` GL64_PARTIALLY_REDUCED) that removes
  some per-round reductions, which no open source implements for this exact round structure;
- a different lane count per thread than 1 or 12 (e.g. splitting the 12 lanes over 2-3 threads) --
  the lane-parallel result suggests any cross-thread coupling via shuffles is too expensive on
  Turing to pay off.

## How to reproduce

```
# on the box, in tools/cuda-quantus/q2-harness
nvcc -O3 -arch=sm_75 -std=c++17 -I. -DQTC_SUB=1 -DQTC_SQR3=2 qtc.cu -o q_base
nvcc -O3 -arch=sm_75 -std=c++17 -I. -DQTC_SUB=1 -DQTC_SQR3=2 -DQTC_GFMA2=2 qtc.cu -o q_gfma2
./q_base test && ./q_gfma2 test
./q_base cyc 256 ; ./q_gfma2 cyc 256     # interleave several rounds
```
