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

## Where this leaves the 1.34x / 1.38x-per-clock gap to RGminer

The planned restructurings are exhausted and refuted on sm_75. Because the kernel is issue-bound
with a saturated multiply pipe, closing the gap needs **~28% fewer instructions per hash** (our
~25.6k vs PeakMiner ~17.7k), which the shape changes above do not deliver. Remaining untested
direction: a **lane-parallel** layout (one field element / one lane per thread, 12 threads or a warp
per hash, row sum via shuffles) instead of 1 hash/thread in 96-118 registers. That is the opposite
direction from NPT and is the only structural change not yet tried. It is a research-scale rewrite.

## How to reproduce

```
# on the box, in tools/cuda-quantus/q2-harness
nvcc -O3 -arch=sm_75 -std=c++17 -I. -DQTC_SUB=1 -DQTC_SQR3=2 qtc.cu -o q_base
nvcc -O3 -arch=sm_75 -std=c++17 -I. -DQTC_SUB=1 -DQTC_SQR3=2 -DQTC_GFMA2=2 qtc.cu -o q_gfma2
./q_base test && ./q_gfma2 test
./q_base cyc 256 ; ./q_gfma2 cyc 256     # interleave several rounds
```
