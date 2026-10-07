# Quantus CUDA (Turing sm_75) carry-chain experiments: results

Branch: `perf/qpow-cuda-turing`. Box: CMP 50HX, sm_75, driver 610.43.03, CUDA 13.3, ~1500 MHz
effective under the 225 W cap. Harness: `tools/cuda-quantus/q2-harness/qtc.cu`, mode `cyc` (SM
cycles per hash via clock64), interleaved A/B, 6 rounds.

## What the open references suggested (and how it held up on sm_75)

The plan (`docs/qpow_cuda_turing_carry_chain_plan.md`) rested on three shape changes that the
official `quantus-miner`, KanQ and the AMD backend converged on. On sm_75 all three are refuted:

| Variant | Switch | Result on sm_75 | Note |
|---|---|---|---|
| signed reduce (KanQ `fold128_lazy32` / official `combined_reduce`) | `QTC_RED_SIGNED=1` | **worse** (~+8% cycles) and fails the exact field/hash test | official reported +5% on sm_86 |
| native `__umul64hi` product (official v5) | `QTC_MULHI=1` | **worse** (~+4% cycles) | official +7% on samsung 3060 Ti (sm_86) |
| split internal product (sum + separate mul/add128) | `QTC_INT_SPLIT=1` | **worse** (~+4% cycles) | official rejected the fused form on sm_86; on sm_75 the fused form wins |

Conclusion: the carry-chain structure that is optimal on Ampere (sm_86) is **not** optimal on Turing
(sm_75). The two microarchitectures' integer pipes differ; results do not transfer between them.

## What actually helps on sm_75

Baseline = the miner's own shape: `QTC_SUB=1` (nonce line) + `QTC_SQR3=2` (funnel-shift squaring),
`QTC_INT_W=1`, `QTC_GFMA2=0` (fused `gfma_w`). Interleaved, cyc 256, 6 rounds:

| Variant | cycles/hash (median) | delta |
|---|---|---|
| base | 652.0 | — |
| **`QTC_GFMA2=2` (`gfma_y`, even/odd product chains)** | **642.2** | **-1.5%** (consistent, all 6 rounds) |
| `QTC_INT_SPLIT=1` | 677.5 | +3.9% |
| `QTC_MULHI=1` | 681 | +4.4% |

`gfma_y` computes the even chain (a0*b0 + w, a1*b1 + w.t) and the odd chain (a0*b1 + a1*b0) on
separate natural register pairs and merges at word 1, instead of the fused single mad chain. It is
the gECC idea from the ledger (Q7: "-3.5% SASS, +0.8% per clock, not integrated"); on sm_75 it
measures -1.5% cycles, reproducibly. Integrated into the kernel (see below).

Everything else tried in the ledger (unroll depth, peephole, occupancy, maxrregcount, 24-bit limbs,
ALU-only reduction) stays rejected.

## Where this leaves the 1.34x / 1.38x-per-clock gap to RGminer

The planned restructurings are exhausted; they do not close the gap on sm_75. The remaining gap is
not instruction count, not occupancy, not icache. Open item from `50-next-steps.md` rank 5 still
stands: microbenchmark the exact IMAD.WIDE + carry-predicate constructs on sm_75 to find whether the
limit is latency or pipe width, then restructure accordingly (e.g. `sppark` GL64_PARTIALLY_REDUCED
interleaving, register-bank-aware operand order).

## How to reproduce

```
# on the box, in tools/cuda-quantus/q2-harness
nvcc -O3 -arch=sm_75 -std=c++17 -I. -DQTC_SUB=1 -DQTC_SQR3=2 qtc.cu -o q_base
nvcc -O3 -arch=sm_75 -std=c++17 -I. -DQTC_SUB=1 -DQTC_SQR3=2 -DQTC_GFMA2=2 qtc.cu -o q_gfma2
./q_base test && ./q_gfma2 test
./q_base cyc 256 ; ./q_gfma2 cyc 256     # interleave several rounds
```
