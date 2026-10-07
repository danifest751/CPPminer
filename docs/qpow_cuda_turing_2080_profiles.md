# Quantus/Pearl vs closed miners on sm_75 — profiling and lever experiments (2080 Ti)

Platform: RTX 2080 Ti (sm_75, ncu-capable), driver 615.71.09, CUDA 13.4, Nsight Compute 2026.3.
Rates: ours Quantus 312 MH/s, RGminer ~366-373, PeakMiner ~347 (CMP). Pearl: ours ~79 TMAC/s,
RG ~93. Both closed miners are ~1.18-1.24x, and the same ~1.2x gap shows in registers/occupancy terms.

## Our Quantus kernel profile (ncu)

Latency-bound: Compute (SM) 58.2%, memory 0.02%, issue slots busy 50.9%, ALU the busiest pipe 58.2%,
Executed IPC 2.07, Active warps/scheduler 4.80, Eligible 1.37, no-eligible 48.3%, warp cycles per
issued instruction 9.29, REG 70, no spills. ncu estimates ~42% local speedup available from stalls.

Our Pearl kernel (`cp_turing_scan_kernel`): tensor-bound 85.2% (Tensor pipe), memory 53%, IPC 1.29,
2.0 active warps/scheduler, REG 225, 56 KB smem → 1 block/SM.

## Levers tried on the real kernel (2080 Ti and CMP)

| lever | harness (qtc) | real miner |
|---|---|---|
| `__launch_bounds__(256,4)` (64 reg) | 266 vs 306 cyc (-13%) | **worse**: 303 vs 312 (2080), 269 vs 279 (CMP) |
| `k_tpb=128` (more resident warps) | 285 vs 306 cyc (-7%) | **worse**: 308 vs 312 (2080), 274 vs 279 (CMP) |
| `minb` 5/6/8 | 286 (worse than 4) | — |
| ptxas flags (-O3, expensive-opt, dlcm, edv, -O2) | all 306 (no change) | — |
| maxrregcount 64/48 | ignored (stayed 96) | — |

Key point: the occupancy wins seen in the standalone harness (whose kernel is a 96-register variant)
do **not** transfer to the miner kernel, which already sits at its occupancy optimum (70 reg, 3
blocks/SM). Forcing more blocks or smaller blocks adds spills / reduces work-per-thread and loses.

## Cross-check: RG/PeakMiner fields

`cuobjdump -res-usage`: RG `quantus_search_classic` REG 78, `quantus_probe` REG 80,
`_kernel_quantus_0` REG 64; RG Pearl main kernel REG ~210. Ours: Quantus REG 70, Pearl REG 225. So the
closed miners win at **similar register/occupancy** — the edge is instruction scheduling / IPC, not
resources.

## Toolchain

ptxas version test blocked: the pip `nvidia-cuda-nvcc-cu12` ships only `ptxas` 12.9 (no nvcc), and
nvcc 13.4 emits PTX `.version 9.4` which ptxas 12.9 rejects. The handoff notes sm_75 SASS is identical
across CUDA 12.x/13.x anyway.

## Conclusion

Every source-level lever available to us on the Quantus kernel (field-op shape, ILP, occupancy, block
size, launch bounds, ptxas flags) is neutral or worse. The ~18-24% gap to RG/PeakMiner is a
whole-kernel codegen/schedule difference at comparable resources. Closing it to within 2-3% would
require reproducing their compiled schedule (a rewrite), not a parameter change. Pearl is
tensor-core-bound at 85% and likewise ~18% behind, with the same character.

The extraction artifacts, cubins, SASS and these profiles are in `extracted-miners/`.
