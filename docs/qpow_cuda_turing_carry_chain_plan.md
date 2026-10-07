# Quantus CUDA (Turing sm_75): carry-chain optimization plan

Branch: `perf/qpow-cuda-turing` (from `feat/amd-r9700 @ ca259ff`).
Target: CMP 50HX, Quantus, currently ~278 MH/s mining vs RGminer 372 (gap 1.34x; 1.38x per clock).
Owner's OC limit: core offset <= +200.

## Where the time goes (measured)

- ~305 SM cycles per hash, ~32.6k "units" per hash (ALU 1, IMAD/MOV ~1.4, IMAD.WIDE ~2).
  Unit budget: internal rounds ~43%, full-round S-boxes ~38%, external layers ~19%.
- All kernel variants have ~7.1k IMAD.WIDE per hash and roughly the same speed. Cutting ~3-4% of
  instructions gave only ~1%. So the limiter is NOT issue width / instruction count / occupancy
  (2 warps per scheduler already saturates) / icache (25 KB costs ~4%). It is the 64-bit
  multiply-accumulate + carry-chain structure.
- `gsqr` measures as MORE expensive than `gmul` (14.4 vs 11.5 SASS, 7.27 vs 7.54 ops/clk), which is
  abnormal and points at the carry structure, not the multiply count.

## The one thing that differs from every open reference

Our internal round fuses the row sum into the mad chain (`gfma_w`, 9 PTX ops in one carry chain)
and our `reduce128` is a hand-written 6-op PTX carry chain. Three independent open sources agree on
a different shape:

| Source | reduce128 | internal round |
|---|---|---|
| ours (`cp_qpow_cuda.cu`) | 6-op PTX carry chain (`mad.lo.cc`/`madc.hi.cc`/`addc`/`sub`) | `gfma_w` (row sum inside mad chain) |
| official `quantus-miner` (`qm_mining.cu`, PR#104) | **`combined_reduce`: signed 64-bit add/sub chain** (+5% on 3080 Ti) | **split**: sum lo/hi, then separate `mul64wide` + `add128_wide` |
| KanQ (`kangq_qpow.cuh`, `kangq_goldilocks32.cuh`) | **`fold128_lazy32`: signed `long long` form** | **split** (`int_round_p`) |
| AMD ours (`qpow_mining.cl`) | `QV_RED=2` signed form already exists | — |

Official note (`qm_vast.md`): "Kept: software-pipelined internal rounds. Rejected: the row sum
riding in `mad.wide.u32` accumulators (-5% on ptxas 12.8, -3% on 13.2, 16 B spill)." That is exactly
our `gfma_w`.

## Plan (ordered by ROI, all A/B via the `qtc.cu` harness `cyc` mode)

Validation gate for every variant: `qtc test` passes (field + hash vs CPU), then `qtc cyc` for
cycles/hash, then `qtc bench` for MH/s. Compare at the 225 W cap or in cycles/hash.

### P1 - reduce128 as a signed / combined chain  (highest confidence)
- Replace the 6-op PTX `red128` with the signed form from KanQ `fold128_lazy32`
  (`X = r0 - r2 - r3`, `Y = r1 + r2 + (X>>32)`, fold `y_hi`), or the official `combined_reduce`.
- Already present in the harness as `QTC_RED_ALU` (that one is the ALU-form, not the signed form);
  add a new switch `QTC_RED_SIGNED=1`.
- Evidence: official +5% on 3080 Ti; KanQ uses it; AMD uses `QV_RED=2`. Mathematically the same
  +/-EPS lazy tolerance, host re-verification unchanged.
- Expected: several percent (reduction runs on every one of ~1470 multiplications per nonce).

### P2 - internal round without `gfma_w` (split mul and sum)
- Replace `gfma_w` with: sum lanes 1..11 in 32-bit halves (IADD3 tree), then `mul64wide(lane,
  c_diag[i])` independently, then `add128_wide`. Matches `qm_mining.cu:298` / `kangq_qpow.cuh:245`.
- Harness switch `QTC_INT_SPLIT=1`.
- Evidence: official rejected our exact shape (-3..5%, spill); two codes use the split.
- Expected: +3..5% plus removal of a spill that costs occupancy.

### P3 - mul64wide via native `__umul64hi`
- Our `mul128` uses 3x `mul.wide.u32` + a `mad` chain. The official moved to native `__umul64hi`
  (v5, +7% on the 3060 Ti). On AMD the analogous 64-bit `mul_hi` path had fewer VALU but spilled;
  on NVIDIA there is no VCC, so it may be a pure win.
- Harness switch `QTC_MULHI=1`: `u64 lo = a*b; u64 hi = __umul64hi(a,b);`.
- Cheap to test; combine with P1/P2.

### P4 - microbench: latency vs throughput of the carry chains
- Extend the Turing microbench (`tools/cuda-quantus/mb-turing-microbench`): 1 chain vs 2/4/8
  independent `mul128 -> red128 -> mul128` chains, back-to-back dependencies, measure cycles.
- Answer decides whether interleaving nonces is worth anything. Note: dual-nonce interleave was
  already rejected on sm_86/sm_89 (official -6.5%), so expectations are low; still worth one run on
  sm_75.

### P5 - GL64_PARTIALLY_REDUCED style (sppark, Apache-2.0)
- `sppark_gl64.cuh` shows the carry fixups can be interleaved with the preceding multiply, folding
  critical paths, provided the multiply result is the 2nd operand of the add (register-bank aware).
- Only pursue if P4 says latency-bound. This is a bigger refactor of the field layer.

## Stop list (already disproven; do not repeat)

- Internal-round unroll depth (0% / -8% icache); full unroll (icache).
- SASS peepholes IMAD.X -> IADD3.X (wrong hashes) / IMAD.MOV -> MOV (crash).
- maxrregcount, occupancy, launch geometry, `__launch_bounds__` min-blocks tuning.
- Carry-free mad.wide multiply, ALU-only / hybrid reduction (slower).
- 24-bit limbs / `mul32` forms.
- `unsigned __int128` (NVRTC rejects without `--device-int128`).
- Removing IMAD.HI from `reduce128` (our reduce has no IMAD.HI; EPS is in `__constant__` already).
- `QV_EPS32` `(x<<32)-x` as 32-bit halves: on AMD LLVM already folds it to a constant mad64 and the
  rewrite is slower; check whether our `sqr128` funnel-shift path is the reason `gsqr` > `gmul`.

## Files

- Kernel: `src/qpow/cuda/cp_qpow_cuda.cu` (worker self-test + resume hook preserved).
- Harness (not yet in the repo): `tools/cuda-quantus/q2-harness/qtc.cu` - standalone verify + bench,
  `test` / `bench SEC BPSM` / `cyc` modes, switches `QTC_RED_ALU`, `QTC_SQR3`, `QTC_INT_W`,
  `QTC_GFMA2`, `QTC_SUB`, `QTC_IUNR`, `QTC_MINB`, `QTC_TPB`.
- Reference (ground truth): `src/qpow/include/qpow/{goldilocks,poseidon2,nonce_line}.hpp`.

## Ground rules

1. Correctness first: `qtc test` (or the miner self-test) must pass before any speed number counts.
2. Compare in cycles/hash at the 225 W cap, or in % of peak at a measured clock.
3. Never connect an ablation build to a real pool.
4. Integrate into `release/fork` only through this branch after the A/B numbers are in.
