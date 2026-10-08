# Quantus CUDA (sm_75): post-ptxas SASS pass with dual-carry IADD3 — results

Branch `perf/qpow-cuda-turing`, 2026-10-08. Tooling in `tools/cuda-quantus/sass-fuse/`. Card: CMP 50HX.
Kernel: the production device code of `perf/qpow-cuda-u128` (`cp_qpow_cuda.cu`, 2152 SASS, 74 regs in
the standalone bench), measured with the `qb.cu` harness in SM cycles per hash (clock64), checksum of
1M outputs must equal `e2e9f81b881715fc`.

## Why this was tried

ncu on a 2080 Ti (the CMP blocks profiling) says the kernel is issue-bound on the math pipes, not
latency-bound: stalls per issue are math_pipe_throttle 2.27, dispatch_stall 1.85, not_selected 1.62,
wait 1.40, no_instruction 0.72 (icache). ptxas never emits the 3-input `IADD3` with two carry-out
predicates for our 96-bit lazy sums (zero in the kernel), although sm_75 has it; five source forms
(PTX `add.cc` chains, `__int128`, plain C) were tried and only `(u64)a0 + b0 + c0` gets a dual-carry
low word. So the fusion was done after ptxas, on the cuasm text, with CuAssembler.

## sm_75 facts measured on the way (all needed for a correct pass)

- `IADD3 Rd, P0, P1, a, b, c`: P0 = (carry >= 1), P1 = (carry >= 2), so P0 + P1 is the carry count
  and `IADD3.X ..., P0, P1` consumes both with weight 1 (`t_dual.cu`).
- Register file: at most two operand reads per bank (bank = register index parity) per instruction.
  A third same-bank read silently returns another operand's value (`IADD3 R13, P0, P1, R6, R6, R2`
  computed a + c + c). ptxas never emits such an instruction; a pass must check it.
- Dependent-issue distances ptxas itself uses (min over the kernel): same pipe 4 cycles, across the
  ALU/FMA pipes 5, `IMAD.WIDE` -> `IMAD.WIDE` 6. A pass that deletes or moves instructions must re-derive
  the stall counts with this table; with a flat 4 the fused kernel hashed wrong (nondeterministically).
- Encodings: the X form with two carry-outs and two carry-ins was learned from nvdisasm-verified
  examples (`gen_examples.py`); fields in the high word: Pin2 bits 13-15 (negate bit 16), Pout1 17-19,
  Pout2 20-22, Pin1 23-25 (negate 26).

## Results (interleaved x3, miner stopped, cold card ~1935 MHz)

| variant | change | cycles/hash |
|---|---|---|
| base | ptxas output | 341.0 |
| f4 | 21 fused IADD3 pairs (both placements), checksum OK | **338.2 (-0.8%)** |
| s1000 | f4 + 134 `SEL` carry materializations moved to `IMAD.X` | 339.2 |

Only 21 of the ~1000 candidate pairs are fusable: 232 are blocked because the intermediate sum is
used twice (ptxas shares `t01`, `t23`, `t0123` across the M4 outputs), 53 because it is read between,
42 because the operands are overwritten between the two adds, 13 by the bank rule, 11 by capacity.
Moving carry materializations onto the FMA pipe does not help (slightly worse), consistent with the
earlier source-level finding that ptxas's pipe balance is already right.

## Follow-up on the fast-reduction kernel (`CP_QPOW_WRED_FAST`, 1960 SASS)

- **Scoreboard waits.** On this kernel two placement-B fusions (fused instruction hoisted to the
  first add's slot) hashed nondeterministically: the hoisted read of the second add's operand moved
  above a scoreboard wait for an `LDC.64` result. The pass now refuses placement B when the second
  add or anything between the two carries a wait mask. Third hardware rule after the carry and
  register-bank ones.
- **CuAssembler encodings.** The fast kernel uses an `IMAD.WIDE.U32 ..., R.reuse` accumulator form
  that was missing from the repository; even the unmodified listing failed to assemble until
  `learn.py` was run on the kernel's own `cuobjdump -sass` output. Run it on every new baseline.
- **Result.** 18 fusions, checksum stable over 5 runs, 328.0 -> 325.1 cycles/hash (-0.9%).
- **SEL folding is not possible.** `sel_scan.py` classifies the 99 carry materializations
  (`SEL R, RZ, 0x1, !P`): about 65 feed an `IADD3.X R, A, RZ, Rsel, Pa, Pb` that already uses both
  carry-in slots (the SEL is a third carry), 14 feed `IMAD.WIDE` as the `t` of a lazy sum, and the
  remaining `IMAD.X` consumers mostly have the predicate overwritten before the use. About three are
  foldable, so this lever is closed.

## Hand-written S-box microbenchmark (`mb_sbox.cu`, `gen_sbox.py`, `run_sbox.sh`)

Question: can a hand-scheduled SASS S-box beat ptxas by the ~25% per clock that separates us from
the closed miners? `k_ptx` runs the production `sbox()` on 3 independent chains per thread; `k_hand`
has xor placeholders that `gen_sbox.py` replaces with generated SASS, list-scheduled with ptxas's own
latencies. Results are checked against a host `x^7 mod p` (canonical), speed is clock64 S-boxes per
SM clock, 3 blocks x 256 threads per SM for both.

Hand arithmetic (all checked exact on 65536 values):
- product: even/odd chains, the odd chain's carry and the word-2 carry enter one `IADD3.X` as two
  carry-ins (no SEL): 4 WIDE + 3 ALU;
- squaring: the doubled middle product is added by one 3-input `IADD3` with two carry-outs and one
  `IADD3.X` with two carry-ins and two carry-outs (no shifts): 3 WIDE + 3 ALU;
- reduction: `z0 = T0 + ~w3 + !c` and `z1 = T1 - 1 + c + borrow`: 1 WIDE + 2 ALU (no IMAD.X).

| op (per SM clock) | ptxas | hand |
|---|---|---|
| squaring | 5.85 | 5.93 |
| product | 5.09 | 4.87 |
| x^3 | 2.74 | 2.83 |
| **x^7** (38 vs ~48 instructions) | **1.406** | **1.444 (+2.7%)** |

21% fewer instructions buy 2.7%. Both versions have the same 18 IMAD.WIDE per S-box (14 products +
4 reductions, the floor for 32-bit limbs), and those take about 60% of the time: the cost model
(WIDE ~2.9 dispatch cycles, no ALU co-issue, ALU ~1.7) predicts the measured 88 scheduler cycles
per warp-S-box. Scheduling and instruction selection are therefore not where a 25% gap can come
from; it would need fewer wide products per S-box, i.e. different arithmetic.

Hardware/tooling rules learned while building it:
- the kernel's register count lives in the top byte of the text section's `sh_info`
  (`.__section_info 0x12000011` = 18 registers) as well as in `SHI_REGISTERS`; CuAssembler uses the
  former;
- the declared count needs slack above the highest register used: R62 faulted with 64 declared,
  R64 with 66, R64 ran with 72 (illegal instruction otherwise);
- a loop's `ISETP` -> `BRA` needs ~12 cycles; when the splice left them adjacent the branch read the
  stale predicate and the loop ran once;
- `IADD3.X` with two carry-outs and two carry-ins, `~R` operands and negated carry-ins (`!P`) are
  all legal on sm_75 even though ptxas never emits some of them.

## Conclusion

The dual-carry peephole is correct and reusable but worth under 1% on this kernel; the sums would
have to be re-associated at the source level into single-use 3-input trees for it to matter, and
ptxas re-associates them back. Not integrated into the build (the cubin swap in `hackbuild2.sh` is a
bench-only flow).

## Reproduce (inside the CUDA container, `/w/bench/sass2`, `PYTHONPATH=/w/bench/CuAssembler`)

```
awk '/^typedef uint32_t u32;/{on=1} /^\/\* -+ host side/{on=0} on' cp_qpow_cuda.cu > qk.inc
./hackbuild2.sh q_base - -O3 -arch=sm_75 -std=c++17          # q_base.orig.cubin
python3 $CUASM/bin/cuasm.py q_base.orig.cubin -o base.cuasm
python3 gen_examples.py ex.sass && python3 learn.py ex.sass   # once: dual-carry IADD3.X encodings
./run_fuse.sh f4                                              # fuse -> assemble -> build -> checksum
```

## Other inputs to ptxas (same production kernel, CMP, cycles/hash, checksum identical everywhere)

| front end / ptxas | SASS | regs | cycles/hash |
|---|---|---|---|
| nvcc 12.6 (production) | 1960 | 74 | 328.0 |
| clang 18 (LLVM NVPTX PTX) -O2 / -O3 -> ptxas 12.6 | 1952 | 76 | 329.3 / 329.2 |
| nvcc + ptxas 11.8 | 1960 | 74 | 328.0 |
| nvcc + ptxas 11.0.3 (carries via IMAD.X, 4 SEL instead of 99) | 1960 | 74 | 326.5 |

All within +-0.5%, as the hand-SASS bound predicts: the kernel's arithmetic is inline PTX, so the
front end hardly changes the SASS, and no schedule removes the 18 IMAD.WIDE per S-box. CUDA 10.x
images are no longer on Docker Hub. Script: `tools/cuda-quantus/sass-fuse/fe_build.sh`.

## Open-source GPU Goldilocks vs ours (same S-box microbenchmark, CMP 50HX)

Built with `-DWITH_OSS -Iossh --expt-relaxed-constexpr`, where `ossh/` holds the unmodified headers
of era-boojum-cuda `native/` (MIT/Apache-2.0: goldilocks, common, carry_chain, memory, ptx .cuh) and
sppark's `gl64_t.cuh` as vendored in Polygon's goldilocks repo (file header Apache-2.0; the repo
itself is AGPL, so nothing from it should be copied into the miner). Every variant matches the host
`x^7 mod p` on 65536 values.

| S-box implementation | S-boxes per SM clock |
|---|---|
| **ours** (production `sbox()`) | **1.41** |
| sppark `gl64_t`, `GL64_PARTIALLY_REDUCED` | 0.99 |
| sppark `gl64_t` | 0.94 |
| era-boojum-cuda (u64 mul_lo/mul_hi, ALU reduction) | 0.90 |
| era-boojum-cuda, lazy 96-bit `field<3>` state as in its Poseidon2 | 0.89 |

Our arithmetic is 1.4-1.6x faster than the public GPU state of the art on Turing. ICICLE's public
tree has no CUDA field backend (only CPU and PQC); its Goldilocks reduction is the same identity
with branches. One structural difference worth knowing: Boojum's Poseidon2 uses an internal
diagonal of the form 1 + 2^k, so its internal products are shifts. Quantus fixes Plonky3's random
64-bit diagonal, so that trick changes the hash and is not available.

## Search for less known open miners (2026-10-08)

GitHub code search for the first internal-diagonal constant `c3b6c08e23ba9300` (unique to width-12
Goldilocks Poseidon2 with Plonky3's parameters) in `.cu/.cuh/.cl/.metal`, plus repository search on
GitHub, GitLab, Codeberg and Gitee. Everything with source falls into the families already measured:

| source | arithmetic | status |
|---|---|---|
| Yose144/Zion-v3.0.0 (MIT) `miner/csrc/cuda,opencl/poseidon2_kernel` | port of the official quantus-miner "G2" kernel: same reduce128 as ours, `__umul64hi` product (+4% cycles here) | nothing new |
| jshojan/quantus-5060-miner, emanwrxsti/f4pool-quantus-miner (Apache-2.0) | forks of official `engine-cuda/mining.cu`; f4pool identical, 5060 fuses one mul+add | nothing new |
| okx/zeknox (Apache-2.0), 0xPolygonHermez/pil2-proofman (Apache/MIT, has a PoW grinding kernel) | sppark `gl64_t` | 0.94-0.99 vs our 1.41 |
| SihaoLiu/Lzvm, dloghin/plonky3-gpu | plain `__int128` / example code | slower by construction |
| benjamin920101/quantus-ascend-miner | Huawei Ascend NPU | other hardware |
| 0xMiden/miden-signature, elliottech/lighter-prover | Metal (Apple) | other hardware |

Binary-only: kryptex/krig-miner, heishiqing/Vminer, 8gkgcom/QuantusMiner, amdpowgit/quan-amd-miner,
DankMiner, FearMiner, longcipher/quantus-miner-perf, gitlab home-group2050535/quantus (custom
SRBMiner and quanpool HiveOS archives). No public code carries the closed miners' advantage.
