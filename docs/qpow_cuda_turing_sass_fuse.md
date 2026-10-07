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
