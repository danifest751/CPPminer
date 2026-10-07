# PeakMiner Quantus kernel extraction (sm_75) — findings

Goal: understand how PeakMiner reaches ~1.24x our rate on the CMP 50HX, to port the technique
into our own kernel. Artifacts in `tools/cuda-quantus/peak-extract/`.

## Measured rates (CMP 50HX, exclusive GPU, 225 W, ~1770 MHz)

| Miner | Quantus rate | note |
|---|---|---|
| ours (CUDA, this branch) | 279.3 MH/s | `--mock`, device 0 |
| RGminer 1.1.2 | ~372 MH/s | handoff |
| **PeakMiner 2.17.2** | **347.5 MH/s** | live pool, 1 accepted share, 0 invalid, eff 100% |

So PeakMiner is 1.24x our kernel on the same card/clock.

## How the kernel was obtained

PeakMiner is a static stripped Rust binary (`sstrip`, no section headers). It calls
`prctl(PR_SET_DUMPABLE, 0)`: under `strace` it self-inspects `/proc/self/maps` and never brings up
CUDA (no socket, no `/dev/nvidia`, no `libcuda`), and `/proc/<pid>/maps` is unreadable even for the
owning user. Run as root (sudo) the process memory is readable; scanning the RW anon mappings for
`\x7fELF` + `.nv.info` yields the CUDA modules the driver has loaded, including the fatbin with the
Quantus kernels (`cubin.sm_75.cubin`).

Artifacts:
- `peakm_quantus_sm75.cubin` — the sm_75 fatbin carved from process memory.
- `peakm_quantus_sm75.sass` — `cuobjdump -sass` of it.

The kernels (C++-mangled, obfuscated names in namespace `qk`):
- `qk::z188d3cec54ef5d3f(const u8* header, const u8* nonce, u8* out)` — 21856 SASS: the full Quantus
  hash (both permutations), fully unrolled.
- `qk::z5300d4a04b44c629(...)` — 2568 SASS: the scan loop + HitBuf.

All 12 `MDS_DIAG` constants are present verbatim in the cubin, so it is the same Poseidon2/Goldilocks
Quantus algorithm; only the implementation differs.

## What is different in the implementation

Opcode histogram of the 21856-instruction hash function:

| opcode | count |
|---|---|
| IADD3 / IADD3.X | 4689 / 4824 |
| IMAD.WIDE.U32 / IMAD.WIDE.U32.X | 2979 / 800 |
| IMAD.HI.U32 | 1988 |
| IMAD.MOV(.U32) | 2183 |
| IMAD.X | 1422 |
| IMAD.IADD | 998 |
| SEL | 599 |
| LDC.64 | 125 |
| SHF.* | ~130 |

Observations vs our kernel:
1. **~3.8k IMAD.WIDE per hash vs our ~7.1k.** The field multiply build uses roughly half the wide
   32x32->64 products. The `.X` forms (`IMAD.WIDE.U32.X Rd, Ra, Rb, Rc, Pn`) accumulate one
   partial product into a 64-bit pair with the predicate carry, so a 128-bit product is assembled
   from a few wide MACs with IADD3 carry instead of the extra wide multiplies we spend.
2. **EPS is an immediate, not a constant-bank load.** The SASS contains
   `IMAD.HI.U32 R19, P5, R12, -0x1, R42` — the `-0x1` immediate is `0xFFFFFFFF = EPS`. Our kernel
   deliberately moved EPS to `__constant__` to avoid `IMAD.HI`; PeakMiner uses `IMAD.HI` (1988 of
   them) on Turing and wins anyway.
3. **Everthing is unrolled** (36 BRA in 21856 instructions). Our kernel keeps both round loops
   rolled (`#pragma unroll 1`). PeakMiner's code overflows the icache and is still faster, i.e. the
   ILP from static scheduling outweighs the icache cost here.
4. Very heavy `IADD3` (9.5k) with predicate carry chains — the carry work is spread across ALU
   slots rather than concentrated in the multiply pipe.

## Implication

The 1.24x is not algorithmic and not layout (we already refuted layout/ILP changes on our kernel).
It is a different **field-multiply/reduction codegen**: half the wide multiplies, immediate EPS via
`IMAD.HI`, and a fully-unrolled schedule that fills both pipes. That is the thing to port.

## Attempts to port the technique (all measured on the CMP, exclusive GPU)

Decoded from the SASS: PeakMiner computes `x*EPS` as `(x<<32)-x` using `IMAD.HI.U32 r,-0x1`
(gives `x-1`) plus `IMAD.X r,r,0x1,Rb,P` to rebuild the word, i.e. no wide multiply for the
reduction. Implemented as `QTC_RED_SHIFT` in the harness:

| variant (base = SUB=1,SQR3=2,GFMA2=2) | cycles/hash |
|---|---|
| base (constant-EPS `mad.lo`/`madc.hi` reduction) | 300.6 |
| `QTC_RED_SHIFT` (shift/borrow EPS, like PeakMiner) | 311.5 (+3.6%) |
| `QTC_RED_ALU` (sub-based, existing) | 311.0 (+3.5%) |

So the EPS formulation is not the lever on its own; our constant-EPS form stays best.

Full unrolling (`QTC_IUNR=22`) is also worse on our kernel: 362.6 vs 300.7 cycles. PeakMiner is
fully unrolled and faster, so their unrolled schedule does not blow register pressure / icache the
way ours does — the difference is inside their code shape, not the unroll flag.

## Conclusion

PeakMiner is ~1.24x our kernel and we now have its exact sm_75 SASS, but none of the field-op
shapes we can express in our kernel reproduces the gain: sign/mulhi/split/fused/ALU/shift
reductions are all neutral or worse, NPT ILP is worse, lane-parallel is 3.9x worse, full unroll is
worse. Their advantage lives in a specific codegen (roughly 2.5 IMAD.WIDE per field multiply vs our
~3, EPS folded as an immediate, and an unrolled schedule that keeps both pipes full) that would have
to be ported op-by-op from the disassembly. That is a rewrite of the field layer, not a tweak.

## Next step (if pursued)

Isolate one full field-multiply + reduction block from `peakm_quantus_sm75.sass` (the repeating
`IMAD.WIDE[.X]` + `IMAD.HI -0x1` + `IADD3` pattern spans ~150-250 instructions because several
independent multiplies are interleaved), express it as one `gmul` in the harness, and A/B. Expected
effort: high; payoff: the ~20% instruction-count gap if the schedule transfers.
