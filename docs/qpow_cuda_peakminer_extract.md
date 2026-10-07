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

## Next step

Reconstruct the field multiply + reduction block from the extracted SASS (the repeating IMAD.WIDE /
IMAD.WIDE.U32.X / IADD3 / IMAD.HI pattern) and implement it as a variant in
`tools/cuda-quantus/q2-harness/qtc.cu`, then A/B on the CMP against the current `gmul`/`red128`.
