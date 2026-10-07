# RGminer Quantus kernel extraction (sm_75)

Companion to `qpow_cuda_peakminer_extract.md`. Artifacts in `tools/cuda-quantus/peak-extract/`:
`rg_quantus_sm75.cubin`, `rg_quantus_sm75.sass.gz`.

## How it was obtained

RGminer runs in Docker (`palmatorro/rgminer:1.1.2`). The launcher decrypts the backend and writes
it to `/root/.cache/rgminer-dual/<sha256>/rgminer.cuda12.turing`, then **unlinks it** (the file is
gone, so `docker cp` of the path fails). The running process still maps it, so
`sudo cp /proc/<pid>/exe rg_turing.bin` recovers the backend. It is a normal dynamically-linked
stripped ELF (28 MB) with a `.nv_fatbin` section; there is no anti-`ptrace` (unlike PeakMiner).

The loaded module was also dumped from process memory (regions with `.nv.info`), which is where the
kernel symbols are readable.

## Kernels found (RGminer names are NOT obfuscated)

| function | SASS | what |
|---|---|---|
| `_kernel_quantus_0` | 19712 | 15149 are NOP — real code ~4.5k |
| `quantus_search_classic` | 31424 | search kernel (classic) |
| `quantus_probe` | 51432 | probe/bench kernel |
| `quantus_reduce_probe` | 32 | isolated field reduction |
| `quantus_mulacc_probe` | 40 | isolated multiply-accumulate |
| `quantus_arithmetic` | 48 | two isolated field multiplies |

The author shipped their own microbenchmarks (`*_probe`, `quantus_arithmetic`) — the same exercise we
have been doing — which makes their field arithmetic readable in isolation.

## Decoded primitives

Reduce (`quantus_reduce_probe`), 128-bit -> 64-bit mod p:
```
IMAD.HI.U32 R8, P0, R7, -0x1, R4 ;   // R8 = hi(R7*EPS + R4); EPS = 0xFFFFFFFF as -0x1
IMAD.IADD   R4, R4, 0x1, -R7 ;       // R4 = r0 + 1 - r2
IADD3.X     R5, RZ, R6, RZ, P0, !PT ;
IADD3       R4, P1, R4, -R5, RZ ;
IADD3.X     R5, RZ, -0x1, R8, P1, P0 ;
ISETP.GT / SEL / IADD3 ...           // canonicalize: if result >= p, subtract p
```
So RG uses `IMAD.HI.U32 x, -0x1` for the `x*EPS` high word (like PeakMiner) plus an explicit
canonical subtract, not the `mad.lo.cc`/`madc.hi.cc` chain we use.

Multiply (`quantus_arithmetic`, storing two products): schoolbook
`p00 = a_lo*b_lo`, `p01 = a_lo*b_hi`, `p10 = a_hi*b_lo` (accumulated into `p01`),
`p11 = a_hi*b_hi` — 4x `IMAD.WIDE.U32`, then the reduce above with one more `IMAD.WIDE.U32 ...., -0x1`.
Our `mul128` is Karatsuba (3 wide) + a 2-IMAD reduce; theirs is schoolbook (4 wide) + wide-EPS reduce.

Search kernel `quantus_search_classic`: `IMAD.WIDE.U32 8833` (+1500 `.X`), `IADD3 5822`+`.X 6651`,
`IMAD.X 3126`, `IMAD.HI.U32 112`, `SHF.R.U32.HI 738`, `LEA 870`. The batching differs from ours, so
the per-hash wide count is not directly comparable.

## Rate note

See "Corrected rates" below: RGminer measures 337 MH/s (Quantus) / 82.4 TH/s (Pearl) on an idle box.
An earlier reading of ~154 MH/s was caused by a leftover `ncu` process holding the GPU.

## RGminer Pearl (same backend exe)

Running the container with `--algo pearl` loads the same `rgminer.cuda12.turing` backend (same
BuildID) but different modules. Measured **82.4 TH/s** on the CMP (PeakMiner Pearl: 83.5 TH/s — a
tie).

RGminer's Pearl kernels (namespace `pearl::cuda`, mangled but recognizable):
- `pearl::cuda::sm75::fingerprint_u32_words_kernel` (an sm_75-specific kernel)
- `pearl::cuda::proof::*` (q1scg2kf9d2wvj, qezr980ztbgub, q19b2hu17kf856, q2fgoogsut993e, q2xsbn3632lizl)
- `pearl::cuda::common::{generate_uniform_kernel, generate_permutation_kernel,
  generate_permutation_digest_kernel, transpose_b_to_bt_kernel, derive_b_noise_seed_kernel, ...}`
- `pearl::cuda::sm80_plus::detail::generate_*`

Opcode evidence: RGminer's Pearl uses the Turing **int8 tensor cores** —
`IMMA.8816.S8.S8` (7168 in the dumped kernels) plus `LDSM.16.M88.4` (96). PeakMiner's (equally
fast) Pearl build does not use IMMA in the kernels we captured.

Artifacts: `rg_pearl_reg_2_60.*`, `rg_pearl_reg_3_162.*`, `rg_pearl_reg_6_281.*`.

## Corrected rates (clean GPU)

Earlier in this session the RG container reported 154 MH/s / 40 TH/s; that was an artifact of a
leaked `ncu` process from an aborted profiling run holding the GPU. With the box idle the measured
rates on the CMP 50HX are:

| miner | Quantus | Pearl |
|---|---|---|
| ours (this branch) | 279.3 MH/s | — |
| RGminer 1.1.2 | 337 MH/s | 82.4 TH/s |
| PeakMiner 2.17.2 | 347.5 MH/s | 83.5 TH/s |

So both closed miners are 1.21-1.24x our Quantus kernel; the earlier 3x-looking gap was the leak.

## Takeaway

Both closed miners are now extracted (PeakMiner by memory dump past its anti-debug, RGminer via
`/proc/pid/exe`), and both do the same Goldilocks field work with a different multiply/reduce
formulation than ours: `IMAD.HI`/`IMAD.WIDE` with EPS as the immediate `-0x1`, and schoolbook
products. This is the concrete material to port; none of the black-box shape variants we tried
reproduced the gain.
