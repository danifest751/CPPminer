# CUDA research validation ledger

Started 2026-10-04 on `perf/cuda-incremental-a`, from `08176ba`.
Target: the existing CMP 50HX (sm_75), with the original 225 W limit and
CUDA 12.6. Production remains v0.5-fork.4 throughout this research.

## Ordered experiments

| ID | Question and adaptation | Correctness and measurement | Status |
| --- | --- | --- | --- |
| R1 | Does the measured `.cg` load benefit survive full miner work, and combine with predicated jackpot state? Compare unchanged control, `.cg`, predicated state, and both. | Full proof/mock checks; balanced complete 131072x131072x4096 attempts, preparation included; dense and incremental A. | Complete: combined +2.41% dense / +2.64% incremental in this run. |
| R2 | Should A and B use different cache policies? Test default/default, cg/default, default/cg, and cg/cg, retaining the 128-byte L2 hint. | Every required prefix XOR and final BLAKE3 against INT64/C reference; balanced full-kernel panel timing. | Complete: cg/cg remains best (+1.05% panel). |
| R3 | Does another output-block traversal improve L2 operand reuse? Tune existing CUTLASS identity grouping and an alternative orientation, retaining logical proof coordinates. | Exact milestone/hash checks; balanced full-kernel panels; full miner confirmation for a winning candidate. | Complete: existing group 8 remains best; no new winner. |
| R4 | Can a deeper software pipeline hide loads on sm_75? Use ordinary vector global loads, shared stores and unchanged synchronization semantics; adjust K tile/depth to fit 64 KiB. | Exact checks at all 32 ordered 128-term boundaries; compare actual SASS, registers/shared storage and full kernel timing. | Complete: nine adapted candidates were exact but slower. |
| R5 | Can a smaller independent Tensor Core implementation improve operand/fragment handling? Adapt the minimal CuTe/other Pearl kernel ideas to our exact scattered 8x8 proof geometry. | Independent INT64 prefix/hash oracle, including full-range INT8 and actual noisy inputs; full valid kernel timing. | Complete: exact raw-MMA prototypes lost by roughly 19–21x. |
| R6 | Can a genuinely different native schedule beat ptxas? Start from our cubin, check assembler round-trip, then bounded guarded load/MMA interleavings. | Byte/instruction round-trip evidence, exact prefix/hash oracle for every executed candidate, balanced full-kernel measurements. | Complete: four exact schedules; 256x128 slower, 128x128 within variation. |

This order is authorized as one research suite. Each question is assessed
separately before combining changes. An unsuccessful prototype is recorded,
not enabled in the production miner. Newly discovered implementation constraints
may reduce a search space, but an untested direction is not marked completed.

## Common requirements

- Preserve signed INT8 products, INT32 cumulative values, all 32 milestone XORs,
  the 16-word rotate/XOR state, keyed BLAKE3 and proof coordinates.
- Keep allocation, CPU oracle work and diagnostic output outside kernel timing.
  Time all arithmetic/hash work required by the candidate; no incomplete kernel
  is eligible for adoption.
- Use deterministic zero-target fixed work, warmup and balanced block order.
  Record raw timings and active temperature, clocks and power. Normal Boost is
  part of the measured result; a small mean delta inside repeat variation is not
  a speedup.
- Build in private source/header/dependency copies. Preserve upstream licenses.
  Never patch shared dependency trees or the installed release.
- Bounded server-local runners save private process state and restore the exact
  original binary, args, environment, cwd and log destinations in `finally`.
  Independently verify worker uniqueness and resumed mining.
- No GitHub Actions, rented GPU, upstream PR or experimental release deployment.
  Wallet-bearing logs, credentials and restoration snapshots stay server-local.

## Applicability checks

FP8/approximate quantization is outside the exact arithmetic requirement. Native
INT4 is not a direct replacement for our full INT8 operands. TMA/WGMMA require
newer hardware; sm_75 pipelines use ordinary loads/stores. The current 256x128x64
operand stage uses 24 KiB, so three shared stages would exceed Turing's 64 KiB
limit. R4 must change geometry or use another prefetch arrangement.

Full RL training and a wholesale language/compiler migration are not needed to
test the underlying ideas. R6 uses a bounded native-schedule search rather than
assuming that CuAsmRL's A100 result applies to this CMP50.

## Research sources

- [Turing SASS GEMM study (IPDPS 2020)](https://cse.hkust.edu.hk/~weiwa/papers/yan-ipdps20.pdf)
- [GAS, instruction-order study (PPoPP 2021)](https://www.cse.ust.hk/~weiwa/papers/gas-ppopp21.pdf)
- [CuAsmRL (CGO 2025)](https://arxiv.org/html/2501.08071v1)
- [CuAssembler](https://github.com/cloudcores/CuAssembler)
- [CUTLASS efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
- [Triton grouped GEMM](https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py)
- [Turing INT8/INT4 implementation](https://github.com/Chennesxu/triton-turing/blob/8da2085caa00c79cf7ce99c7927bb72135e36af1/python/tutorials/12-turing-integer-matmul.py)
- [Pearl sm_75 reference](https://github.com/akoyapool/akoya-miner/blob/0e42b2a50b4cb62cb8995b77420226c2182137ce/native/pearl-gemm/csrc/portable/transcript_gemm_sm75.cu)
- [NVIDIA Turing limits](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)
- [PTX sm75 INT8 fragment mapping](https://docs.nvidia.com/cuda/archive/12.6.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-8816)

Published gains concern different precisions, shapes, kernels and toolchains.
They motivate experiments, not expected Pearl speedup percentages.

## Results

### R1: complete miner cache/jackpot factorial

All four binaries passed verified mock shares for both CUDA threadblock shapes
and both A modes (16 cases). The combined binary additionally passed the
production-pipeline alignment oracle at 8192x8192; the local CTest suite passed.

After 40 conditioning attempts, each mode used the balanced order
baseline / cg / predicated / combined / combined / predicated / cg / baseline.
Each block completed 36 zero-target attempts; the first four were discarded.
The following values are means of two block rates. Rates account for the
reported prep + scan wall time; with overlap, preparation occurs during scanning
and the residual prep field rounds to zero milliseconds. The complete workload
was executed; this is not standalone GEMM-only timing.

| Variant | Dense TMAC/s | Dense delta | Incremental TMAC/s | Incremental delta |
| --- | ---: | ---: | ---: | ---: |
| Unchanged control | 60.3749 | — | 61.1470 | — |
| `.cg` A+B | 60.8513 | +0.789% | 61.7905 | +1.052% |
| Predicated jackpot | 60.8069 | +0.715% | 61.7143 | +0.928% |
| Combined | 61.8312 | +2.412% | 62.7601 | +2.638% |

The combined result repeated in forward/reverse blocks: dense 61.8236 / 61.8388,
incremental 62.7470 / 62.7732. This supports continuing with that candidate;
two blocks per mode are not an exhaustive variance study or an earnings claim.
Normal Boost and the original power limit were retained. The longer final
confirmation below compares rebuilt control and combined binaries after R2–R6.

The original release was restored exactly. Independent audit matched executable
SHA, argv, environment, cwd and both log destinations, found exactly one CMP
worker and counted 10 fresh completed attempts in 12 seconds.

### R2: selective operand cache policies

The candidate builder clones the CUTLASS global-load specialization and operand
iterator under distinct names, then selects the A and B iterator types separately.
It preserves the original address arithmetic, predication and 128-byte L2 hint.
The shared vendor tree is not changed.

All four variants passed all 32 prefix XORs and final keyed BLAKE3 digests for
both shapes, for zero, -128, random full-range INT8 and real-noise input fixtures.
Four balanced panel blocks per variant (60 measured kernels after 40 warmups)
gave these means of block medians:

| A policy | B policy | Kernel ms | Rate delta |
| --- | --- | ---: | ---: |
| Default | Default | 35.618635 | — |
| cg | Default | 35.525648 | +0.262% |
| Default | cg | 35.499072 | +0.337% |
| cg | cg | 35.246900 | +1.055% |

The panel was 4096x131072x4096 and included all milestones and final BLAKE3.
All variants used 216 registers, 128 local bytes, 49152 shared bytes and one
predicted resident CTA per SM. Single-operand deltas are small and are not
presented as a reliable production gain. Both-operands cg remained the preferred
policy and agrees with R1 and the earlier independent cg confirmation.
The original miner was restored and independently audited before R3.

### R3: output block traversal

All six traversals passed exhaustive prefix XOR and BLAKE3 checks on both CUDA
shapes and all four fixtures. The alternative orientation transposes the launch
grid, not matrix arithmetic: logical coordinates still address the original
scattered proof tiles.

| Traversal | Mean block-median ms | Rate delta versus group 8 |
| --- | ---: | ---: |
| Existing identity group 8 | 35.563632 | — |
| Group 1 | 35.703680 | -0.392% |
| Group 2 | 35.653904 | -0.253% |
| Group 4 | 35.825044 | -0.730% |
| Group 16 | 35.591656 | -0.079% |
| Transposed block grid | 37.268520 | -4.575% |

Resources remained identical to R2. There was no winning candidate to promote
to a whole-miner confirmation. Group 16 is essentially tied within repeat
variation; the transposed orientation clearly lost on this panel. These results
are specific to the CMP50 and this panel, not a universal swizzle comparison.
The installed miner was restored and independently audited before R4.

### R4, first part: adapted shared-stage pipelines

All five variants passed the exact milestone and BLAKE3 oracle. The Sm80-tagged
multistage mainloop uses the sm75 MMA atom and ordinary synchronous vector
global/shared copies on this GPU; it does not execute cp.async. Reducing CTA K
from 64 to 32 makes three/four shared stages fit within the actual 64 KiB limit.
The small K64 candidate uses a 128x128 CTA rather than exceeding that limit.

| Candidate | Kernel ms | Registers | Shared bytes | Rate delta |
| --- | ---: | ---: | ---: | ---: |
| Existing K64 / 2 stages / 256x128 | 35.625177 | 216 | 49152 | — |
| K32 / 2 stages / 256x128 | 43.606376 | 203 | 24576 | -18.303% |
| K32 / 3 stages / 256x128 | 84.505096 | 212 | 36864 | -57.843% |
| K32 / 4 stages / 256x128 | 83.699968 | 222 | 49152 | -57.437% |
| K64 / 3 stages / 128x128 | 75.738863 | 218 | 49152 | -52.963% |

All retain 128 local bytes and one predicted resident CTA per SM. Lower shared
storage alone does not increase residency because register usage remains high.
The synchronous-copy fallback is a tested negative result, not proof that every
manual Turing software pipeline is slower. A separate register-lookahead
prototype now preloads two future operand fragments while retaining the existing
two-stage shared ring. It refills a register slot only after its old fragment is
stored. Phase unrolling gives constant fragment indices in source, with the aim
of reducing array-index overhead. The generated resource use was measured rather
than assuming that the compiler scalarized the arrays successfully.

### R4, second part: register lookahead and adaptation

All five additional prototypes passed the exact oracle. Preloading both operands
increased resource pressure; moving the refill after the CTA barrier and then
prefetching only B did not produce a competitive kernel either. Each subexperiment
used its own balanced baseline rather than comparing readings across runs.

| Prototype | Kernel ms | Registers | Local bytes | Control ms |
| --- | ---: | ---: | ---: | ---: |
| Two future A+B fragments, K64 | 104.404425 | 255 | 400 | 35.694968 |
| Two future A+B fragments, K32 | 86.709037 | 255 | 288 | 35.694968 |
| Post-barrier B lookahead, K64 | 77.131971 | 255 | 304 | 35.694420 |
| Post-barrier B lookahead, K32 | 88.024816 | 247 | 288 | 35.694420 |
| Post-barrier A+B lookahead, K64 | 109.801208 | 255 | 400 | 35.694420 |

The two-stage shared ring was retained in these prototypes (49152 bytes for K64,
24576 for K32); predicted residency remained one CTA per SM. None is enabled.
This rejects these nine implementations, not every possible software pipeline.
The original miner was restored and independently audited between subexperiments
and before R5. These are complete kernels, including every milestone and BLAKE3.

### R5: independent small-fragment MMA prototype

The raw PTX prototype gives one warp an exact scattered 8x8 proof tile. It loads
the full signed INT8 operands directly into the documented m8n8k16 fragments,
accumulates in INT32, reduces all 64 cells every 128 K terms, and performs the
same 16-word rotate/XOR transcript, keyed BLAKE3 and target check. This adapts the
minimal-Tensor-Core idea without importing a reference miner's different proof
geometry or omitting Pearl's checkpoint work.

All three independent implementations passed the existing INT64/C-BLAKE3 oracle
for both nominal shape selections and all four fixtures. Both selections launch
the same one-warp-per-proof kernel in this experiment. The timing uses four
warmups and eight measured kernels per block, equally for its control and each
candidate; four balanced blocks were used. The shorter block avoids spending
minutes repeating a clearly slower kernel.

| Implementation | Kernel ms | Registers | Local bytes | Shared bytes | Resident CTAs/SM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing fused control | 35.598680 | 216 | 128 | 49152 | 1 |
| Independent direct MMA | 679.982651 | 44 | 64 | 0 | 4 |
| One-step register lookahead | 685.276199 | 45 | 64 | 0 | 4 |
| Lookahead with cg loads | 761.330406 | 45 | 64 | 0 | 4 |

The direct prototype is 19.10x slower despite much lower register/shared usage
and higher predicted residency. Its source gives up broad operand reuse and
assigns one whole warp to the checkpoint/hash work of a single proof tile;
the measured result does not support replacing the current fused design with
this small-fragment implementation. It does not rule out a future independently
tiled implementation with comparable reuse. The installed miner was restored
and independently audited before R6.

### R6: bounded native instruction ordering

The driver harness loads the exact sm75 cubin with the original CUTLASS parameter
ABI. This control includes both cg loads and predicated jackpot state, so native
results are compared with the best known source candidate rather than added to
an unrelated baseline. All allocation, module loading and diagnostic checks are
outside CUDA event timing.

CuAssembler was pinned to `96a9f72baf00f40b9b299653fcef8d3e2b4a3d49` in a private
server directory, with sympy 1.13.3 and pyelftools 0.29. Full reassembly failed on
a CUDA 12.6 `UISETP.EQ.OR.EX` modifier absent from the assembler's instruction
repository. This is an assembler compatibility result, not a timed speed result.
Control-word splitting and merging nevertheless reproduced every original text
section byte for byte. The actual schedule candidates move original 128-bit
instruction words directly, preserving non-text section data and resource
headers. No branch, CTA barrier or shared matrix-load instruction is moved.

The guarded search found six eligible adjacent ALU/IMMA–global-load pairs per
cubin: one in the 256x128 mining loop, three in the 128x128 dump loop, and two in
its mining loop. Register supersets must be disjoint, scoreboard identifiers
must not conflict, and load inputs must have at least 16 explicit scheduled
cycles since any earlier register touch. Reuse fields stay attached to their
original instruction words. These guards define the search; exact GPU oracles
are still mandatory and are not replaced by a claim of a complete hazard model.

`load_1` moves one pair per loop (three total); `load_4` moves all six available
pairs. `load_8` produces identical cubins to `load_4` and is excluded as a
duplicate. Each has two stall treatments: preserve the moved ALU's downstream
scheduled delay, or retain its original stall when its result uses a dynamic
scoreboard. The latter is explicitly a tested candidate rather than an assumed
latency rule. The 256x128 code has only one eligible pair, so the two pair limits
share that shape's schedule; they differ in the 128x128 code.

All four schedules passed every small-fixture prefix/hash check for both shapes,
including zero, -128, full-range random INT8 and actual noisy operands. They also
passed the large-panel sampled oracle in each timing process. Four balanced
blocks of 60 measured kernels followed 40 warmups per block.

| Schedule | 256x128 mean block-median ms | Rate delta |
| --- | ---: | ---: |
| Combined source control | 34.711852 | — |
| One pair/loop, compensated stalls | 35.295728 | -1.654% |
| All pairs, compensated stalls | 35.303880 | -1.677% |
| One pair/loop, original scoreboard stalls | 35.288059 | -1.633% |
| All pairs, original scoreboard stalls | 35.291348 | -1.642% |

The 256x128 control and candidates have 246 registers, zero local bytes,
49152 shared bytes and one predicted resident CTA per SM. An earlier balanced
run of the compensated schedules independently lost by 1.62–1.65%. Native
schedules remain isolated research artifacts.

The independent 128x128 run used the same four schedules and the same balanced
protocol. Its combined control uses 254 registers, zero local bytes, 32768 shared
bytes and two predicted resident CTAs per SM.

| Schedule | 128x128 mean block-median ms | Rate delta |
| --- | ---: | ---: |
| Combined source control | 37.463112 | — |
| One pair/loop, compensated stalls | 37.456232 | +0.018% |
| All pairs, compensated stalls | 37.444087 | +0.051% |
| One pair/loop, original scoreboard stalls | 37.460548 | +0.007% |
| All pairs, original scoreboard stalls | 37.475000 | -0.032% |

These small differences are within the observed block variation. No native
schedule qualifies as a winner for whole-miner integration. This rejects this
bounded search, not every possible SASS schedule. After each native run, an
independent process audit confirmed the exact original release and ten fresh
completed mining attempts in twelve seconds.

## Final complete-miner confirmation

The public builder rebuilt the four factorial binaries in the private checkout.
Rebuilt control and combined candidates passed CTest, the 8192x8192
production-pipeline alignment oracle, and eight verified mock-proof cases across
both CUDA shapes and both A modes. This independently validates the new binary
hashes rather than assuming that equal generated-header hashes certify a rebuild.

After 40 conditioning attempts, each mode ran baseline / combined / combined /
baseline. Each block completed 65 fixed zero-target attempts and discarded its
first five, leaving 60 measured attempts per block: 240 per mode, 480 total.
The complete attempt dimensions were 131072x131072x4096, with overlap enabled
and 4096 A updates. The reported prep + scan metric follows the R1 definition.

| A mode | Control TMAC/s | Combined TMAC/s | Rate delta |
| --- | ---: | ---: | ---: |
| Dense | 60.2206 | 61.7653 | +2.565% |
| Incremental | 61.0778 | 62.6972 | +2.651% |

Combined block rates were 61.7541 / 61.7766 in dense and 62.6865 / 62.7079 in
incremental. Control blocks were 60.2446 / 60.1965 and 61.0213 / 61.1344.
This corroborates the initial factorial result on the CMP50 under normal Boost;
it is not a measured RTX3090 result or a guaranteed pool-income percentage.

The original release was restored, then independently audited: binary SHA,
argv, environment, working directory and both log destinations matched;
exactly one original worker remained, with ten fresh attempts in twelve seconds.
No production defaults, release artifact, CI configuration or upstream PR changed.
The candidate retained for subsequent production integration is cg on both
operands plus predicated jackpot state. R2 adds cache-policy evidence, not an
additional additive gain; R3–R6 produced no further qualifying winner.

## Reproducing the suite

Use a private Linux checkout based on `08176ba`, with these research scripts and
headers copied in and the repository-pinned dependencies populated. CUTLASS and
the CUDA wrapper directory must be actual private copies. The whole-miner builder
rejects symlinked headers and restores its original header text in `finally`.
Use the real proof FFI, common/noise routines and C BLAKE3 implementation.

On the measured server the checkout is `/w/experiment-research-suite`; the
installed release is `/w/releases/v0.5-fork.4`. The protected runners deliberately
check that release SHA and worker before stopping it. They are specific to this
deployment; another server needs its own verified process identity and restore
configuration. Do not copy private restoration snapshots or wallet-bearing logs
into the repository.

```sh
# Complete miner factorial, then the longer independent confirmation.
python3 scripts/build_cuda_research_miners.py \
  --root /w/experiment-research-suite --output build/research-miners
python3 scripts/run_cuda_research_miners.py
python3 scripts/run_cuda_research_miners.py \
  --confirmation --output results/confirmation

# Prepare the real host objects once, then build/run each group in table order.
python3 scripts/build_cuda_data_feed_profile.py --output build/feed-profile
python3 scripts/build_cuda_research_profile.py \
  --group cache --output build/cache --objects build/feed-profile
python3 scripts/run_cuda_research_profile.py \
  --build build/cache --output results/cache
# Repeat for swizzle, pipeline, lookahead, lookahead_postbar and minimal.
# Use --repeats 8 --seconds 1800 for minimal; its kernels are much slower.

# Native control includes cg + predicated jackpot.
python3 scripts/build_cuda_research_profile.py \
  --group native --output build/native --objects build/feed-profile
python3 scripts/build_cuda_native_variants.py \
  --build build/native --assembler build/native-tools/CuAssembler \
  --packages build/native-tools/packages
python3 scripts/run_cuda_research_profile.py \
  --build build/native --output results/native
python3 scripts/run_cuda_research_profile.py \
  --build build/native --output results/native128 --tile 128x128

# Independent restore audit after each runner; only its numeric result is public.
python3 scripts/check_cuda_research_restoration.py \
  --snapshot results/native128/restore.json --output results/native128-restoration.json
```

The native tool checkout and its Python packages are private test dependencies;
they are not part of miner packaging. Manifests identify compiled binaries,
generated headers, cubins, reused objects and compiler arguments. The public
whole-miner builder reproduced all four helper/memory header hashes; rebuilt
binary hashes differ from R1 and the rebuilt baseline/combined binaries therefore
receive their own correctness checks and confirmation rather than inheriting R1.

Final script hardening adds explicit native text bounds/resource-header guards
and an import-safe whole-miner runner entry point. An independent audit checked
all 36 changed instruction pairs across the four full/verification cubin pairs:
every ELF header and non-text section stayed identical, each load word stayed
exact, and moved ALU words differed only in permitted stall bits. These guards
and the Python entry-point refactor do not change generated CUDA code or flags.

Hardware performance counters were unavailable (`ERR_NVGPUCTRPERM`), and the
installed sanitizer could not instrument the GPU with debugging disabled.
Reported exactness comes from executed CPU prefix/hash and proof oracles; these
runs are not presented as a successful sanitizer check. Bottleneck explanations
are source/resource-based hypotheses rather than measured counter attribution.

## Raw evidence

- [R1 complete-miner factorial](benchmarks/cuda-research-r1-2026-10-04.json)
- [R2 operand cache](benchmarks/cuda-research-r2-2026-10-04.json)
- [R3 traversal](benchmarks/cuda-research-r3-2026-10-04.json)
- [R4 shared stages](benchmarks/cuda-research-r4-pipeline-2026-10-04.json),
  [register lookahead](benchmarks/cuda-research-r4-lookahead-2026-10-04.json),
  [post-barrier adaptation](benchmarks/cuda-research-r4-postbar-2026-10-04.json)
- [R5 independent MMA](benchmarks/cuda-research-r5-2026-10-04.json)
- [R6 initial native schedules](benchmarks/cuda-research-r6-initial-2026-10-04.json),
  [256x128 follow-up](benchmarks/cuda-research-r6-256-2026-10-04.json),
  [128x128](benchmarks/cuda-research-r6-128-2026-10-04.json)
- [Independent restoration and native-word audits](benchmarks/cuda-research-restoration-2026-10-04.json)
- [Rebuilt complete-miner confirmation](benchmarks/cuda-research-confirmation-2026-10-04.json)
