# Changelog

## v0.5-fork.8
- HiveOS custom-miner package `cppminer-0.5_fork.8.tar.gz` (one miner per GPU, per-GPU hashrate, shares, temperatures and fans) and Docker images `ghcr.io/danifest751/cppminer:nvidia|amd|intel`.
- Stats API: `--api-port N` serves `/summary` (JSON) and `/hiveos`; `--api-bind` sets the address.
- Pearl OpenCL: 25x faster keyed matrix hash, zero-B job seed on the GPU, 256-wide fused A prepack; RDNA4 half-wave reduce-scatter. Radeon AI PRO R9700 88 -> 95-96 TMAC/s at the pool; Arc A380 prep 750 -> 70-90 ms per nonce.
- Quantus: gfx12 inline-assembly reduction, two nonces per work-item and 64-bit linear layers on RDNA4 (R9700 162 -> 211 MH/s); multiply-chain row sums and skipped rare carry fold on OpenCL and CUDA (CMP 50HX CUDA 286.5 -> 293.6, OpenCL 202 -> 213; A380 30.2 -> 31.2 MH/s).
- Quantus is still mined without a developer fee.
- Pool agent `cppminer/0.5-fork.8`.
- Detailed release notes: [v0.5-fork.8](docs/releases/v0.5-fork.8.md).

## v0.5-fork.7
- Quantus CUDA backend (`--algo quantus --backend cuda`): CMP 50HX 101 (OpenCL) → 288 MH/s on Kryptex, RTX 2080 Ti 316, RTX 3090 419. Start-up self-test against the CPU reference; every candidate is re-hashed on the CPU.
- Quantus OpenCL kernel rewritten: per-device variant probe, carry-free 22-bit limbs in the linear layers, `__builtin_addc` carry chains on AMD, registers instead of scratch on AMD, two launches in flight. Arc A380 30 MH/s, Radeon AI PRO R9700 162 MH/s.
- Pearl: WMMA by default on AMD RDNA4 (gfx12), layout picked by a start-up self-test: R9700 88.5 TMAC/s. WMMA milestone without dynamic private indexing.
- Quantus is mined without a developer fee in this release.
- Pool agent `cppminer/0.5-fork.7`; start script `start-quantus-kryptex`.
- Detailed release notes: [v0.5-fork.7](docs/releases/v0.5-fork.7.md).

## v0.5-fork.6
- New CUDA scan kernels on packed noisy operands for Turing (`mma.m8n8k16`), Ampere/Ada and Blackwell GeForce (`mma.m16n8k32`, 4-stage `cp.async`), replacing the CUTLASS kernels there. The prep writes every 256/128-row block and 32-wide k-tile as one contiguous record. Same hash tiles, words and hit coordinates as CUTLASS.
- RTX 4090 (450 W): 243.8 → 271–275 TMAC/s, 271 live. RTX 3090 (350 W): 99.0 → 112.4–113.5 TMAC/s kernel, 108–110 live. CMP 50HX (225 W): 62.5 → 68.4–69.5. RTX 5070 (250 W): 106.6 → 115.3, 116.8 live.
- `--align-test` checks the packed kernel against the CUTLASS reference. `CP_CUDA_PACKED=0` returns to CUTLASS; `CP_CUDA_PACKED=1` enables the packed kernel on Hopper and datacenter Blackwell (not measured yet).
- Builds use CUDA 12.9: native code for sm_75/86/89, sm_120 as PTX that the driver compiles (its compiler gives faster RTX 50 code than CUDA 12.9 ptxas).
- Pool agent `cppminer/0.5-fork.6`.
- Detailed release notes: [v0.5-fork.6](docs/releases/v0.5-fork.6.md).

## v0.5-fork.5
- Intel Arc support in the release builds: the `onednn` backend now ships in the Linux and Windows packages.
- ESIMD XMX scan for Intel GPUs (`kernels/libcp_esimd.so` on Linux, `kernels/cp_esimd.dll` on Windows): one kernel per panel does the int8 GEMM (`dpasw`), milestone tile XOR, fold, keyed BLAKE3 and target compare; the GPU prep writes the operands in DPAS block layouts. Arc A380 on Linux at 131072²: 18.79 TMAC/s per full attempt, vs 3.20 for OpenCL DPAS. Both packages bundle the SYCL runtime, so only the Intel GPU driver is needed; without the library the backend falls back to gemmstone. Details: [ESIMD](docs/intel_esimd.md).
- oneDNN/gemmstone: XeHPG systolic (XMX) kernels enabled (milestone XOR inside larger K unrolls, fallback crash fixed), measured-best kernel preferred, 1024x1024 panels, register-resident jackpot kernel and a pipelined panel scan. Arc A380: 16.66 TMAC/s. Details: [oneDNN on A380](docs/intel_a380_onednn.md).
- OpenCL prep: work-group noise kernels and 16-wide signal generation (A prep 200 → 96 ms on A380, per-job B ~200 → ~80 ms); also used by the OpenCL backends.
- Linux `build.sh` vendors the oneDNN deps itself (`src/onednn/prepare_onednn_deps.sh`); new `-DCP_ENABLE_ESIMD=ON`; Windows `build.ps1 -EnableEsimd` builds `cp_esimd.dll` and the release package bundles its Intel runtime and UR OpenCL adapter. Windows GPU execution remains unverified.
- Portable Linux library lookup via `$ORIGIN`; Windows smoke-test process completion and CTest path fixed. Extracted Windows packages are checked without toolchain paths, including CPU proof verification and ESIMD ABI/dependency loading.
- Tools: register-only DPAS ceiling benchmark, device-wide Level Zero metric sampler, Intel scan benchmark script.
- Pool agent `cppminer/0.5-fork.5`.
- Fix duplicated CUDA multi-GPU work with device-local A matrices, keys, commitments and incremental caches; fetch proof inputs from the winning GPU, widen aggregate tile totals and check cancellation between column batches. Single-GPU throughput regression check: 63.171 → 63.132 TMAC/s (-0.062%). Physical two-NVIDIA-GPU scaling remains unvalidated; [details](docs/cuda_multi_gpu.md).
- Integrate the opt-in Intel XMX/DPAS OpenCL backend (`--backend opencl --ocl-dot dpas`) with automatic lane-layout self-tests, exact milestone/proof validation and checked environment bounds. Arc A380 full-attempt testing at 32768²: KHR 0.886 → DPAS 2.773 TMAC/s; [details](docs/intel_a380_dpas.md).
- Promote the measured sm75 `.cg` operand-load policy and constant-index, predicated jackpot fold into the normal CMake CUDA build.
- Move the sm75 TensorOp milestone callback before the next K tile's loads, retaining the exact prefix order and final callback.
- Include opt-in incremental signal-A generation and cached keyed BLAKE3 trees (`CP_CUDA_A_MODE=incremental`, 4096 updates). Dense remains the default; the CMP50 test server runs incremental mode.
- Verify the integrated build on CMP50HX: exact CUDA oracles, alignment, proof checks and CTest pass; complete-attempt ABBA throughput is 60.46 → 63.18 TMAC/s (+4.50%) against v0.5-fork.4.
- CUDA validation details: [integrated CUDA validation](docs/cuda_integrated_improvements.md).
- Detailed release notes: [v0.5-fork.5](docs/releases/v0.5-fork.5.md).

## v0.5-fork.4
- Pool session recovery: authorization/first-job deadlines, per-request submit ACK tracking and oldest-share timeout.
- Bounded latest-job delivery for Pearl and Quantus, complete work identities, preserved early jobs and cancellation across the idle/active transition.
- Connection-local Pearl difficulty and immutable job targets; strict JSON, Unicode, authorization and numeric-field validation.
- Quantus login correlation, full-width sequence parsing and final cancellation/connection checks before nonce submission.
- Correct gzip negotiation and recovery from failed compression without sending the wrong encoding.
- In-memory proof verification, including targetless jobs; checked dry-run diagnostics scoped to process and job.
- Explicit backend/resource failure exits, bounded multi-address TCP connection attempts and monotonic deadlines/rates.
- Strict command-line validation, local regression coverage and pool agent `cppminer/0.5-fork.4`.
- Fix TCP connection-budget compilation with MSVC and the Windows SDK `min` macro.
- Detailed release notes: [v0.5-fork.4](docs/releases/v0.5-fork.4.md).

## v0.5 (tentative)
- Pearl parses difficulty notifications with arbitrary JSON whitespace and dispatches exact method names. Reject invalid explicit targets, empty/oversized job IDs and invalid headers before starting work; retain object and legacy array notifications.
- Quantus rechecks cancellation, full work identity and connection state immediately before submitting a found nonce; disconnected sessions do not start queued work.
- Pearl negotiates gzip only from a direct authorize-response type or a direct result.type, ignoring unrelated nested metadata (outer type takes precedence).
- Quantus work identity includes the full mining hash, target and extranonce; equivalent hex casing remains a duplicate.
- Quantus login matches the request ID, rejects explicit errors/failed status and preserves the latest early job. Login and initial work share a fixed 30-second deadline.
- When a pool requires gzip proofs, a compression/allocation failure drops that share instead of submitting the wrong encoding; the proof queue returns its buffers and processes subsequent shares.
- Fix Quantus current-job buffer overflow and make job handlers retain array bounds.
- Pearl work identity includes the complete header, effective target and certificate version; ignore true duplicates and drop cancelled proofs after building/encoding. Clear queued work on reconnect.
- Pool TCP connections try all resolved IPv4/IPv6 addresses within a shared 10-second budget, reserving time for later addresses. Validate pool URI host/port lengths and accept bracketed IPv6. A broken connection returns a send error instead of terminating Linux mining with SIGPIPE.
- Use monotonic clocks for mining rates and pool deadlines on every platform.
- Pool recovery: track each share ACK by JSON-RPC id before sending, enforce the oldest unacknowledged share's 60-second deadline, and reconnect on rejected authorization or missing authorization/first job (30 seconds each). Fee-pool handshake failures trigger the existing fallback. Add deterministic session tests and loopback integration tests.
- Experimental OneDNN backend for intel GPU
- Fix OpenCL dot product extension on intel GPU
- Shrink opencl macro size to 64x64 in 4x8 tile mode, prevent to many work item per work group 
- Introduce quantus algorithm
- Quantus wgpu backend via quantus-miner GpuEngine FFI
- Quantus OpenCL Poseidon2 worker under src/qpow/opencl
- Pearl wgpu backend
- Reduce host memory usage (CPU, CUDA, OpenCL, OneDNN, Wgpu)
- MinGW and MSYS2 support (thanks to @danifest751)
- AVX512-VNNI CPU kernel (`--simd avx512vnni`, auto-selected on Zen4-class CPUs): ~2x the AVX2 kernel per core
- Base AVX-512 CPU kernel (AVX512F + AVX512BW, no VNNI; `--simd avx512`, auto-selected on e.g. Skylake-X/SP)
- Configurable matrix size: `--m` / `--n` in units of 1024 (default 128x128)
- Kryptex "Pearl stratum gzip protocol" (v2): `mining.authorize` offers `"type":"v2"`; when the pool's response carries `"type":"v2"` the `plain_proof` is submitted as base64 of the gzip stream (flate2/miniz_oxide in `cp-proof-ffi`, `cp_proof_gzip_b64`). Pools that do not answer `type v2` (LuckyPool) keep plain proofs. `--pool-pass STR` sets the authorize `password` (Kryptex custom difficulty `d=N`); on a kryptex host the wallet is also sent as `WALLET.worker`
- Pearl CPU: scan macro blocks scheduled one at a time (a chunk of 4096 left an 8k×8k scan on a single thread: 118 -> ~660 GMAC/s on a Zen4 8-core); `--threads N` now also sets the Pearl CPU pool; threads are pinned, one per logical CPU by default (`--no-smt` for one per physical core); `OMP_PLACES` / `OMP_PROC_BIND` disable the built-in pinning

### Pearl wgpu
- vec4<u32> A/B panel loads in the GEMM shader
- Disable naga loop bounding on the GEMM shader (~36x faster, GTX 1070 86 GMAC/s -> ~4 TMAC/s)
- Rewrite prepack_a (one 256-WI group per 8 rows, noise hashed once, packed u32 stores); per-attempt prep 1.3s -> ~0.13s on GTX 1070
- GEMM accumulator tile as named vec4<i32> locals instead of array<i32, 64>; fixes Intel iGPU (UHD 770 35 -> ~540 GMAC/s)
- Single-buffered LDS GEMM (one 32 KiB k-block panel per barrier pair), `--wgpu-lds on|off`, default on for discrete GPUs (GTX 1070 ~4.0 -> ~5.0 TMAC/s)
- `--wgpu-tile 4x4|4x8|8x8|8x16[/64x64|/128x128]` and `--wgpu-macro`, same tiles/hash tiles as OpenCL; shader tile code generated at engine init (default stays 8x8/128)
- Bind a_pre / b_pre / a_sig in windows under `max_storage_buffer_binding_size` (512 MiB buffers vs 256 MiB on Mali); scan batches split only where a macro column exceeds the limit

### Host memory reduction
- GPU backend: per-hit D2H 512 MiB -> ~0.3-0.8 MiB, no 2x512 MiB host A/B buffers. See proof.md (CUDA, OpenCL, OneDNN, Wgpu)
- Wgpu: readback staging buffer 512 MiB -> 256 KiB (host-visible; was sized to the whole A matrix)
- CPU, OpenCL `--cpu-gen` and oneDNN host fallback: drop the 512 MiB all-zero host B^T buffer; proofs use zero-matrix Merkle sub-roots cached per job
- CPU `--prepack fused` is now the default (~1.5 GiB steady vs ~2.5 GiB for separate)
- Host-matrix proofs hash A/B^T in place: transient per-share peak ~1.5 GiB -> < 1 MiB (no flatten/pad/MerkleTree copies of the 512 MiB matrix)

## v0.4
- ARM CPU + NEON support.
- AVX-VNNI support.
- Termux OpenCL support.
- Dynamic openmp schedule for CPUs with big and small cores.
- Lightweight random matrix generation on CPU, only sparsely fill matrix elements (one random write per column).
- Zero-B on cuda worker (reduce VRAM usage, 2GiB -> 1.5 GiB).


## v0.3
- Switch to cert V3 to align with the salted seed hardfork. Old cert V2 shares are no longer accepted by the network.
- Separate OpenCL kernel compilation process with error handling. Improve compatibility for driver/compiler that crashes on specific compilation failure.
- Skip matrix prep kernel if --cpu-gen is on. This fixes unsupported intrinsics on some drivers like beignet.
- Add OpenCL **4×8** hash tile (`--ocl-tile 4x8`) to further reduce register pressure on small GPUs; make **4×8** the default OpenCL tile (AMD still auto-selects **8×16**).
- OpenCL issue mode `--ocl-issue auto|broadcast|packed` (default **auto** = DP4A detection then broadcast fallback).
- OpenCL int8 promotion selection `--ocl-cpm-type float|int`: broadcast B scalar in float (default) or int32. 

OpenCL mode note:
Broadcast+float issue may improve performance since some GPUs are weak in int but strong in float. For example, on UHD 630, broadcast+float yields 115GH/s, packed+int yields 100GH/s, and broadcast+int yields 90GH/s.

## v0.2.1
- Refactor CUTLASS GEMM main loop to reduce XOR overhead. This buys back the lost performance in the last version due to more frequent XOR boundary. 8.0TH -> 9.1TH on a GTX 1070.

## v0.2
- Use rank=128 to avoid pearl rank penalty. This adds a bit more overhead compared to rank=256, may hurt hashrate slightly.
- SSE fallback for non-AVX CPU.
- Add CPU thread affinity and prioritizing physical cores than SMT threads.
- Use 8x8 tile as default for the opencl worker, lowering register pressure for integrated GPUs. Use 8x16 tile if detects AMD GPU.

## v0.1

First public release of **CPPminer** — a cross-platform Pearl (LuckyPool plain_proof) miner in C++.

### Package contents

| Binary | Backends | Notes |
|--------|----------|--------|
| `cppminer_all.exe` | CUDA + OpenCL + CPU | Pick at runtime with `--backend` |
| `cppminer_opencl.exe` | OpenCL + CPU | AMD / OpenCL GPUs |
| `cppminer_cpu.exe` | CPU only | AVX2 x86_64 |

Also ship next to the CUDA-capable binary:

- `cudart64_12.dll` (CUDA runtime)

CUDA Toolkit is **not** required to run.

### Hardware support

- **Legacy NVIDIA** — Pascal cards via the CUDA CUTLASS fused kernel (e.g. GTX 10-series).
- **AMD GPUs** — OpenCL worker.
- **CPU** — AVX2 OpenMP worker.

### Basic usage

```powershell
# CPU
.\cppminer_cpu.exe --backend cpu --pool stratum+tcp://HOST:PORT --wallet prl1... --worker rig01

# NVIDIA (CUDA) — use cppminer_all.exe
.\cppminer_all.exe --backend cuda --pool stratum+tcp://pearl-cpu-eu1.luckypool.io:3370 --wallet prl1... --worker rig01

# AMD / OpenCL
.\cppminer_opencl.exe --backend opencl --pool stratum+tcp://pearl-eu1.luckypool.io:3360 --wallet prl1... --worker rig01
```
