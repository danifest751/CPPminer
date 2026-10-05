# CPPminer — danifest751 fork

[![Latest release](https://img.shields.io/github/v/release/danifest751/CPPminer?label=release)](https://github.com/danifest751/CPPminer/releases/latest)
[![Windows build](https://github.com/danifest751/CPPminer/actions/workflows/windows-cuda.yml/badge.svg?branch=release%2Ffork)](https://github.com/danifest751/CPPminer/actions/workflows/windows-cuda.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A C++ miner for **Pearl (PRL)**, with **Quantus** as a second algorithm. It mines on NVIDIA GPUs (CUDA), on AMD, Intel and mobile GPUs (OpenCL) and on CPUs (x86 and ARM).

This is a fork of [1640675651/CPPminer](https://github.com/1640675651/CPPminer) by @foolzhz. It adds:
- faster kernels for most hardware;
- support for the Kryptex and HeroMiners pools;
- ready-to-run Windows and Linux builds.

Changes are developed in this fork. Selected changes are offered upstream as [pull requests](https://github.com/1640675651/CPPminer/pulls?q=is%3Apr+author%3Adanifest751).

**Download:** [latest release](https://github.com/danifest751/CPPminer/releases/latest) — Windows x64 zip, Linux x64 tar.gz.

### Hashrate at a glance

Pearl, full pool size, default settings, stock power limit (1 TMAC/s = 1 TH/s on the pool):

| GPU | Power | TMAC/s |
|---|---:|---:|
| RTX 5070 | 250 W | 115–117 |
| RTX 3090 | 350 W | 108–113 |
| CMP 50HX | 225 W | 68–69 |
| Arc A380 | stock | 18.8 |
| Radeon 780M (iGPU) | laptop | 6.4 |
| RX 580 4 GB | stock | 1.84 |

More cards and CPUs: [Supported hardware and speed](#supported-hardware-and-speed).

---

## Contents

- [What this fork adds](#what-this-fork-adds)
- [Supported hardware and speed](#supported-hardware-and-speed)
- [Quick start](#quick-start)
- [Pools](#pools)
- [Usage examples](#usage-examples)
- [Command-line options](#command-line-options)
- [Environment variables](#environment-variables)
- [Building from source](#building-from-source)
- [Troubleshooting](#troubleshooting)
- [Developer fee](#developer-fee)
- [How Pearl mining works](#how-pearl-mining-works)
- [Project layout](#project-layout)
- [Credits and license](#credits-and-license)

---

## What this fork adds

| Area | Change |
|---|---|
| **NVIDIA Turing** (RTX 20xx, CMP 40HX/50HX) | Own INT8 tensor-core kernel (`mma.m8n8k16`) on packed operands instead of dp4a: CMP 50HX 21 → 69 TMAC/s |
| **NVIDIA Ampere / Ada / Blackwell** (RTX 30xx, 40xx, 50xx, A-series) | Own `mma.m16n8k32` kernel with a 4-stage `cp.async` ring on packed operands: RTX 3090 31 → 113 TMAC/s, RTX 5070 107 → 115. It also runs on cards where upstream did not start |
| **All CUDA** | Noisy matrices stored in a packed layout, so every k-tile of a threadblock is one contiguous read (less L2/DRAM traffic, higher power-limited clocks); lighter milestone code; the next attempt's matrix is prepared while the current one is scanned |
| **AMD RDNA3** (RX 7000, Radeon 780M/760M) | WMMA matrix cores (`v_wmma_i32_16x16x16_iu8`): 780M 3.65 → 6.4 TMAC/s. Before that, `v_dot4` and work-group swizzle |
| **AMD Polaris and older GCN** (RX 470/480/570/580, Fiji, Tonga) | Dedicated 24-bit multiply-add kernel: RX 580 1.38 → 1.84 TMAC/s |
| **Intel Arc** (Xe-HPG) | `--backend onednn`: an ESIMD XMX kernel does the GEMM, milestone XOR and the whole jackpot in one pass, on operands the prep writes in DPAS layout (A380 3.2 → 18.8 TMAC/s); gemmstone systolic kernels as the fallback (16.7) |
| **Qualcomm Adreno** | The prep program compiles now, and a 4x4 register tile is used: ~15 → ~700 GMAC/s |
| **x86 CPU** | Work is spread over all threads (small matrices used to run on one), pinned threads, `--threads` / `--smt` / `--no-smt`. Kernels: AVX-512 VNNI, base AVX-512 (F+BW), AVX2 without register spills |
| **ARM CPU** | DotProd rewrite (+50%), new I8MM (`smmla`) kernel, faster NEON fallback (+79%) |
| **Pools** | Kryptex gzip stratum v2 (also used by HeroMiners), `--pool-pass d=N` custom difficulty, `WALLET.worker` on Kryptex, TCP keepalive and a submit-reply watchdog |
| **Robustness** | One miner per GPU, clean exit when no GPU works, fixed share-target scaling for non-8x16 hash tiles, MinGW static runtime, stdout buffering on MinGW |
| **Releases** | Windows and Linux builds with CUDA, OpenCL, oneDNN, CPU and the ESIMD scan with its bundled SYCL runtime, plus start scripts for Kryptex and HeroMiners |

---

## Supported hardware and speed

Hashrate is in **MAC/s**: multiply-accumulates per second of the int8 matrix product. 1 TMAC/s here is what pools show as **1 TH/s**. All numbers are at the full pool problem size (131072 × 131072 × 4096) unless marked otherwise, with the default settings of the latest release.

### GPUs measured by the fork

| GPU | Architecture | Backend / kernel | TMAC/s | Notes |
|---|---|---|---|---|
| RTX 4090 | Ada, sm_89 | CUDA, tensor cores | 157–229 | rented card, v0.5-fork.1; varies with clocks |
| RTX 5070 | Blackwell, sm_120 | CUDA, packed `m16n8k32` kernel | 115–117 | 250 W; 107 with v0.5-fork.5 |
| RTX 3090 | Ampere, sm_86 | CUDA, packed `m16n8k32` kernel | 108–113 | 350 W; 99 with v0.5-fork.5 |
| CMP 50HX | Turing, sm_75 | CUDA, packed `m8n8k16` kernel | 68–69 | 225 W; 62.5 with v0.5-fork.5 |
| Arc A380 | Xe-HPG (DG2) | oneDNN, ESIMD XMX scan | 18.8 | Linux; 16.7 with gemmstone, 3.2 with OpenCL DPAS |
| Radeon 780M (iGPU) | RDNA3, gfx1103 | OpenCL, WMMA | 6.4 | laptop, shared memory |
| RX 580 4 GB | Polaris, gfx803 | OpenCL, GCN kernel | 1.84 | AMD Windows driver |
| Adreno 830 | Snapdragon 8 Elite | OpenCL, 4x4 tile | ~0.7 | 8192² |

### CPUs measured by the fork

| CPU | Kernel | TMAC/s | Notes |
|---|---|---|---|
| Ryzen 7 8745HS (Zen 4, 8C/16T) | AVX-512 VNNI | 1.48 | full size; 1.55 at 8192² |
| Ryzen 5 5500 (Zen 3, 6C/12T) | AVX2 | ~0.7 | 8192² |
| Snapdragon 8 Elite phone (8 cores) | ARM I8MM | 0.86–0.95 | 8192², NDK build over adb |

### Upstream measurements

Measured by the upstream author on upstream builds; the fork's kernels for these devices are the same or faster.

| Device | Backend | TMAC/s |
|---|---|---|
| GTX 1070 | CUDA dp4a | ~9.1 |
| Radeon Pro 5500M | OpenCL dp4a | ~5.0 |
| Intel Xe-LPG 64EU (Core Ultra 9 275HX) | oneDNN | ~3.0 |
| Intel UHD 770 | oneDNN | ~1.3 |
| Core i9 12900K | CPU AVX-VNNI | ~2.3 |
| Core i5 12490F | CPU AVX-VNNI | ~1.1 |
| Core i9 9980HK | CPU AVX2 | ~0.44 |

### Which backend to use

| Hardware | Backend | Kernel picked automatically |
|---|---|---|
| NVIDIA GTX 10xx (Pascal) | `cuda` | dp4a |
| NVIDIA RTX 20xx, CMP 40HX / 50HX (Turing) | `cuda` | tensor cores, 256x128 |
| NVIDIA GTX 16xx, CMP 30HX (Turing without tensor cores) | `cuda` | not measured; `--cuda-mma simt` is the safe choice |
| NVIDIA RTX 30xx / 40xx, A-series (Ampere, Ada) | `cuda` | tensor cores, multistage |
| AMD RX 7000 / 780M / 760M (RDNA3) | `opencl` | WMMA |
| AMD RX 9000 (RDNA4) | `opencl` | `v_dot4`; WMMA is opt-in with `--ocl-dot wmma` (not yet verified on hardware) |
| AMD RX 6000, Radeon VII (Vega 20) | `opencl` | `v_dot4` |
| AMD RX 5000 (RDNA1) | `opencl` | `v_dot4` where the chip has it, otherwise scalar |
| AMD Vega 56 / 64 | `opencl` | scalar |
| AMD RX 400 / 500, Fiji, Tonga (GCN) | `opencl` | GCN 24-bit kernel |
| Intel Arc (Xe-HPG) | `onednn` | ESIMD XMX scan when built with `-DCP_ENABLE_ESIMD=ON`, else gemmstone systolic XMX |
| Intel Arc, Intel iGPU | `opencl` | `cl_khr_integer_dot_product`, or scalar |
| Qualcomm Adreno, ARM Mali | `opencl` | 4x4 tile on Adreno |
| Any CPU | `cpu` | best of AVX-512 VNNI / AVX-VNNI / AVX-512 / AVX2 / SSSE3, or I8MM / DotProd / NEON |

---

## Quick start

### Windows

1. Download `cppminer-win64-cuda.zip` from the [latest release](https://github.com/danifest751/CPPminer/releases/latest) and unpack it.
2. Open `start-herominers.bat` or `start-kryptex.bat` in Notepad and set `WALLET`:
   - HeroMiners: your Pearl address `prl1...`;
   - Kryptex: your Kryptex account (`krx...`) or a Pearl address.
3. Intel Arc: use `start-herominers-intel.bat` (`--backend onednn`). AMD GPU: change `--backend cuda` to `--backend opencl`. CPU only: `--backend cpu`.
4. Double-click the script. It restarts the miner if it exits.

Requirements:
- NVIDIA: a driver that supports CUDA 12. The CUDA runtime and the MSVC DLLs are in the zip.
- AMD / Intel: the normal GPU driver; it includes OpenCL.
- Intel ESIMD: `kernels/cp_esimd.dll` and its Intel runtime are bundled; oneAPI is not required. Windows package loading and CPU proof verification are checked without toolchain paths. Intel GPU execution on Windows still needs hardware validation; the A380 performance figures are from Linux.

### Linux

```sh
tar xzf cppminer-linux-x64-cuda.tar.gz
cd cppminer-linux-x64-cuda
nano start-herominers.sh            # set WALLET (and --backend for AMD/CPU)
# Intel Arc: start-herominers-intel.sh (--backend onednn)
./start-herominers.sh
```

The Linux build is made on Ubuntu 24.04 (glibc 2.39), and `libcudart` is bundled. OpenCL needs an ICD loader (`ocl-icd-libopencl1`) plus the vendor driver:
- AMD: ROCm, or Mesa rusticl;
- Intel: `intel-opencl-icd`. `--backend onednn` runs the ESIMD scan through `kernels/libcp_esimd.so` and the SYCL runtime bundled next to it (tested with compute-runtime 23.43 from Ubuntu 24.04 and 26.31); without it the backend uses gemmstone.

### Check before mining

```sh
cppminer --backend cuda --list-devices      # or opencl / cpu
cppminer --backend cuda --mock              # mines offline until the first share and verifies it
```

`--mock` prints the kernel that was picked and `attempt timing: ... TMAC/s`. Seeing `verify OK` means the whole pipeline works on your machine.

---

## Pools

| Pool | Stratum address | Wallet | Notes |
|---|---|---|---|
| **HeroMiners** | `stratum+tcp://ru.pearl.herominers.com:1200` (other regions: [pearl.herominers.com](https://pearl.herominers.com)) | Pearl address `prl1...` | Worker name from `--worker`; gzip proofs |
| **Kryptex** | `stratum+tcp://prl.kryptex.network:7048` | Kryptex account or Pearl address | Worker sent as `WALLET.worker`; gzip proofs; `--pool-pass d=N` sets the share difficulty |
| **Kryptex (Russia)** | `stratum+tcp://prl-ru.kryptex.network:7048` | same | Use this one from Russia (see below) |
| **LuckyPool** | `stratum+tcp://pearl-eu1.luckypool.io:3360` (GPU), `stratum+tcp://pearl-cpu-eu1.luckypool.io:3370` (CPU) | Pearl address | Plain (uncompressed) proofs |

**Proof compression.** Kryptex and HeroMiners speak the gzip stratum v2 ([spec](https://gist.github.com/maxmalysh/eaaf4332dbc5ca99d0a78f24a733fffe)), and the miner enables it automatically when the pool answers `"type":"v2"`. A GPU share then takes ~40 KB instead of ~130 KB.

**Connection recovery.** The miner waits up to 30 seconds for an accepted authorization and another 30 seconds for a valid first job. Rejected authorization or a missing response/job causes a reconnect; failures on the fee pool count toward its three-attempt fallback. Each submitted share is tracked by its JSON-RPC id and must receive a reply within 60 seconds. New shares and unrelated replies do not extend that deadline.

**Work delivery.** Pearl and Quantus keep one pending job: the latest valid job replaces older pending work and cancels a different active job. A job received before authorization is retained. Pearl difficulty resets to 32 for each connection; a targetless job keeps the difficulty and computed target from the moment it arrived. Malformed JSON and invalid authorization results are rejected. Backend or resource failures terminate the miner with status 1 so a supervisor can restart it. See [pool reliability changes and tests](docs/pool_reliability.md).

TCP connection attempts share a 10-second budget across the pool's resolved IPv4/IPv6 addresses, with time reserved for later addresses. DNS resolution uses the system resolver and is outside that budget. Timeouts and hashrate measurements use a monotonic clock, so adjusting the system clock does not change them.

**Russia.** Some Russian ISPs let a connection to foreign hosting pass its first ~15 KB and then drop every packet. Login and jobs work, but shares never arrive, and the pool shows no hashrate. Use `prl-ru.kryptex.network` or `ru.pearl.herominers.com`.

**How many shares to expect.** At the default difficulty 2097152 (Kryptex, HeroMiners) one share is expected every `2^53 / hashrate` seconds:

| Hashrate | Average time per share |
|---|---|
| 1 TMAC/s | ~2.5 h |
| 10 TMAC/s | ~15 min |
| 60 TMAC/s | ~2.5 min |
| 100 TMAC/s | ~1.5 min |

The pool's hashrate graph is therefore noisy over short windows; compare it with the miner over several hours. On Kryptex, `--pool-pass d=N` lowers the difficulty for slow devices.

---

## Usage examples

```sh
# NVIDIA GPU on HeroMiners
cppminer --backend cuda --devices 0 --pool stratum+tcp://ru.pearl.herominers.com:1200 --wallet prl1... --worker rig1

# Intel Arc on HeroMiners (ESIMD XMX scan, gemmstone fallback)
cppminer --backend onednn --devices 0 --pool stratum+tcp://ru.pearl.herominers.com:1200 --wallet prl1... --worker arc

# AMD GPU on Kryptex
cppminer --backend opencl --devices 0 --pool stratum+tcp://prl.kryptex.network:7048 --wallet krx... --worker rig1

# CPU, physical cores only, on LuckyPool
cppminer --backend cpu --no-smt --pool stratum+tcp://pearl-cpu-eu1.luckypool.io:3370 --wallet prl1... --worker cpu1

# Several NVIDIA GPUs in one process (independent A attempts per device)
cppminer --backend cuda --devices 0,1,2 --pool ... --wallet ... --worker rig1

# Quantus on CPU
cppminer --algo quantus --backend cpu --threads 8 --pool stratum+tcp://HOST:PORT --wallet qzpp... --worker rig1
```

OpenCL uses one device per process. For several AMD/Intel GPUs, start one miner per GPU with different `--devices` and `--worker`.

Several NVIDIA GPUs in one process scan independent A matrices and fetch proof data from
the winning device. See [multi-GPU behavior and validation](docs/cuda_multi_gpu.md).
For Intel Arc, `--backend onednn` runs an ESIMD XMX scan (18.8 TMAC/s on an A380, Linux;
see [ESIMD](docs/intel_esimd.md)) or gemmstone's systolic XMX kernels (16.7 TMAC/s),
versus 3.2 for `--backend opencl --ocl-dot dpas`. `CP_INTEL_GEMM=esimd|gemmstone` forces
one of them. See [oneDNN on A380](docs/intel_a380_onednn.md) and [DPAS validation](docs/intel_a380_dpas.md).
Run Intel and NVIDIA in separate processes with distinct workers.

---

## Command-line options

`cppminer --help` prints the same list for the backends compiled into your build.

### General

| Option | Description |
|---|---|
| `--algo NAME` | `pearl` (default) or `quantus` |
| `--backend NAME` | `cpu`, `cuda`, `opencl`, `onednn` or `wgpu` (must be compiled in) |
| `--devices N[,M]` | CUDA device ids, OpenCL flat index, or wgpu adapter indices (default 0; OpenCL prefers a discrete GPU) |
| `--list-devices` | List devices of the selected backend and exit |

### Pool

| Option | Description |
|---|---|
| `--pool URI` | `stratum+tcp://host:port` (IPv6: `stratum+tcp://[address]:port`) |
| `--wallet ADDR` | Wallet address or pool account |
| `--worker NAME` | Worker name (default `rig01`) |
| `--pool-pass STR` | `mining.authorize` password (default `x`). Kryptex: `d=N` sets the share difficulty (default 2097152) |
| `--agent NAME` | Agent string sent to the pool (default: the miner version) |

### Matrix and scan

| Option | Description |
|---|---|
| `--m N`, `--n N` | Matrix rows / columns in units of 1024 (default 128; each ≤ 256, `m*n` ≤ 16384). Pools set the real size; use these for benchmarks |
| `--batch-size N` | Launch batch: Pearl column / macro panel (default 1024); Quantus GPU nonces per launch (default 1000000). Aliases `--period-batch`, `--col-period-batch` |
| `--row-period-batch N` | CUDA: row CTAs per launch (default 32, max 1024) |
| `--max-nonce N` | Stop after N attempts per job |
| `--cert-version N` | Force the certificate version (1/2 legacy, 3 salted; default 3, the pool's `cert_version` wins otherwise) |

### CPU

| Option | Description |
|---|---|
| `--threads N` | OpenMP threads (default: all hardware threads; `OMP_NUM_THREADS` overrides) |
| `--smt` / `--no-smt` | One pinned thread per logical CPU (default) or per physical core. AVX2 gains nothing from SMT, AVX-512 VNNI ~30% on Zen 4 |
| `--simd ISA` | `auto` (default), `hybrid`, `avx512vnni`, `avxvnni`, `avx512` (`avx512bw`), `avx2`, `ssse3`, `i8mm`, `dotprod`, `neon`, `scalar` |
| `--prepack MODE` | `fused` (default), `reuse`, `separate` |

### CUDA

| Option | Description |
|---|---|
| `--cuda-mma MODE` | `auto` (default): `tensorop80` on sm_80+, `tensorop` on sm_75, dp4a `simt` below. Also `simt`, `tensorop`, `tensorop80`, `tensoropms` |
| `--cuda-tb TILE` | Tensor-core threadblock: `128x128`, `256x128` or `128x256` (default 256x128 on sm_75, 128x128 on sm_80+) |
| `--cutlass-fused` | Fused CUTLASS GEMM + jackpot (default) |
| `--cublas-period`, `--no-cutlass-fused`, `--no-period-gemm` | Debug paths |

### OpenCL

| Option | Description |
|---|---|
| `--ocl-platform P` | Only enumerate OpenCL platform `P` |
| `--ocl-dot MODE` | `auto` (default), `wmma` (AMD gfx11/gfx12), `sudot`, `sdot4`, `khr`, `force-khr`, `asm`, `off` |
| `--ocl-tile MxN[/MmMm]` | Register tile `4x4`, `4x8` (default), `8x8`, `8x16` (auto on AMD); optional `/64x64` or `/128x128` macro |
| `--ocl-macro MxN` | Macro block `64x64` or `128x128` (default) |
| `--ocl-issue MODE` | `auto` (default), `broadcast`, `packed` |
| `--ocl-cpm-type T` | Broadcast accumulator `float` (default) or `int` |
| `--ocl-lds on/off` | Stage A/B in local memory (default off) |
| `--cpu-gen` | Prepare the matrices on the CPU instead of the GPU (slower; debugging) |

### wgpu and oneDNN

oneDNN (Intel GPU) is in the release builds; wgpu is not, build it yourself (see [Building](#building-from-source)).

| Option | Description |
|---|---|
| `--wgpu-tile MxN[/MmMm]`, `--wgpu-macro MxN`, `--wgpu-lds on/off` | wgpu register tile, macro block and workgroup-memory staging |
| `--fused-jackpot` / `--no-fused-jackpot` | oneDNN: single fused kernel, or GEMM + separate jackpot (default) |
| `--onednn-layout NAME` | oneDNN A/B layout `TN` (default), `TT`, `NT`, `NN` |

### Testing and diagnostics

| Option | Description |
|---|---|
| `--mock` | Offline: mine a fixed job until the first share, verify it, exit |
| `--mock-diff D` | Mock difficulty (default 58 for Pearl; higher = longer) |
| `--align-test` | Check the GPU kernel against the CPU reference and exit |
| `--align-test-prod` | Same at the full `--m/--n` size (slow, ~1 GiB RAM) |
| `--simd-test` | Compare every CPU SIMD kernel with scalar (use with `--mock`) |
| `--prepack-test` | Check CPU prepack modes against each other |
| `--profile-scan [N]`, `--profile-prep [N]` | Time GEMM vs jackpot, or OpenCL matrix prep |
| `--dry-run` | Build proofs without submitting; save `pp_<pid>_<job>_header.bin` and `pp_<pid>_<job>_proof.b64` beside the executable |
| `--verify` | Verify each proof in-process before submitting |

Normal mining keeps proof data in memory; `--verify` checks it before submitting. Neither writes proof diagnostics. `--dry-run` requires a writable executable directory and reports failed diagnostic writes. Unknown options, missing values, malformed numbers and values that would be truncated are rejected before connecting to a pool. Value options also accept `--option=value`; `--threads 0` selects automatic thread count and `--max-nonce 0` removes the nonce limit.

---

## Environment variables

| Variable | Effect |
|---|---|
| `CP_CUDA_TB` | Same as `--cuda-tb` |
| `CP_CUDA_OVERLAP` | `1` (default) prepares the next attempt during the scan; `0` turns it off (saves 2 × m × 4096 bytes of VRAM) |
| `CP_OCL_GCN` | `1` / `0` force the GCN (Polaris) OpenCL kernel on or off |
| `CP_OCL_WMMA_PIPELINE` | `0` disables the WMMA register double buffer |
| `CP_OCL_WMMA_SELFTEST` | `1` runs the WMMA layout self-test at startup (useful on new AMD GPUs) |
| `CP_OCL_BUILD_LOG` | `1` prints the full OpenCL compiler log for failed kernel probes |
| `CP_ALLOW_SHARED_GPU` | `1` allows two miners on one OpenCL device |
| `CP_OCL_DUMP_BIN` | Path: dump the built OpenCL program binaries (debugging) |
| `CP_SIMD` | Same as `--simd` |
| `CP_CPU_AFFINITY` | `0` disables CPU thread pinning |
| `OMP_NUM_THREADS`, `OMP_PLACES`, `OMP_PROC_BIND` | Standard OpenMP thread count and placement (override the miner's pinning) |
| `CP_PYTHON` | Python for the optional host proof bridge |

---

## Building from source

The release builds use these exact steps. You need:
- CMake;
- a C++17 compiler;
- Rust (`cargo`) for the proof library;
- for CUDA, the CUDA Toolkit 12.x. CUTLASS 2.11 is fetched by the build scripts.

### Windows (MSVC, all release backends)

```powershell
./scripts/setup_windows_oneapi.ps1 -Prefix "$PWD/build/oneapi-esimd"
$env:CP_ONEAPI_ROOT = "$PWD/build/oneapi-esimd"
$env:PATH = "$env:CP_ONEAPI_ROOT/Library/bin;$env:PATH"
powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,Cuda,OpenCl,OneDnn -EnableEsimd -CudaArch "75;86;89"
./scripts/smoke_windows.ps1
./scripts/package_windows.ps1 -CudaRoot $env:CUDA_PATH -RuntimeRoot $env:CP_ONEAPI_ROOT
./scripts/validate_windows_package.ps1 -PackageDir ./cppminer-win64-cuda
```

`cppminer.exe` ends up in the repo root. `-EnableEsimd` builds `kernels/cp_esimd.dll` with Intel DPC++ 2026.1.1 (`icx`); omit it and the oneAPI setup for gemmstone only. Use `-CudaArch 61` for Pascal, `-Backend Cpu,OpenCl` without the CUDA Toolkit, and add `Wgpu` for that backend. The packaging script requires a fresh output directory.

### Windows (MSYS2 UCRT64, CPU + OpenCL)

```sh
export PATH=/c/msys64/ucrt64/bin:$PATH
MSYSTEM=UCRT64 ./build.sh --backend cpu,opencl
```

The binary is `build/cmake/cppminer.exe`. It is linked statically, so it does not depend on the MinGW DLLs.

### Linux

```sh
./build.sh --backend cpu,cuda,opencl,onednn --cuda-arch "75;86;89"
./build.sh --backend cpu,opencl          # without the CUDA Toolkit
```

The ESIMD scan library needs oneAPI's `icpx`; build it with CMake, then bundle its runtime:

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -DCP_ENABLE_OPENCL=ON -DCP_ENABLE_ONEDNN=ON -DCP_ENABLE_ESIMD=ON
cmake --build build -j4
scripts/package_esimd_runtime.sh build/kernels
```

### Android (Termux / NDK)

Cross-building with the NDK works with `-DCP_PROOF_FFI=OFF`. That build can benchmark (`--mock` without verify) but cannot build proofs, so it cannot mine on a pool. To mine on a phone, build natively in Termux, where the Rust proof crate also builds. See [`docs/termux_note.md`](docs/termux_note.md).

### CMake options

| Option | Default | Meaning |
|---|---|---|
| `CP_ENABLE_CPU` | ON | CPU backend |
| `CP_ENABLE_CUDA` | OFF | CUDA / CUTLASS backend |
| `CP_ENABLE_OPENCL` | OFF | OpenCL backend |
| `CP_ENABLE_ONEDNN` | OFF | Intel GPU oneDNN backend |
| `CP_ENABLE_ESIMD` | OFF | Intel XMX ESIMD scan library (needs oneAPI `icpx`/`icx`) |
| `CP_ENABLE_WGPU` | OFF | wgpu backends (Pearl and Quantus) |
| `CP_ENABLE_CUBLAS` | OFF | cuBLAS debug path |
| `CP_CUDA_ARCH` | native | e.g. `61`, or `"75;86;89"` |
| `CP_PROOF_FFI` | ON | Link the Rust proof library (turn off only for cross builds) |

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| The pool shows less hashrate than the miner | Few shares per hour make the pool's estimate noisy, and restarts count as zero. Compare over several hours (see [share times](#pools)) |
| Pool connected, jobs arrive, but no hashrate at all (Russia) | The ISP drops large packets to foreign pool servers. Use `prl-ru.kryptex.network:7048` or `ru.pearl.herominers.com:1200` |
| `CUDA driver version is insufficient` | Update the NVIDIA driver to one that supports CUDA 12 |
| `No OpenCL GPU or CPU devices found` | The GPU driver is not installed or has crashed. Check Device Manager, reboot, reinstall the driver |
| `another cppminer is already mining on OpenCL device N` | A second miner was started on the same GPU. Close it, or use a different `--devices` |
| `backend init failed; exiting` | No usable device for that backend; the start script retries every 10 s |
| `[ocl] ... not supported by this GPU/driver, trying the next kernel` | Normal on GPUs without the newer AMD int8 instructions; the next kernel is used |
| PC freezes with an RX 470/480/570/580 | The new GCN kernel draws more power. Lower the Power Limit in AMD Adrenalin by 15–20%, or set `CP_OCL_GCN=0` |
| Speed much lower than the table | Check that the expected kernel is printed at startup (`--mock`), that nothing else uses the GPU, and the GPU's clocks and power limit |
| `[fee] ...` lines | The developer fee scheduler, see below |

---

## Developer fee

The miner keeps upstream's **1% developer fee**: one matrix in about 101 is mined for the developer. In this fork's builds the fee goes to the **fork maintainer**:
- **Normally.** For each fee cycle the miner reconnects to HeroMiners (`ru.pearl.herominers.com:1200`) and mines to `prl1pp9k3spr6l0c0s00mlmcnpktm5yfp0up9lu92deuvj2hnp38s8x0svyseuq` (worker `devfee`). Then it returns to your pool.
- **If HeroMiners can't be reached** three times in a row, the fee is mined on your own pool instead: Kryptex account `krxX8QJ872` on Kryptex, the same Pearl address elsewhere.

The schedule is tile-based. Every scanned tile adds to a debt; once it reaches 100 matrices, the miner works one matrix-worth for the fee and returns. `[fee]` log lines show each switch.

If you'd rather support the original author, build upstream's `dev` branch.

---

## How Pearl mining works

A Pearl job asks for an int8 matrix product C = A·B:
- the size is 131072 × 131072 with inner dimension 4096;
- A is random per attempt, B comes from the job;
- both carry low-rank (r = 128) noise derived from the job.

While the product is computed, every 8×16 (or 8×8) block of C is folded into XOR "milestones" every 128 steps of the inner dimension. Each block's milestones are then hashed with BLAKE3. A block whose hash is below the pool target is a share. The miner then builds a `plain_proof` (Merkle openings of the rows and columns involved), which the pool verifies with a zero-knowledge check.

Almost all the work is the int8 multiply-accumulate. That is why hashrate is counted in MAC/s, and why tensor cores, matrix cores and dot-product instructions matter so much. See [`docs/hashrate_calculation.md`](docs/hashrate_calculation.md) and [`docs/proof.md`](docs/proof.md).

---

## Project layout

```
include/                 public headers (pool, mining loop, workers, fee, config)
src/common/              pool client, job loop, fee scheduler, worker dispatch, main
src/cpu/                 CPU backend: fused GEMM + XOR, SIMD kernels (x86 and ARM), thread pinning
src/cuda/                CUDA backend; cutlass/ holds the fused CUTLASS kernels (SIMT, Sm75, Sm80)
src/opencl/              OpenCL backend; kernels/ holds the GEMM (dot4 / WMMA / GCN / scalar) and prep kernels
src/onednn/              Intel GPU oneDNN / gemmstone backend
src/pearl/wgpu/, rust/cp-pearl-wgpu-ffi/   Pearl wgpu backend (WGSL)
src/qpow/                Quantus (Poseidon2) on CPU, OpenCL and wgpu
rust/cp-proof-ffi/       plain_proof Merkle proof, gzip, zk-pow verify (Rust, linked in)
third_party/             BLAKE3, pearl-blake3, zk-pow, plonky2, OpenCL headers, CUTLASS
scripts/                 benchmark sweeps (sweep_ampere.sh), optional Python proof bridge
docs/                    notes on hashrate, proofs, SIMD, OpenCL issue shapes, Termux
.github/workflows/       Windows release CI
```

Fork branches (what goes upstream, what stays in the fork, how releases are cut) are described in [FORK.md](FORK.md).

---

## Credits and license

- **Miner:** [CPPminer](https://github.com/1640675651/CPPminer) by **@foolzhz**. This fork builds on it, and its changes are offered back upstream.
- **Libraries:**
  - [CUTLASS](https://github.com/NVIDIA/cutlass) (NVIDIA);
  - [BLAKE3](https://github.com/BLAKE3-team/BLAKE3);
  - [plonky2](https://github.com/0xPolygonZero/plonky2);
  - Pearl's `zk-pow` and `pearl-blake3`;
  - oneDNN gemmstone.
- **Protocols:** the Kryptex gzip stratum protocol by maxmalysh.

[MIT License](LICENSE).
