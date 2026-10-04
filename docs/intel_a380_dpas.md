# Intel Arc A380: opt-in XMX/DPAS backend

The unreleased fork candidate incorporates the Intel OpenCL DPAS prototype from
`perf/opencl-dpas` (five commits ending at `9c018e2`), preserving the current fork's
GCN fallback and quiet automatic dot-backend probing. Select it explicitly:

```sh
./cppminer --backend opencl --devices 0 --ocl-dot dpas --verify \
  --pool stratum+tcp://POOL:PORT --wallet WALLET --worker arc-a380
```

The Intel `auto` policy remains unchanged. DPAS requires the Intel subgroup matrix
multiply extension and an 8x16 hash tile. An automatic tile becomes 8x16; an
incompatible explicit tile or LDS staging is rejected. There is no silent scalar
fallback. The kernel uses `intel_sub_group_i8_i8_matrix_mad_k32` for exact integer
matrix multiplication, retains cumulative rank-128 milestones and feeds the
existing jackpot/proof pipeline.

Every DPAS kernel initialization runs its lane-layout self-test before adoption.
Compile with `-DCP_ENABLE_OPENCL=ON`; CMake copies the required OpenCL kernels next
to the executable. CUDA-only binaries do not expose this backend.
The device's minimum subgroup size is required: 8 for the tested A380. SG16 code
is retained for newer devices but has not been hardware-validated on this server;
its selected layout must also pass the automatic self-test. Environment integers
are parsed strictly and tile dimensions are bounded before multiplication.

| Optional variable | Purpose |
|---|---|
| `CP_OCL_DPAS_TM`, `CP_OCL_DPAS_TN` | Hash tiles per subgroup; positive powers of two, product at most subgroup size |
| `CP_OCL_DPAS_SG` | Explicit 8 or 16; must agree with the device minimum |
| `CP_OCL_DPAS_AK` | SG16 A packing variant; 0 is the default |
| `CP_OCL_DPAS_FORCE=1` | Explicit test override for a device reporting no hardware DPAS units |
| `CP_OCL_DPAS_EMULATE=8\|16` | Existing AMD functional-model experiment; does not measure Intel hardware |

## Local hardware validation, 2026-10-04

Arc A380 (8086:56a5), approximately 6 GiB VRAM, Ubuntu host kernel 6.8.0-142 with
i915, GuC/HuC and Resizable BAR active. The upstream slot runs at PCIe 2.0 x1.
An isolated Ubuntu 24.04 test container exposes only Intel's `renderD129` and
installs the distribution's `intel-opencl-icd` userspace runtime. The host kernel,
NVIDIA driver and generation services were preserved. Container restart policy
is `no`; this setup does not introduce miner autostart.

Runtime versions: `intel-opencl-icd` 23.43.27642.40-1ubuntu3, IGC
1.0.15468.25-2ubuntu0.1, GMM 22.3.17+ds1-1ubuntu1. A subsequent three-attempt
131072² DPAS scan test passed at 3.195 TMAC/s for the complete attempt (first
attempt discarded), with no allocation failure on the 6 GiB card.

KHR and DPAS alignment checks passed. DPAS SG8's builtin/layout self-test passed,
and every one of 131072 cumulative GEMM milestone words agreed with the CPU
reference. Every tested tile variant also built and verified a full mock share
through the real Rust proof implementation at difficulty 40.

Full zero-target scans at 32768², five attempts per configuration with the first
discarded, measured:

| Backend | Full attempt, TMAC/s | Scan, TMAC/s | Register spill, bytes/work-item |
|---|---:|---:|---:|
| KHR integer dot, 4x8 | 0.886 | 0.931 | not queried |
| DPAS 4x1 hash tiles/subgroup | 2.773 | 3.371 | 512 |
| DPAS 2x1 | 2.556 | 3.056 | 0 |
| DPAS 1x1 | 2.152 | 2.480 | 0 |

The default DPAS 4x1 setting was fastest on this driver despite register spills:
about 3.13 times KHR's full-attempt rate in these short sequential blocks. These
are measured complete scans, not extrapolated early-hit rates. They do not establish
performance on other Intel models or drivers.

Evidence: [alignment/proof cases](benchmarks/intel-a380-validation-2026-10-04.json),
[full scan timings](benchmarks/intel-a380-dpas-2026-10-04.json).
The full-size check is recorded [separately](benchmarks/intel-a380-size-2026-10-04.json).
Invalid environment/tile combinations were rejected normally and CTest passed;
see [boundary checks](benchmarks/intel-a380-boundaries-2026-10-04.json).

To reproduce the four backend/tile cases with an OpenCL build and an available
Intel GPU, use `python3 scripts/run_intel_dpas_validation.py --binary build/cppminer
--output build/intel-dpas-results`. This test uses a loopback pool and an artificial
wallet, checks alignment and Rust proofs, and measures complete zero-target scans.

For Intel plus NVIDIA, run separate OpenCL and CUDA processes with distinct worker
names. A single miner process selects one backend; CUDA `--devices` cannot include
an Intel adapter. Their independently generated A matrices give independent work
on the same pool job.

## Manual pool deployment

On 2026-10-04 the tested Intel binary was started manually on the local server
with `--ocl-dot dpas --verify`, full 131072² matrices and worker `arc-a380`, using
the same pool and wallet as the simultaneously running CMP50HX worker `cmp50hx`.
Its code comes from `6365a7040920f724a95efb6a6ad20ae8f7a5b0af`. The Intel process
uses its isolated OpenCL container; both mining containers have restart policy
`no`, with no miner startup service. ComfyUI remained active and its system-stats
endpoint, Ollama, Open WebUI, Speaches, DubPipe and the generation applications
responded to read-only health requests after deployment.

The short simultaneous mining audit confirms fresh completed attempts, executable
identity and launch settings, with no computation/proof error markers. Offline
alignment and real Rust mock-proof checks establish correctness independently;
the live sample does not establish long-term pool acceptance or sustained rates.
The final 40-second sample recorded two completed Intel attempts, with no recorded
errors and no submitted shares. During the same sample, CMP completed 37 attempts
and its locally verified share was accepted by the pool.

Evidence: [deployment](benchmarks/intel-a380-deployment-2026-10-04.json),
[simultaneous mining audit](benchmarks/intel-a380-combined-audit-2026-10-04.json).
