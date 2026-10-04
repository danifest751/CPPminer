# Intel Arc A380: oneDNN systolic (XMX) backend

On Arc A380 the oneDNN/gemmstone backend with XeHPG systolic kernels mines at
**15.47 TMAC/s** per complete 131072² attempt, **4.8x** the OpenCL DPAS path
(3.20 TMAC/s) measured the same day on the same card. Use it on Arc:

```sh
cppminer --backend onednn --devices 0 --verify \
  --pool stratum+tcp://POOL:PORT --wallet WALLET --worker arc-a380
```

A pool share mined this way was accepted by HeroMiners (`"result":true`).
Results apply to this A380 and driver; other Arc models have not been tested.

## What changed

The `onednn` backend existed for Xe-LP/Xe-LPG iGPUs, which have no XMX. On the
A380 it had never selected a systolic kernel:

- **Milestone check.** oneDNN's catalog ranks ten systolic int8 kernels for
  DG2. Each unrolls K by 256–512 (`ks64`/`ks128` × `cab3`/`cab4`), while the
  Pearl milestone XOR fired only at unrolled-panel boundaries and therefore
  required `unrollK | 128`. All ten were rejected. When `unrollK` exceeds the
  128-term milestone, the XOR is now scheduled every 128 k inside the panel
  (`case5XorEveryK`, `k_loop.cxx`).
- **Fallback crash.** The non-catalog fallback candidates were flagged as
  catalog entries with a null entry and segfaulted in `evaluate()`.
- **Kernel choice.** oneDNN's model ranks plain GEMM. With the milestone fold,
  its first choice (`catalog-334`, 16x16) scanned at 9.7 TMAC/s. The 8x4
  work-group 16x32 entry (`catalog-333`/`352`) is now moved to the front for
  XeHPG; it is matched by strategy string and unroll, because the catalog
  repeats the string with a 16x16 unroll (`catalog-343`).
- **Panels.** Systolic kernels default to 1024x1024 hash tiles per panel
  (16384² GEMM per launch) unless `--batch-size`/`--row-period-batch` are
  given. `--batch-size 1024` was previously mistaken for the default and
  replaced by 256.
- **Matrix prep.** The row-major noise kernel used one work item per 4 KiB
  row with lane-strided byte accesses. A row-parallel uniform pass plus one
  work group per row (local-memory lookup, 16-byte stores) cuts A noise from
  154 to 56 ms, signal generation from 21 to 12.5 ms, prep from 200 to 96 ms
  per attempt, and per-job noisy B from ~200 to ~80 ms. Pool jobs arrived
  every 15–20 s during the test. The OpenCL DPAS backend shares this code.
- **Build.** `src/onednn/prepare_onednn_deps.sh` vendors oneDNN v3.13.2 on
  Linux; the oneDNN jackpot kernel is copied when OpenCL is also enabled.

## Measurements, 2026-10-04

Arc A380 (8086:56a5), Ubuntu kernel 6.8.0-142 with i915, PCIe 2.0 x1,
`intel-opencl-icd` 23.43.27642.40, IGC 1.0.15468.25, isolated container.
Complete zero-target scans on a loopback pool (no early hit), first attempt
discarded. Each case first builds and verifies a mock share through the Rust
proof verifier. Script: `scripts/run_intel_scan_bench.py`.

**Final, 131072², default flags, 5 attempts:**

| Backend | Full attempt, TMAC/s | Scan, TMAC/s |
|---|---:|---:|
| `onednn` (catalog-333, 1024x1024 panels) | **15.468** | 15.783 |
| `opencl --ocl-dot dpas` | 3.196 | 3.360 |

**Systolic catalog kernels, 32768², 256x256 panels:**

| Kernel | Unroll / work group | Full | Scan |
|---|---|---:|---:|
| catalog-333 / 352 | 16x32 / 8x4 | **11.387** | 13.178 |
| catalog-343 | 16x16 / 8x4 | 8.985 | 10.064 |
| catalog-353 / 334 | 16x16 / 8x8 | 8.705 | 9.714 |
| catalog-335 | 8x16 / 8x4 | 5.984 | 6.446 |
| catalog-337 | 8x8 / 8x8 | 3.651 | 3.818 |
| catalog-339 | 8x8 / 8x4 | 3.602 | 3.764 |
| catalog-341 | 16x4 / 2x8 | — | 16x4 hash tile; proof build fails |

**Panels for catalog-333 at 131072²:** 256x1024 hash tiles 14.30, 512x512
14.39, 512x1024 14.81, 1024x1024 **15.15** TMAC/s per full attempt (before
the prep change).

**Where the time goes:**

- Milestone XOR: skipping the fold (`CASE5_XOR_NOP=1`, diagnostic only) gives
  11.86 vs 11.39 TMAC/s, about 4%.
- Compute-bound: under load the card holds ~52 W of its 55 W `power1_max` and
  runs at 2000 MHz instead of 2450. Capping the clock at 1600 MHz lowers the
  scan rate from 15.79 to 12.55 TMAC/s (65536²), the same 1.25x ratio as the
  clocks, so DRAM bandwidth is not the limit.
- Larger custom tiles (32x32, 16x64 per thread; 8x8 work groups) either do
  not fit DG2's registers/SLM or are slower. Hilbert walk order, other
  B access widths and dropping `sr`/`pab` changed results within ±3%;
  removing boustrophedon walk order cost 27%. The fused in-kernel jackpot
  (`--fused-jackpot`) is 7% slower than the separate jackpot pass.

Raw data: [final](benchmarks/intel-a380-onednn-final-2026-10-04.json),
[kernels](benchmarks/intel-a380-onednn-kernels-b-2026-10-04.json),
[panels](benchmarks/intel-a380-onednn-panels-2026-10-04.json),
[XOR ceiling](benchmarks/intel-a380-onednn-xor-nop-2026-10-04.json),
clock [1600](benchmarks/intel-a380-onednn-clock1600-2026-10-04.json) /
[2450](benchmarks/intel-a380-onednn-clock2450-2026-10-04.json).

## Validation

- 20 mock shares with the final kernel built and verified through the Rust
  proof verifier, with winning tiles in both 16x16 halves of the 16x32 unroll
  (plus 13 with other catalog kernels).
- New noise kernels are byte-identical to the per-row kernels on full
  131072x4160 A and B (`CP_OCL_PREP_CHECK=1`).
- OpenCL DPAS mock proofs and `--align-test` still pass.
- One live pool share accepted.

Not established: long-term acceptance rate, other Arc models, the Windows
build of these changes, and layouts other than the default TN (an `NT` mock
proof failed and was not investigated).

## Tuning hooks

| Variable | Purpose |
|---|---|
| `CASE5_KERNEL=catalog-N` | Keep one catalog candidate |
| `CASE5_STRATEGY`, `CASE5_UNROLL=MxN` | Try a custom gemmstone strategy first (`_` may replace spaces) |
| `CASE5_XOR_NOP=1` | Diagnostic ceiling without the fold; shares will not verify |
| `CASE5_DEBUG_SELECT=1` | Log every candidate and why it was rejected |
| `CP_OCL_PREP_TIMING=1` | Per-stage prep timings |
| `CP_OCL_PREP_CHECK=1` | Compare the work-group noise path with the per-row kernel |
| `CP_OCL_PREP_WG=0` | Use the per-row noise kernels |

## Reproduce

```sh
cd src/onednn && ./prepare_onednn_deps.sh && cd ../..
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCP_ENABLE_OPENCL=ON -DCP_ENABLE_ONEDNN=ON
cmake --build build -j4
python3 scripts/run_intel_scan_bench.py --binary build/cppminer --output build/intel-bench \
  --size 128 --attempts 6 --case "onednn=--backend onednn" \
  --case "dpas=--backend opencl --ocl-dot dpas"
```

The remaining large lever is the 55 W power limit: at 2450 MHz the scan rate
should rise up to ~22%, if the card's power delivery allows it.
