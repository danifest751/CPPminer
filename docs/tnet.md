# `--algo tnet` — TNet v1 (Requant)

Mining of [Requant](https://github.com/requant-network/requant) in a Requant pool or against a `requantd` node.
The work function is TNet v1 (`SPEC.md` there): a header-seeded int8 network, made for tensor cores; every
256-byte piece of an output row is a lottery ticket. `requant` is accepted as an alias of `tnet`.

## Build

Any CUDA build includes it; the int8 GEMM is CUTLASS, compiled into the miner (no cuBLAS or other
libraries to ship):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCP_ENABLE_CUDA=ON -DCP_CUDA_ARCH="75;86;89"
cmake --build build --target cppminer
```

Two GEMM kernels: `mma.m8n8k16` for Turing (RTX 20xx, CMP 30HX–90HX) and `mma.m16n8k32` with `cp.async`
stages for Ampere, Ada and Blackwell. At start-up the miner checks the kernel against the CPU on random
matrices; if the newer one disagrees it takes the Turing kernel, which runs on every GPU since Turing.
`CP_TNET_GEMM=sm75|sm80` chooses by hand. A build with `-DCP_ENABLE_CUBLAS=ON` can also run cuBLAS
(`CP_TNET_CUBLAS=1`) for comparisons.

The requantization of each layer (`clamp((y M + 2^23) >> 24, -128, 127)`) runs in the GEMM's epilogue, so
the int32 product never goes to memory; the miner checks it against the CPU at start-up (both clamps) and
falls back to a separate pass if it disagrees. `CP_TNET_FUSED=0` forces the separate pass.

## OpenCL (AMD, Intel and other GPUs)

A build with `-DCP_ENABLE_OPENCL=ON` mines TNet on any OpenCL GPU: `--backend opencl` (the default only where
CUDA is not built), `--device N` the flat index from `cppminer --backend opencl --list-devices`. The kernels
(`kernels/tnet.cl`) run the int8 GEMM with the requantization in its epilogue, on AMD's dot instructions
(`v_dot4_i32_iu8` on RDNA3/4, `v_dot4_i32_i8` on GCN5/RDNA2), the KHR integer dot product, or plain
multiply-adds: at start-up the miner builds them in turn and keeps the first that agrees with the CPU
(`CP_TNET_OCL_DOT=0..3` forces one). The network and the mining loop are the same as on CUDA.

```sh
cppminer --algo tnet --backend opencl --selftest
cppminer --algo tnet --backend opencl --bench --batch 1024
cppminer --algo tnet --backend opencl --rpc 193.187.93.29:19340 --payee <key hash> --batch 1024
```

On Intel GPUs with XMX the GEMM runs on the matrix engines: through `libcp_esimd` (`kernels/libcp_esimd.so`,
`cp_esimd.dll`; the ESIMD library of the Pearl scan, which now also has `cp_esimd_tnet_gemm`) when it loads, else
through OpenCL DPAS (`cl_intel_subgroup_matrix_multiply_accumulate`). Both read the operands packed in DPAS blocks:
the weights once per epoch, the activations as the expansion and every layer but the last write them. The ESIMD GEMM
takes the batch in whole 512-row tiles (`--batch 8192` or any multiple of 512). `CP_TNET_NOESIMD=1` and
`CP_TNET_OCL_NODPAS=1` turn either off.

| GPU | GEMM | ns per ticket | TMAC/s | Pool |
|---|---|---:|---:|---|
| Arc A380 | ESIMD `dpasw` (`--batch 8192`) | 1131 | 14.8 | 18 shares in 300 s, none rejected |
| Arc A380 | OpenCL DPAS, packed operands | 9890 | 1.7 | |
| Arc A380 | OpenCL KHR dot product | 16 270 | 1.0 | 2 shares and a block in 600 s |
| Radeon 780M | OpenCL `sudot4` | 7 000 | 2.4 | 8 shares in 600 s, none rejected |

The OpenCL DPAS path is limited the same way as for Pearl (2.8-3.2 TMAC/s on the A380): Intel's OpenCL compiler
does not schedule DPAS well; ESIMD with `dpasw` and L1 prefetch reaches the matrix engines. The 780M has no XMX; its
next step would be WMMA (about 1.7x over sudot4 for Pearl).

## Phones (Android) and other CPUs

`--backend cpu` is the default on Android. It needs no GPU and no libraries: the package
`cppminer-android-arm64.tar.gz` (`scripts/build_android.sh`) is one arm64 file for Android 8 or newer, run from
[Termux](https://termux.dev) or `adb shell`.

```sh
tar xzf cppminer-android-arm64.tar.gz && cd cppminer-android-arm64
./cppminer --algo tnet --selftest
./cppminer --algo tnet --bench --seconds 20
./cppminer --algo tnet --rpc pool.requant.network:19340 --payee <key hash> --worker phone
```

It takes about 700 MiB of memory: 512 MiB of weights, the rest for a batch of rows. `--threads N` uses fewer cores and
runs cooler; `--batch` is 256 rows here, a pass of about a second. The miner runs at low priority, so the phone stays
usable, but all its cores are busy: expect heat and a battery that drains even on a weak USB port. Keep it on a charger
and out of a case. Android slows the cores down by itself when they get hot, so the rate falls within a minute or two.
At 45 C at the battery the miner pauses until it is down to 42 C (`CP_TNET_MAX_TEMP` sets the limit in degrees, 0 turns it
off); where Android hides the battery's temperature this check does nothing.

The kernel is AArch64 DotProd (`sdot`; Cortex-A55, A75 and newer), chosen at start-up. Other CPUs run a plain C++ kernel
that the compiler vectorizes: it is right, but much slower, and is not meant for mining on x86. `CP_TNET_CPU_KC` and
`CP_TNET_CPU_NC` change the block sizes; on the phone below 512-4096 and 64-256 made no difference above the noise.

Android switches cores off and on, and a thread's affinity stays narrowed to those that were on when it started, so the
miner asks for all of them again before each pass. Without this it ended up on three slow cores and a middle one
(233 us per ticket against 121).

| Device | Backend | us per ticket | Tickets/s | GMAC/s |
|---|---|---:|---:|---:|
| Snapdragon 860, cool | cpu, DotProd, 8 threads | 78 | 12 800 | 214 |
| Snapdragon 860, after minutes of mining | cpu, DotProd, 8 threads | 120-145 | 7 000-8 300 | 115-140 |
| Adreno 640, the same phone | opencl, `rq_gemm_direct` | 972 | 1 030 | 17 |
| Adreno 640, the kernel other GPUs use | opencl, `rq_gemm_rq` | 14 700 | 68 | 1.1 |

A pool share is 2^24 tickets (`share_bits` in the log), so the hot phone finds one in about 35 minutes. The package mined block 11820 of the test network this way, from a POCO X3 Pro over `adb shell` on mobile data.

### A phone's GPU

`--backend opencl` needs a build that links the vendor's `libOpenCL.so`, so the package leaves it out: build in Termux
(`./build.sh --backend cpu,opencl`, see `docs/termux_note.md`), or link the phone's library in a cross build and run with
`LD_LIBRARY_PATH=/vendor/lib64:/system/lib64`. It is not the default, as on the phone measured the CPU is 7 to 12 times
faster. Adreno 640 has no integer dot product (43 G multiply-adds per second, several of them per product of bytes).

On Adreno and Mali the miner uses `rq_gemm_direct`, without local memory or barriers, which were 13 times slower there,
and cuts each pass into pieces of about 40 ms (`--batch` defaults to 64 rows). This is not only speed: Android cannot
draw the screen while the GPU holds a long pass, and one of several seconds froze the phone until it rebooted.

## Run

```sh
# in the test-network pool (no node needed); rewards to the key hash, statistics per --worker
cppminer --algo tnet --rpc 193.187.93.29:19340 --payee <key hash> --worker rig1

# solo, against your own node
requant-wallet create miner.wallet                 # a wallet; `address miner.wallet` prints the key hash
requantd --network test                            # JSON-RPC on 127.0.0.1:19334
cppminer --algo tnet --rpc 127.0.0.1:19334 --payee <key hash> [--device 0] [--batch 8192]

cppminer --algo tnet --selftest                    # SHA-256, expansion, ticket hash and both GEMMs vs the CPU
cppminer --algo tnet --bench [--seconds 20]        # offline: ns per ticket and the time of each phase
```

- `--batch` is the number of rows per GPU pass (default 8192). Memory: 512 MiB of weights plus about
  `2 * batch * 8192` bytes (`6 *` with `CP_TNET_FUSED=0`); lower it on small GPUs. Rows are independent, so any batch is valid.
- Network threads submit winning tickets (all of them, up to 16 per pass) and watch the tip, so the GPU
  never waits for the network. The tip is watched by a long poll (`getwork payee longpollid`, answered
  when the next block arrives; Requant node 0.13.0 and pool 0.14.0 or newer), else by asking every
  second; the log says which (`tip watched by long poll` / `no long poll here`). Work is refreshed on a new tip and every 60 s; a share
  found on a tip the miner has left is counted as `stale` and not sent. The pool or node re-verifies
  every claim.
- No miner fee: Requant funds development in its consensus rules (CHAIN.md §8 of the Requant repository).

## Measured

CMP 50HX (Turing, 225 W): 4.18 M tickets/s (239 ns per ticket), in the test-network pool, every share
accepted (52 shares and 4 blocks in 240 s). `--bench` on the same card, ns per ticket:

| Version | ns/ticket | Batch of 8192 rows: expand / 8 layers / hash, ms |
|---|---:|---:|
| separate requantization pass, 128 x 256 tile, byte-wise SHA kernels (v0.5-fork.10) | 277.0 | 2.71 / 68.5 / 1.36 |
| requantization in the epilogue | 256.0 | 2.76 / 63.0 / 1.36 |
| + unrolled SHA-256, vector stores, precomputed ticket tail | 244.1 | 0.46 / 63.1 / 0.37 |
| + 256 x 128 tile (128 x 128: 273.5) | 239.1 | 0.46 / 61.8 / 0.37 |

The layers now run at 71 TMAC/s (the card's square int8 GEMM reaches about 78). cuBLAS gave 3.42 M
tickets/s with the separate pass. Larger batches do not help (16384-65536 rows: 240-243 ns).

RTX 3090 (Ampere, rented VM, sm80 kernel), `--bench` with the SM clock locked at 1110 MHz so that heat does
not skew the comparison: requantization as a separate pass 208.4 ns/ticket, in the epilogue 193.1 ns/ticket
(5.18 M tickets/s, 87 TMAC/s, about 92% of the card's tensor peak at that clock). At its stock clocks the
card reaches 152-156 ns/ticket while cool, and settles near 180 ns once the core is at 83 C and the clock
drops to about 1200 MHz: there the limit is cooling and power, not the kernel. The kernels match the Rust reference byte for byte on
Turing and Ampere. Ampere, Ada and Blackwell use the second kernel, not yet measured against cuBLAS.
