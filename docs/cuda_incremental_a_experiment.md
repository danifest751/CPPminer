# CUDA incremental signal-A experiment

This experiment is based on v0.5-fork.4 and is developed on
`perf/cuda-incremental-a`. It is opt-in and is not part of the published release.
The default CUDA path remains dense random signal-A generation.

## Hypothesis

Use spare VRAM to retain a signal matrix and its complete keyed BLAKE3 CV tree.
Change one random entry per column between attempts, then recompute only dirty
leaves and ancestors. A similar sparse mutation strategy already exists in the
CPU worker, but the incremental GPU tree is new.

Changing the signal commitment changes the A noise seed. The noisy A matrix is
still regenerated and the complete GEMM/jackpot scan still runs. Old jackpot
results cannot be reused across these attempts.

## Experimental controls

| Environment variable | Meaning |
|---|---|
| `CP_CUDA_A_MODE=dense` | Existing full random A generation and full hashing; default. |
| `CP_CUDA_A_MODE=sparse` | Persistent signal A, one random row/value write per column, full hashing. |
| `CP_CUDA_A_MODE=incremental` | Same mutation strategy, persistent CV tree, dirty-path hashing. |
| `CP_CUDA_A_CHECK=1` | Compare each experimental root and every witness subroot with full GPU hashing. Exclude this mode from performance measurements. |
| `CP_CUDA_OVERLAP=0` | Serial preparation and scanning, exposing preparation costs. |
| `CP_CUDA_OVERLAP=1` | Production default: next-A preparation overlaps current scanning. |

Sparse and incremental modes require a power-of-two number of 1024-byte chunks.
The tested CLI sizes `--m 1`, `--m 8` and `--m 128` satisfy this requirement.
Each signal buffer has its own cache; a different job key rebuilds that cache.
The tree contains non-root CVs. The ROOT flag is applied separately to produce
the final digest, preserving the witness subroots' existing representation.

At `m=131072`, the signal is 512 MiB. The CV tree uses 32 MiB and dirty flags
use 4 MiB per buffer. The normal two-buffer overlap adds 72 MiB in this mode.
No signal/noisy matrix is saved solely to retain rejected jackpot hashes.

## Independent correctness test

The standalone CUDA test compares roots with the vendored CPU BLAKE3
implementation and subroots with the existing full GPU hashing kernel. It covers
all-zero initialization, repeated mutations, overlapping dirty paths, re-keying,
signal bounds and clearing dirty flags. Chunk counts are 2, 4, 128, 256, 512,
4096, 32768 and 524288; the largest case is the full 512 MiB signal matrix.

Build on Linux, from the repository root with CUDA available:

```sh
mkdir -p build/incremental-test
for name in blake3 blake3_dispatch blake3_portable; do
    cc -O3 -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_AVX2 \
       -DBLAKE3_NO_AVX512 -Ithird_party/blake3 \
       -c third_party/blake3/$name.c -o build/incremental-test/$name.o
done
nvcc -std=c++17 -O3 -arch=sm_75 -Iinclude -Ithird_party/blake3 \
    tests/cp_cuda_incremental_a_test.cu build/incremental-test/*.o \
    -o build/incremental-test/cp_cuda_incremental_a_test
build/incremental-test/cp_cuda_incremental_a_test
compute-sanitizer --tool memcheck --error-exitcode 1 \
    build/incremental-test/cp_cuda_incremental_a_test --quick
```

Compute Sanitizer reports that GPU debugging features are disabled on the tested
CMP50. Its instrumentation could not run; this is not a passed memory check.
The independent CPU/GPU comparisons are available on that hardware.

## Benchmark method

Compare the published v0.5-fork.4 binary, experimental dense control, sparse/full
hash and incremental variants on the same CMP50, full `131072 × 131072 × 4096`
problem, fixed loopback header and all-zero target. No share can finish a scan
early. Run 18 complete attempts per variant, discard the first two for warmup,
and repeat in reverse order. Test both serial and overlapping preparation.

Report throughput over preparation plus scanning, in addition to scan-only
throughput. Keep the 225 W power limit and unchanged driver clock policy; record
temperature and clock observations. Instantaneous power samples are not an
energy-efficiency measurement. Validation checks are disabled during timing.

The production process is stopped only for hardware tests. Its command line,
environment, working directory and log paths are saved for restoration. A final
live test uses the same wallet/pool with a temporary worker and `--verify`.
The v0.5-fork.4 miner is restored after the experiment, including failures.

## Results

Measured on 2026-10-04 (Asia/Yekaterinburg), NVIDIA CMP 50HX with 20 GiB VRAM, CUDA 12.6.85, 225 W limit. Memory clock observations were 7000 MHz; steady SM clock observations were 1935 MHz.

| Mode | Serial effective TMAC/s | Serial prep | Overlap effective TMAC/s | Overlap gain vs release |
|---|---:|---:|---:|---:|
| published | 59.68 | 26.81 ms | 60.20 | +0.00% |
| dense | 59.68 | 26.75 ms | 60.20 | +0.01% |
| sparse | 59.92 | 22.97 ms | 60.46 | +0.44% |
| incremental | 60.55 | 8.97 ms | 61.05 | +1.42% |

The 16 groups contain 288 complete attempts, of which 256 are included after warmup. The dense experimental control agrees with the published binary. Incremental mode improves the warmed overlap result by 1.42% while reducing serial preparation from about 27 to 9 ms. This is a short fixed-work benchmark, not a measurement of pool earnings or broad GPU compatibility.

Independent checks passed 172 CPU-root/full-GPU-subroot comparisons. Six verified mock runs covered 1K, 8K and 128K matrices, with and without overlap. Another 25 prepared trees passed full-reference checks during repeated attempts. Duplicate/changed-job identity and active cancellation checks passed. The three existing CTest executables also passed.

The live HeroMiners test produced a share that passed local proof verification and was accepted by the pool with gzip encoding. The test performed 322 full-reference tree checks across 307 scan attempts, with no verification failure. One accepted share validates this example; it does not measure the long-run accepted-share rate.

The accepted proof compressed from 90,560 to 2,240 base64 characters (40.4x).
The sparse signal strips contain many zero bytes, reducing the wire payload;
this one proof is not a measurement of average compression across shares.

The original v0.5-fork.4 executable, wallet, pool, worker, environment and arguments were restored and checked against the published binary checksum. GitHub Actions were not used. The experiment remains opt-in on its own branch; no release binary is replaced by this change.

Recommendation: retain the prototype for a longer pool comparison and GPU memory instrumentation on hardware that supports Compute Sanitizer before considering a default change. The measured improvement is modest because production already overlaps preparation with scanning.

Raw measurements: [CMP50 experiment results](benchmarks/cmp50-incremental-a-2026-10-04.json).
