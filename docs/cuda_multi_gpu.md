# Independent CUDA attempts on multiple GPUs

The unreleased fork candidate fixes duplicated Pearl work in `--devices 0,1,2`.
Previously GPU0 prepared signal A, then copied its noisy A and jackpot key to every
other device. Each device scanned the same tile coordinates and the same matrix,
so additional GPUs repeated existing work. Proof readback also always selected GPU0.

Each CUDA device now prepares a fresh signal A, derives its own noise/jackpot key,
and retains its own Merkle commitment, sub-roots and incremental hash caches.
Noisy B remains shared because the zero-B construction makes it constant for a job.
Its one-time transfer uses explicit source/destination CUDA device IDs.
Scanning the same tile coordinates on different A matrices is independent work.
The winning device supplies both signal and witness readback. Aggregate scanned
tiles count every independent attempt and use 64-bit totals, including 8–16 GPUs.

Single-GPU prefetch and pipelined scanning retain the previous behavior. Multi-GPU
preparation currently runs serially before concurrent device scans; next-attempt
prefetch is disabled because its state is single-device. CPU-prepared/external
matrix inputs retain their original identity and distribute disjoint batches,
counted once. That compatibility path currently synchronizes each batch and does
not provide concurrent acceleration across devices. Duplicate CUDA device IDs
are rejected. Job cancellation is checked between both row and column batches.

## Validation on 2026-10-04

The server has one NVIDIA CMP50HX and one Intel Arc A380. Intel cannot act as a
second CUDA device. The harness therefore uses two independent logical contexts
with separate allocations on CMP50; it does **not** validate inter-device transfer
or simultaneous execution on two physical NVIDIA GPUs.

Dense, sparse and incremental modes passed on all three CUDA scan paths (CUTLASS
fused, period GEMM and the direct kernel), with the following checks across two jobs:

- Distinct A commitments and jackpot keys on every tested attempt.
- Aggregate work accounting for independent and shared-input scans.
- An actual GPU1 jackpot with no GPU0 hit, followed by winner signal/witness fetch.
- Winner root against CPU BLAKE3, byte-identical witness/full-host Rust proofs,
  and Rust verification against the actual jackpot target.
- Rejection of a witness paired with another device's commitment.
- Both CPU-upload APIs, cancellation and resource cleanup.

Production alignment, four dense/incremental × overlap off/on proof-verification
cases and CTest passed. A separate 40-attempt ABBA comparison at 131072², with two
warm-up attempts per block, measured 63.171 TMAC/s for the preceding integrated
candidate and 63.132 TMAC/s for this candidate (-0.062%). This short regression
check found no material single-GPU throughput change; it is not a multi-GPU
scaling measurement.

Evidence: [checks](benchmarks/cuda-multigpu-validation-2026-10-04.json),
[build](benchmarks/cuda-multigpu-build-2026-10-04.json),
[regression timings](benchmarks/cuda-multigpu-regression-2026-10-04.json).

## Manual deployment on the local server

The CUDA candidate built from `6365a7040920f724a95efb6a6ad20ae8f7a5b0af`
replaced the preceding candidate on CMP50HX on 2026-10-04. It retains the existing
pool, wallet, `cmp50hx` worker and incremental-A/single-GPU overlap configuration.
The previous binary remains available for rollback. An independent Intel OpenCL
process runs alongside it under worker `arc-a380`; see the
[Intel validation](intel_a380_dpas.md).

Both processes were started manually. Both containers have restart policy `no`,
and no miner startup service was installed. The fork branch holds the candidate;
no new release, tag or upstream PR was published and no GitHub workflow was run.
The simultaneous mining audit checks fresh completed attempts, executable identity,
launch settings and computation/proof error markers. Its short sample does not
establish long-term pool acceptance or physical multi-NVIDIA scaling.
The final 40-second sample recorded 37 completed CMP attempts and one locally
verified share accepted by the pool, with no rejected shares or recorded errors.

Evidence: [deployment](benchmarks/cuda-multigpu-deployment-2026-10-04.json),
[simultaneous mining audit](benchmarks/cuda-multigpu-combined-audit-2026-10-04.json).

## Reproducing the ownership test

Build a CUDA miner with real Rust proof FFI using CMake's Unix Makefiles generator,
then link the harness against its objects:

```sh
cmake --build build -j4
python3 scripts/build_cuda_multigpu_test.py --build build
CP_CUDA_A_MODE=dense build/cp_cuda_multigpu_test
CP_CUDA_A_MODE=sparse CP_CUDA_A_CHECK=1 build/cp_cuda_multigpu_test
CP_CUDA_A_MODE=incremental CP_CUDA_A_CHECK=1 build/cp_cuda_multigpu_test
CP_CUDA_A_MODE=incremental CP_CUDA_A_CHECK=1 build/cp_cuda_multigpu_test --period
CP_CUDA_A_MODE=incremental CP_CUDA_A_CHECK=1 build/cp_cuda_multigpu_test --scalar
# On a host with two CUDA GPUs, use distinct physical devices 0 and 1:
CP_CUDA_A_MODE=incremental CP_CUDA_A_CHECK=1 build/cp_cuda_multigpu_test --physical
```

The single-device logical setup exists only inside the test translation unit;
the production miner has no duplicate-device or injected-hit testing switch.
