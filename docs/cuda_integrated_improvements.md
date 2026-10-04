# Integrated CUDA improvements for the next fork release

Validated on 2026-10-04 with the authorized NVIDIA CMP 50HX (sm75, 20 GiB),
CUDA 12.6.85 and the existing 225 W limit. The changes are included in
`release/fork` for the next release. No new tag, GitHub release or upstream PR
is created. GitHub Actions is not used.

## Included changes

1. **Operand loads and jackpot state.** The normal CMake CUDA build generates
   a private CUTLASS `memory.h` override in its build directory. The measured
   16-byte operand load uses `.cg` for sm75 code. The sm75 jackpot fold uses
   constant indices with predicates, retaining the sixteen transcript words
   in registers rather than dynamically indexing them. Vendor headers remain
   intact; configuration fails if the pinned load hook changes unexpectedly.
2. **Milestone placement.** The sm75 TensorOp loop folds the previous complete
   milestone at the start of the next K tile, before loads and before the next
   MMA modifies the accumulator. The post-loop callback flushes the last
   milestone once, with no extra tile. SIMT retains its previous placement.
3. **Incremental signal-A and keyed BLAKE3 trees.** The existing working-branch
   implementation is included in the next release source. It mutates the
   configured signal bytes and updates the affected cached-tree paths. The
   deployed CMP50 uses 4096 updates. Dense remains the software default;
   incremental mode is explicitly enabled on this server.

The kernel changes are guarded by `__CUDA_ARCH__ == 750`, which selects a
**compiled sm75 kernel**, not the physical GPU at runtime. Native sm86 and sm89
code retains the previous load/fold/milestone paths. A forward-JIT sm75 PTX path
can also carry the optimization onto another GPU; performance there is not
validated. No RTX3090 measurement or Windows runtime validation is claimed.

## Integrated build and correctness

The candidate is built from tracked source at base commit
`717823f5d6b20953e48738a3c27d85dc1ed01f37` plus the three production file changes
recorded in the [build manifest](benchmarks/cuda-integrated-build-2026-10-04.json).
Documentation and archived research tools added afterward do not change the
compiled production sources.

```sh
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release \
  -DCP_ENABLE_CUDA=ON '-DCP_CUDA_ARCH=75;86;89' \
  -DCP_ENABLE_CPU=ON -DCP_ENABLE_OPENCL=OFF \
  -DCP_ENABLE_CUBLAS=OFF -DCP_ENABLE_WGPU=OFF
cmake --build build/cmake -j4
ctest --test-dir build/cmake --output-on-failure
```

The build uses populated pinned dependencies and the real Rust
`libcp_proof_ffi.a`; the proof stub is rejected by the build helper. The
published v0.5-fork.4 binary is used as the comparison baseline.

Validation passed before deployment:

- Independent INT64 prefix-XOR and CPU keyed-BLAKE3 oracles for both
  128×128 and 256×128 tiles, four input patterns, including the null-dump mining
  branch: **393,216 comparisons**.
- Jackpot fold oracle: **1,048,576 state words and 65,536 BLAKE3 digests**.
- Incremental mutation, CPU roots, full-GPU subroots, re-keying, byte range and
  dirty-flag checks: **860 mutation rounds**, including full-size trees.
- All three CTest tests, the production 8192×8192 GPU alignment check and
  verified full-size mock proofs. The integrated miner passes dense and
  incremental mode for both tile sizes; incremental mock checks enable the
  full reference tree comparison.

Diagnostic capture exists only in private validation headers, not the deployed
miner. GPU counters are unavailable and sanitizer instrumentation was not
available in the earlier experiments; these are not claimed as passed checks.

## Complete-attempt performance

Fixed loopback work uses `131072 × 131072 × 4096`, rank 128, a zero target,
256×128 tiles and overlap enabled. The sequence is old/new/new/old, 30 complete
attempts per block, discarding the first four in each. A separate 24-attempt
conditioning block precedes the comparison. The table pools the elapsed
preparation-plus-scan time of **52 measured attempts per build**.

| Build and signal-A mode | Effective TMAC/s | Change |
|---|---:|---:|
| Published v0.5-fork.4, dense | 60.4612 | Reference |
| Integrated candidate, incremental / 4096 | 63.1840 | +4.50% |

This bounded integrated confirmation supports the earlier isolated improvements;
their individual percentages are not simply added together. It measures
completed computation, not earnings or accepted-share probability. ABBA order
and recorded telemetry help control thermal/clock drift, but do not replace a
longer soak or measurements on other GPUs.

Full numerical observations, timings, telemetry and oracle output:
[integrated miner results](benchmarks/cuda-integrated-miner-2026-10-04.json).
Prior evidence: [R1 integration candidates](cuda_research_validation.md),
[incremental validation](pearl_optimization_suite.md), and
[milestone scheduling](cuda_xor_milestones.md).

## Active CMP50 deployment

The candidate executable is `/w/working-three-improvements/build/cppminer`,
SHA256 `63e79a0cdf66aafcf5540a1b328d36024d1bd47daf54fb2974620b9a5d5547d2`.
It runs the existing pool, wallet and `cmp50hx` worker, with local proof
verification enabled. Launch settings:

```sh
CP_CUDA_A_MODE=incremental
CP_CUDA_A_UPDATES=4096
CP_CUDA_A_CHECK=0
CP_CUDA_OVERLAP=1
CP_CUDA_TB=256x128
```

The working directory and log destinations are preserved. The published v4
artifact remains available for rollback. Active deployment metadata is stored
at `/w/current-miner.json` and `/w/working-three-improvements/deployment-result.json`;
the candidate PID is in `/w/working-three-improvements/miner.pid`. The v4
deployment record is marked inactive. Wallet-bearing rollback and process
snapshots are private, mode 0600, and stay on the server.

The initial 30-second check confirms 25 fresh mining attempts, the incremental
cache initialization and exactly one mining worker. See the
[deployment audit](benchmarks/cuda-integrated-deployment-2026-10-04.json).
An independent follow-up audit verifies the executable hash, complete arguments
and environment, working directory, log destinations and unique worker, with
11 further attempts in 12 seconds:
[active-process audit](benchmarks/cuda-integrated-active-audit-2026-10-04.json).
