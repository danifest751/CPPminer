# RTX 30 series / sm86 experiment bundle

This is a research package based on the fork's working CUDA implementation.
Production kernel defaults, releases, CI and deployed miners are unchanged.
Prepare every binary on the local build server before renting an RTX 3090.
Speed and GPU correctness of the new kernels remain unverified until an sm86
GPU executes the bundled tests. Compilation and CPU tests are separate evidence.

## Candidates

| Variant | Difference from the control |
|---|---|
| baseline | Current sm86 kernel, 64x64 warp, three stages, group-8 traversal |
| before_loads | Defer the previous completed milestone to before the next tile's shared loads |
| async_shadow | Issue the next async copy group, fold the previous milestone, then load shared fragments |
| predicated | Constant-index transcript updates on sm80+ instead of dynamic indexing |
| before_predicated | Before-loads scheduling plus constant-index transcript |
| shadow_predicated | Async-shadow scheduling plus constant-index transcript |
| redux | Native subgroup XOR instead of shuffle reduce-scatter |
| redux_predicated | Native XOR plus constant-index transcript |
| shadow_redux_predicated | Native XOR, async-shadow scheduling and constant-index transcript |
| stages2 | Two-stage CUTLASS specialization; report its actual pipeline type |
| stages4 | Four stages; assess occupancy and register pressure |
| warp32 | 32x64 warp tile instead of 64x64; explicit launch bounds keep large CTAs within the register limit |
| cache_a_ca | Async A copies use `.ca` rather than `.cg` |
| cache_b_ca | Async B copies use `.ca` rather than `.cg` |
| cache_ab_ca | Both operands use `.ca` |
| group4 | Group four N tiles in the traversal |
| group16 | Group sixteen N tiles in the traversal |

Every candidate includes 128x128, 256x128 and 128x256 threadblock shapes.
The native XOR candidate executes every reduction uniformly within its
eight-lane subgroup. Calling a reduction only for a lane's owned partial would
mix different inputs across the subgroup and is deliberately avoided.

Callback candidates preserve every prefix before the next MMA changes the live
accumulator. The last callback is flushed once without an extra K iteration.
No candidate removes milestones, transcript words or BLAKE3 computation.

## Prepare before renting

Requires Linux, CUDA 12.6, CMake, a C/C++ compiler, populated pinned dependencies
and a real `rust/cp-proof-ffi/target/release/libcp_proof_ffi.a`.
The builder rejects a proof stub and edits only its own scratch headers.

```sh
python3 -B tests/cp_ampere_research_test.py
python3 scripts/build_cuda_ampere.py --output build/ampere --jobs 2
# Resume an interrupted build only if all source/dependency hashes still match:
python3 scripts/build_cuda_ampere.py --output build/ampere --jobs 2 --resume
# For Python-only research/tool edits, compare regenerated headers/profile and
# exact compiler commands before reusing a previously compiled variant:
python3 scripts/build_cuda_ampere.py --output build/ampere --jobs 2 --refresh-tools
```

The output is `build/ampere/ampere-sm86.tar.gz`. It carries seventeen standalone
profilers, seventeen exact-oracle executables, seventeen full miners with the real
Rust proof verifier, the runner, loopback pool helper, CUDA runtime and a SHA256
manifest. Source/build commands and generated-header hashes are recorded.
No compiler or project checkout is needed on the rented server.

The prepared binaries target Linux x86_64 / Ubuntu 24.04 (glibc 2.39 or newer),
Python 3.10+, libgomp/libstdc++ and an NVIDIA driver compatible with CUDA 12.6.
Select an idle RTX 3090 or another sm86 RTX 30 GPU. This bundle deliberately
rejects CMP50HX, A100, Ada and other architectures. It does not establish that
an optimal 3090 configuration is optimal for every RTX 30 card.
For older host userspace, the included Dockerfile provides Ubuntu 24.04 through
NVIDIA Container Toolkit. It can be built before the rental; it has no compiler.

```sh
# From the directory containing the unpacked bundle/:
docker build -f bundle/Dockerfile -t cppminer-ampere-test .
mkdir -p results-3090
docker run --rm --gpus all -v "$PWD/results-3090:/results" cppminer-ampere-test
```

## Run on the rental

Upload the archive, then:

```sh
tar -xzf ampere-sm86.tar.gz
cd bundle
python3 run_cuda_ampere.py --device 0 --output results-3090 --budget-seconds 2700
```

The 45-minute budget reserves time for upload, inspection and downloading
results during a one-hour rental. Runtime varies by host; a budget exhaustion
preserves partial results with `completed: false`. It never reports an
unfinished suite as a confirmed winner. If necessary, select a focused subset:

```sh
python3 run_cuda_ampere.py --output results-focused --variants \
  baseline before_loads async_shadow predicated before_predicated shadow_predicated
```

Default sequence:

1. Validate all bundle SHA256 hashes, physical GPU architecture and idle state.
2. Verify every candidate on four input patterns, all three shapes, every prefix
   XOR and every keyed CPU BLAKE3 digest, in both dump and null-dump mining paths.
   Each candidate checks 589,824 words (10,027,008 for seventeen candidates).
   A failed candidate is recorded and excluded; a failed baseline stops the run.
3. Screen the valid candidates in four alternating forward/reverse rounds.
   Each panel has CPU prefix samples, forty warmups and forty timed launches.
   Record registers, local bytes, shared bytes, active blocks/SM, actual pipeline,
   temperature, clocks and power. All timed kernels retain the complete hash.
4. Select three candidate/shape pairs. Run production GPU alignment and real
   Rust-verified mock proofs for them and the control in dense/incremental modes.
5. Confirm complete-attempt rates against the original baseline/128x128 in ABBA
   order, separately for dense and incremental A. Each block has 24 attempts,
   with four discarded warmup attempts. Rates include preparation plus scan.

All fixed work goes to a zero-target loopback job with M=N=131072, K=4096,
rank 128 and 4096 signal-A updates. No real wallet or remote pool is used.
The runner never stops unrelated processes or changes clocks/power limits.
SIGINT, SIGTERM and the deadline stop only its own test children.

Optional `--ncu` captures one baseline Nsight Compute report if installed and
GPU performance counters are permitted. Missing tooling/permissions are
reported, without inventing a stall attribution from static SASS counts.

## Results and selection

Download the entire result directory before ending the rental. `results.json`
contains per-run measurements, proofs, rejected candidates, ABBA comparisons
and telemetry; `SUMMARY.md` contains confirmed full-attempt rate differences.
All raw test logs remain beside them. A panel-only improvement is insufficient
to change the production default, and accepted-share counts are not a speed
metric. Review both ABBA pairs and dense/incremental behavior before integration.

The next integration should contain only confirmed winners, with another
correctness pass after combining them. This package does not make a release or
send a PR upstream.

## Preparation checks (2026-10-04)

- All seventeen native sm86 variants compile, including complete miners with
  the real proof FFI. Compilation does not validate execution on RTX hardware.
- Six CPU tests cover distributed XOR equivalence for both warp geometries,
  incomplete/nonfinite result rejection, bundle integrity, full-attempt accounting
  and current source hooks.
- All seventeen miners' CPU mock shares build and pass the actual Rust verifier.
- Baseline, before-loads and async-shadow callback schedules pass 196,608 word
  comparisons each through the sm75 synchronous multistage fallback, including
  every prefix and both dump/mining digest paths. This validates callback/K
  ordering; it does not validate native Ampere async behavior or native REDUX.
- The original native profile uses a 128-byte stack frame. The predicated
  variant removes it, while scheduling/predication combinations can spill.
  Native XOR reduces registers in the compiled control. These are compiler
  observations, not measured performance gains.

The clean Ubuntu 24.04 runtime image also passes bundle integrity, dependency
resolution, the real CPU proof and the sm75 rejection guard through the NVIDIA
container runtime. The production CMP and Intel processes remain running.

See [preparation evidence](https://github.com/danifest751/CPPminer/blob/research/ampere-sm86/docs/benchmarks/ampere-preparation-2026-10-04.json).
