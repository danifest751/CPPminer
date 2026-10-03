# Pool reliability fixes for the next fork release

The changes on `fix/pool-session-watchdog` are intended for the preparing fork release.
They must be integrated into `release/fork` before building a new release; an existing
release binary is not updated by pushing this branch. No upstream PR is part of this batch.

| Fix | Result |
|---|---|
| Latest work and the idle/active transition | Pearl and Quantus share a bounded single pending job slot. Newer jobs replace older pending jobs. Publishing before or after mining begins cancels stale work. |
| Connection-local difficulty | Pearl resets difficulty to 32 on connect. Each received job owns its target and difficulty snapshot. |
| Backend/resource errors | Preparation, entropy, device and witness transfer failures propagate as `CP_JOB_ERROR` and terminate with status 1. Cancellation and exhausting `--max-nonce` remain separate outcomes. |
| Verification of targetless jobs | `--verify` checks a proof against the target computed when its job arrived, even without a target field in the notification. |
| Proof diagnostics | Verification uses the share's header/proof/target in memory. Only `--dry-run` writes diagnostics, with names scoped to process and job and checked write/close results. |
| Authorization result types | Numeric/string/array results and objects with a failed or malformed status cannot authorize mining. Boolean true and successful object responses remain supported. |
| Strict JSON | Token boundaries, object/array grammar, number syntax, UTF-8, Unicode escapes and surrogate pairs are validated. Duplicate decoded object keys are rejected. Parsing depth and network frame size are bounded. |
| Quantus numeric fields | `seq` is parsed directly as uint64, preserving values above 2^53. Negative, fractional, overflowing or wrong-type sequence values and non-positive/non-finite difficulty values are rejected. |
| Command-line validation | Unknown options, missing/empty values, malformed/overflowing numbers, partial dimension/device lists and overlong strings fail before backend or pool startup. |

## Local verification

The normal Windows CPU/OpenCL build and Linux CPU/CUDA sm_75 build pass all three
CTest executables. The Linux CPU ASan/UBSan build also passes those tests.
All three builds pass the 26 real-miner loopback integration scenarios, including
deadline expiry, job replacement, ACK tracking, compressed proofs, targetless
verification, concurrent dry-run artifacts and CLI validation.

```sh
cmake --build build -j4
ctest --test-dir build --output-on-failure
python3 -B tests/cp_pool_session_integration.py build/cppminer
```

Linux fault injection verifies that losing `/dev/urandom` during CPU matrix preparation,
host matrix preparation or Quantus nonce preparation exits with status 1:

```sh
gcc -shared -fPIC tests/cp_entropy_fault.c -ldl -o build/libcp_entropy_fault.so
python3 -B tests/cp_pool_session_integration.py build/cppminer build/libcp_entropy_fault.so
```

The fixture is used only by this local test command; it is not linked into the miner.
The integration suite takes about 90 seconds because it checks the real 30/60-second
timeouts. It needs a runnable CPU build, loopback sockets and Python; no real wallet,
external pool or GPU is needed. On Windows, use the `.exe` path and omit the Linux
fault library.

CMP50 hardware checks passed for duplicate/full job identity, cancellation of active
work and CUDA offline mining with proof verification. A live HeroMiners gzip share
was accepted. The original miner was then restored with the same wallet, pool and worker.

GPU device-loss recovery is intentionally not attempted. Resource-failure exits are
tested; recreating a lost GPU context would require a separately verified recovery
path. The local checks do not establish performance or compatibility on untested GPUs
(including RTX 3090, OpenCL hardware, oneDNN and wgpu).
DNS resolution timeouts and additional CPU checking of Quantus OpenCL shares are
outside this batch.

GitHub workflow files are unchanged. Commits use `[skip ci]`; no release build or
workflow dispatch is required for these local checks.
