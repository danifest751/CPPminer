# Stats API

`--api-port N` starts a read-only HTTP API on `127.0.0.1:N` (`--api-bind 0.0.0.0` to listen on
all interfaces, e.g. in a container). Mining continues if the port cannot be opened.

| Path | Content |
|---|---|
| `/summary` (also `/`) | full JSON summary |
| `/hiveos` | `{"khs": ..., "stats": {...}}` as a HiveOS `h-stats.sh` expects |

Example `/summary`:

```json
{
  "miner": "cppminer", "version": "0.5-fork.13", "algo": "quantus", "backend": "cuda",
  "worker": "rig01", "pool": "qtc-ru.kryptex.network:7049", "pool_connected": true,
  "uptime": 200,
  "hashrate": {"unit": "H/s", "windows": [10, 60, 900], "total": [133824333.9, 134612016.4, 134497562.2]},
  "devices": [{"id": 0, "name": "NVIDIA CMP 50HX", "pci": "0000:01:00.0", "bus": 1,
               "hashrate": [133824333.9, 134612016.4, 134497562.2]}],
  "shares": {"accepted": 5, "rejected": 0, "last_share_ago": 36}
}
```

- Hashrate is in H/s over the last 10 s, 60 s and 15 min. For Pearl one hash is one int8
  multiply-accumulate, so the value in TH/s equals the TMAC/s the miner logs.
- Per-device rates are the process total divided by its devices (exact with one GPU per process,
  which `packaging/common/cppminer-multi.sh` sets up).
- Shares count the pool's replies to this miner's submits; dev-fee sessions are not counted.
- `pci` is empty and `bus` is -1 when the driver does not report a PCI address (CPU backend).

## Optional owned-device control API (TNet CUDA)

The stats listener above remains read-only. TNet CUDA can additionally expose a
separate loopback-only control listener with explicit opt-in flags:

```text
--control-port PORT --control-owner WORKER_SERVICE_PUBLIC_KEY_HEX
--control-key-env PRIVATE_OWNER_CREDENTIAL_ENVIRONMENT_VARIABLE
```

The miner and AI worker remain separate CLI processes. This listener never
starts an AI runtime, changes pool payouts, changes clocks/power settings or
controls another process. Without these flags, ordinary mining is unchanged.
An enabled miner starts in `awaiting_operator` before CUDA initialization. The
owner must independently confirm model idleness and explicitly resume its fresh
instance/bootstrap lease before mining starts.

All requests require `Authorization: Bearer ...` and `X-ReQuant-Miner-Owner`.
Status discovery is `GET /v1/control/status`; mutations require the exact current
`X-ReQuant-Miner-Instance` and `X-ReQuant-Miner-Lease`.

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/v1/control/status` | Version, process instance, owner, device, state, lease and `gpu_quiescent` |
| POST | `/v1/control/quiesce` | Request a pause for one owned lease; wait for `quiescent` with `gpu_quiescent: true` before dispatching AI work |
| POST | `/v1/control/resume` | Resume only that exact instance/owner/lease after owned model cleanup is confirmed |

A pause is acknowledged only at a mining batch boundary after successful CUDA
synchronization. It retains mining allocations; pausing does not promise free
VRAM. The worker must still measure memory headroom before loading its model.
Uncertain model termination or a lost resume reply requires explicit matching
status/reconciliation. There is no expiry that resumes mining into potentially
active AI kernels. Credentials are separate from pool/wallet/customer keys.

For TNet mining, `--api-port PORT` now enables the existing stats listener on
loopback. The normalized TNet worker name, physical device PCI/UUID, work counter
and share responses populate `/summary`. `uuid` is additive; older callers may
continue using `pci`. UUID remains authoritative where a virtualized environment
does not expose a usable PCI address. TNet work counts tickets; these telemetry
counters are not accepted-target work proofs, payout authority or verified demand.
Stats are opt-in; the mining kernels, work encoding and block/pool rules are not
changed. CPU HTTP/control tests do not qualify a new CUDA binary or performance.

CPU-only protocol checks (the last harness stubs GPU mining):

```sh
c++ -std=c++17 -pthread -Wall -Wextra -Werror -Iinclude \
  src/common/cp_miner_control.cpp tests/cp_miner_control_test.cpp -o control-host
./control-host
python3 tests/cp_miner_control_http_test.py ./control-host
c++ -std=c++17 -pthread -Wall -Wextra -Werror -DCP_ENABLE_CUDA=1 -Iinclude \
  src/common/cp_api.cpp src/common/cp_miner_control.cpp src/tnet/cp_tnet.cpp \
  tests/cp_api_tnet_test.cpp -o stats-host
python3 tests/cp_api_tnet_http_test.py ./stats-host
```
