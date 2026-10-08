# cppminer on HiveOS

cppminer runs on HiveOS as a custom miner, for Pearl and Quantus, on NVIDIA (CUDA) and AMD
(OpenCL) rigs. It starts one process per GPU and reports hashrate, accepted/rejected shares,
temperatures and fans per GPU in the HiveOS dashboard.

## Flight sheet

Create a flight sheet with **Miner: Custom** and **Setup Miner Config**:

| Field | Value |
|---|---|
| Miner name | filled in from the URL (`cppminer`) |
| Installation URL | the `cppminer-<version>.tar.gz` link from the GitHub release |
| Hash algorithm | `pearl` or `quantus` |
| Wallet and worker template | `%WAL%.%WORKER_NAME%` |
| Pool URL | e.g. `stratum+tcp://prl-ru.kryptex.network:7048` (Pearl) or `stratum+tcp://qtc-ru.kryptex.network:7049` (Quantus) |
| Pass | `x` (Kryptex Pearl accepts `d=N` for a fixed share difficulty) |
| Extra config arguments | optional cppminer options, see below |

For Kryptex, the wallet is your Kryptex account (e.g. `krxXXXXXXX`); a Pearl address
(`prl1...`) works on pools that pay to addresses.

## Extra config arguments

Anything here is passed to every GPU process, e.g.:

- `--backend opencl` to mine NVIDIA GPUs through OpenCL instead of CUDA;
- `--devices 0` to run a single process on one GPU only (no per-GPU split);
- `--batch-size N` to set the Quantus launch size.

## What it does

- `h-run.sh` finds the GPUs (nvidia-smi for NVIDIA, the miner's OpenCL device list for AMD and
  Intel) and starts `cppminer --backend cuda|opencl --devices N` for each, with a stats API on
  port 4068, 4069, ... A process that exits is restarted after 10 s.
- `h-stats.sh` reads `http://127.0.0.1:PORT/hiveos` from every process and adds the HiveOS GPU
  temperatures and fans by PCI bus.
- The log is `/var/log/miner/custom/cppminer.log`; each line starts with `[gpuN]`.

Pearl hashrate is shown in H/s where one hash is one int8 multiply-accumulate, so TH/s on the
dashboard equals the TMAC/s the miner logs (the unit pools use for Pearl).

## Requirements

- HiveOS on Ubuntu 20.04 or newer (the binary needs glibc 2.31).
- NVIDIA: driver 525 or newer (CUDA 12). Turing (RTX 20, CMP) and newer.
- AMD: an image whose OpenCL runtime supports the GPU; RDNA4 (RX 9000, AI PRO R9700) needs
  ROCm 6.4 or newer.

## Fee

Pearl mining includes a 1% dev fee (the miner switches to the dev pool for about 1% of the work
and reports that in its log). Quantus mining has no fee.
