# cppminer in Docker

Three images, one per GPU vendor. Each runs one miner process per GPU and serves the stats API
(`/summary`, `/hiveos`, see `docs/api.md`) on port 4068 for the first GPU, 4069 for the second...

| Image | GPUs | Host needs |
|---|---|---|
| `ghcr.io/danifest751/cppminer:nvidia` | NVIDIA, CUDA (Turing / RTX 20 / CMP and newer) | driver 525+ and the NVIDIA Container Toolkit |
| `ghcr.io/danifest751/cppminer:amd` | AMD, OpenCL via ROCm 7.2 (RDNA2/3/4, CDNA) | amdgpu kernel driver (RDNA4: Linux 6.12+ or DKMS) |
| `ghcr.io/danifest751/cppminer:intel` | Intel Arc, OpenCL | i915/xe kernel driver |

Tags `nvidia`, `amd`, `intel` follow the latest release; `nvidia-0.5-fork.8` etc. pin a version.

## Run

Pearl on all NVIDIA GPUs:

```bash
docker run -d --name cppminer --restart unless-stopped --gpus all -p 4068:4068 \
  -e WALLET=YOUR_KRYPTEX_ACCOUNT_OR_PRL_ADDRESS -e ALGO=pearl \
  ghcr.io/danifest751/cppminer:nvidia
```

Quantus on AMD GPUs:

```bash
docker run -d --name cppminer --restart unless-stopped \
  --device /dev/kfd --device /dev/dri -p 4068:4068 \
  -e WALLET=YOUR_KRYPTEX_ACCOUNT -e ALGO=quantus \
  ghcr.io/danifest751/cppminer:amd
```

Intel Arc: `--device /dev/dri` and the `intel` image. The miner runs as root inside the
container, so the passed devices need no extra groups.

Logs: `docker logs -f cppminer`. Stats: `curl http://localhost:4068/summary`.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `WALLET` | required | Kryptex account or Pearl address |
| `ALGO` | `pearl` | `pearl` or `quantus` |
| `POOL` | Kryptex RU (`prl-ru` :7048 / `qtc-ru` :7049) | `stratum+tcp://host:port` |
| `WORKER` | container hostname | worker name |
| `PASS` | `x` | pool password (Kryptex Pearl: `d=N` fixed share difficulty) |
| `BACKEND`, `DEVICES` | auto | force a backend or a device list (one process then) |
| `EXTRA_ARGS` | empty | any other cppminer options |
| `API_PORT` | `4068` | first API port |

Instead of variables you can pass cppminer options as the container command:
`docker run ... ghcr.io/danifest751/cppminer:nvidia --pool stratum+tcp://... --wallet ... --algo quantus`.

`docker-compose.yml` here is a compose example.

## Build

From an unpacked Linux package (for example the HiveOS archive's `cppminer/` directory):

```bash
packaging/docker/build.sh /path/to/cppminer 0.5-fork.8 nvidia amd intel
```

## Fee

Pearl mining includes a 1% dev fee (the miner logs when it mines for it). Quantus has no fee.
