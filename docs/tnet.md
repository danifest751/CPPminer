# `--algo tnet` — TNet v1 (Requant)

Mining of [Requant](https://github.com/danifest751/requant) in a Requant pool or against a `requantd` node.
The work function is TNet v1 (`SPEC.md` there): a header-seeded int8 network on tensor cores; every
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

## Run

```sh
# in the test-network pool (no node needed); rewards to the key hash, statistics per --worker
cppminer --algo tnet --rpc 193.187.93.29:19340 --payee <key hash> --worker rig1

# solo, against your own node
requant-wallet create miner.wallet                 # a wallet; `address miner.wallet` prints the key hash
requantd --network test                            # JSON-RPC on 127.0.0.1:19334
cppminer --algo tnet --rpc 127.0.0.1:19334 --payee <key hash> [--device 0] [--batch 8192]

cppminer --algo tnet --selftest                    # SHA-256, expansion and the int8 GEMM vs the CPU
```

- `--batch` is the number of rows per GPU pass (default 8192). Memory: 512 MiB of weights plus about
  `6 * batch * 8192` bytes; lower it on small GPUs. Rows are independent, so any batch is valid.
- Network threads submit winning tickets (all of them, up to 16 per pass) and watch the tip, so the GPU
  never waits for the network. The tip is watched by a long poll (`getwork payee longpollid`, answered
  when the next block arrives; Requant node 0.13.0 and pool 0.14.0 or newer), else by asking every
  second; the log says which (`tip watched by long poll` / `no long poll here`). Work is refreshed on a new tip and every 60 s; a share
  found on a tip the miner has left is counted as `stale` and not sent. The pool or node re-verifies
  every claim.
- No miner fee: Requant funds development in its consensus rules (CHAIN.md §8 of the Requant repository).

## Measured

CMP 50HX (Turing, 225 W): 3.61 M tickets/s (277 ns per ticket) with the CUTLASS GEMM, 3.42 with cuBLAS,
in the test-network pool, every share accepted. The kernels match the Rust reference byte for byte on
Turing and Ampere. Ampere, Ada and Blackwell use the second kernel, not yet measured against cuBLAS.
