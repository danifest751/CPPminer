# `--algo tnet` — TNet v1 (Requant)

Solo mining of [Requant](https://github.com/danifest751/requant) against a `requantd` node. The work
function is TNet v1 (`SPEC.md` there): a header-seeded int8 network on tensor cores; every 256-byte piece
of an output row is a lottery ticket. `requant` is accepted as an alias of `tnet`.

## Build

The backend needs CUDA and cuBLAS (int8 GEMM):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCP_ENABLE_CUDA=ON -DCP_ENABLE_CUBLAS=ON -DCP_CUDA_ARCH="75;86;89"
cmake --build build --target cppminer
```

Without cuBLAS the option is present but reports that the build cannot mine TNet.

## Run

```sh
requant-wallet keygen miner.key --network test          # prints the key hash
requantd --network test                                  # JSON-RPC on 127.0.0.1:19334
cppminer --algo tnet --rpc 127.0.0.1:19334 --payee <key hash> [--device 0] [--batch 8192]
cppminer --algo tnet --selftest                          # device SHA-256 and expansion vs host
```

- `--batch` is the number of rows per GPU pass (default 8192). Memory: 512 MiB of weights plus about
  `6 * batch * 8192` bytes; lower it on small GPUs. Rows are independent, so any batch is valid.
- The miner asks the node for work (`getwork`), polls the tip every second, refreshes work every 60 s and
  submits winning tickets (`submitwork`). The node re-verifies every claim.
- No miner fee: Requant funds development in its consensus rules (CHAIN.md §8 of the Requant repository).

## Measured

On a CMP 50HX (Turing) and an RTX 3090 (Ampere) the kernels match the Rust reference byte for byte; the
attempt costs 292 and 159.5 ns per ticket (Requant `RATIONALE.md`). End to end, the miner mined 20 regtest
blocks and 15 blocks at the TNet v1 parameters into `requantd`, every claim accepted by the node.
