# danifest751/CPPminer — fork layout

This fork tracks [1640675651/CPPminer](https://github.com/1640675651/CPPminer) `dev` and publishes its own builds.

## Branches

| Kind | Naming | Based on | Goes upstream? |
|---|---|---|---|
| Upstream PRs | `fix/*`, `perf/*`, `feat/*`, `build/*` | upstream `dev` | Offered as a PR; the maintainer decides |
| Fork-only | `fork/*` | upstream `dev` | No — kept here on purpose |
| Release | `release/fork` | merge of the above | No — public builds are cut from it |
| Personal | `perf/all` | merge of the above + `fork/no-fee` | No — the maintainer's own rigs |

A release does not wait for upstream: every PR branch is in `release/fork` whether or not it has been merged upstream. When upstream merges or changes something, `dev` is merged into `release/fork` and `perf/all` (merge, never rebase — those branches are published).

## Fork-only branches

| Branch | What | Why it stays in the fork |
|---|---|---|
| `fork/dev-fee` | The 1% dev fee goes to the fork maintainer: Kryptex account `krxX8QJ872` (worker `devfee`) on Kryptex hosts, `prl1pk6ak3hy4rnv2gnkskrgc7xd0v5ngmkz993j75nswcqwf007t4jxqf8zx9y` elsewhere | Upstream keeps its own fee wallet |
| `fork/release-ci` | Windows CUDA/OpenCL/CPU build workflow (`.github/workflows/windows-cuda.yml`), this file | Release infrastructure of the fork |
| `fork/no-fee` | `--no-fee` flag | Personal builds only; never in public releases |

## Updating a release

```sh
git fetch origin                     # upstream
git switch release/fork
git merge origin/dev                 # upstream changes
git merge <new PR branches> <fork/* branches>
git push fork release/fork           # CI builds the Windows zip
git tag v0.5-fork.N && git push fork v0.5-fork.N
```

Linux builds are made in an Ubuntu 24.04 container with CUDA 12.6 (`./build.sh --backend cpu,cuda,opencl --cuda-arch "75;86;89"`).
