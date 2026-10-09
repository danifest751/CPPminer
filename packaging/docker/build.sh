#!/usr/bin/env bash
# build.sh PKG_DIR TAG [nvidia|amd|intel ...]
#
# Build the cppminer images from an unpacked Linux package directory (cppminer, kernels/,
# bundled .so files, cppminer-multi.sh), e.g. the HiveOS archive's cppminer/ directory.
#   ./build.sh /tmp/cppminer 0.5-fork.9            -> cppminer:nvidia-0.5-fork.9, :amd-..., :intel-...
set -euo pipefail
PKG=$1
TAG=$2
shift 2
VARIANTS=("$@")
[[ ${#VARIANTS[@]} -eq 0 ]] && VARIANTS=(nvidia amd intel)
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

CTX=$(mktemp -d)
trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/pkg"
cp -a "$PKG"/. "$CTX/pkg/"
# HiveOS-only files are not needed in the image
rm -f "$CTX"/pkg/h-*.sh "$CTX"/pkg/h-manifest.conf "$CTX"/pkg/README-hiveos.md
cp "$ROOT/packaging/common/cppminer-multi.sh" "$CTX/pkg/"
cp "$HERE/entrypoint.sh" "$CTX/"
for v in "${VARIANTS[@]}"; do
    docker build -f "$HERE/Dockerfile.$v" -t "cppminer:$v-$TAG" "$CTX"
done
docker image ls cppminer
