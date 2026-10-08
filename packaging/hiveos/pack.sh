#!/usr/bin/env bash
# pack.sh BUILD_DIR VERSION OUT_DIR [LIB ...]
#
# Build the HiveOS custom-miner archive OUT_DIR/cppminer-<VERSION with '-' -> '_'>.tar.gz from a
# Linux build (BUILD_DIR/cppminer and BUILD_DIR/kernels/*.cl) plus shared libraries to bundle
# (libcudart.so.12, libOpenCL.so.1, libgomp.so.1, ...). HiveOS takes the miner name from the
# archive name up to the last '-', so the version must not contain one.
#
# The binary should be built on an old glibc (Ubuntu 20.04, glibc 2.31) with
# LDFLAGS="-static-libstdc++ -static-libgcc" so it runs on current HiveOS images.
set -euo pipefail
BUILD=$1
VERSION=$2
OUT=$3
shift 3
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
HV=${VERSION//-/_}

STAGE=$(mktemp -d)
P="$STAGE/cppminer"
mkdir -p "$P/kernels"
cp "$BUILD/cppminer" "$P/"
cp "$BUILD"/kernels/*.cl "$P/kernels/"
for lib in "$@"; do cp -L "$lib" "$P/"; done
cp "$ROOT/packaging/common/cppminer-multi.sh" "$P/"
cp "$HERE/h-config.sh" "$HERE/h-run.sh" "$HERE/h-stats.sh" "$P/"
sed "s/@VERSION@/$VERSION/" "$HERE/h-manifest.conf" > "$P/h-manifest.conf"
cp "$HERE/README.md" "$P/README-hiveos.md"
cp "$ROOT/LICENSE" "$P/" 2>/dev/null || true
chmod 755 "$P/cppminer" "$P"/*.sh
chmod 644 "$P/h-manifest.conf" "$P"/kernels/*.cl

mkdir -p "$OUT"
tar -C "$STAGE" --owner=0 --group=0 -czf "$OUT/cppminer-$HV.tar.gz" cppminer
rm -rf "$STAGE"
ls -la "$OUT/cppminer-$HV.tar.gz"
