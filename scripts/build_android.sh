#!/usr/bin/env bash
# The Android package for phones: one arm64 file with no libraries beside it, CPU backend, run from Termux or adb.
# Needs CMake, Ninja and the NDK (ANDROID_NDK_HOME, or ANDROID_NDK_LATEST_HOME on GitHub's runners).
# The proof library is left out (it needs a Rust cross build), so Pearl cannot mine in this package: TNet and Quantus can.
# OpenCL is left out because the binary would then refuse to start where the vendor's libOpenCL.so cannot be loaded;
# docs/tnet.md says how to build with it inside Termux.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
ndk="${ANDROID_NDK_HOME:-${ANDROID_NDK_LATEST_HOME:-}}"
[[ -n "$ndk" && -f "$ndk/build/cmake/android.toolchain.cmake" ]] || { echo "ANDROID_NDK_HOME is not an NDK" >&2; exit 1; }
out="$root/build/android-arm64"
package="cppminer-android-arm64"

blake3="$root/third_party/blake3"
if [[ ! -f "$blake3/blake3_neon.c" ]]; then
    mkdir -p "$blake3"
    for f in blake3.c blake3.h blake3_dispatch.c blake3_portable.c blake3_impl.h blake3_neon.c \
             blake3_sse2.c blake3_sse41.c blake3_avx2.c blake3_avx512.c; do
        curl -fsSL --retry 3 -o "$blake3/$f" "https://raw.githubusercontent.com/BLAKE3-team/BLAKE3/1.8.5/c/$f"
    done
fi

cmake -S "$root" -B "$out" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release -DCP_ENABLE_CPU=ON -DCP_ENABLE_OPENCL=OFF -DCP_PROOF_FFI=OFF -DBUILD_TESTING=ON
cmake --build "$out" --target cppminer cp_tnet_cpu_test

rm -rf "${out:?}/$package"
mkdir -p "$out/$package"
cp "$out/cppminer" "$out/cp_tnet_cpu_test" "$root/tests/data/tnet-v1-small.jsonl" "$root/LICENSE" "$out/$package/"
tar -C "$out" -czf "$root/$package.tar.gz" "$package"
echo "$root/$package.tar.gz"
