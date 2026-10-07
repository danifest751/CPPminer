#!/bin/bash
# check.sh: correctness on every AMD GPU before mining.
#   Pearl:   --align-test (GPU GEMM milestone words vs the CPU, plus the WMMA self-test that
#            picks the gfx12 operand layout) and a mock run to a verified zk-pow share.
#   Quantus: mock run (start-up self-test of the kernel variants, then a verified share).
# Results: logs/check-<time>/ and a summary at the end.
set -u
cd "$(dirname "$0")"
. ./common.sh
OUT="$LOGS/check-$(stamp)"; mkdir -p "$OUT"
devs=$(amd_devices)
[ -z "$devs" ] && { echo "[check] no AMD OpenCL device; run ./setup.sh first"; exit 1; }
echo "[check] devices: $devs   logs: $OUT"
summary="$OUT/summary.txt"; : > "$summary"
for d in $devs; do
  echo "[check] GPU $d: Pearl align-test"
  CP_OCL_WMMA_SELFTEST=1 timeout 900 "$BIN/cppminer" --backend opencl --devices "$d" --align-test > "$OUT/pearl-align-$d.log" 2>&1
  res=$(grep -qE "all tests passed" "$OUT/pearl-align-$d.log" && echo PASS || echo FAIL)
  kern=$(grep -m1 -oE "GEMM kernel: .*" "$OUT/pearl-align-$d.log")
  wmma=$(grep -E "WMMA self-test" "$OUT/pearl-align-$d.log" | tr '\n' ';')
  echo "GPU $d pearl align-test: $res | $kern | $wmma" | tee -a "$summary"

  echo "[check] GPU $d: Pearl mock share (zk-pow verify, a few minutes)"
  timeout 1800 "$BIN/cppminer" --backend opencl --devices "$d" --mock --mock-diff 16 > "$OUT/pearl-mock-$d.log" 2>&1
  res=$(grep -qiE "verify OK" "$OUT/pearl-mock-$d.log" && echo PASS || echo FAIL)
  rate=$(grep -oE "attempt timing: .*" "$OUT/pearl-mock-$d.log" | tail -1)
  echo "GPU $d pearl mock: $res | $rate" | tee -a "$summary"

  echo "[check] GPU $d: Quantus mock share"
  timeout 600 "$BIN/cppminer" --algo quantus --backend opencl --devices "$d" --mock --mock-diff 2e9 > "$OUT/qtc-mock-$d.log" 2>&1
  res=$(grep -qE "PASS: first share mined and verified" "$OUT/qtc-mock-$d.log" && echo PASS || echo FAIL)
  kern=$(grep -m1 -oE "kernel mul=[0-9]+ red=[0-9]+ local=[0-9]+" "$OUT/qtc-mock-$d.log")
  echo "GPU $d quantus mock: $res | $kern" | tee -a "$summary"
done
echo "[check] done:"; cat "$summary"
grep -q FAIL "$summary" && { echo "[check] some checks FAILED, send ./collect.sh output"; exit 1; }
echo "[check] all passed. Next: ./bench.sh"
