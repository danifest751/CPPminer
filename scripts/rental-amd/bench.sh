#!/bin/bash
# bench.sh [SECONDS]: offline speed of every kernel variant on every AMD GPU, then all GPUs at
# once with the defaults. Pearl reports TMAC/s per attempt, Quantus MH/s. rocm-smi telemetry
# (power, clocks, temperature) is sampled alongside when available.
# Results: logs/bench-<time>/summary.txt
set -u
cd "$(dirname "$0")"
. ./common.sh
SEC=${1:-120}
OUT="$LOGS/bench-$(stamp)"; mkdir -p "$OUT"
devs=$(amd_devices)
[ -z "$devs" ] && { echo "[bench] no AMD OpenCL device; run ./setup.sh first"; exit 1; }
summary="$OUT/summary.txt"; : > "$summary"
echo "[bench] devices: $devs, $SEC s per run, logs: $OUT"

telemetry_start() {
  if command -v rocm-smi >/dev/null; then
    ( while true; do echo "== $(date +%T)"; rocm-smi --showpower --showclocks --showtemp 2>/dev/null |
        grep -E "^(GPU|card)\[" ; sleep 5; done ) > "$1" 2>&1 &
    TELE=$!
  else TELE=; fi
}
telemetry_stop() { [ -n "${TELE:-}" ] && kill "$TELE" 2>/dev/null; }

# Median of the last half of the "attempt timing" rates (Pearl) in a log, in TMAC/s.
pearl_rate() {
  local vals n
  vals=$(grep -oE "attempt timing: .*" "$1" | grep -oE "[0-9.]+ [TG]MAC/s" |
         awk '{print ($2 == "GMAC/s") ? $1 / 1000 : $1}')
  n=$(printf "%s\n" "$vals" | grep -c .)
  [ "$n" -eq 0 ] && { echo "n/a"; return; }
  printf "%s\n" "$vals" | tail -n $(( (n + 1) / 2 )) | sort -n |
    awk -v n="$n" '{a[NR] = $1} END {printf "%.2f TMAC/s (%d attempts)", a[int((NR + 1) / 2)], n}'
}
qtc_rate() { grep -oE "\[qpow\] [0-9.]+ MH/s" "$1" | tail -1 | grep -oE "[0-9.]+ MH/s" || echo n/a; }

run_pearl() { # dev name env... -- args...
  local d=$1 name=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env "${envs[@]}" timeout "$SEC" "$BIN/cppminer" --backend opencl --devices "$d" --mock --mock-diff 1e12 "$@" \
    > "$OUT/pearl-$name-$d.log" 2>&1
  local kern; kern=$(grep -m1 -E "^\[ocl\] .*: (OK|BUILD FAILED|SELF-TEST FAILED)$" "$OUT/pearl-$name-$d.log" | sed 's/^\[ocl\] //')
  echo "GPU $d pearl $name: $(pearl_rate "$OUT/pearl-$name-$d.log") | $kern" | tee -a "$summary"
}

telemetry_start "$OUT/telemetry-single.txt"
for d in $devs; do
  run_pearl "$d" wmma           -- --ocl-dot auto
  run_pearl "$d" wmma-nopipe    CP_OCL_WMMA_PIPELINE=0 -- --ocl-dot auto
  run_pearl "$d" sudot4         -- --ocl-dot sudot
  timeout "$SEC" "$BIN/cppminer" --algo quantus --backend opencl --devices "$d" --mock --mock-diff 1e15 \
    > "$OUT/qtc-$d.log" 2>&1
  echo "GPU $d quantus: $(qtc_rate "$OUT/qtc-$d.log") | $(grep -m1 -oE "kernel mul=[0-9]+ red=[0-9]+ local=[0-9]+" "$OUT/qtc-$d.log")" | tee -a "$summary"
done
telemetry_stop

# All GPUs at once (power / thermal limits, PCIe and host sharing).
if [ "$(echo $devs | wc -w)" -gt 1 ]; then
  telemetry_start "$OUT/telemetry-all.txt"
  for d in $devs; do
    timeout "$SEC" "$BIN/cppminer" --backend opencl --devices "$d" --mock --mock-diff 1e12 > "$OUT/pearl-all-$d.log" 2>&1 &
  done; wait
  for d in $devs; do echo "GPU $d pearl all-GPUs: $(pearl_rate "$OUT/pearl-all-$d.log")" | tee -a "$summary"; done
  for d in $devs; do
    timeout "$SEC" "$BIN/cppminer" --algo quantus --backend opencl --devices "$d" --mock --mock-diff 1e15 > "$OUT/qtc-all-$d.log" 2>&1 &
  done; wait
  for d in $devs; do echo "GPU $d quantus all-GPUs: $(qtc_rate "$OUT/qtc-all-$d.log")" | tee -a "$summary"; done
  telemetry_stop
fi
echo "[bench] summary ($summary):"; cat "$summary"
