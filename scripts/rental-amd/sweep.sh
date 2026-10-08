#!/bin/bash
# sweep.sh [SECONDS] [DEVICE]: offline speed of the kernel variants that are selectable at run
# time (CP_OCL_EXTRA_OPTS / CP_QPOW_OCL_OPTS / CP_*_MUL ...), on one GPU, one after another.
# For every Pearl variant the built code object is dumped and its VGPR / scratch usage and
# instruction mix are recorded (llvm-objdump from ROCm), so the speed can be read next to
# what the real ROCm compiler produced. Results: logs/sweep-<time>/summary.txt
set -u
cd "$(dirname "$0")"
. ./common.sh
SEC=${1:-90}
DEV=${2:-0}
OUT="$LOGS/sweep-$(stamp)"; mkdir -p "$OUT"
summary="$OUT/summary.txt"; : > "$summary"
OBJDUMP=$(ls /opt/rocm/llvm/bin/llvm-objdump 2>/dev/null || command -v llvm-objdump || true)
READELF=$(ls /opt/rocm/llvm/bin/llvm-readelf 2>/dev/null || command -v llvm-readelf || true)
echo "[sweep] GPU $DEV, $SEC s per variant, logs: $OUT" | tee -a "$summary"

pearl_rate() {
  local vals n
  vals=$(grep -oE "attempt timing: .*" "$1" | grep -oE "[0-9.]+ [TG]MAC/s" |
         awk '{print ($2 == "GMAC/s") ? $1 / 1000 : $1}')
  n=$(printf "%s\n" "$vals" | grep -c .)
  [ "$n" -eq 0 ] && { echo "n/a"; return; }
  printf "%s\n" "$vals" | tail -n $(( (n + 1) / 2 )) | sort -n |
    awk -v n="$n" '{a[NR] = $1} END {printf "%.2f TMAC/s (%d att)", a[int((NR + 1) / 2)], n}'
}
qtc_rate() { grep -oE "\[qpow\] [0-9.]+ MH/s" "$1" | tail -1 | grep -oE "[0-9.]+ MH/s" || echo n/a; }

# resources of kernel K in code object F: "vgpr=.. scratch=.. wmma=.. scratch_ops=.. vmem=.. ds=.."
code_stats() {
  local f=$1 k=$2
  [ -s "$f" ] || { echo "no dump"; return; }
  local meta=""
  [ -n "$READELF" ] && meta=$("$READELF" --notes "$f" 2>/dev/null | awk -v K="$k" '
      /\.name:/ {n=$2} /\.vgpr_count:/ {v=$2} /\.private_segment_fixed_size:/ {p=$2}
      /^ *- \./ {if (n == K && !d) {print "vgpr=" v " scratch=" p; d=1}}
      END {if (n == K && !d) print "vgpr=" v " scratch=" p}')
  local isa=""
  if [ -n "$OBJDUMP" ]; then
    "$OBJDUMP" -d --mcpu="$(amd_gfx "$DEV")" "$f" 2>/dev/null | awk "/<$k>:/,/s_endpgm/" > "$f.s"
    isa="wmma=$(grep -c v_wmma "$f.s") scratch_ops=$(grep -cE "scratch_" "$f.s") vmem=$(grep -cE "(global|buffer)_load" "$f.s") ds=$(grep -cE "ds_(load|store)" "$f.s") insts=$(grep -cE "^\s+[sv]_|^\s+(global|buffer|scratch|ds)_" "$f.s")"
  fi
  echo "$meta $isa"
}

pearl_variant() { # name pipeline "extra opts"
  local name=$1 pipe=$2 opts=$3 log="$OUT/pearl-$1.log"
  rm -f "$OUT/pearl-$name.co"*
  CP_OCL_WMMA_PIPELINE=$pipe CP_OCL_EXTRA_OPTS="$opts" CP_OCL_DUMP_BIN="$OUT/pearl-$name.co" \
    timeout "$SEC" "$BIN/cppminer" --backend opencl --devices "$DEV" --mock --mock-diff 1e12 \
    --ocl-dot wmma > "$log" 2>&1
  local co; co=$(grep -l "case33_macro_gemm_xor" "$OUT"/pearl-$name.co* 2>/dev/null | head -1)
  printf "pearl %-14s pipe=%s %-46s %s | %s\n" "$name" "$pipe" "[$opts]" "$(pearl_rate "$log")" \
    "$(code_stats "${co:-none}" case33_macro_gemm_xor)" | tee -a "$summary"
}

qtc_variant() { # name "env assignments" "extra opts"
  local name=$1 envs=$2 opts=$3 log="$OUT/qtc-$1.log"
  rm -f "$OUT/qtc-$name.co"*
  env $envs CP_QPOW_OCL_OPTS="$opts" CP_OCL_DUMP_BIN="$OUT/qtc-$name.co" timeout "$SEC" "$BIN/cppminer" \
    --algo quantus --backend opencl --devices "$DEV" --mock --mock-diff 1e15 > "$log" 2>&1
  # the worker builds several variants while probing; the newest dump is the one it kept
  local co; co=$(ls -t "$OUT"/qtc-$name.co* 2>/dev/null | head -1)
  printf "qtc   %-14s %-24s %-28s %s | %s | %s\n" "$name" "[$envs]" "[$opts]" "$(qtc_rate "$log")" \
    "$(grep -m1 -oE "kernel mul=[0-9]+ red=[0-9]+ local=[0-9]+.*self-test [a-z]+" "$log")" \
    "$(code_stats "${co:-none}" qpow_scan)" | tee -a "$summary"
}

# --- Pearl (WMMA). Defaults on gfx12: pipeline 0, msg in LDS, full k unroll, no LDS staging.
pearl_variant base        0 ""
pearl_variant msg-vgpr    0 "-DCASE32_WMMA_MSG_LDS=0"
pearl_variant k1          0 "-DCASE32_WMMA_KUNROLL=1"
pearl_variant k2          0 "-DCASE32_WMMA_KUNROLL=2"
pearl_variant lds         0 "-DCASE32_WMMA_LDS=1"
pearl_variant lds-k1      0 "-DCASE32_WMMA_LDS=1 -DCASE32_WMMA_KUNROLL=1"
pearl_variant lds-k2      0 "-DCASE32_WMMA_LDS=1 -DCASE32_WMMA_KUNROLL=2"
pearl_variant lds-k4      0 "-DCASE32_WMMA_LDS=1 -DCASE32_WMMA_KUNROLL=4"
pearl_variant pipe        1 ""

# --- Quantus. Default on AMD: QV_OVF=2 (5ac910f); the worker probes mul 3/1 x local 64/256.
qtc_variant auto      ""                   ""
qtc_variant ovf1      ""                   "-DQV_OVF=1"
qtc_variant mul3      "CP_QPOW_OCL_MUL=3"  ""
qtc_variant mul1      "CP_QPOW_OCL_MUL=1"  ""
qtc_variant mul3red2  "CP_QPOW_OCL_MUL=3 CP_QPOW_OCL_RED=2" ""
# 2026-10-08 additions: rare-carry skip in wred (default on) vs exact, two/three nonces per
# work-item (offline gfx1201: 184/192 VGPR, no scratch), lazy 96-bit sums vs 22-bit limbs.
qtc_variant wred-exact ""                  "-DQV_WRED_FAST=0"
qtc_variant ext64     ""                   "-DQV_EXT22=0"
qtc_variant npw2      ""                   "-DQV_NPW=2"
qtc_variant npw2ext64 ""                   "-DQV_NPW=2 -DQV_EXT22=0"
qtc_variant npw3      ""                   "-DQV_NPW=3"
qtc_variant npw2mul1  "CP_QPOW_OCL_MUL=1"  "-DQV_NPW=2"

echo "[sweep] done: $summary"
