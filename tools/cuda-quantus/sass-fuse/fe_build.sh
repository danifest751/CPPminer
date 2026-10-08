#!/bin/bash
# fe_build.sh: production kernel bench (qb.cu + qk.inc) through different front ends into ptxas.
#   q_nv     nvcc 12.6 (reference)
#   q_clang  clang-18 (LLVM NVPTX) PTX -> ptxas 12.6
#   q_clangO2 / q_ptxO1 ... extra variants
cd /w/bench/sass2
export PYTHONPATH=/w/bench/CuAssembler
mix() {
  cuobjdump -sass "$1" | awk '/Function : .*qb_scan/{f=1;next} /Function :/{f=0} f' \
    | grep -oE '/\*[0-9a-f]{4}\*/ +(@!?P[0-9T] +)?[A-Z0-9_.]+' | awk '{print $NF}' | sort | uniq -c | sort -rn \
    | awk '{t+=$1; if (NR<=7) s=s" "$2":"$1} END{print "tot="t s}'
}
SRC=qb.cu ./hackbuild2.sh q_nv - -O3 -arch=sm_75 -std=c++17 2>&1 | tail -1
for fl in "O3:-O3" "O2:-O2" "Os:-Os"; do
  n=${fl%%:*}; o=${fl#*:}
  clang++-18 -x cuda --cuda-gpu-arch=sm_75 --cuda-path=/usr/local/cuda $o -std=c++17 --cuda-device-only -S qb.cu \
      -o qb_clang$n.ptx -Wno-unknown-cuda-version 2>&1 | grep -E "error" | head -5
  ptxas -arch=sm_75 -O3 qb_clang$n.ptx -o qb_clang$n.cubin 2>&1 | head -3
  SRC=qb.cu ./hackbuild2.sh q_clang$n qb_clang$n.cubin -O3 -arch=sm_75 -std=c++17 2>&1 | tail -1
done
for b in q_nv q_clangO3 q_clangO2 q_clangOs; do
  printf "%-10s %s  " $b "$(cuobjdump -res-usage $b | grep -A1 qb_scan | grep -oE 'REG:[0-9]+')"; mix $b
done
for b in q_nv q_clangO3 q_clangO2 q_clangOs; do printf "%-10s " $b; ./$b 1; done
