#!/bin/bash
# t_dual.sh: build t_dual.cu, patch the carry-count materialization to read one predicate at a time, run all
set -e
cd /w/bench/sass2
export PYTHONPATH=/w/bench/CuAssembler
SRC=t_dual.cu ./hackbuild2.sh td_base - -O3 -arch=sm_75 2>&1 | tail -1
cuobjdump -sass td_base | grep -E "IADD3" | sed -E "s@/\*[0-9a-f]+\*/@@; s@/\* 0x[0-9a-f]+ \*/@@"
python3 /w/bench/CuAssembler/bin/cuasm.py td_base.orig.cubin -o td.cuasm 2>&1 | grep -iE "error" || true
grep -n "IADD3" td.cuasm | sed -E "s@/\* 0x.*@@"
for v in p0 p1 p1p0; do
  case $v in
    p0) sed -E 's/(IADD3\.X R[0-9]+, RZ, RZ, RZ, )P0, P1/\1P0, !PT/' td.cuasm > td_$v.cuasm ;;
    p1) sed -E 's/(IADD3\.X R[0-9]+, RZ, RZ, RZ, )P0, P1/\1P1, !PT/' td.cuasm > td_$v.cuasm ;;
    p1p0) sed -E 's/(IADD3\.X R[0-9]+, RZ, RZ, RZ, )P0, P1/\1P1, P0/' td.cuasm > td_$v.cuasm ;;
  esac
  grep -c "RZ, RZ, RZ, P" td_$v.cuasm
  python3 /w/bench/CuAssembler/bin/cuasm.py td_$v.cuasm -o td_$v.cubin 2>&1 | grep -iE "error" || true
  SRC=t_dual.cu ./hackbuild2.sh td_$v td_$v.cubin -O3 -arch=sm_75 2>&1 | tail -1
done
echo "== base (P0+P1)"; ./td_base
echo "== P0 only"; ./td_p0 | awk '{print $5}' | tr '\n' ' '; echo
echo "== P1 only"; ./td_p1 | awk '{print $5}' | tr '\n' ' '; echo
echo "== P1,P0 swapped"; ./td_p1p0 | awk '{print $5}' | tr '\n' ' '; echo
