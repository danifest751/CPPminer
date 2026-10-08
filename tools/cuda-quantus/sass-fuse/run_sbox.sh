#!/bin/bash
# run_sbox.sh NAME [gen_sbox.py options]: mb0.cuasm -> NAME.cuasm (hand sbox) -> cubin -> binary -> check + speed
set -e
cd /w/bench/sass2
export PYTHONPATH=/w/bench/CuAssembler
N=$1; shift
python3 gen_sbox.py mb0.cuasm $N.cuasm "$@"
python3 /w/bench/CuAssembler/bin/cuasm.py $N.cuasm -o $N.cubin 2>&1 | grep -iE "error|unknown|fail" | head -5 || true
test -f $N.cubin
SRC=mb_sbox.cu ./hackbuild2.sh $N $N.cubin -O3 -arch=sm_75 -std=c++17 2>&1 | tail -1
cuobjdump -res-usage $N 2>/dev/null | grep -A1 k_hand | grep -oE "REG:[0-9]+" || true
./$N ${BPS:-3} 8 ${ITERS:-1024}
