#!/bin/bash
# run_fuse.sh NAME [fuse.py extra args]: base.cuasm -> NAME.cuasm (fused) -> NAME.cubin -> q_NAME binary -> checksum run
# runs inside cppminer-dev at /w/bench/sass2
set -e
cd /w/bench/sass2
export PYTHONPATH=/w/bench/CuAssembler
N=$1; shift
python3 fuse.py base.cuasm $N.cuasm _Z7qb_scan6ParamsPx "$@"
python3 /w/bench/CuAssembler/bin/cuasm.py $N.cuasm -o $N.cubin 2>&1 | grep -iE "error|fail|cannot|invalid" | head -5 || true
ls -la $N.cubin
./hackbuild2.sh q_$N $N.cubin -O3 -arch=sm_75 -std=c++17 2>&1 | tail -1
cuobjdump -sass q_$N | awk '/Function : .*qb_scan/{f=1;next} /Function :/{f=0} f' | grep -oE '/\*[0-9a-f]{4}\*/ +(@!?P[0-9T] +)?[A-Z0-9_.]+' | awk '{print $NF}' | sort | uniq -c | sort -rn | awk '{t+=$1; if (NR<=8) s=s" "$2":"$1} END{print "tot="t s}'
./q_$N 1
