#!/bin/bash
# sass_ab.sh bin... : stop the QTC miner (any binary name), run /w/bench/sass2/BIN 4 inside cppminer-dev
# interleaved x3, then restart the same miner binary that was running.
P=$(docker exec cppminer-tc129 sh -c "ps -eo pid,args | grep -E \"cppminer[_a-z0-9]* --algo quantus\" | grep -v grep | head -1")
PID=$(echo "$P" | awk "{print \$1}"); BIN=$(echo "$P" | awk "{print \$2}"); BIN=${BIN:-./cppminer_wf}
if [ -n "$PID" ]; then docker exec cppminer-tc129 kill $PID; sleep 3; fi
for rep in 1 2 3; do for b in "$@"; do printf "%-8s " $b; docker exec -w /w/bench/sass2 cppminer-dev ./$b 4; done; done
docker exec -d cppminer-tc129 sh -c "cd /root/u128 && nohup $BIN --algo quantus --backend cuda --pool stratum+tcp://qtc-ru.kryptex.network:7049 --wallet krxX8QJ872 --worker cmpqtc-cuda --no-fee >> /root/qtc-mine.log 2>&1 &"
sleep 2; docker exec cppminer-tc129 sh -c "ps -eo args | grep -E \"cppminer[_a-z0-9]* --algo quantus\" | grep -v grep | cut -c1-40"
