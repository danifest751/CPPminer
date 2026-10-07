#!/bin/bash
# sass_ab.sh bin... : stop the QTC miner, run /w/bench/sass2/BIN 4 inside cppminer-dev interleaved x3, restart the miner
MINER_PID=$(docker exec cppminer-tc129 sh -c "pgrep -f 'cppminer --algo quantus' | head -1")
if [ -n "$MINER_PID" ]; then docker exec cppminer-tc129 kill $MINER_PID; sleep 3; fi
for rep in 1 2 3; do for b in "$@"; do printf "%-8s " $b; docker exec -w /w/bench/sass2 cppminer-dev ./$b 4; done; done
docker exec -d cppminer-tc129 sh -c "cd /root/u128 && nohup ./cppminer --algo quantus --backend cuda --pool stratum+tcp://qtc-ru.kryptex.network:7049 --wallet krxX8QJ872 --worker cmpqtc-cuda --no-fee >> /root/qtc-mine.log 2>&1 &"
sleep 2; docker exec cppminer-tc129 sh -c "pgrep -f 'cppminer --algo quantus' >/dev/null && echo miner-restarted || echo MINER-NOT-RUNNING"
