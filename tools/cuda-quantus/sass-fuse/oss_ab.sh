#!/bin/bash
# oss_ab.sh "ENV=.. ./bin args" ... : stop the QTC miner, run each command 2x inside cppminer-dev (/w/bench/sass2),
# restart the miner binary that was running.
P=$(docker exec cppminer-tc129 sh -c "ps -eo pid,args | grep -E \"cppminer[_a-z0-9]* --algo quantus\" | grep -v grep | head -1")
PID=$(echo "$P" | awk '{print $1}'); BIN=$(echo "$P" | awk '{print $2}'); BIN=${BIN:-./cppminer_wf}
if [ -n "$PID" ]; then docker exec cppminer-tc129 kill $PID; sleep 3; fi
for rep in 1 2; do for c in "$@"; do echo "## $c"; docker exec -w /w/bench/sass2 cppminer-dev sh -c "$c" 2>&1 | grep -vE "^\s*$"; done; done
docker exec -d cppminer-tc129 sh -c "cd /root/u128 && nohup $BIN --algo quantus --backend cuda --pool stratum+tcp://qtc-ru.kryptex.network:7049 --wallet krxX8QJ872 --worker cmpqtc-cuda --no-fee >> /root/qtc-mine.log 2>&1 &"
sleep 2; docker exec cppminer-tc129 sh -c "ps -eo args | grep -E \"cppminer[_a-z0-9]* --algo quantus\" | grep -v grep | cut -c1-40"
