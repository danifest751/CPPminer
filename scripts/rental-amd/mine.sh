#!/bin/bash
# mine.sh pearl|quantus: one miner process per AMD GPU (each with its own pool connection and
# worker name), restarted automatically if it exits. Runs in the background; logs in
# logs/mine-<algo>-<gpu>.log. Stop with ./stop.sh, watch with ./status.sh.
set -u
cd "$(dirname "$0")"
. ./common.sh
algo=${1:-}
case "$algo" in
  pearl)   pool=$PEARL_POOL; extra=(--backend opencl);                  envs=$PEARL_ENV ;;
  quantus) pool=$QTC_POOL;   extra=(--algo quantus --backend opencl);   envs=$QTC_ENV ;;
  *) echo "usage: $0 pearl|quantus"; exit 1 ;;
esac
devs=$(amd_devices)
[ -z "$devs" ] && { echo "[mine] no AMD OpenCL device; run ./setup.sh first"; exit 1; }
./stop.sh >/dev/null 2>&1
short=$([ "$algo" = pearl ] && echo pearl || echo qtc)
for d in $devs; do
  log="$LOGS/mine-$short-$d.log"
  nohup bash -c '
    while true; do
      echo "=== start $(date)"
      env '"$envs"' "'"$BIN"'/cppminer" '"${extra[*]}"' --devices '"$d"' \
        --pool '"$pool"' --wallet '"$WALLET"' --worker '"$WORKER-$short-$d"' --no-fee
      echo "=== exited with $? at $(date), restarting in 10 s"
      sleep 10
    done' >> "$log" 2>&1 &
  echo $! >> "$LOGS/mine.pids"
  echo "[mine] GPU $d: $algo as $WORKER-$short-$d -> $pool (log $log)"
done
echo "[mine] running. ./status.sh shows hash rates and shares, ./stop.sh stops everything."
