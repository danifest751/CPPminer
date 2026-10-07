#!/bin/bash
# stop.sh: stop every miner started by mine.sh (the restart loops first, then the miners).
cd "$(dirname "$0")"
. ./common.sh
if [ -f "$LOGS/mine.pids" ]; then
  while read -r p; do kill "$p" 2>/dev/null; done < "$LOGS/mine.pids"
  rm -f "$LOGS/mine.pids"
fi
pkill -f "$BIN/cppminer" 2>/dev/null
sleep 1
pgrep -f "$BIN/cppminer" >/dev/null && { pkill -9 -f "$BIN/cppminer"; echo "[stop] killed"; } || echo "[stop] stopped"
