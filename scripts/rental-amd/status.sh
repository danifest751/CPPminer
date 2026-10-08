#!/bin/bash
# status.sh: last hash rate and share counts of every mine.sh log, plus GPU telemetry.
cd "$(dirname "$0")"
. ./common.sh
for log in "$LOGS"/mine-*.log; do
  [ -f "$log" ] || continue
  name=$(basename "$log" .log)
  rate=$(grep -oE "\[qpow\] [0-9.]+ MH/s|attempt timing: .* [0-9.]+ [TG]MAC/s" "$log" | tail -1 | grep -oE "[0-9.]+ (MH/s|[TG]MAC/s)")
  ok=$(grep -cE '"status":"OK"|submit response:.*"result":true|share accepted|accepted' "$log")
  bad=$(grep -ciE "rejected|invalid share|low difficulty|stale share" "$log")
  restarts=$(grep -c "=== exited" "$log")
  echo "$name: ${rate:-n/a}  shares ok=$ok rejected=$bad  restarts=$restarts"
done
command -v rocm-smi >/dev/null && rocm-smi --showpower --showclocks --showtemp --showuse 2>/dev/null | grep -E "^(GPU|card)\["
