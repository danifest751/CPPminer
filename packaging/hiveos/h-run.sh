#!/usr/bin/env bash
# HiveOS runs this in the miner's screen session: one cppminer process per GPU.
cd "$(dirname "$(readlink -f "$0")")"
. ./h-manifest.conf

if [[ ! -f $CUSTOM_CONFIG_FILENAME ]]; then
    echo "cppminer: $CUSTOM_CONFIG_FILENAME is missing; apply the flight sheet again"
    exit 1
fi
. "$CUSTOM_CONFIG_FILENAME"

mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"

args=(--pool "$POOL" --wallet "$WALLET" --worker "$WORKER")
[[ $ALGO == quantus ]] && args+=(--algo quantus)
[[ -n $PASS && $PASS != x ]] && args+=(--pool-pass "$PASS")
# extra options from the flight sheet, split like a shell command line
eval "extra=($EXTRA)"

./cppminer-multi.sh --api-base "$CUSTOM_API_PORT" --ports-file "$CUSTOM_PORTS_FILE" -- \
    "${args[@]}" "${extra[@]}" 2>&1 | tee -a "$CUSTOM_LOG_BASENAME.log"
