#!/usr/bin/env bash
# Container entry point: one cppminer process per GPU (see cppminer-multi.sh).
#
# Either pass cppminer options as the container command, or set:
#   WALLET (required), POOL (default: Kryptex Pearl RU), ALGO=pearl|quantus, WORKER,
#   PASS, BACKEND, DEVICES, EXTRA_ARGS, API_PORT (default 4068; one port per GPU from there)
set -u
cd /opt/cppminer

API_PORT=${API_PORT:-4068}
if [[ $# -gt 0 ]]; then
    args=("$@")
else
    if [[ -z ${WALLET:-} ]]; then
        echo "Set WALLET (and POOL, ALGO, WORKER), or pass cppminer options as the command." >&2
        echo "Example: -e WALLET=YOUR_KRYPTEX_ACCOUNT_OR_PRL_ADDRESS -e ALGO=pearl" >&2
        exit 2
    fi
    ALGO=${ALGO:-pearl}
    if [[ -z ${POOL:-} ]]; then
        if [[ $ALGO == quantus ]]; then POOL=stratum+tcp://qtc-ru.kryptex.network:7049
        else POOL=stratum+tcp://prl-ru.kryptex.network:7048; fi
    fi
    args=(--pool "$POOL" --wallet "$WALLET" --worker "${WORKER:-$(hostname)}")
    [[ $ALGO == quantus ]] && args+=(--algo quantus)
    [[ -n ${PASS:-} && ${PASS:-} != x ]] && args+=(--pool-pass "$PASS")
    [[ -n ${BACKEND:-} ]] && args+=(--backend "$BACKEND")
    [[ -n ${DEVICES:-} ]] && args+=(--devices "$DEVICES")
    # shellcheck disable=SC2206
    [[ -n ${EXTRA_ARGS:-} ]] && args+=(${EXTRA_ARGS})
fi

exec ./cppminer-multi.sh --api-base "$API_PORT" --api-bind 0.0.0.0 \
    --ports-file /tmp/cppminer.ports -- "${args[@]}"
