#!/usr/bin/env bash
# HiveOS sources this before every miner start: turn the flight sheet into cppminer.conf.
#
# Flight sheet (custom miner):
#   Hash algorithm   pearl | quantus   (empty: quantus when the pool URL looks like a Quantus pool)
#   Wallet template  %WAL%.%WORKER_NAME%   (or just %WAL%; the rig name is then the worker)
#   Pool URL         stratum+tcp://prl-ru.kryptex.network:7048
#   Pass             x   (Kryptex Pearl takes d=N for a fixed share difficulty)
#   Extra config     any cppminer options, e.g. --backend opencl or --batch-size 8000000

[[ -z $CUSTOM_TEMPLATE ]] && echo -e "${YELLOW}cppminer: the wallet template is empty${NOCOLOR}" && return 1
[[ -z $CUSTOM_URL ]] && echo -e "${YELLOW}cppminer: the pool URL is empty${NOCOLOR}" && return 1

cppminer_pool=$(head -n 1 <<< "$CUSTOM_URL" | awk '{print $1}')
[[ $cppminer_pool != *://* ]] && cppminer_pool="stratum+tcp://$cppminer_pool"

if [[ $CUSTOM_TEMPLATE == *.* ]]; then
    cppminer_wallet=${CUSTOM_TEMPLATE%%.*}
    cppminer_worker=${CUSTOM_TEMPLATE#*.}
else
    cppminer_wallet=$CUSTOM_TEMPLATE
    cppminer_worker=${WORKER_NAME:-rig}
fi

cppminer_algo=$(tr '[:upper:]' '[:lower:]' <<< "$CUSTOM_ALGO")
case $cppminer_algo in
    quantus|qpow|qtc) cppminer_algo=quantus ;;
    pearl|prl|pearlhash) cppminer_algo=pearl ;;
    "")
        if [[ $cppminer_pool == *qtc* || $cppminer_pool == *quantus* ]]; then
            cppminer_algo=quantus
        else
            cppminer_algo=pearl
        fi
        ;;
    *)
        echo -e "${YELLOW}cppminer: unknown algorithm '$CUSTOM_ALGO', using pearl${NOCOLOR}"
        cppminer_algo=pearl
        ;;
esac

mkdir -p "$(dirname "$CUSTOM_CONFIG_FILENAME")"
{
    printf 'ALGO=%q\n' "$cppminer_algo"
    printf 'POOL=%q\n' "$cppminer_pool"
    printf 'WALLET=%q\n' "$cppminer_wallet"
    printf 'WORKER=%q\n' "$cppminer_worker"
    printf 'PASS=%q\n' "${CUSTOM_PASS:-x}"
    printf 'EXTRA=%q\n' "$CUSTOM_USER_CONFIG"
} > "$CUSTOM_CONFIG_FILENAME"
