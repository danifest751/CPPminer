#!/usr/bin/env bash
# cppminer-multi.sh: run one cppminer process per GPU, each with its own stats API port.
#
#   cppminer-multi.sh [--api-base PORT] [--api-bind ADDR] [--ports-file FILE] -- MINER ARGS...
#
# GPUs are found with nvidia-smi (CUDA backend) and with the miner's own OpenCL device list
# (AMD / Intel discrete GPUs; integrated GPUs only when there is no discrete one). Each process
# gets --backend and --devices for its GPU and --api-port API_BASE+N, restarts 10 s after an exit,
# and its output lines are prefixed with [gpuN]. The ports go to --ports-file for stats scripts.
#
# If the miner arguments already pick devices (--devices), the CPU backend or --mock, a single
# process runs with them unchanged (plus --api-port API_BASE).
#
# Used by the HiveOS package (h-run.sh) and the Docker images.

set -u
cd "$(dirname "$(readlink -f "$0")")"

API_BASE=4068
API_BIND=""
PORTS_FILE=/tmp/cppminer.ports
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-base) API_BASE=$2; shift 2 ;;
        --api-bind) API_BIND=$2; shift 2 ;;
        --ports-file) PORTS_FILE=$2; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done
ARGS=("$@")

export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# CUDA device numbers in nvidia-smi order
export CUDA_DEVICE_ORDER=PCI_BUS_ID

has_arg() {
    local a
    for a in "${ARGS[@]}"; do [[ $a == "$1" || $a == "$1="* ]] && return 0; done
    return 1
}
arg_value() {
    local i
    for ((i = 0; i < ${#ARGS[@]}; i++)); do
        [[ ${ARGS[i]} == "$1" ]] && { echo "${ARGS[i + 1]:-}"; return; }
        [[ ${ARGS[i]} == "$1="* ]] && { echo "${ARGS[i]#*=}"; return; }
    done
}

BACKEND=$(arg_value --backend)
WANT_CUDA=1
WANT_OCL=1
[[ $BACKEND == cuda ]] && WANT_OCL=0
[[ -n $BACKEND && $BACKEND != cuda ]] && WANT_CUDA=0

PROCS=() # "backend device" pairs; "- -" = one process with the arguments as given
if has_arg --devices || has_arg --device || has_arg --mock || [[ $BACKEND == cpu ]]; then
    PROCS=("- -")
else
    if [[ $WANT_CUDA == 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
        n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c '^[0-9]')
        for ((i = 0; i < n; i++)); do PROCS+=("cuda $i"); done
    fi
    if [[ $WANT_OCL == 1 ]]; then
        # "  [N] name" then "      platform[k]=NAME  discrete GPU ..."
        list=$(./cppminer --backend opencl --list-devices 2>/dev/null)
        disc=() integ=()
        idx=""
        while IFS= read -r line; do
            if [[ $line =~ ^\ \ \[([0-9]+)\]\  ]]; then
                idx=${BASH_REMATCH[1]}
            elif [[ -n $idx && $line =~ platform\[[0-9]+\]= ]]; then
                # NVIDIA GPUs run on CUDA unless --backend opencl was asked for
                if [[ $line == *NVIDIA* && $BACKEND != opencl ]]; then :
                elif [[ $line == *"discrete GPU"* ]]; then disc+=("$idx")
                elif [[ $line == *"integrated GPU"* ]]; then integ+=("$idx")
                fi
                idx=""
            fi
        done <<< "$list"
        sel=("${disc[@]}")
        [[ ${#sel[@]} -eq 0 && ${#PROCS[@]} -eq 0 ]] && sel=("${integ[@]}")
        for i in "${sel[@]}"; do PROCS+=("opencl $i"); done
    fi
    [[ ${#PROCS[@]} -eq 0 ]] && PROCS=("- -")
fi

: > "$PORTS_FILE" 2>/dev/null || PORTS_FILE=/dev/null
trap 'trap - TERM INT; kill 0; exit 0' TERM INT

k=0
for p in "${PROCS[@]}"; do
    read -r be dev <<< "$p"
    port=$((API_BASE + k))
    echo "$port" >> "$PORTS_FILE"
    extra=(--api-port "$port")
    [[ -n $API_BIND ]] && extra+=(--api-bind "$API_BIND")
    if [[ $be != - ]]; then
        # drop a --backend the user gave, then set this GPU's backend and device
        own=()
        skip=0
        for a in "${ARGS[@]}"; do
            if [[ $skip == 1 ]]; then skip=0; continue; fi
            [[ $a == --backend ]] && { skip=1; continue; }
            [[ $a == --backend=* ]] && continue
            own+=("$a")
        done
        cmd=(./cppminer "${own[@]}" --backend "$be" --devices "$dev" "${extra[@]}")
        tag="gpu$k"
    else
        cmd=(./cppminer "${ARGS[@]}" "${extra[@]}")
        tag="miner"
    fi
    echo "[cppminer-multi] $tag: ${cmd[*]}"
    (
        while :; do
            "${cmd[@]}" 2>&1 | sed -u "s/^/[$tag] /"
            echo "[$tag] miner exited, restarting in 10 s"
            sleep 10
        done
    ) &
    k=$((k + 1))
    # stagger start-up so GPU inits and pool logins do not all land at once
    sleep 2
done
wait
