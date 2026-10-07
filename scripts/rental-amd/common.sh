# Shared helpers, sourced by the kit scripts.
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$KIT/config.env"
BIN="$KIT/bin"
LOGS="$KIT/logs"
mkdir -p "$LOGS"
export LD_LIBRARY_PATH="$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

miner() { "$BIN/cppminer" "$@"; }

# OpenCL flat indices of AMD GPUs, or $DEVICES when set.
amd_devices() {
  if [ -n "$DEVICES" ]; then echo "$DEVICES" | tr ',' ' '; return; fi
  # "  [N] name" lines, each followed by "platform[P]=... type ..."; keep N when either
  # line names AMD or a gfx target.
  miner --backend opencl --list-devices 2>/dev/null | awk '
    /^ *\[[0-9]+\]/ { line = $0; sub(/^ *\[/, "", line); idx = line; sub(/\].*/, "", idx);
                      amd = (tolower($0) ~ /amd|advanced micro|gfx[0-9]/); next }
    /platform\[/    { if (idx != "" && (amd || tolower($0) ~ /amd|advanced micro/)) print idx; idx = "" }
  ' | sort -un | tr '\n' ' '
}

stamp() { date +%Y%m%d-%H%M%S; }

# gfx target of an AMD GPU (for llvm-objdump --mcpu), default gfx1201.
amd_gfx() {
  local g
  g=$(rocminfo 2>/dev/null | grep -oE "gfx[0-9a-f]{3,4}" | sort -u | head -1)
  echo "${g:-gfx1201}"
}
