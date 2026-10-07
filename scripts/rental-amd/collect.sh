#!/bin/bash
# collect.sh: pack logs and system information into logs/r9700-report-<time>.tar.gz to send back.
cd "$(dirname "$0")"
. ./common.sh
OUT="$LOGS/sysinfo"; mkdir -p "$OUT"
{ uname -a; . /etc/os-release; echo "$PRETTY_NAME"; lscpu | head -20; free -g; } > "$OUT/system.txt" 2>&1
lspci -nn 2>/dev/null | grep -iE "vga|display|3d" > "$OUT/lspci.txt"
command -v clinfo >/dev/null && clinfo > "$OUT/clinfo.txt" 2>&1
command -v rocm-smi >/dev/null && rocm-smi -a > "$OUT/rocm-smi.txt" 2>&1
command -v rocminfo >/dev/null && rocminfo > "$OUT/rocminfo.txt" 2>&1
dmesg 2>/dev/null | grep -iE "amdgpu|kfd" | tail -200 > "$OUT/dmesg-amdgpu.txt"
"$BIN/cppminer" --backend opencl --list-devices > "$OUT/devices.txt" 2>&1
tarball="$LOGS/r9700-report-$(stamp).tar.gz"
tar czf "$tarball" -C "$KIT" logs --exclude='*.tar.gz' config.env
echo "[collect] $tarball ($(du -h "$tarball" | cut -f1))"
