#!/bin/bash
# setup.sh: make sure the AMD OpenCL runtime (ROCm) sees the GPUs. Run as root (or with sudo).
# Installs the ROCm OpenCL runtime with amdgpu-install only if no AMD OpenCL device is found.
set -u
cd "$(dirname "$0")"
. ./common.sh

say() { echo "[setup] $*"; }

say "OS: $(. /etc/os-release; echo "$PRETTY_NAME"), kernel $(uname -r)"
say "GPUs on the PCI bus:"
lspci -nn 2>/dev/null | grep -iE "vga|display|3d" | grep -i amd || say "  (lspci found no AMD display device)"
if [ -e /dev/kfd ]; then say "/dev/kfd present (amdgpu KFD driver loaded)"; else say "WARNING: /dev/kfd missing: amdgpu kernel driver not loaded"; fi

have_amd_cl() { [ -n "$(amd_devices)" ]; }

if have_amd_cl; then
  say "AMD OpenCL devices already visible: $(amd_devices)"
else
  say "no AMD OpenCL device visible, installing the ROCm OpenCL runtime"
  if [ "$(id -u)" != 0 ]; then echo "[setup] run me as root: sudo ./setup.sh"; exit 1; fi
  . /etc/os-release
  apt-get update -qq
  apt-get install -y -qq wget ca-certificates clinfo pciutils >/dev/null
  base="https://repo.radeon.com/amdgpu-install/latest/ubuntu/${VERSION_CODENAME}/"
  deb=$(wget -qO- "$base" | grep -oE 'amdgpu-install_[^"]+_all\.deb' | sort -V | tail -1)
  if [ -z "$deb" ]; then echo "[setup] could not find amdgpu-install for ${VERSION_CODENAME} at $base"; exit 1; fi
  say "installing $deb"
  wget -q "$base$deb" -O "/tmp/$deb" && apt-get install -y -qq "/tmp/$deb" >/dev/null
  # --no-dkms: keep the kernel driver the host already runs (a DKMS build would need a reboot).
  amdgpu-install -y --usecase=opencl --no-dkms
  for u in ${SUDO_USER:-} root; do [ -n "$u" ] && usermod -aG render,video "$u" 2>/dev/null; done
  ldconfig
fi

command -v clinfo >/dev/null && { say "clinfo:"; clinfo -l 2>/dev/null | sed 's/^/  /'; }
command -v rocm-smi >/dev/null && { say "rocm-smi:"; rocm-smi --showproductname 2>/dev/null | grep -E "GPU\[|Card" | sed 's/^/  /'; }
say "cppminer sees:"
miner --backend opencl --list-devices 2>&1 | sed 's/^/  /'
if have_amd_cl; then
  say "OK: AMD OpenCL devices $(amd_devices). Next: ./check.sh"
else
  say "FAILED: still no AMD OpenCL device. Check 'dmesg | grep -i amdgpu' and that the user is in the render group."
  exit 1
fi
