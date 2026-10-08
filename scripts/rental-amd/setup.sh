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
  # Ubuntu 25.04+ (e.g. 26.04 "resolute", kernel 7.0) ships ROCm itself and its in-kernel amdgpu
  # runs RDNA4; amdgpu-install has no packages for it. rocm-opencl-icd does not pull the code
  # object manager, without which the runtime lists 0 devices ("Failed to load COMGR library").
  if [ -e /dev/kfd ] && apt-cache show rocm-opencl-icd >/dev/null 2>&1; then
    # the -rocm build matches rocm-opencl-icd (tested: Ubuntu 26.04, libamd-comgr3-rocm 7.1.0)
    comgr=$(apt-cache search --names-only '^libamd-comgr[0-9]+-rocm$' | awk '{print $1}' | sort -V | tail -1)
    [ -z "$comgr" ] && comgr=$(apt-cache search --names-only '^libamd-comgr[0-9]+$' | awk '{print $1}' | sort -V | tail -1)
    say "installing the distribution's ROCm OpenCL: rocm-opencl-icd $comgr rocminfo llvm"
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rocm-opencl-icd $comgr rocminfo llvm >/dev/null
    ldconfig
  fi
fi

if ! have_amd_cl; then
  if [ "$(id -u)" != 0 ]; then echo "[setup] run me as root: sudo ./setup.sh"; exit 1; fi
  . /etc/os-release
  base="https://repo.radeon.com/amdgpu-install/latest/ubuntu/${VERSION_CODENAME}/"
  deb=$(wget -qO- "$base" | grep -oE 'amdgpu-install_[^"]+_all\.deb' | sort -V | tail -1)
  if [ -z "$deb" ]; then echo "[setup] could not find amdgpu-install for ${VERSION_CODENAME} at $base"; exit 1; fi
  say "installing $deb"
  wget -q "$base$deb" -O "/tmp/$deb" && apt-get install -y -qq "/tmp/$deb" >/dev/null
  # The in-kernel amdgpu of older kernels cannot initialize RDNA4: on the Ubuntu 24.04 kernel
  # 6.8 an R9700 (gfx1201) logs "amdgpu: Fatal error during GPU init" and has no /dev/kfd.
  # Then the amdgpu DKMS module is needed (and a reboot); otherwise keep the running driver.
  need_dkms=0
  if [ ! -e /dev/kfd ] || dmesg 2>/dev/null | grep -qi "fatal error during gpu init"; then need_dkms=1; fi
  if [ "$need_dkms" = 1 ]; then
    say "amdgpu kernel driver not working: installing the amdgpu DKMS module + OpenCL (reboot needed)"
    apt-get install -y -qq "linux-headers-$(uname -r)" >/dev/null 2>&1 || true
    amdgpu-install -y --usecase=dkms,opencl
  else
    amdgpu-install -y --usecase=opencl --no-dkms
  fi
  for u in ${SUDO_USER:-} root; do [ -n "$u" ] && usermod -aG render,video "$u" 2>/dev/null; done
  ldconfig
  if [ "$need_dkms" = 1 ]; then
    say "DKMS driver installed. REBOOT now (sudo reboot), then run ./setup.sh again."
    exit 2
  fi
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
