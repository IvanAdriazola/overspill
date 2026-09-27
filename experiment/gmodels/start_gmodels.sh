#!/bin/bash
# (Git Bash) Keep the gmodels helper distro alive with /models bind-mounted at /mnt/wsl/gmodels, then check Ubuntu sees it.
# Must be re-run after every `wsl --shutdown` (e.g. each RAM-cap change). /keep.sh lives inside the helper distro:
#   mkdir -p /mnt/wsl/gmodels; mountpoint -q /mnt/wsl/gmodels || mount --bind /models /mnt/wsl/gmodels; exec sleep 2147483647
if ! MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- test -e /mnt/wsl/gmodels/.gmodels; then
  powershell.exe -NoProfile -Command "Start-Process wsl.exe -WindowStyle Hidden -ArgumentList '-d gmodels -u root -- /bin/sh /keep.sh'"
  for i in $(seq 1 30); do
    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- test -e /mnt/wsl/gmodels/.gmodels && break
    sleep 2
  done
fi
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash -c 'ls /mnt/wsl/gmodels && df -h /mnt/wsl/gmodels | tail -1'
