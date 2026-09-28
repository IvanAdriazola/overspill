#!/bin/bash
# (Git Bash) Create (first time) and start a busybox helper WSL distro whose /models is bind-mounted at
# /mnt/wsl/<name>, so Ubuntu reads a model disk on another drive natively (ext4, not 9p).
# Usage: start_helper.sh <name> <windows dir for the distro disk>     e.g. start_helper.sh gmodels 'G:\wsl\gmodels'
# Needs the rootfs from make_rootfs.sh at G:\wsl\gmodels-rootfs.tar. Re-run after every `wsl --shutdown`.
NAME=$1; DIR=$2
ROOTFS='G:\wsl\gmodels-rootfs.tar'
if ! wsl.exe -l -q | tr -d '\0\r' | grep -qx "$NAME"; then
  mkdir -p "$(cygpath -u "$DIR")"
  wsl.exe --import "$NAME" "$DIR" "$ROOTFS" --version 2 | tr -d '\0'
  MSYS_NO_PATHCONV=1 wsl.exe -d "$NAME" -u root -- /bin/sh -c "printf '#!/bin/sh\nmkdir -p /mnt/wsl/$NAME\nmountpoint -q /mnt/wsl/$NAME || mount --bind /models /mnt/wsl/$NAME\nexec sleep 2147483647\n' > /keep.sh; chmod +x /keep.sh"
fi
if ! MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- test -e "/mnt/wsl/$NAME/.gmodels"; then
  powershell.exe -NoProfile -Command "Start-Process wsl.exe -WindowStyle Hidden -ArgumentList '-d $NAME -u root -- /bin/sh /keep.sh'"
  for i in $(seq 1 30); do
    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- test -e "/mnt/wsl/$NAME/.gmodels" && break
    sleep 2
  done
fi
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash -c "chown ivan:ivan /mnt/wsl/$NAME; df -h /mnt/wsl/$NAME | tail -1"
