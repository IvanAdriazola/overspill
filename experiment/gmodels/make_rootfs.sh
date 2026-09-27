#!/bin/bash
# (inside Ubuntu WSL, as root) Build a tiny busybox rootfs tarball for the "gmodels" helper distro.
# The helper distro's ext4 disk lives on G: and holds the benchmark models; it bind-mounts its /models into
# /mnt/wsl/gmodels (shared across WSL distros, as Docker Desktop does) so Ubuntu can read them natively.
# Why a helper distro: `wsl --mount --vhd` needs admin; `wsl --import` doesn't.
set -e
OUT=${1:-/mnt/g/wsl/gmodels-rootfs.tar}
command -v busybox >/dev/null && busybox --list | grep -qx mount || { apt-get update -qq && apt-get install -y -qq busybox-static; }
R=$(mktemp -d)
mkdir -p "$R"/{bin,sbin,etc,proc,sys,dev,tmp,root,models,mnt/wsl}
touch "$R/models/.gmodels"
cp "$(command -v busybox)" "$R/bin/busybox"
for a in sh mount umount mountpoint mkdir sleep ls df cat chown touch; do ln -s busybox "$R/bin/$a"; done
echo 'root:x:0:0:root:/root:/bin/sh' > "$R/etc/passwd"
echo 'root:x:0:' > "$R/etc/group"
mkdir -p "$(dirname "$OUT")"
tar -C "$R" -cf "$OUT" .
rm -rf "$R"
ls -la "$OUT"
