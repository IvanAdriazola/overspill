#!/bin/bash
# (inside WSL, as root) Set the block-device read-ahead (KB) of the disk behind /mnt/wsl/gmodels; print before/after.
# mmap page faults read around the fault by this window, so it sets how much llama.cpp pulls in per expert miss.
KB=${1:-128}
dev=$(basename "$(findmnt -no SOURCE /mnt/wsl/gmodels | sed 's/\[.*//')")
echo "$dev read_ahead_kb: $(cat /sys/block/$dev/queue/read_ahead_kb) -> $KB"
echo "$KB" > /sys/block/$dev/queue/read_ahead_kb
