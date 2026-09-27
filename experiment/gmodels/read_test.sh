#!/bin/bash
# (inside WSL, as root) Uncached read speed of a model file: sequential (4 GB) and random 2.7 MB chunks (one
# Flash-Next NVFP4 expert) with O_DIRECT, so the page cache is bypassed. Usage: read_test.sh <file> [label]
F=$1; L=${2:-$1}
SZ=$(stat -c %s "$F")
seq_s=$( { /usr/bin/time -f %e dd if="$F" of=/dev/null bs=4M count=1024 iflag=direct status=none; } 2>&1 )
CH=$((2816 * 1024)); N=400
t0=$(date +%s.%N)
for i in $(seq 1 $N); do
  off=$(( (RANDOM * 32768 + RANDOM) % (SZ / CH - 1) ))
  dd if="$F" of=/dev/null bs=$CH skip=$off count=1 iflag=direct status=none
done
t1=$(date +%s.%N)
awk -v l="$L" -v s="$seq_s" -v t0="$t0" -v t1="$t1" -v n=$N -v ch=$CH \
  'BEGIN { printf "%s: sequential %.0f MB/s, random 2.7MB chunks %.0f MB/s (%.1f ms each)\n", l, 4096/s, n*ch/1048576/(t1-t0), (t1-t0)*1000/n }'
