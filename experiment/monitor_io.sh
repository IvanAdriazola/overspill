#!/bin/bash
# (inside WSL, root) 5 s samples, system-wide: disk read MB/s (all sd* devices), page cache GiB, available GiB.
echo "time,read_mb_s,pagecache_gb,avail_gb"
sectors() { awk '$3 ~ /^sd[a-z]+$/ {s += $6} END {print s}' /proc/diskstats; }
prev=$(sectors)
while true; do
  sleep 5
  cur=$(sectors)
  cached=$(awk '/^Cached:/{print $2}' /proc/meminfo); avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  echo "$(date +%H:%M:%S),$(( (cur - prev) * 512 / 5 / 1048576 )),$(( cached / 1048576 )),$(( avail / 1048576 ))"
  prev=$cur
done
