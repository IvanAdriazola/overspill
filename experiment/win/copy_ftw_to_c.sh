#!/bin/bash
# (inside WSL) Copy the Flash-Next FTW from the G: helper disk to C:\AIModels, 6 files at a time
# (the WSL->Windows 9p bridge is slow per stream). Re-copies any file whose size differs.
SRC=/mnt/wsl/gmodels/flashnext_ftw; DST=/mnt/c/AIModels/flashnext_ftw
pkill -f "cp -r /mnt/wsl/gmodels/flashnext_ftw" ; sleep 1
mkdir -p "$DST"; t0=$(date +%s)
cd "$SRC" && ls -S | xargs -P 6 -I{} sh -c '[ "$(stat -c %s "{}")" = "$(stat -c %s "'"$DST"'/{}" 2>/dev/null)" ] || cp "{}" "'"$DST"'/{}"'
echo "copied in $(( $(date +%s) - t0 ))s"
cd "$SRC" && find . -type f -printf '%P %s\n' | sort > /tmp/s.txt; cd "$DST" && find . -type f -printf '%P %s\n' | sort > /tmp/d.txt
diff /tmp/s.txt /tmp/d.txt && echo "all $(wc -l < /tmp/s.txt) files match in size"
