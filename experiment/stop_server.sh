#!/bin/bash
# (inside WSL, as root) Stop ft serve by PID - a pkill -f pattern would also match this script's own shell.
pids=$(pgrep -f '\.venv/bin/ft serve')
[ -n "$pids" ] && kill -INT $pids
for i in $(seq 1 20); do pgrep -f '\.venv/bin/ft serve' >/dev/null || exit 0; sleep 1; done
pids=$(pgrep -f '\.venv/bin/ft serve'); [ -n "$pids" ] && kill -KILL $pids; true
