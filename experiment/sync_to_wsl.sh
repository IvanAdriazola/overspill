#!/bin/bash
# (inside WSL) Mirror the committed experiment tree to ~/src/freetoken-exp and reuse the prebuilt extensions.
set -e
SRC=/mnt/c/GIT/Freetoken-colibri-experiment
if [ ! -d ~/src/freetoken-exp/.git ]; then git clone -q "$SRC" ~/src/freetoken-exp; fi
cd ~/src/freetoken-exp && git fetch -q origin && git checkout -q -B main origin/main && git reset -q --hard origin/main
cp -n ~/src/FreeToken/python/freetoken/kernel/*.so python/freetoken/kernel/
git log --oneline -1
