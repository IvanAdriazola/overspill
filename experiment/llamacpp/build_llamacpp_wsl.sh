#!/bin/bash
# (inside WSL, as ivan) Check out llama.cpp tag ${1:-b11205} (same as the native-Windows binary) and rebuild llama-server
# with the existing build dir config (CUDA sm_86, GGML_NATIVE=ON, FA, CUDA graphs).
set -e
cd ~/src/llama.cpp
git fetch -q --tags origin
git checkout -q "${1:-b11205}"
git describe --tags
export PATH=/usr/local/cuda/bin:$PATH
cmake --build build --config Release -j 10 --target llama-server 2>&1 | tail -3
build/bin/llama-server --version 2>&1 | tail -2
