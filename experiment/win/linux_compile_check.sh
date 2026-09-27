#!/bin/bash
# (inside WSL) Compile-check the cross-platform C++ extensions on Linux after Windows-port edits.
set -e
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/src/freetoken-exp
~/ft/.venv/bin/python - <<'PY'
import os
from torch.utils.cpp_extension import load
for name, src, flags in [("_cpu_moe", "cpu_moe/cpu_moe_ext.cpp", ["-O3", "-std=c++17", "-pthread"]),
                         ("_ple_store", "ple_store/ple_store_ext.cpp", ["-O3", "-std=c++17"])]:
    m = load(name=name + "_chk", sources=["python/freetoken/kernel/csrc/" + src], extra_cflags=flags,
             extra_include_paths=["/usr/local/cuda/include"], extra_ldflags=["-L/usr/local/cuda/lib64", "-lcudart"],
             build_directory=os.path.expanduser("~/src/chk_" + name), verbose=False)
    print("linux build OK:", name)
PY
