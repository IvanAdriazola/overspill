#!/bin/bash
# (inside WSL) Rebuild only the CPU MoE extension of the experiment tree, in place.
set -e
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/src/freetoken-exp
~/ft/.venv/bin/python - <<'PY'
import os, shutil, glob
from torch.utils.cpp_extension import load
src = "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp"
mod = load(name="_cpu_moe", sources=[src], extra_cflags=["-O3", "-std=c++17", "-pthread"],
           extra_include_paths=["/usr/local/cuda/include"], extra_ldflags=["-L/usr/local/cuda/lib64", "-lcudart"],
           build_directory=os.path.expanduser("~/src/cpu_moe_build"), verbose=False)
so = mod.__file__
dst = "python/freetoken/kernel/" + os.path.basename(glob.glob("python/freetoken/kernel/_cpu_moe*.so")[0])
shutil.copy(so, dst)
print("built", so, "->", dst, "has set_prefetch:", hasattr(mod.CpuMoeExecutor, "set_prefetch"))
PY
