"""Rebuild the CPU MoE executor (_cpu_moe) with clang-cl instead of MSVC - the same compiler family as the
Linux/WSL build (GCC-style per-function AVX-512 targets and __builtin_cpu_supports instead of the MSVC shim).

    call experiment\\win\\env_win.bat && python experiment\\win\\build_cpu_moe_clang.py [msvc|clang]

"clang" builds with D:\\tools\\llvm-23 and installs it over python/freetoken/kernel/_cpu_moe*.pyd, keeping the MSVC
build as _cpu_moe.msvc.pyd.bak; "msvc" restores that backup. A/B: rerun the same quick decode test after each.
"""

import glob
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KDIR = os.path.join(REPO, "python", "freetoken", "kernel")
LLVM = os.environ.get("LLVM_HOME", r"D:\tools\llvm-23")
target = glob.glob(os.path.join(KDIR, "_cpu_moe.cp*-win_amd64.pyd"))[0]
backup = os.path.join(KDIR, "_cpu_moe.msvc.pyd.bak")

which = sys.argv[1] if len(sys.argv) > 1 else "clang"
if which == "msvc":
    shutil.copy(backup, target)
    print("restored MSVC build ->", target)
    sys.exit(0)

os.environ["PATH"] = os.pathsep.join([os.path.join(LLVM, "clwrap"), os.path.join(LLVM, "bin"), os.environ["PATH"]])  # clwrap/cl.exe is a copy of clang-cl: torch always calls "cl"
os.environ["CXX"] = os.environ["CC"] = "clang-cl"
from torch.utils.cpp_extension import load  # noqa: E402  (env first)

rt = glob.glob(os.path.join(LLVM, "lib", "clang", "*", "lib", "windows", "clang_rt.builtins-x86_64.lib"))
cuda = os.environ["CUDA_HOME"]
mod = load(
    name="_cpu_moe",
    sources=[os.path.join(KDIR, "csrc", "cpu_moe", "cpu_moe_ext.cpp")],
    extra_cflags=["/O2", "/clang:-O3", "/std:c++17", "/EHsc", "/DNOMINMAX"],
    extra_include_paths=[os.path.join(cuda, "include")],
    extra_ldflags=[f"/LIBPATH:{os.path.join(cuda, 'lib', 'x64')}", "cudart.lib"] + rt,
    build_directory=os.path.join(REPO, "build", "cpu_moe_clang"),
    verbose=False,
)
if not os.path.exists(backup):
    shutil.copy(target, backup)
shutil.copy(mod.__file__, target)
print("built with clang-cl:", mod.__file__, "->", target)
