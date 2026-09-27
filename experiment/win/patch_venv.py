"""Patch third-party bugs in .venv-win that only bite on native Windows. Idempotent; run after (re)installing.

1. apache-tvm-ffi 0.1.13.post3 (FreeToken's JIT for its CUDA kernels): the Windows default nvcc flags are
   ["-Xcompiler", "/std:c++17", "/O2"] -- -Xcompiler only forwards the NEXT token, so nvcc takes "/O2" as a second
   input file ("A single input file is required for a non-link phase"). Linux passes -std/-O2 to nvcc itself;
   do the same on Windows and forward only /O2 to cl. No -std here: FreeToken passes -std=c++20 itself, and a second
   -std=c++17 made nvcc parse its C++20 requires-clauses as C++17.
"""

import pathlib
import sys

venv = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / ".venv-win")
p = venv / "Lib" / "site-packages" / "tvm_ffi" / "cpp" / "extension.py"
s = p.read_text(encoding="utf-8")
bad = 'default_cuda_cflags = ["-Xcompiler", "/std:c++17", "/O2"]'
good = 'default_cuda_cflags = ["-Xcompiler", "/O2", "-O2"]  # patched: see experiment/win/patch_venv.py'
if bad in s:
    p.write_text(s.replace(bad, good), encoding="utf-8")
    print("patched", p)
elif good in s:
    print("already patched", p)
else:
    print("WARNING: tvm_ffi default_cuda_cflags line not found; check", p)
