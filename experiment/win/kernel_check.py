"""Compile every AOT CUDA kernel spec (the ones FreeToken JIT-builds at startup) and report per-kernel pass/fail
with the first compiler errors -- one pass instead of one server launch per error. Run via kernel_check.bat."""

import re
import sys
import tempfile
import pathlib

from freetoken.kernel.aot import default_kernel_specs

only = sys.argv[1] if len(sys.argv) > 1 else ""
specs = [s for s in default_kernel_specs() if only in s.name]
ok = 0
seen_errors: dict[str, str] = {}
for spec in specs:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:  # the built DLL stays loaded
        try:
            spec.build(pathlib.Path(d))
            ok += 1
            print("OK  ", spec.name, flush=True)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            errs = [l.strip() for l in msg.splitlines() if re.search(r"error|fatal", l, re.I) and "warning" not in l]
            key = errs[0] if errs else msg[:200]
            first = key not in seen_errors
            seen_errors.setdefault(key, spec.name)
            print("FAIL", spec.name, "--", key if first else f"(same as {seen_errors[key]})", flush=True)
            if first:
                for l in errs[1:6]:
                    print("      ", l)
print(f"{ok}/{len(specs)} kernels built")
