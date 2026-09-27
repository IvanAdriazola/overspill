"""Simulate a machine with less RAM: lock N GiB of physical memory until this process exits.

    python ram_hog.py <GiB to lock>        e.g. 64 GB box -> "32 GB box": ram_hog.py 32

Allocates N GiB, raises this process's minimum working set (SetProcessWorkingSetSize - SeIncreaseWorkingSetPrivilege,
which standard users hold) and VirtualLock()s it chunk by chunk. Locked pages are never paged out nor usable as file
cache, so the OS and every other process (both inference engines) really have ~N GiB less. Prints progress, then
"LOCKED <n> GiB" and blocks; kill the process (or Ctrl+C) to release everything.
"""

import ctypes
import sys
import time
from ctypes import wintypes

GiB = 1 << 30
CHUNK = 256 << 20

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetCurrentProcess.restype = wintypes.HANDLE
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
k32.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
k32.VirtualLock.restype = wintypes.BOOL
k32.SetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD]
k32.SetProcessWorkingSetSizeEx.restype = wintypes.BOOL
MEM_COMMIT_RESERVE, PAGE_READWRITE = 0x3000, 0x04
QUOTA_LIMITS_HARDWS_MIN_ENABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE = 0x1, 0x8


def main() -> int:
    want = float(sys.argv[1])
    total = int(want * GiB)
    proc = k32.GetCurrentProcess()
    # locked pages count against the minimum working set: make room for all of them (+256 MiB slack)
    if not k32.SetProcessWorkingSetSizeEx(proc, total + (256 << 20), total + (512 << 20),
                                          QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE):
        print(f"SetProcessWorkingSetSizeEx failed: {ctypes.get_last_error()}", flush=True)
        return 1
    locked = 0
    while locked < total:
        n = min(CHUNK, total - locked)
        p = k32.VirtualAlloc(None, n, MEM_COMMIT_RESERVE, PAGE_READWRITE)
        if not p:
            print(f"VirtualAlloc failed at {locked / GiB:.2f} GiB: {ctypes.get_last_error()}", flush=True)
            return 1
        ctypes.memset(p, 1, n)  # touch every page so it is really backed by RAM
        if not k32.VirtualLock(p, n):
            print(f"VirtualLock failed at {locked / GiB:.2f} GiB: {ctypes.get_last_error()}", flush=True)
            return 1
        locked += n
    print(f"LOCKED {locked / GiB:.2f} GiB", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
