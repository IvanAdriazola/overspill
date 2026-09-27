"""Native-Windows stand-ins for the Linux system probes the engine uses (sysfs topology, sysconf).

Only imported on ``os.name == "nt"``; every function degrades to None / [] instead of raising.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes


def physical_memory_bytes() -> int | None:
    """Total physical RAM (GlobalMemoryStatusEx), or None."""

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(st)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return int(st.ullTotalPhys)


class _MemRange(ctypes.Structure):  # WIN32_MEMORY_RANGE_ENTRY
    _fields_ = [("VirtualAddress", ctypes.c_void_p), ("NumberOfBytes", ctypes.c_size_t)]


def prefetch_ranges(ranges: list[tuple[int, int]]) -> bool:
    """PrefetchVirtualMemory: queue async reads for the non-resident pages of (address, nbytes) ranges of
    file mappings -- Windows' madvise(WILLNEED). Returns False if the call failed or is unavailable."""
    if not ranges:
        return True
    arr = (_MemRange * len(ranges))(*[_MemRange(a, n) for a, n in ranges])
    k32 = ctypes.windll.kernel32
    # explicit types: the default int conversion truncates the -1 pseudo-handle to 32 bits on x64
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.PrefetchVirtualMemory.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.POINTER(_MemRange), wintypes.ULONG]
    k32.PrefetchVirtualMemory.restype = wintypes.BOOL
    return bool(k32.PrefetchVirtualMemory(k32.GetCurrentProcess(), len(ranges), arr, 0))


def physical_core_first_cpus() -> list[int]:
    """The lowest logical CPU of each physical core (GetLogicalProcessorInformation), or []."""
    RelationProcessorCore = 0

    class SLPI(ctypes.Structure):  # SYSTEM_LOGICAL_PROCESSOR_INFORMATION (x64: 32 bytes)
        _fields_ = [("ProcessorMask", ctypes.c_size_t), ("Relationship", ctypes.c_int),
                    ("_union", ctypes.c_ubyte * 16)]

    k32 = ctypes.windll.kernel32
    need = wintypes.DWORD(0)
    k32.GetLogicalProcessorInformation(None, ctypes.byref(need))
    if need.value == 0:
        return []
    n = need.value // ctypes.sizeof(SLPI)
    buf = (SLPI * n)()
    if not k32.GetLogicalProcessorInformation(buf, ctypes.byref(need)):
        return []
    firsts = []
    for e in buf:
        if e.Relationship == RelationProcessorCore and e.ProcessorMask:
            m = e.ProcessorMask
            firsts.append((m & -m).bit_length() - 1)
    return sorted(firsts)
