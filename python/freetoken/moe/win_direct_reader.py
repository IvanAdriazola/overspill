"""Native-Windows direct reads for the whole-layer pageable prefill copy of file-backed expert banks.

Long-prompt prefill streams every CPU-served layer's expert banks to the GPU. Through the file mapping,
Windows serves that as page faults in small reads (~500 MB/s on DeepSeek-V4 vs ~800 MB/s for the same
path under WSL's 8 MB readahead). Here the bytes are read with large parallel unbuffered ReadFile calls
(FILE_FLAG_NO_BUFFERING, positional OVERLAPPED offsets) straight into a pinned staging buffer, then copied
to the GPU asynchronously. Two staging buffers alternate: while layer L's copy and compute run, layer
L+1 (the next CPU-served layer, wrapping for the next prefill chunk) is read in the background.

FT_WIN_DIRECT=0 disables it (falls back to the mapping + PrefetchVirtualMemory path).
"""

from __future__ import annotations

import ctypes
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from ctypes import wintypes

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

ENABLED = os.name == "nt" and os.getenv("FT_WIN_DIRECT", "1").strip().lower() not in {"0", "false", "no", "off"}
_ALIGN = 4096
_CHUNK = 8 << 20
_WORKERS = int(os.getenv("FT_WIN_DIRECT_WORKERS", "8"))

if os.name == "nt":
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateFileW.restype = wintypes.HANDLE
    _k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                 wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                              ctypes.c_void_p]
    _k32.ReadFile.restype = wintypes.BOOL

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]

_GENERIC_READ, _FILE_SHARE_READ, _OPEN_EXISTING = 0x80000000, 0x1, 3
_FILE_FLAG_NO_BUFFERING, _FILE_FLAG_SEQUENTIAL_SCAN = 0x20000000, 0x08000000
_INVALID = ctypes.c_void_p(-1).value


class _Handles:
    """One unbuffered read handle per shard file (positional reads are safe across threads)."""

    def __init__(self) -> None:
        self._h: dict[str, int] = {}
        self._lock = threading.Lock()

    def get(self, path: str) -> int:
        h = self._h.get(path)
        if h is None:
            with self._lock:
                h = self._h.get(path)
                if h is None:
                    h = _k32.CreateFileW(path, _GENERIC_READ, _FILE_SHARE_READ, None, _OPEN_EXISTING,
                                         _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_SEQUENTIAL_SCAN, None)
                    if h in (None, _INVALID):
                        raise OSError(f"CreateFile({path}) failed: {ctypes.get_last_error()}")
                    self._h[path] = h
        return h


def _read_at(handle: int, dst: int, nbytes: int, offset: int) -> None:
    done = 0
    while done < nbytes:
        ov = _OVERLAPPED()
        pos = offset + done
        ov.Offset, ov.OffsetHigh = pos & 0xFFFFFFFF, pos >> 32
        got = wintypes.DWORD(0)
        if not _k32.ReadFile(handle, ctypes.c_void_p(dst + done), nbytes - done, ctypes.byref(got), ctypes.byref(ov)):
            raise OSError(f"ReadFile at {pos} failed: {ctypes.get_last_error()}")
        if got.value == 0:
            raise OSError(f"ReadFile hit EOF at {pos}")
        done += got.value


class WinDirectLayerReader:
    """Reads one layer's file-backed banks into pinned staging, double-buffered with a one-layer lookahead."""

    def __init__(self, banks: list[tuple[list[torch.Tensor], torch.Tensor]], unpinned: frozenset[int]) -> None:
        from freetoken.moe.host_banks import FILE_BANK_SOURCES

        self.order = sorted(unpinned)
        self.srcs: dict[int, list[tuple[str, int, int]]] = {}
        per_layer_bytes = 0
        for lid in self.order:
            entries = []
            for per_layer, _cache in banks:
                src = FILE_BANK_SOURCES.get(per_layer[lid].data_ptr())
                if src is None:
                    raise KeyError(f"layer {lid}: bank is not file-backed")
                entries.append(src)
            self.srcs[lid] = entries
            per_layer_bytes = max(per_layer_bytes, sum(-(-n // _ALIGN) * _ALIGN for _, _, n in entries))
        self.handles = _Handles()
        self.pool = ThreadPoolExecutor(_WORKERS, thread_name_prefix="win-direct")
        self.bg = ThreadPoolExecutor(1, thread_name_prefix="win-direct-next")
        self.buf = [torch.empty(per_layer_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        self.done = [torch.cuda.Event(), torch.cuda.Event()]
        self.done_recorded = [False, False]
        self.slot_of: dict[int, int] = {}
        self.pending: dict[int, Future] = {}
        self.next_slot = 0
        logger.info_rank0(f"[exp] Windows direct prefill reads: {len(self.order)} layers, 2 x "
                          f"{per_layer_bytes / 2**30:.2f} GiB pinned staging, {_WORKERS} readers")

    def _read_layer(self, lid: int, slot: int) -> None:
        if self.done_recorded[slot]:
            self.done[slot].synchronize()  # the previous H2D copy out of this buffer must have finished
        base = self.buf[slot].data_ptr()
        jobs, pos = [], 0
        for path, off, n in self.srcs[lid]:
            h = self.handles.get(path)
            span = -(-n // _ALIGN) * _ALIGN  # unbuffered reads need sector-multiple lengths (FTW pads to 4 KiB)
            for c in range(0, span, _CHUNK):
                jobs.append(self.pool.submit(_read_at, h, base + pos + c, min(_CHUNK, span - c), off + c))
            pos += span
        for j in jobs:
            j.result()

    def _start(self, lid: int) -> None:
        if lid in self.pending or lid in self.slot_of:
            return
        slot = self.next_slot
        self.next_slot ^= 1
        for k, v in list(self.slot_of.items()):
            if v == slot:
                del self.slot_of[k]
        self.slot_of[lid] = slot
        self.pending[lid] = self.bg.submit(self._read_layer, lid, slot)

    def copy_layer(self, lid: int, banks: list[tuple[list[torch.Tensor], torch.Tensor]], num_experts: int) -> None:
        self._start(lid)
        self.pending.pop(lid).result()
        slot = self.slot_of[lid]
        stream = torch.cuda.current_stream()
        pos = 0
        for (per_layer, cache), (_, _, n) in zip(banks, self.srcs[lid]):
            src = per_layer[lid]
            view = self.buf[slot][pos:pos + n].view(src.dtype).view(src.shape)
            cache[:num_experts].copy_(view, non_blocking=True)
            pos += -(-n // _ALIGN) * _ALIGN
        self.done[slot].record(stream)
        self.done_recorded[slot] = True
        del self.slot_of[lid]  # consumed: the next prefill chunk re-reads it (the page cache is not used)
        i = self.order.index(lid)
        self._start(self.order[(i + 1) % len(self.order)])
