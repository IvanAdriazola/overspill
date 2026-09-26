# FreeToken disk tier (experimental preview)

**Run MoE models larger than your RAM on a consumer GPU.** This fork adds a disk tier to
[FreeToken](https://github.com/FlashML-org/FreeToken). Experts that don't fit in RAM are served
straight from NVMe through the OS page cache, and the rest of FreeToken's fast execution path is unchanged.

> **Status: experimental.** It works and is measured on one machine and one model (details below).
> Expect rough edges, hard-coded defaults, and missing tests. Issues and reports from other hardware
> are very welcome.

## Why

Stock FreeToken keeps every routed expert in pinned host RAM, so a model whose experts don't fit in RAM
is refused. Disk-streaming engines such as [Colibri](https://github.com/JustVugg/colibri) do run those
models, but slowly. This fork combines the two: FreeToken's GPU execution, with a disk -> RAM -> VRAM
hierarchy under it.

## Results

DeepSeek-V4-Flash REAP-150B (`puwaer/DeepSeek-V4-Flash-0731-reap-150b`, 85 GB with FP4 experts). The model
is **bigger than RAM**: the machine has 64 GB and WSL2 is capped at 48 GB. Hardware: RTX 3060 12 GB,
Ryzen 9 7900, DDR5-6000, NVMe (through WSL2's virtual disk). All runs start cold (page cache dropped), use
greedy decoding, and use the same three prompts.

| | this fork | Colibri (cold) | Colibri (warm) | llama.cpp (MXFP4_MOE, experts mmap'd on CPU) |
|---|---|---|---|---|
| decode, short prompt | **2.78 tok/s** | 1.17 | 1.19 | 0.54 |
| decode, coding prompt | **3.10 tok/s** | 1.20 | 1.24 | 0.49 |
| decode, after a 6.4k prompt | **2.69 tok/s** | 1.12 | 1.16 | 0.38 |
| time to first token, 6.4k-token prompt | **99 s** | 1565 s | 1557 s | 373 s |
| time to first token, short prompt | 27-34 s | 17.5-25 s | 18-26 s | 40-44 s |

That is about **2.5x Colibri and 5-7x llama.cpp on decode**, and **3.8x llama.cpp / 16x Colibri on a
6.4k-token prompt**. llama.cpp (`81bc6b8`, `-ngl 99 --cpu-moe`, mmap, flash attention) uses the same model at
the same FP4 expert precision; for a model bigger than RAM, mmap is its only load mode that works. Colibri
"warm" = second session with a warm page cache.

**Correctness:** on Qwen3.6-35B-A3B (fits in RAM, so stock FreeToken can serve it too) the disk-tier path
produces **byte-identical greedy output** to stock FreeToken on 5/5 prompts.

## What it changes

All of it is opt-in through environment variables; with none set, FreeToken behaves exactly as upstream.

| switch | what it does |
|---|---|
| `FT_FILE_BANKS=1` | Non-pinned expert layers of an FTW checkpoint are `mmap`ed from the file instead of read into locked RAM. The page cache is the RAM tier; the NVMe holds the rest. |
| (automatic with `FT_FILE_BANKS`) | The CPU MoE executor calls `madvise(MADV_WILLNEED)` on every routed expert as soon as a layer's routing is known, so the kernel reads whole experts with large parallel I/Os instead of page fault by page fault (**4x faster expert loading**; this took decode from 0.9 to about 3 tok/s). |
| `FT_EMBED_HOST=1`, `FT_HEAD_HOST=1` | The token embedding and LM head live in pinned host RAM and the GPU reads them over PCIe (UVA). This frees 2 GiB of VRAM on DeepSeek-V4, which is what makes it fit a 12 GB card. |
| `FT_SWA_RATIO` | Sizes DeepSeek-V4's sliding-window pool. A bigger pool allows bigger prefill chunks, and every chunk re-streams all experts once (6.4k prompt: 17 chunks / 486 s -> 3 chunks / 99 s). |
| `FT_CPU_PREFILL_MAX` | Short prefill chunks on disk-backed layers run on the CPU executor instead of streaming the whole layer to the GPU (work in progress). |

Also fixed: FTW conversion of a model bigger than RAM ran out of memory. Shared anonymous mappings were
"released" with `MADV_DONTNEED`, which doesn't free shmem pages; the fix uses `MADV_REMOVE`.

## Quick start (DeepSeek-V4-Flash REAP-150B, 12 GB GPU)

```bash
# 1. convert once (streams layer by layer, fine on 48 GB RAM)
ft checkpoint --model <hf_dir> --out <ftw_dir>
# 2. serve
FT_FILE_BANKS=1 FT_EMBED_HOST=1 FT_HEAD_HOST=1 FT_SWA_RATIO=0.7 \
ft serve --model <ftw_dir> --moe-strategy cpu --num-tokens 8192 \
  --cuda-graph-max-bs 1 --max-running-requests 1
```
The CPU executor change is C++ (`python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp`); rebuild the
extension after installing (`experiment/build_cpu_moe.sh` shows how).

## Limitations

- Measured on one machine and one model; Linux/WSL2 only (it relies on `mmap`/`madvise`).
- Decode is bounded by expert misses from disk: faster NVMe (or native Linux instead of WSL2's
  virtual disk) and more RAM help directly.
- Short prompts still pay a full expert stream; the CPU-prefill path addresses this and is being measured.
- Single request at a time (`--max-running-requests 1`) is the tested configuration.

## Credits

- [FreeToken](https://github.com/FlashML-org/FreeToken) (Apache-2.0): the engine this is built on.
  Everything here is a modification of it, and the modified files are marked in the git history.
- [Colibri](https://github.com/JustVugg/colibri) by Vincenzo Fornaro (Apache-2.0): the disk -> RAM -> VRAM
  tiering ideas (expert heat, lookahead). No Colibri code is used.

## License

Apache-2.0, like FreeToken. See `LICENSE`.
