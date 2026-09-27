# Overspill

**Run Mixture-of-Experts models larger than your RAM on a consumer GPU, fast.**

Overspill adds a disk tier to the [FreeToken](https://github.com/FlashML-org/FreeToken) MoE engine: experts that
don't fit in RAM *overspill* to NVMe and stream back through the OS page cache, while FreeToken's GPU
execution path stays as it is. Inspired by the disk -> RAM -> VRAM tiering in
[Colibri](https://github.com/JustVugg/colibri).


> **Status: experimental preview.** It works and is measured on one machine and one model (details below).
> Expect rough edges, hard-coded defaults, and missing tests. Issues and reports from other hardware
> are very welcome.

## Why

Stock FreeToken keeps every routed expert in pinned host RAM, so a model whose experts don't fit in RAM
is refused. Disk-streaming engines such as [Colibri](https://github.com/JustVugg/colibri) do run those
models, but slowly. Overspill combines the two: FreeToken's GPU execution, with a disk -> RAM -> VRAM
hierarchy under it.

## Results

DeepSeek-V4-Flash REAP-150B (`puwaer/DeepSeek-V4-Flash-0731-reap-150b`, 85 GB with FP4 experts). The model
is **bigger than RAM**. Hardware: RTX 3060 12 GB, Ryzen 9 7900, DDR5-6000, NVMe, 64 GB of RAM. Greedy decoding,
same three prompts for every engine: a short explanation, a coding question, and a long Python-module summary
(6,446 tokens for the WSL runs, 5,971 tokens for the native llama.cpp run; the published
[`experiment/bench_openai.py`](experiment/bench_openai.py) defaults to a FreeToken source file of that size).

- **Overspill** ran inside WSL2 capped at 48 GB, starting cold (page cache dropped).
- **llama.cpp** ran **natively on Windows** with all 64 GB, warm, with its best configuration from a sweep
  (`--cpu-moe -ub 2048`, build b11205). This is the fair baseline: see the note below on why llama.cpp must
  not be benchmarked inside WSL.
- **Colibri** ran inside the same 48 GB WSL2 VM as Overspill.

| | Overspill (WSL, 48 GB, cold) | llama.cpp (native, 64 GB, warm, best config) | Colibri (WSL, cold) | Colibri (WSL, warm) |
|---|---|---|---|---|
| decode, short prompt | **3.21 tok/s** | 2.43 | 1.17 | 1.19 |
| decode, coding prompt | **3.37 tok/s** | **3.38** | 1.20 | 1.24 |
| decode, after the long prompt | **2.75 tok/s** | 2.40 | 1.12 | 1.16 |
| time to first token, long prompt | **102 s** | 371 s | 1565 s | 1557 s |
| time to first token, first short prompt | 43 s (cold start) | 31 s (warm) | **25 s** | 26 s |
| time to first token, next short prompt | **10 s** | 26 s | 18 s | 18 s |

Against llama.cpp at its best, Overspill decodes **up to 32% faster** (and is even with it on the coding prompt) and reaches
the first token of a long prompt **3.6x faster** (2.6x on the second short prompt), despite running in a VM
with 16 GB less RAM. Against Colibri it is **2.7x faster on decode** and **15x faster on the long prompt**.
Colibri is quickest to the very first token after a cold start. All engines use the same FP4 experts; the
~8 GB of non-expert weights differ slightly (llama.cpp's GGUF stores them in Q8_0, FreeToken and Colibri use
DeepSeek's original FP8), so outputs are not bit-identical. Exact commands are
[below](#how-each-engine-was-run); raw logs and analysis are in
[`experiment/DESIGN_NOTES.md`](experiment/DESIGN_NOTES.md) (Validations 7 and 8).

> **Correction (2026-09-27).** An earlier version of this table compared against llama.cpp run *inside* the
> same WSL2 VM (0.54 / 0.49 / 0.38 tok/s, 373 s to the first token of the long prompt) and claimed "6-7x
> llama.cpp". That was unfair to llama.cpp: WSL2's virtual disk defaults to an 8 MB read-ahead, which cripples
> llama.cpp's mmap page-fault reads of experts that are not in RAM (on another MoE model it took llama.cpp from
> ~7 tok/s to 0.8 tok/s). The table above uses llama.cpp natively instead, at its best.

**Hardware matters a lot here.** In the disk path the expert math runs on the CPU (FreeToken's CPU
executor, which used its AVX-512 path on this Zen 4 Ryzen 9 7900), and experts stream from disk through
RAM. So the numbers depend heavily on the CPU, RAM speed and storage, not just the GPU. CPUs without
AVX-512 (many Intel consumer chips), fewer cores, slower RAM, a slower SSD or less RAM for the page cache
will likely be slower. Treat ~3 tok/s as what one fairly strong CPU + DDR5 + NVMe machine gets.

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
| `FT_CPU_PREFILL_MAX=256` | Prefill chunks up to 256 tokens on disk-backed layers run on the CPU executor instead of streaming the whole layer's experts to the GPU. Short-prompt TTFT drops from 27 s to 10 s, and decode improves because the page cache is no longer flushed by full-model streams. |

Also fixed: FTW conversion of a model bigger than RAM ran out of memory. Shared anonymous mappings were
"released" with `MADV_DONTNEED`, which doesn't free shmem pages; the fix uses `MADV_REMOVE`.

## Try it

Tested path: DeepSeek-V4-Flash REAP-150B on an RTX 3060 12 GB under WSL2. Expect **2-4 hours**, mostly waiting on
the download and the one-time conversion.

**You need:**
- Linux or WSL2 with an NVIDIA GPU (12 GB VRAM tested) and a CUDA 13 toolkit with `nvcc` on `PATH`
  (FreeToken JIT-compiles its kernels; see [docs/install.md](docs/install.md)).
- ~48 GB of RAM or more for WSL2, and ideally a CPU with AVX-512. **Give WSL2 as much RAM as Windows can spare**
  (`memory=` in `%UserProfile%\\.wslconfig`): the page cache is Overspill's RAM tier, so every extra GB means fewer
  expert reads from disk. Leave Windows ~10-12 GB: on a 64 GB machine 52 GB works, while 58 GB made Windows page
  the VM itself out and decode collapsed.
- **~170 GB of free disk:** the 85 GB download plus the 80 GB converted copy. You can delete the download
  after converting. Use a native Linux filesystem (in WSL2: your home directory, not `/mnt/c`).
- [uv](https://docs.astral.sh/uv/).

**1. Install (~5-15 min)**
```bash
git clone https://github.com/IvanAdriazola/overspill.git && cd overspill
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --no-sources -e ".[accel]"
```
`--no-sources` installs `torch` and `sglang-kernel` from PyPI (the same CUDA 13 builds). Without it uv uses the
package's pinned custom indexes, and at the time of writing the `sglang-kernel` wheel there fails with a hash
mismatch. Everything else, including the modified C++ CPU executor, is built by this step.

**2. Download the model (85 GB)**
```bash
hf download puwaer/DeepSeek-V4-Flash-0731-reap-150b --local-dir ~/models/dsv4-reap
```

**3. Convert it to FreeToken's FTW format (~20 min, one time)**
```bash
ft checkpoint --model ~/models/dsv4-reap --out ~/models/dsv4-reap-ftw
# optional, once it finishes: rm -rf ~/models/dsv4-reap
```
This streams layer by layer, so it works on a machine with less RAM than the model (that needed a fix in
this repo; stock FreeToken runs out of memory here).

**4. Serve (~1-2 min to load)**
```bash
# WSL2 with an RTX 30xx: the kernel JIT can't detect the GPU architecture, so set it (8.6 = RTX 30xx)
export TVM_FFI_CUDA_ARCH_LIST=8.6 TORCH_CUDA_ARCH_LIST=8.6 FLASHINFER_CUDA_ARCH_LIST=8.6

FT_FILE_BANKS=1 FT_EMBED_HOST=1 FT_HEAD_HOST=1 FT_SWA_RATIO=0.7 FT_CPU_PREFILL_MAX=256 \
ft serve --model ~/models/dsv4-reap-ftw --served-model-name dsv4-reap --moe-strategy cpu \
  --num-tokens 8192 --cuda-graph-max-bs 1 --max-running-requests 1
```

**5. Ask it something** (the first request after a start takes ~40 s while the cache fills)
```bash
curl -s http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "dsv4-reap",
  "messages": [{"role": "user", "content": "In one sentence: what is a Mixture-of-Experts model?"}],
  "max_tokens": 80, "chat_template_kwargs": {"enable_thinking": false}}'
```
It's an OpenAI-compatible API on port 1919, so any OpenAI client works. Drop `enable_thinking: false` to
get DeepSeek's reasoning mode.

## How each engine was run

Same PC and model files, greedy decoding. Overspill and Colibri ran in the same WSL2 VM (48 GB), page cache
dropped before each run; llama.cpp ran natively on Windows (see the results note). The
benchmark client is [`experiment/bench_openai.py`](experiment/bench_openai.py), and raw logs are in
[`experiment/results/`](experiment/results/).

**Overspill** (this repo, FreeToken v0.1.3 base):
```bash
FT_FILE_BANKS=1 FT_EMBED_HOST=1 FT_HEAD_HOST=1 FT_SWA_RATIO=0.7 FT_CPU_PREFILL_MAX=256 \
ft serve --model <dsv4-reap-ftw> --moe-strategy cpu --num-tokens 8192 --cuda-graph-max-bs 1 --max-running-requests 1
```

**llama.cpp** (release b11205, native Windows CUDA build), model
`puwaer/DeepSeek-V4-Flash-0731-reap-150b-gguf` / `DeepSeek-V4-Flash-0731-reap-150b-MXFP4_MOE.gguf` (85.05 GB),
warm (after a warm-up run), all 64 GB of RAM:
```bash
llama-server --model DeepSeek-V4-Flash-0731-reap-150b-MXFP4_MOE.gguf --ctx-size 8192 --parallel 1 \
  --n-gpu-layers 99 --cpu-moe -ub 2048 --flash-attn on --load-mode mmap --threads 11 --threads-batch 11 \
  --jinja --no-mmproj
```
| flag | meaning |
|---|---|
| `--n-gpu-layers 99` | every layer on the GPU... |
| `--cpu-moe` | ...except the expert weights, which stay in system memory and run on the CPU (= `--n-cpu-moe` for all layers) |
| `-ub 2048` | prompt micro-batch: 4x faster long-prompt processing than the default (1362 s -> 371 s to the first token); decode is unaffected |
| `--load-mode mmap` | memory-map the file: the only mode that works when the model is bigger than RAM, since experts page in from disk on demand |
| `--flash-attn on` | faster, memory-efficient attention |
| `--ctx-size 8192`, `--parallel 1` | the same 8k context and single request as Overspill |
| `--threads 11`, `--threads-batch 11` | the same 11 CPU threads as Overspill's CPU executor |

Sweep: `--cpu-moe` at `-ub 512` decodes the same (prompt processing is 4x slower); `--n-cpu-moe 42` (one layer's
experts on the GPU) is slower on this 12 GB card.

**Colibri** (commit `ce370e8`, DeepSeek-V4 engine, built with `CUDA=1 CUDA_ARCH=sm_86`), reading the original
HF checkpoint:
```bash
CUDA_DENSE=1 COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1 DSV4_CUDA_VRAM_RESERVE_MB=2500 \
V4_MOE_REFILL_GROUP=12 V4_LOADER_LANES=3 \
python3 c/coli serve --model <DeepSeek-V4-Flash-0731-reap-150b> --gpu 0 --ram 38 --ctx 12288
```
These follow Colibri's documented 10-12 GB VRAM configuration. The 6.4k prompt needs `--ctx` above the default
4096, and `--ram 38` leaves headroom in the 48 GB VM.

## Limitations

- Measured on one machine and one model. **Linux or WSL2 is the supported path.** A native Windows port exists
  ([`experiment/win/`](experiment/win/), see DESIGN_NOTES Validation 8) but is experimental: it decodes ~12% slower
  than WSL on DeepSeek-V4 today.
- Decode is bounded by expert misses from disk: faster NVMe (or native Linux instead of WSL2's
  virtual disk) and more RAM help directly.
- The first request after a cold start is slow (~40 s) while the page cache fills.
- Single request at a time (`--max-running-requests 1`) is the tested configuration.

## Credits

Overspill is a modified version of FreeToken; the FreeToken README is kept as [README_FREETOKEN.md](README_FREETOKEN.md).


- [FreeToken](https://github.com/FlashML-org/FreeToken) (Apache-2.0): the engine this is built on.
  Everything here is a modification of it, and the modified files are marked in the git history.
- [Colibri](https://github.com/JustVugg/colibri) by Vincenzo Fornaro (Apache-2.0): the disk -> RAM -> VRAM
  tiering ideas (expert heat, lookahead). No Colibri code is used.

## License

Apache-2.0, like FreeToken. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The benchmark harness, design notes and raw result logs are in [`experiment/`](experiment/).
