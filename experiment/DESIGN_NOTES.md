# Design notes (carry these into the clean rewrite)

## Sim 1 - placement policies on real Workshop traces (2026-09-26)
Traces: 10 Workshop tasks on Qwen3.6-35B-A3B, 25,139 decode tokens, 384k prompt tokens (`~/traces/workshop10`,
`sim_workshop10.txt`). Simulated "model bigger than RAM" by shrinking the RAM tier; VRAM = 3347 slots (real).
Time model calibrated to measured FreeToken (62 tok/s = 13.0 ms compute + 3.2 ms PCIe stalls).

Findings:
1. **Lookahead prefetch helps even when everything fits in RAM**: LRU VRAM hit 86.7% -> 94.1% (8 guesses)
   / 96.5% (16) -> est. 62 -> 69.6 / 72.5 tok/s (+12-17%) on FreeToken as-is.
2. **Use different policies per tier.** VRAM: LRU (86.7%) beats heat/LFU-with-periodic-rebalance (78.8%);
   the working set shifts too fast for a 16-token rebalance. RAM tier: heat/LFU beats LRU (disk misses
   at 75% RAM: 0.97% vs 2.87%). Colibri-style LFU on both tiers is the worst combo.
3. **Speculative disk reads hurt.** Prefetching wrong guesses from disk costs more than it saves at
   <=50% RAM. Prefetch should pull only RAM-resident guesses into VRAM (disk prefetch only for
   high-confidence guesses). Not simulated yet: needs a "prefetch from RAM only" variant.
4. **Prefill union vs whole-layer**: with a heat-managed RAM tier, prompt reading stays near the compute
   bound (870-1014 tok/s at 75% RAM) vs 4 tok/s measured on Colibri's CPU fallback. Union matters most
   when RAM is small.
5. Best predicted decode (Qwen-scale experts), WSL virtual disk / native NVMe:
   - 75% RAM: 61.5 / 66.9 tok/s (LRU VRAM + LFU RAM + 8-guess prefetch)
   - 50% RAM: ~43 / 60 tok/s; 35% RAM: ~34 / 55 tok/s
   - vs FreeToken-style LRU everywhere: 37 / 25 / 20 tok/s on the WSL disk (75/50/35%) -> **~1.7x from placement alone**.

Caveats: my LFU is a simple variant (inserts only if room, rebalances every 16 tokens), so it may undersell
Colibri's approach. Qwen experts are small (1.78 MB, 3B active); for DeepSeek-150B (~13B active, larger
experts) only the ratios transfer, not the tok/s. Contention for RAM bandwidth is not modelled. Belady
(hindsight optimum, no prefetch) = 94.1% VRAM hit, the same as LRU + 8-guess lookahead.

## Prototype 1 - router-lookahead prefetch in FreeToken (2026-09-26): NEGATIVE on RTX 3060
Real A/B, Qwen3.6-35B-A3B, CUDA graphs on, greedy (`ab1_k*.log`). Outputs identical (5/5), so the change is correct.
| k (guesses/layer) | short decode tok/s | long-prompt decode |
|---|---|---|
| 0 (baseline) | 75-79 | 69-72 |
| 8 | 70-72 (-9%) | 63-65 |
| 16 | 55-56 (-29%) | 52 |
The simulator assumed prefetch copies overlap compute for free. They don't here: FreeToken's copy is an SM
kernel (zero-copy loads), and the 3060 has only 28 SMs, so side-stream copies steal the GEMM's SMs. Each layer
also adds a router GEMV + topk + lru_ensure. Wrong guesses (~20% at k=8) also waste PCIe.
**Lesson for the engine:** overlap only helps with the DMA copy engine (cudaMemcpyAsync, no SMs used), or
when the bottleneck is disk rather than PCIe. For disk (two-miss) prefetch, the host issues reads, so it
doesn't compete with GPU SMs at all. Keep the lookahead idea for the disk tier, not for RAM->VRAM on this GPU.

## Prototype 2 - DeepSeek-V4-Flash REAP-150B (85 GB) on FreeToken with a disk tier (2026-09-26)
Setup: FTW conversion (fixed a shmem-release OOM bug in the converter); all 43 expert layers file-mapped
(`FT_FILE_BANKS`, page cache = RAM tier); expert math on FreeToken's CPU executor; embedding and LM head
read from pinned host RAM over UVA (`FT_EMBED_HOST`, `FT_HEAD_HOST`, frees 2 GiB of VRAM); a one-layer GPU
prefill buffer; `--max-running-requests 1`; a larger DSV4 window pool (`FT_SWA_RATIO=0.7`) so prompts
chunk at ~2.3k tokens instead of 384 (each chunk re-streams every layer's experts).
Same prompts as the Colibri cold run:
| | FreeToken v10 | Colibri cold |
|---|---|---|
| 6.4k-token prompt, time to first token | **102 s** | 1565 s (15x slower) |
| prefill tok/s (long) | 63 | 4.1 |
| decode tok/s (short / coding / long) | 0.94 / 0.93 / 0.82 | 1.17 / 1.20 / 1.12 |
v8 (384-token chunks, 17 chunks): TTFT 486 s. Chunk size is the whole prefill story: each chunk costs
one pass over all 70 GiB of experts (disk + page cache).
Decode still trails Colibri. Disk reads during decode run at only ~0.9 GB/s, against 2.2 GB/s during
prefill streaming. The CPU executor pulls experts in through page faults. Next: an explicit WILLNEED /
parallel read of the routed experts.

### v11/v12 - WILLNEED prefetch of routed experts: GOAL MET (faster than Colibri)
Microbench (`fault_bench.py`, cold cache, real DSV4 expert files, 6 random experts per layer, 12 threads):
page faults 0.46-0.50 GiB/s (~155 ms/layer) vs madvise(WILLNEED)-then-touch 2.00 GiB/s (37 ms/layer) -> 4x.
Implemented in the C++ CPU executor (`prefetch_routed` in `submit`, rebuilt via `build_cpu_moe.sh`).
Cold run (page cache dropped first, like Colibri's cold run), same prompts:
| | FreeToken v12 cold | Colibri cold | |
|---|---|---|---|
| short decode | 2.78 tok/s | 1.17 | 2.4x |
| coding decode | 3.10 tok/s | 1.20 | 2.6x |
| long decode | 2.69 tok/s | 1.12 | 2.4x |
| 6.4k prompt, time to first token | 98.6 s | 1565 s | 15.9x |
| short prompt, time to first token | 27-34 s | 17.5-25 s | Colibri better |
The warm run (v11, ~43 GB already in the page cache) gave the same numbers, so the gain isn't a caching artifact.
Colibri's warm run (with its saved usage history) was never measured, so its warm decode could be higher than 1.2.

Remaining levers, cheapest first:
1. Short-prompt TTFT: prefill streams ALL experts of every layer even for a 38-token prompt. Union-only
   prefill (copy just the routed experts) should bring short TTFT to a few seconds.
2. Next-layer disk prefetch (the lookahead idea, host-side this time, so it takes no GPU SMs): WILLNEED the
   predicted experts of layer L+1 while layer L computes.
3. Hot-expert residency: popularity-based (LFU) mlock of the hottest experts so the page cache can't evict
   them (sim: LFU beats LRU for the RAM tier).
4. Raise WSL's memory cap (48 -> ~56 GB) for more page cache.

## Validation 1 - the tier layer does not change the model's math (2026-09-26)
Qwen3.6-35B-A3B FTW, `--moe-strategy cpu`, greedy, 5 prompts (3 short, 1 long 6k, 1 coding):
stock vs tier (FT_FILE_BANKS + WILLNEED + FT_EMBED_HOST + FT_HEAD_HOST) vs tier_nohead -> **byte-identical
outputs, 5/5 for both** (`qwentier_*.log`). Speed on a model that fits in RAM is unchanged within noise
(stock 25-27, tier 22-28, tier_nohead 28-31 tok/s).

## Validation 2 - vs Colibri warm and llama.cpp, same model (2026-09-26)
Colibri warm (prime session + warm page cache; no usage-history file was written): decode 1.19 / 1.24 / 1.16,
TTFT 26 / 18 / 1557 s -> same as cold.
llama.cpp `81bc6b8` (CUDA sm_86), `DeepSeek-V4-Flash-0731-reap-150b-MXFP4_MOE.gguf` (85.05 GB, same FP4
experts), `-ngl 99 --cpu-moe --flash-attn on --load-mode mmap -t 11`, cold: decode 0.54 / 0.49 / 0.38 tok/s,
TTFT 44 / 40 / 373 s, ~1.0 GB/s average disk reads. Answers correct.
=> this fork: ~5-7x llama.cpp decode, 3.8x long-prompt TTFT, and faster short-prompt TTFT than llama.cpp.
Cross-engine token equality isn't expected (llama.cpp keeps attention in Q8_0, FreeToken in FP8); layer
correctness rests on Validation 1 (byte-identical to stock FreeToken).

## Validation 3 - CPU short-prefill, clean cold run (v14)
`FT_CPU_PREFILL_MAX=256`: decode 3.21 / 3.37 / 2.75 tok/s (vs 2.78 / 3.10 / 2.69 without it), TTFT 43 s (first
request after a cold start; was 34 s), 10.2 s (second short prompt; was 27.5 s), 102 s (6.4k; was 99 s).
Short prompts no longer stream all 70 GB through the page cache, which evicted the experts decode needs,
so decode improves too. Shipped on by default. (The v13 run was contaminated by a concurrent 85 GB download.)

## Validation 4 - real Workshop agent traffic on DSV4 through the disk tier
3 Workshop tasks (tool calls, grammar-constrained JSON, multi-turn, prefix cache hits ~1.8k tokens/call):
9.1 PASS (474 s), 8.1 PASS (544 s), 2.1 FAIL (87 s): on daring's free-text first call the model wrote its own
`<tool_call>` markup instead of Workshop's format - a model/prompt-format mismatch, not an engine error.

## Validation 5 - llama.cpp `--n-cpu-moe 42` (suggested by a Reddit commenter)
Same build and flags as the published run, but `--n-cpu-moe 42` instead of `--cpu-moe` (the experts of
1 of 43 layers go on the GPU, which is all that fits next to ~9.4 GB of non-expert weights on 12 GB). Cold:
decode 0.58 / 0.54 / 0.49 tok/s (vs 0.54 / 0.49 / 0.38), TTFT 42 / 39 / 349 s (vs 44 / 40 / 373 s).
Caveat: the long prompt here is 5,937 tokens (the bench was decoupled from the private module used for the
6,446-token runs), so that row is close but not identical. Net: +10-25% for llama.cpp; Overspill still ~5-6x faster.
