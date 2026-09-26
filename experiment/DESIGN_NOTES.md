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
