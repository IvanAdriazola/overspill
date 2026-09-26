# FreeToken x Colibri tiering experiment

**Throwaway prototype.** This tree is a fork of [FreeToken](https://github.com/FlashML-org/FreeToken) (Apache 2.0, see
`LICENSE`), used to test whether ideas from FreeToken and from [Colibri](https://github.com/JustVugg/colibri) (Vincenzo Fornaro, Apache 2.0)
combine into a faster engine for MoE models larger than RAM. Code here may be derived from either
project and keeps their notices. If the idea holds, the real engine is written from scratch
from `experiment/DESIGN_NOTES.md`, and this tree is archived.

## Question
FreeToken runs the model fast, but it needs every expert in RAM. Colibri can serve experts from disk,
but its compute path is slow. Would a smarter disk -> RAM -> VRAM placement, combined with
FreeToken's execution path, make models bigger than RAM usable on an RTX 3060 12 GB with 64 GB RAM?

Background and measured numbers: `C:\GIT\chatbot\benchmark\moe_engines\FREETOKEN_COLIBRI_EXPLAINED.md`
and `ENGINES_DOSSIER.md`.

## Plan (cheapest first)
0. **Offline placement simulation** (`experiment/`):
   1. Capture real per-layer, per-token routing traces (topk expert ids) from FreeToken serving
      Qwen3.6-35B-A3B on Workshop-style requests.
   2. Replay the traces against placement policies (LRU, heat+hysteresis, cost-aware, router-lookahead
      prefetch, per-role warm start) with artificially small VRAM/RAM budgets. This simulates a model
      that doesn't fit.
   3. Output: hit rates and disk bytes per token, which give predicted tok/s for each policy.
   Go/no-go: this sizes the prize before any engine work.
1. **Prototype in this tree**: RAM tier becomes a cache backed by disk, with a smarter placement
   policy, next-layer prefetch, and batched prefill that reads only the routed experts.
2. **Measure on a real bigger-than-RAM model** against Colibri (1.2 tok/s decode measured on
   DeepSeek-V4-Flash REAP-150B).
