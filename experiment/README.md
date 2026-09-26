# experiment/ - how Overspill was built and measured

This is the author's working harness, kept for transparency and reproducibility. Scripts contain paths
specific to the author's machine (Windows + WSL2 Ubuntu-24.04, repo at `C:\GIT\...`, models in `~/models`);
adapt them before use.

- `DESIGN_NOTES.md` - every experiment, including the ones that failed (e.g. RAM->VRAM lookahead prefetch
  was correct but 9-29% *slower* on an RTX 3060), with the numbers.
- `results/` - raw benchmark logs behind the README tables (`dsv4_v12cold.log`, `dsv4_v14cpuprefill.log`,
  `llamacpp_cold.log`, the Qwen correctness A/B `qwentier_*.log`, ...).
- `serve_dsv4.sh`, `bench_dsv4.sh` - serve/benchmark DeepSeek-V4-Flash REAP-150B with the disk tier.
- `bench_llamacpp.sh`, `llamacpp/` - the llama.cpp comparison on the same model.
- `bench_openai.py` - the streaming benchmark client (TTFT, prefill and decode tok/s).
- `fault_bench.py` - page-fault vs madvise(WILLNEED) expert loading microbenchmark.
- `simulate.py`, `serve_trace.sh` - routing-trace capture and the offline expert-placement simulator.
- `build_cpu_moe.sh` - rebuild the CPU MoE extension after changing its C++.
