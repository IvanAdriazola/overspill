# Handoff: state when this workspace was created (2026-09-26 ~01:40)

## Done
- This repo is a local clone of `C:\GIT\FreeToken`, branch `workshop-constrained-decoding`
  (FreeToken v0.1.3 + our xgrammar constrained-JSON patch), on the new branch `colibri-tier-experiment`.
  Remote `freetoken-fork` = `C:\GIT\FreeToken`.
- `EXPERIMENT.md`: the goal, the plan, and attribution.
- `python/freetoken/experiment_trace.py`: routing-trace recorder (`FT_ROUTE_TRACE=<dir>`).
  **Not wired in yet.** It still needs:
  1. In `models/qwen3_5_moe/moe.py` `Qwen3_5MoE`: store `layer_id`, call
     `experiment_trace.register_gate(layer_id, self.gate)` in `__init__`, and call
     `experiment_trace.record(self.layer_id, hidden_states, router_logits, top_k)` in `forward` right
     after `router_logits` is computed, guarded by `if experiment_trace.ENABLED`.
  2. Call `flush()` at shutdown (atexit).
  3. Serve with `--graph 0`, because CUDA-graph replays skip Python.

## Next steps (plan step 0: offline simulation, no engine work)
1. Wire the tracer, install this tree in WSL (see below), and serve Qwen3.6-35B-A3B-NVFP4 with tracing.
   Drive it with the Workshop benchmark batch
   (`C:\GIT\chatbot\benchmark\moe_engines\bench_workshop_freetoken.sh`) so the traces come from
   real agent requests.
2. `experiment/simulate.py`: replay the traces against placement policies: LRU (FreeToken today),
   heat+hysteresis (Colibri), cost-aware, lookahead prefetch using `pred_next`, and prefill
   routed-union vs whole-layer streaming. Use shrunken VRAM/RAM budgets to simulate a model bigger
   than RAM. Report hit rates and disk bytes per token.
3. Go/no-go on the prototype.

## Environment facts (from the chatbot workspace)
- Machine: Ryzen 9 7900, 64 GB RAM (WSL VM capped at 48 GB via `.wslconfig`), RTX 3060 12 GB, CUDA 13.0 in WSL.
- WSL distro `Ubuntu-24.04`, user `ivan`. Never touch the `docker-desktop` distro. Don't run `wsl --shutdown`
  while other work runs.
- The installed FreeToken lives in `~/ft` inside WSL (venv `~/ft/.venv`, editable from `~/src/FreeToken`).
  The serve script is `C:\GIT\chatbot\benchmark\moe_engines\serve_ft.sh`: it needs
  `TVM_FFI_CUDA_ARCH_LIST=8.6 TORCH_CUDA_ARCH_LIST=8.6 FLASHINFER_CUDA_ARCH_LIST=8.6`. The model is
  `~/models/Qwen3.6-35B-A3B-NVFP4`, served on port 1919.
  To test this tree, make a separate editable install; don't disturb `~/ft`, which Workshop uses.
- Colibri: `~/colibri` (engines built: qwen36, deepseek_v4 for sm_86). Models: `~/models/qwen36_i4_gs64`
  and `~/models/dsv4_reap150b` (85 GB, bigger than RAM).
- **GPU rule:** one GPU user at a time, via `C:\GIT\chatbot\benchmark\_with_gpu_lock.py <owner> -- <cmd>`
  (lock file `benchmark/.gpu_lock`, 90-min cap).
- From Windows, call WSL with `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u ivan -- bash <script>`.
  A server started with nohup dies with its wsl.exe, so keep it in the foreground of a live wsl.exe.
  `pkill -f` patterns can match your own shell: kill by PID from a script file.

## Pitfalls hit while capturing traces (2026-09-26)
- **Windows→Git Bash mangles absolute POSIX args:** `/home/ivan/x` passed from Windows Python became
  `C:/Program Files/Git/home/ivan/x`. Pass bare names and let the WSL-side script build paths
  (`serve_trace.sh <trace_name>` → `~/traces/<name>/`).
- **The Python GPU-lock wrapper died about 32 min after launch as a background Bash job, with no message.**
  Its children (the server and the benchmark) kept running and the lock file stayed. My hypothesis, not
  confirmed: the background shell killed its direct child only. For long GPU jobs, check the processes;
  don't trust the wrapper's exit code. Stop the server with `stop_server.sh` and remove
  `benchmark/.gpu_lock` by hand.
- Task 1.3 printed no result line in the traced run (no result JSON either). Its routing trace is still valid.
- Without CUDA graphs (`--graph 0`), decode runs at ~10-24 tok/s. The routing is identical, because decoding is greedy.

## Measured baselines (Qwen3.6-35B-A3B, RTX 3060)
| engine | decode tok/s | prefill tok/s |
|---|---|---|
| Ollama Q4_K_M | 45 | 263 |
| FreeToken NVFP4 | 61-66 | 810-1020 |
| Colibri int4 | 9-12 | 26 |

Colibri on DSV4-Flash REAP-150B (85 GB, bigger than RAM): decode 1.1-1.2 tok/s, prefill 4 tok/s
(6.4k-token prompt = 26 min).

FreeToken in Workshop: 41% of LLM time goes to prefill and overhead, and the prefix cache hits only 3.5% of prompt tokens.

Full docs: `C:\GIT\chatbot\benchmark\moe_engines\{FREETOKEN_COLIBRI_EXPLAINED,ENGINES_DOSSIER,COLIBRI_BIG_MODEL}.md`.

## User preferences that carry over
- One task at a time. Commit after each verified piece. Cheapest experiment first.
- In autonomous work: make the call, log it, keep going.
- End goal: the user's own engine written from scratch; this tree is a throwaway prototype.
  Credit FreeToken and Colibri for the ideas.

## Status update (2026-09-26 ~05:15): DeepSeek-V4-Flash REAP-150B runs faster than Colibri
- Serve: `bench_dsv4.sh <label> --num-tokens 8192 --cuda-graph-max-bs 1 --max-running-requests 1`,
  launched with `WSLENV=FT_SWA_RATIO FT_SWA_RATIO=0.7` under the GPU lock (serve config: `serve_dsv4.sh`).
- The model is `~/models/dsv4_reap150b_ftw` (FTW, 80 GB). The rebuilt CPU MoE .so lives in `~/src/freetoken-exp`
  (built by `build_cpu_moe.sh`; `sync_to_wsl.sh` keeps it, since `cp -n` won't overwrite it).
- Results and the lever list: see `DESIGN_NOTES.md`, "Prototype 2".
- Pitfalls: env vars reach WSL only via `WSLENV`; `MSYS_NO_PATHCONV=1` is inherited, so use `C:/` paths
  in scripts; Python on Windows writes CRLF (use `newline="\n"` for .sh files); FreeToken's API outlives a
  dead backend (it answers 503), so check the serve log.
