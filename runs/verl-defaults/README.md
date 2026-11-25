# Check at verl's rollout defaults (2026-11-24)

One A100 SXM 80 GB, SGLang 0.5.23 (`lmsysorg/sglang:v0.5.23`), Qwen3.5-9B in bf16, TP 1.

The server was started by `scripts/run_verl_defaults.sh` with `--mem-fraction-static 0.5 --max-running-requests 256 --enable-session-radix-cache`. Those two values are verl's `gpu_memory_utilization` and `max_num_seqs`. Everything else was left at SGLang's defaults.

SGLang lowered the running limit to 42 because of the Mamba state pool (212 slots); see `serve.log`.

## Files

- `g128-s{seed}-{policy}.json`: one GRPO-shaped step from `scripts/replay_grpo.py`.
  - 128 prompts × 8 samples, with prompt lengths log-uniform in 1k–16k tokens.
  - 512 new tokens per sample, and all requests sent at once.
  - Policies:
    - `p0`: send as is;
    - `warm`: prefill each prompt once, then send its samples;
    - `hold`: warm, plus a session on the prompt that is closed when the group's last sample starts decoding;
    - `-w32`: at most 32 groups in flight.
- `f16-r8000-s{seed}-{p0,p0root}.json`: 16 tree-shaped groups with 8k roots, from `scripts/replay_forest.py`. `p0root` holds each tree's root with a session.
- `summary.txt`: the tables, from `python3 scripts/summarize_verl_defaults.py runs/verl-defaults`.
- `runner.log`, `serve.log`, `server_info.json`, `gpu.csv`: the runner, the SGLang server log, resolved server settings, and GPU use every 5 s.

## Gaps

- Seed 2 has no `warm` run and no forest runs, because of a time cap.
- In seed 0, `p0` and `warm` each lost the last 3 of 1,024 samples to the client's open-file limit. The limit was raised before the later runs.
- The capacity probe and smoke runs failed. Their `/flush_cache` arrived while SGLang's warm-up request was still running.
