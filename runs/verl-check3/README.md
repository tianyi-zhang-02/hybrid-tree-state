# Third check: exact timing and lpm (2026-11-25)

Two A100 SXM 80 GB, TP 2, SGLang 0.5.23, Qwen3.5-9B in bf16. Server flags as in `../verl-followup/README.md`. Runner: `scripts/run_verl_check3.sh`.

Runs: one GRPO step, 128 prompts × 8 samples, 512 new tokens, p0 (all requests at once).

- `../verl-timing-fcfs`: default `--schedule-policy fcfs`, seeds 0 and 1.
- `../verl-timing-lpm`: `--schedule-policy lpm`, seed 0.

Every sample record carries SGLang's `forward_entry_time` and `prefill_finished_time`, kept by `scripts/replay_tree.py` since this check.

`miss_timing_exact.txt` comes from `python3 scripts/miss_timing_exact.py` on the three runs. No missed sample entered a forward batch after its group's first prefill finished, the moment SGLang puts the prompt's state into the cache, so the misses are concurrent prefills, not evictions. 388–455 misses per run still reused part of the prompt, a chunk of a sibling's prefill that was still running.

This folder also has `verl-check3.log` (the runner log) and `check3.out`.
