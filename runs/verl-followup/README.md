# Follow-up: TP 2 and a larger state pool (2026-11-25)

Two A100 SXM 80 GB, SGLang 0.5.23 (`lmsysorg/sglang:v0.5.23`), Qwen3.5-9B in bf16. Runner: `scripts/run_verl_followup.sh`.

Every server used `--mem-fraction-static 0.5 --max-running-requests 256 --context-length 36864 --enable-session-radix-cache`. Every run is one GRPO step from `scripts/replay_grpo.py`: 8 samples per prompt, 512 new tokens, either `p0` (all requests at once) or `hold` with `--window 32`. The policies are described in `../verl-defaults/README.md`.

| Folder | Server | State slots | Running limit | KV tokens |
| --- | --- | --- | --- | --- |
| `../verl-tp2` | `--tp 2` | 592 | 118 | 1,039,040 |
| `../verl-ratio4` | TP 1, `--mamba-full-memory-ratio 4` | 359 | 71 | 142,739 |
| `../verl-ssm-bf16` | TP 1, `--mamba-ssm-dtype bfloat16` | 416 | 83 | not run |

`verl-ssm-bf16` has only its server log. The runner's readiness check flushed twice: the first flush succeeded, and the second hit SGLang's own warm-up request and got a 400.

This folder holds:

- `verl-followup.log`: the runner log;
- `verl-followup-gpu.csv`: GPU use every 5 s;
- `miss_timing.txt`: from `python3 scripts/miss_timing.py` over the p0 and warm runs here and in `../verl-defaults`. It splits misses into each group's first sample, siblings estimated as scheduled before that sample's first token, and siblings estimated as scheduled after it.
  - The estimate is arrival plus `queue_time`, and it runs early by up to about 1–2 s. Most hits also land in "before".
  - So "before" mixes concurrent prefills with early evictions. Only the "after" bucket is reliable: 1 miss of 408.
