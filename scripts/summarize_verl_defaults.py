"""Tables for the check at verl's defaults (runs/verl-defaults).

GRPO runs: step time, prompt tokens recomputed by samples, and misses (samples
that recomputed more than 512 tokens), per policy and seed, with the change in
step time against p0 of the same seed. Forest runs: completion time and
recomputed tokens, p0 against p0root.

  python3 scripts/summarize_verl_defaults.py [runs/verl-defaults]
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "runs/verl-defaults"

grpo = defaultdict(dict)  # seed -> policy -> summary
forest = defaultdict(dict)
for path in sorted(glob.glob(os.path.join(d, "*.json"))):
    name = os.path.basename(path)[:-5]
    m = re.fullmatch(r"g(\d+)-s(\d+)-(p0|warm|hold)(-w\d+)?", name)
    f = re.fullmatch(r"f\d+-r\d+-s(\d+)-(p0|p0root)", name)
    if not (m or f):
        continue
    with open(path) as fh:
        run = json.load(fh)
    s = run["summary"]
    if m:
        grpo[(int(m.group(1)), int(m.group(2)))][m.group(3) + (m.group(4) or "")] = s
    else:
        forest[int(f.group(1))][f.group(2)] = s

order = ["p0", "warm", "hold", "warm-w32", "hold-w32"]
if grpo:
    print("GRPO step, 8 samples per prompt, 512 new tokens")
    print(f"{'prompts':>7} {'seed':>4} {'policy':<9} {'step s':>7} {'vs p0':>7} {'recomp k':>9} "
          f"{'unique k':>9} {'misses':>7} {'samples':>8} problems")
    for key in sorted(grpo):
        base = grpo[key].get("p0", {}).get("step_s")
        for pol in order:
            s = grpo[key].get(pol)
            if not s:
                continue
            rel = f"{100 * (s['step_s'] / base - 1):+.1f}%" if base and pol != "p0" else ""
            print(f"{key[0]:>7} {key[1]:>4} {pol:<9} {s['step_s']:>7.1f} {rel:>7} "
                  f"{s['sample_recomputed'] / 1e3:>9.0f} {s['prompt_tokens_unique'] / 1e3:>9.0f} "
                  f"{s['sample_misses']:>7} {s['samples']:>8} {len(s['problems'])}")
if forest:
    print("\nForest, 16 trees with 8k roots")
    print(f"{'seed':>4} {'policy':<7} {'done s':>7} {'vs p0':>7} {'recomp k':>9} problems")
    for seed in sorted(forest):
        base = forest[seed].get("p0", {}).get("completion_s")
        for pol in ["p0", "p0root"]:
            s = forest[seed].get(pol)
            if not s:
                continue
            rel = f"{100 * (s['completion_s'] / base - 1):+.1f}%" if base and pol != "p0" else ""
            print(f"{seed:>4} {pol:<7} {s['completion_s']:>7.1f} {rel:>7} "
                  f"{s['recomputed_tokens'] / 1e3:>9.0f} {len(s['problems'])}")
