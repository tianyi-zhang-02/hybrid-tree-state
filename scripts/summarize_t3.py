"""Summarize runs/t3 (scripts/replay_t3.py) into one table.

For each budget and policy: completion time relative to the best policy in that row
(mean over seeds, worst seed in brackets), recomputed tokens on action requests (a
held parent leaves only the 2 turn tokens), parent states lost, and how many
expanded parents were held when selected. Tool requests prefill their tool output
in every policy; their recompute is shown once per row as a check.

Policies in a row are compared only on the seeds every one of them has run.

Floor: for each seed and each request, the fewest tokens any run recomputed for it,
summed over the tree. The time per recomputed token is fitted within each seed (both
sides minus the seed's mean), and the floor's time is read off each seed's line.

Usage: summarize_t3.py [runs/t3]
"""

import glob
import json
import statistics as st
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "runs/t3"
POL = ["p0", "p0root", "p2sel", "poracle"]
runs = []
for f in sorted(glob.glob(f"{d}/t3-s*-b*-*.json")):
    x = json.load(open(f))
    a = x["args"]
    runs.append({"seed": a["seed"], "budget": a["budget"], "policy": a["policy"],
                 "s": x["summary"],
                 "rec": {(r["node"], r["kind"]): r["recomputed"] for r in x["records"]}})
if not runs:
    sys.exit(f"no runs in {d}")

for r in runs:
    if r["s"]["problems"]:
        print("PROBLEMS", r["budget"], r["policy"], r["seed"], r["s"]["problems"][:3])

floor = {}
for seed in sorted({r["seed"] for r in runs}):
    best = defaultdict(lambda: float("inf"))
    for r in runs:
        if r["seed"] == seed:
            for key, v in r["rec"].items():
                best[key] = min(best[key], v)
    floor[seed] = sum(best.values())
gm = {s: (st.mean(r["s"]["recomputed_tokens"] for r in runs if r["seed"] == s),
          st.mean(r["s"]["completion_s"] for r in runs if r["seed"] == s)) for s in floor}
dx = [r["s"]["recomputed_tokens"] - gm[r["seed"]][0] for r in runs]
dy = [r["s"]["completion_s"] - gm[r["seed"]][1] for r in runs]
sxx = sum(x * x for x in dx)
b = sum(x * y for x, y in zip(dx, dy)) / sxx if sxx else 0.0
ss = sum(y * y for y in dy)
r2 = 1 - sum((y - b * x) ** 2 for x, y in zip(dx, dy)) / ss if ss else 1.0
ft = {s: gm[s][1] + b * (floor[s] - gm[s][0]) for s in floor}
rate = f"{1 / b / 1000:.1f}k tokens/s" if b > 0 else "no positive slope"
print(f"{len(runs)} runs. Within-seed fit: {rate} (R^2 {r2:.2f}). "
      f"Floor {st.mean(floor.values()) / 1000:.0f}k tokens, {st.mean(ft.values()):.1f} s (mean over seeds).")
print("Per seed floor: " + ", ".join(f"s{s} {floor[s] / 1000:.0f}k tokens {ft[s]:.1f} s" for s in floor))
print()
print("| Budget | Best time | Floor gap | " + " | ".join(POL) + " | tool recompute |")
print("| ---: | ---: | ---: | " + " | ".join("---" for _ in POL) + " | ---: |")
for budget in sorted({r["budget"] for r in runs}):
    cell = {p: [r for r in runs if r["budget"] == budget and r["policy"] == p] for p in POL}
    cell = {p: g for p, g in cell.items() if g}
    common = set.intersection(*({r["seed"] for r in g} for g in cell.values()))
    cell = {p: [r for r in g if r["seed"] in common] for p, g in cell.items()}
    means = {p: st.mean(r["s"]["completion_s"] for r in g) for p, g in cell.items()}
    bestp = min(means, key=means.get)
    best = means[bestp]
    floor_t = st.mean(ft[s] for s in common)
    per_seed = {r["seed"]: r["s"]["completion_s"] for r in cell[bestp]}
    row = []
    for p in POL:
        if p not in cell:
            row.append("")
            continue
        g = cell[p]
        worst = max(r["s"]["completion_s"] / per_seed[r["seed"]] - 1 for r in g)
        act = st.mean(r["s"]["recomputed_action"] for r in g)
        lost = sum(r["s"]["parent_state_lost"] for r in g)
        held = sum(r["s"]["parents_held_when_selected"] for r in g)
        sel = sum(r["s"]["selections"] for r in g)
        rel = "best" if p == bestp else f"+{100 * (means[p] / best - 1):.1f}% (+{100 * worst:.1f}%)"
        row.append(f"{rel} · action {act / 1000:.0f}k · lost {lost} · held {held}/{sel}")
    tool = st.mean(r["s"]["recomputed_tool"] - r["s"]["tool_tokens"] for g in cell.values() for r in g)
    print(f"| {budget} ({len(common)} seeds) | {best:.1f} s | {100 * (best / floor_t - 1):+.1f}% | "
          + " | ".join(row) + f" | +{tool / 1000:.1f}k |")
