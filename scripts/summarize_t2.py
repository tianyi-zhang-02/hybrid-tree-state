"""Summarize runs/t2 (scripts/replay_t2.py) into one table per root.

For each budget and policy: completion time relative to the best policy in that
row (mean over seeds, worst seed in brackets), recomputed tokens, recomputed tokens
on revisits (the first step of a simulation that forks from an old node), full
misses, and how many selected leaves were held when selected.

Policies in a row are compared only on the seeds every one of them has run, so an
unfinished sweep does not compare different trees.

Floor: for each (root, seed) and each request, the fewest tokens any run
recomputed for it; summed over the tree. Tree size differs by seed, so the time
per recomputed token is fitted within each seed (both sides minus the seed's mean),
and the floor's time is read off each seed's line and averaged.

Usage: summarize_t2.py [runs/t2]
"""

import glob
import json
import statistics as st
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "runs/t2"
POL = ["p0", "p0root", "p2sel", "poracle"]
runs = []
for f in sorted(glob.glob(f"{d}/t2-s*-r*-b*-*.json")):
    x = json.load(open(f))
    a = x["args"]
    runs.append({"seed": a["seed"], "root": a["root_len"], "budget": a["budget"],
                 "policy": a["policy"], "s": x["summary"],
                 "rec": {r["node"]: r["recomputed"] for r in x["records"]}})
if not runs:
    sys.exit(f"no runs in {d}")

bad = [(r["root"], r["budget"], r["policy"], r["seed"], r["s"]["problems"])
       for r in runs if r["s"]["problems"]]
for b in bad:
    print("PROBLEMS", b)

for root in sorted({r["root"] for r in runs}):
    rr = [r for r in runs if r["root"] == root]
    # Floor per seed: the fewest tokens any run recomputed for each request.
    floor = {}
    for seed in sorted({r["seed"] for r in rr}):
        best = defaultdict(lambda: float("inf"))
        for r in rr:
            if r["seed"] == seed:
                for n, v in r["rec"].items():
                    best[n] = min(best[n], v)
        floor[seed] = sum(best.values())
    gm = {}
    for seed in floor:
        g = [r for r in rr if r["seed"] == seed]
        gm[seed] = (st.mean(r["s"]["recomputed_tokens"] for r in g),
                    st.mean(r["s"]["completion_s"] for r in g))
    dx = [r["s"]["recomputed_tokens"] - gm[r["seed"]][0] for r in rr]
    dy = [r["s"]["completion_s"] - gm[r["seed"]][1] for r in rr]
    sxx = sum(x * x for x in dx)
    b = sum(x * y for x, y in zip(dx, dy)) / sxx if sxx else 0.0
    ss = sum(y * y for y in dy)
    r2 = 1 - sum((y - b * x) ** 2 for x, y in zip(dx, dy)) / ss if ss else 1.0
    fl = st.mean(floor.values())
    ft = st.mean(gm[s][1] + b * (floor[s] - gm[s][0]) for s in floor)
    rate = f"{1 / b / 1000:.1f}k tokens/s" if b > 0 else "no positive slope"
    print(f"\n## Root {root}: {len(rr)} runs. Within-seed fit: {rate} (R^2 {r2:.2f}). "
          f"Floor {fl / 1000:.0f}k tokens, {ft:.1f} s (mean over seeds).")
    print("| Budget | Best time | " + " | ".join(POL) + " |")
    print("| ---: | ---: | " + " | ".join("---" for _ in POL) + " |")
    for budget in sorted({r["budget"] for r in rr}):
        cell = {}
        for p in POL:
            g = [r for r in rr if r["budget"] == budget and r["policy"] == p]
            if g:
                cell[p] = g
        common = set.intersection(*({r["seed"] for r in g} for g in cell.values()))
        cell = {p: [r for r in g if r["seed"] in common] for p, g in cell.items()}
        means = {p: st.mean(r["s"]["completion_s"] for r in g) for p, g in cell.items()}
        best = min(means.values())
        row = []
        for p in POL:
            if p not in cell:
                row.append("")
                continue
            g = sorted(cell[p], key=lambda r: r["seed"])
            # Worst seed: largest loss against the best policy's run on the same seed.
            bestp = min(means, key=means.get)
            per_seed = {r["seed"]: r["s"]["completion_s"] for r in cell[bestp]}
            worst = max(r["s"]["completion_s"] / per_seed[r["seed"]] - 1
                        for r in g if r["seed"] in per_seed)
            rec = st.mean(r["s"]["recomputed_tokens"] for r in g)
            rev = st.mean(r["s"]["recomputed_revisit"] for r in g)
            fm = sum(r["s"]["full_misses"] for r in g)
            held = sum(r["s"]["leaves_held_when_selected"] for r in g)
            sel = sum(r["s"]["selections"] for r in g)
            rel = "best" if means[p] == best else f"+{100 * (means[p] / best - 1):.1f}% (+{100 * worst:.1f}%)"
            row.append(f"{rel} · {rec / 1000:.0f}k (rev {rev / 1000:.0f}k) · fm {fm}"
                       + (f" · held {held}/{sel}" if p in ("p2sel", "poracle") else ""))
        print(f"| {budget} ({len(common)} seeds) | {best:.1f} s | " + " | ".join(row) + " |")
    print("\nPer seed, recomputed floor: " + ", ".join(f"s{k} {v / 1000:.1f}k" for k, v in floor.items()))
