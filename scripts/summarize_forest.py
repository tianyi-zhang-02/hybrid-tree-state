"""Summarize the forest sweep (several T1 trees at once) into one table.

For each forest (4 trees with 8k roots, 2 trees with 32k roots), budget and policy,
reports over the three seeds: mean time until the last tree finishes (min-max),
mean recomputed tokens, and fallback counts summed over seeds.

Also fits time = a + b * recomputed within each forest across all 27 runs, to
read off the prefill cost per recomputed token and the time at the no-loss
floor (the recomputation a single tree has at full budget, times the number of
trees).

Usage: summarize_forest.py [runs/forest]
"""

import glob
import json
import statistics as st
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "runs/forest"
FORESTS = [("f4-r8000", 4, 8000), ("f2-r32064", 2, 32064)]
BUDGETS = [8, 16, 31]
POLS = ["p0", "p1", "p0root"]

# Single-tree recomputation at full budget (runs/trees, P0 extra_buffer b31):
# 11,648 at 8k and 10,624 at 32k for every shape.
SINGLE = {8000: 11648, 32064: 10624}

rows = {}
for f in glob.glob(f"{d}/f*-extra_buffer-b*-*-s*.json"):
    x = json.load(open(f))
    lab = x["label"]
    forest = "-".join(lab.split("-")[:2])
    b = int(lab.split("-b")[1].split("-")[0])
    pol = lab.split("-")[-2]
    s = x["summary"]
    assert not s["problems"], (lab, s["problems"])
    rows.setdefault((forest, b, pol), []).append(s)

print("| Forest | Budget | Policy | Time, s (min-max) | Recomputed, k tokens | Full misses | Fallbacks (parent/ancestor/root/inside_root/none) |")
print("| --- | ---: | --- | --- | ---: | ---: | --- |")
for forest, k, r in FORESTS:
    for b in BUDGETS:
        for pol in POLS:
            ss = rows[(forest, b, pol)]
            assert len(ss) == 3, (forest, b, pol, len(ss))
            t = [s["completion_s"] for s in ss]
            rc = st.mean(s["recomputed_tokens"] for s in ss) / 1000
            fb = {}
            for s in ss:
                for kk, v in s["fallback_counts"].items():
                    fb[kk] = fb.get(kk, 0) + v
            fbs = "/".join(str(fb.get(kk, 0)) for kk in ("parent", "ancestor", "root", "inside_root", "none"))
            print(f"| {forest} | {b} | {pol} | {st.mean(t):.1f} ({min(t):.1f}-{max(t):.1f}) | {rc:.0f} | {fb.get('none', 0)} | {fbs} |")

print()
for forest, k, r in FORESTS:
    pts = [(s["recomputed_tokens"], s["completion_s"])
           for (fo, b, pol), ss in rows.items() if fo == forest for s in ss]
    n = len(pts)
    mx = st.mean(p[0] for p in pts)
    my = st.mean(p[1] for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    slope = sxy / sxx
    icpt = my - slope * mx
    res = [p[1] - (icpt + slope * p[0]) for p in pts]
    r2 = 1 - sum(e * e for e in res) / sum((p[1] - my) ** 2 for p in pts)
    floor = k * SINGLE[r]
    print(f"{forest}: n={n}, time = {icpt:.1f} s + {slope * 1e6:.1f} s per M recomputed "
          f"({1 / slope / 1000:.1f}k tokens/s), R^2={r2:.3f}; "
          f"floor {floor} tokens -> {icpt + slope * floor:.1f} s; "
          f"max |residual| {max(abs(e) for e in res):.1f} s")
