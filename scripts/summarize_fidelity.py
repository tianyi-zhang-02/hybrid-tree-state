"""Summarize runs/fidelity (scripts/cache_fidelity.py) into two tables.

Part A (sampled): per condition, the decode-vs-recompute gap of each generation
(mean |diff|, mean exp|diff|, max |diff|), averaged over roots; plus each warm
condition minus the cold one on the same prompt, paired by root.

Part B (teacher-forced): per kind of cached state, the logprob difference of a
fixed continuation against a cold recompute; and the chunk-size control.

Usage: summarize_fidelity.py [runs/fidelity]
"""

import json
import os
import statistics as st
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "runs/fidelity"
COND = ["p1_cold", "p1_cold_again", "p1_warm_chunk", "p2_cold", "p2_cold_again", "p2_warm_end", "p2_warm_decode"]
PAIR = {"p1_cold_again": "p1_cold", "p1_warm_chunk": "p1_cold", "p2_cold_again": "p2_cold",
        "p2_warm_end": "p2_cold", "p2_warm_decode": "p2_cold"}

for L in (8000, 32064):
    f = f"{d}/r{L}.json"
    if not os.path.exists(f):
        continue
    x = json.load(open(f))
    rs = x["results"]
    print(f"\n## Root {L}, {len(rs)} roots, server {x['server_info']}")
    print("\nPart A, sampled: decode logprob vs recompute of the same tokens")
    print("| Condition | Cached | Mean abs | Mean exp abs | Max abs | Minus cold, mean exp abs (min..max over roots) |")
    print("| --- | ---: | ---: | ---: | ---: | --- |")
    for c in COND:
        if c not in rs[0]["sampled"]:
            continue
        g = [r["sampled"][c] for r in rs]
        cached = sorted({v["cached"] for v in g})
        ma = st.mean(v["gen_vs_ref"]["mean_abs"] for v in g)
        me = st.mean(v["gen_vs_ref"]["mult_err"] for v in g)
        mx = max(v["gen_vs_ref"]["max_abs"] for v in g)
        if c in PAIR:
            dd = [r["sampled"][c]["gen_vs_ref"]["mult_err"] - r["sampled"][PAIR[c]]["gen_vs_ref"]["mult_err"]
                  for r in rs]
            pair = f"{st.mean(dd):+.4f} ({min(dd):+.4f}..{max(dd):+.4f})"
        else:
            pair = ""
        print(f"| {c} | {cached} | {ma:.4f} | {me:.4f} | {mx:.3f} | {pair} |")
    print("\nPart B, teacher-forced: the same continuation from a cached state vs a cold recompute")
    print("| Kind | Cached | Mean abs | Max abs | First 32 tokens | Last 32 tokens | Roots with any difference |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for k in ("noise_p1", "noise_p2", "chunk", "end", "decode"):
        b = [r["teacher_forced"][k] for r in rs]
        cached = sorted({v["cached"] for v in b})
        nz = sum(1 for v in b if v["max_abs"] > 0)
        print(f"| {k} | {cached} | {st.mean(v['mean_abs'] for v in b):.4f} | "
              f"{max(v['max_abs'] for v in b):.3f} | {st.mean(v['first32_mean'] for v in b):.4f} | "
              f"{st.mean(v['last32_mean'] for v in b):.4f} | {nz}/{len(b)} |")
    fc = f"{d}/control-r{L}.json"
    if os.path.exists(fc):
        c = json.load(open(fc))
        print(f"\nControl, chunked prefill {c['server_info']['chunked_prefill_size']} vs "
              f"{x['server_info']['chunked_prefill_size']}, cold recompute of the same inputs")
        for k in ("p1", "p2", "decode"):
            v = [r["chunking"][k] for r in c["results"]]
            print(f"  {k}: mean abs {st.mean(z['mean_abs'] for z in v):.5f}, max abs "
                  f"{max(z['max_abs'] for z in v):.4f}, roots with any difference "
                  f"{sum(1 for z in v if z['max_abs'] > 0)}/{len(v)}")
