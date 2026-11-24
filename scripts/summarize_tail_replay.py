"""RL-side view of tail_replay_drift.py results.

The rollout engine samples with the rebuilt state and reports its own log-probs ("dec").
The trainer recomputes them from the exact prefix (ref_lp). The per-token ratio
r = pi_trainer / pi_rollout = exp(ref_lp - dec_lp) is what truncated importance sampling
weights by, and what lands outside PPO's clip range before any policy update.

In files from --rollout, each condition sampled its own tokens from its own state, and r is
train_lp / roll_lp on those tokens.

Usage: python scripts/summarize_tail_replay.py runs/tail-replay/r*.json
"""
import json
import math
import statistics as st
import sys

EPS = 0.2  # PPO clip range


def token_stats(ref, lp):
    lr = [a - b for a, b in zip(ref, lp)]  # log r per token
    n = len(lr)
    return {
        "abs": sum(abs(x) for x in lr) / n,
        "out_clip": sum(1 for x in lr if not (math.log(1 - EPS) <= x <= math.log(1 + EPS))) / n,
        "r_gt2": sum(1 for x in lr if x > math.log(2)) / n,
        "r_lt_half": sum(1 for x in lr if x < -math.log(2)) / n,
        "seq_abs": abs(sum(lr)),  # |log| of the whole-sequence ratio
    }


def main(paths):
    pooled = []
    for p in paths:
        d = json.load(open(p))
        L, roots = d["meta"]["root_len"], d["roots"]
        mode = d["meta"].get("mode", "same tokens")
        print(f"\n### {L} tokens, {len(roots)} roots, {d['meta']['seg']} sampled tokens each, {mode}\n")
        print("| Condition | k | mean abs log r | min–max over roots | outside [0.8, 1.2] | r > 2 | r < 0.5 "
              "| median abs log of sequence ratio | KL |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        def lps(r, name):
            c = r["conds"][name]["dec"]
            return (c["train_lp"], c["roll_lp"]) if "roll_lp" in c else (r["ref_lp"], c["lp"])
        for name in roots[0]["conds"]:
            if name == "r100":
                continue
            per = [token_stats(*lps(r, name)) for r in roots]
            kl = sum(r["conds"][name]["dec"]["kl_mean"] for r in roots) / len(roots)
            k = sum(r["conds"][name]["k"] for r in roots) / len(roots)
            a = [s["abs"] for s in per]
            m = lambda key: sum(s[key] for s in per) / len(per)
            label = "exact (no rebuild)" if name == "exact" else name
            print(f"| {label} | {0 if name == 'exact' else k:.0f} | {m('abs'):.3f} | {min(a):.3f}–{max(a):.3f} "
                  f"| {m('out_clip'):.3f} | {m('r_gt2'):.4f} | {m('r_lt_half'):.4f} "
                  f"| {st.median(s['seq_abs'] for s in per):.2f} | {kl:.4f} |")
            if name != "exact":
                pooled.append((k, L, name, m("abs")))
    print("\n### Error against replayed tokens, both lengths\n")
    print("| k (tokens) | root | condition | mean abs log r |")
    print("| ---: | ---: | --- | ---: |")
    for k, L, name, a in sorted(pooled):
        print(f"| {k:.0f} | {L} | {name} | {a:.3f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
