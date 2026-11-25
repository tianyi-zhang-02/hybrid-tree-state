"""Concurrent prefill or eviction? Split from SGLang's server-side times.

Needs runs made with --enable-metrics after 2026-11-25, whose records carry
forward_entry_time and prefill_finished_time. In SGLang 0.5.23:
  - forward_entry_time is set when a request's first forward batch is formed,
    right after its prefix match;
  - prefill_finished_time is set just before the finished prefill's state is
    put into the cache tree.

In each prompt group, the sample whose prefill finished first computed the
prompt ("first"). Every other missed sample (recomputed > 512 tokens) is:
  before  entered a forward batch before that first prefill finished, so it
          was matched before the full prompt's state existed (concurrent
          prefill);
  after   entered after it, so the state had been cached and was gone
          (eviction).

Two side counts:
  - partial: misses that still reused part of the prompt (cached_tokens > 0),
    usually a chunk of a sibling's prefill that was still running;
  - early hits: samples that entered "before" but recomputed at most 512
    tokens. These are partial matches on an earlier chunk with a short
    remainder, not full hits.

  python3 scripts/miss_timing_exact.py runs/verl-timing-fcfs/g128-s0-p0.json ...
"""
import json
import sys
from collections import defaultdict

MISS_TOKENS = 512

print(f"{'run':<42} {'misses':>7} {'first':>6} {'before':>7} {'after':>6} "
      f"{'partial':>8} {'early hits':>10}")
for path in sys.argv[1:]:
    with open(path) as f:
        run = json.load(f)
    groups = defaultdict(list)
    for r in run["records"]:
        if r["kind"] == "sample":
            groups[r["prompt"]].append(r)
    misses = first = before = after = partial = early = 0
    for rs in groups.values():
        if not all(r.get("forward_entry_time") and r.get("prefill_finished_time") for r in rs):
            sys.exit(f"{path}: records lack server times (need --enable-metrics and the "
                     f"2026-11-25 replay_tree.py)")
        f0 = min(rs, key=lambda r: r["prefill_finished_time"])
        t_done = f0["prefill_finished_time"]
        first += f0["recomputed"] > MISS_TOKENS
        for r in rs:
            missed = r["recomputed"] > MISS_TOKENS
            misses += missed
            partial += missed and (r["cached_tokens"] or 0) > 0
            if r is f0:
                continue
            entered_before = r["forward_entry_time"] < t_done
            if missed:
                before += entered_before
                after += not entered_before
            else:
                early += entered_before
    print(f"{path:<42} {misses:>7} {first:>6} {before:>7} {after:>6} {partial:>8} {early:>10}")
