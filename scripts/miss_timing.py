"""When did each missed sample start, relative to its group's first token?

For every GRPO run (replay_grpo.py output) given, counts the misses (samples
that recomputed more than 512 prompt tokens) in three groups:

  first   the first sample of its group to return a token;
  before  a sibling estimated as scheduled before that first token;
  after   a sibling estimated as scheduled after it, so the prompt's state
          had been cached and was lost (eviction).

A sample's schedule time is estimated as its arrival (client first-token time
minus the server's first_token_latency) plus the server's queue_time. The
estimate leaves out the tokenizer-to-scheduler step and the delay between the
first sample's state entering the cache and the client seeing its token, so it
runs early by up to about 1-2 s: most hits land in "before" too. Only "after"
is reliable. SGLang's forward_entry_time and prefill_finished_time (meta_info
with --enable-metrics) would give exact times; replay_tree.stream_generate
does not keep them yet.

  python3 scripts/miss_timing.py runs/verl-tp2/g128-s0-p0.json ...
"""
import json
import sys
from collections import defaultdict

MISS_TOKENS = 512

print(f"{'run':<40} {'first':>6} {'before':>7} {'after':>6}")
for path in sys.argv[1:]:
    with open(path) as f:
        run = json.load(f)
    groups = defaultdict(list)
    for r in run["records"]:
        if r["kind"] == "sample":
            groups[r["prompt"]].append(r)
    first = before = after = 0
    for rs in groups.values():
        rs.sort(key=lambda r: r["client_first"])
        t0 = rs[0]["client_first"]
        first += rs[0]["recomputed"] > MISS_TOKENS
        for r in rs[1:]:
            if r["recomputed"] <= MISS_TOKENS:
                continue
            scheduled = r["client_first"] - r["first_token_latency"] + (r["queue_time"] or 0)
            if scheduled < t0:
                before += 1
            else:
                after += 1
    print(f"{path:<40} {first:>6} {before:>7} {after:>6}")
