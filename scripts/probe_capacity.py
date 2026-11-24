"""How many cached Mamba states does the pool really hold at this server setting?

After /flush_cache, sends `n` unrelated prefill-only requests one at a time (prompt
576 tokens, so each leaves one state), then reads the sglang:mamba_* gauges and
re-queries the newest requests to see how many of them still hit.

With nothing running, every slot can hold a cached state, so the count of cached
states should reach --max-mamba-cache-size. During a tree run, running requests take
their slots out of the same pool; their per-request count comes from the decode log
("mamba num").
"""

import argparse
import json
import random

import requests


def gen(base, ids):
    body = {"input_ids": ids,
            "sampling_params": {"max_new_tokens": 1, "ignore_eos": True}}
    r = requests.post(base + "/generate", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["meta_info"]["cached_tokens"]


def gauges(base):
    out = {}
    for ln in requests.get(base + "/metrics", timeout=10).text.splitlines():
        if ln.startswith("sglang:mamba") or ln.startswith("sglang:num_running_reqs"):
            name, _, value = ln.rpartition(" ")
            out[name.split("{")[0]] = float(value)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--check", type=int, default=16, help="newest requests to re-query")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    info = requests.get(a.base + "/get_server_info", timeout=60).json()
    rng = random.Random(7)
    requests.post(a.base + "/flush_cache", timeout=60).raise_for_status()
    before = gauges(a.base)
    prompts = [[rng.randrange(1000, 100000) for _ in range(576)] for _ in range(a.n)]
    for p in prompts:
        gen(a.base, p)
    after = gauges(a.base)
    # Newest first, so a re-query's own state cannot push out one still to be checked.
    hits = sum(gen(a.base, p + [5]) >= 512 for p in reversed(prompts[-a.check:]))
    res = {"max_mamba_cache_size": info.get("max_mamba_cache_size"),
           "strategy": info.get("mamba_radix_cache_strategy"),
           "states_sent": a.n, "gauges_before": before, "gauges_after": after,
           "newest_checked": a.check, "newest_still_cached": hits}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
