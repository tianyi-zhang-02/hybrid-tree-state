"""Run several trees at once against one SGLang server, as a rollout batch would.

Each tree has its own root (its own prompt) and runs the same program as in
replay_tree.py, under the same policy. All trees start together after one
/flush_cache and share the server's state pool and running-request limit, so a
tree whose requests wait in the queue can lose its states to the others.

Tree i uses shape t1-s{(seed + i) % 3} and root token ids seeded by (seed, i), so a
run is reproducible and the trees differ from each other.

The numbers that matter are the time until the last tree finishes and the
recomputed tokens summed over all trees. Per-tree results are kept as well.

Usage:
  replay_forest.py --trees 4 --root-len 8000 --policy p1 --seed 0 \
      --label f4-r8000-b8-p1-s0 --out runs/forest/x.json
"""

import argparse
import json
import sys
import threading
import time

import requests

from replay_tree import MetricsPoller, post, run_tree

SHAPES = "shapes/t1-s{}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--trees", type=int, required=True)
    ap.add_argument("--root-len", type=int, required=True)
    ap.add_argument("--policy", choices=["p0", "p1", "p0root", "p1root"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shapes", default=SHAPES, help="pattern with {} for the shape seed")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    info = requests.get(a.base + "/get_server_info", timeout=60).json()
    keep = ("version", "context_length", "max_total_num_tokens", "max_running_requests",
            "max_mamba_cache_size", "mamba_radix_cache_strategy", "mamba_track_interval",
            "chunked_prefill_size", "schedule_policy", "disable_overlap_schedule",
            "enable_session_radix_cache", "page_size")
    shapes = [json.load(open(a.shapes.format((a.seed + i) % 3))) for i in range(a.trees)]

    post(a.base, "/flush_cache")
    time.sleep(0.5)
    poller = MetricsPoller(a.base)
    poller.start()
    stamp = int(time.time())
    results = [None] * a.trees
    errors = [None] * a.trees

    def one(i):
        try:
            results[i] = run_tree(a.base, shapes[i], a.root_len, a.policy,
                                  seed=1000 * a.seed + i, tag=f"{a.label}-{stamp}-t{i}",
                                  flush=False, poller=poller)
        except Exception as e:  # keep the other trees running; report at the end
            errors[i] = repr(e)

    t0 = time.time()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(a.trees)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.time()
    poller.stop_flag.set()
    poller.join()

    trees = []
    problems = []
    for i, r in enumerate(results):
        if r is None:
            problems.append(f"tree {i} failed: {errors[i]}")
            continue
        s = r["summary"]
        problems += [f"tree {i}: {p}" for p in s["problems"]]
        trees.append({"tree": i, "shape_seed": shapes[i]["seed"], **s})
    fb = {}
    for t in trees:
        for k, v in t["fallback_counts"].items():
            fb[k] = fb.get(k, 0) + v
    peaks = poller.peaks()
    summary = {
        "trees": a.trees,
        "completion_s": round(t1 - t0, 3),
        "tree_completion_s": [t["completion_s"] for t in trees],
        "recomputed_tokens": sum(t["recomputed_tokens"] for t in trees),
        "fallback_counts": fb,
        "metric_peaks": peaks,
        "kv_peak_usage": peaks.get("sglang:token_usage"),
        "problems": problems,
    }
    out = {"label": a.label, "args": vars(a), "server_info": {k: info.get(k) for k in keep},
           "summary": summary, "trees": trees,
           "per_tree": [{"records": r["records"], "events": r["events"]} if r else None
                        for r in results],
           "metrics": poller.samples}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{a.label}: completion {summary['completion_s']} s "
          f"(trees {summary['tree_completion_s']}), recomputed {summary['recomputed_tokens']}, "
          f"fallback {fb}, kv peak {summary['kv_peak_usage']}"
          f"{'  PROBLEMS: ' + '; '.join(problems) if problems else ''}", flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
