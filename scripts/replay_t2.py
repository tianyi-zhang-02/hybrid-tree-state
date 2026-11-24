"""Run one T2 search tree live against an SGLang server (see tree_shapes.py t2).

The tree program is the same for every policy:

  1. Prefill the shared root once (max_new_tokens=1).
  2. Run the shape's rounds in order. A round's simulations run at once, each in
     its own thread. A simulation is a chain of steps: send the step's 2 children
     together (5 ms apart, streamed, `len` tokens each with ignore_eos), wait for
     both, move to the child the shape follows, repeat until the terminal step.
     A child's prompt is the root plus the outputs its ancestors produced in this
     run. The next round starts when every simulation of this one has finished.
  3. At each round's end, back up the round's values into the visit counts.

Policies differ only in which states they hold with radix-native sessions:

  p0       nothing; the engine's LRU decides.
  p0root   the root, until the tree ends.
  p2sel    the root, plus up to k candidates: generated, non-terminal, childless
           nodes not yet selected. They are ranked by the order in which the next
           round's WU-PUCT selection would pick them, from the visit counts known
           now, and the top k that are still held stay held.
  poracle  the root, plus up to k candidates that the shape will select later,
           soonest first. It uses the future, so it is only a reference.

Mechanism of p2sel and poracle: every child is sent with its own session, so its
state is referenced when it finishes. After each step the program decides: a
terminal child is closed; the followed child stays held until its own children
are admitted; the other child becomes a candidate, and candidates outside the top
k are closed for good (a closed node is not held again). A node's session closes
when its last child returns its first token, as in replay_tree.py.

k = budget - 2 - 1 - 2 * parallel: the engine leaves 2 slots of the pool empty,
the root takes 1, and each running simulation may hold 2 states of its own. When
k is 0, children carry no sessions and the policy is p0root.

Each request's prompt and cached tokens are recorded. A full miss is a request
below the root's children whose cached prefix ends more than 256 tokens before
the root's end, so the root's own state was not reused.

Usage:
  replay_t2.py --shape shapes/t2-s0.json --root-len 8000 --policy p2sel \
      --budget 16 --label t2-s0-8k-b16-p2sel --out runs/t2/x.json
"""

import argparse
import json
import random
import sys
import threading
import time
from collections import defaultdict

import requests

from replay_tree import MetricsPoller, post, stream_generate
from tree_shapes import selection_order

SEND_GAP_S = 0.005
POLICIES = ("p0", "p0root", "p2sel", "poracle")


def run_t2(base, shape, root_len, policy, seed, tag, budget):
    nodes = shape["nodes"]
    rounds = shape["rounds"]
    rng = random.Random(f"t2-{shape['seed']}-{root_len}-{seed}")
    root = [rng.randrange(1000, 100000) for _ in range(root_len)]

    hold_root = policy != "p0"
    k = max(0, budget - 3 - 2 * shape["parallel"]) if policy in ("p2sel", "poracle") else 0
    child_sessions = k > 0

    # When each node is selected by the shape: (round, order within the round).
    when = {}
    for r, sims in enumerate(rounds):
        for i, s in enumerate(sims):
            when[s["leaf"]] = (r, i)

    lock = threading.Lock()
    sid = {}             # node -> session id, for nodes sent with one
    closed = set()
    events = []          # (time, what, node)
    path = {"root": []}  # tokens after the root, per generated node
    generated = {"root"}
    known_children = defaultdict(list)  # children whose requests have finished
    candidates = set()   # generated, non-terminal, childless, not yet selected
    held = set()         # candidates whose session is still open
    selected = set()
    N, W = defaultdict(int), defaultdict(float)
    pending = {}         # parent -> children not yet admitted
    records = []
    sel_log = []         # per simulation: was its leaf held when selected?
    n_open = [0, 0]      # sessions open now, and the peak

    def session(nid):
        if nid == "root":
            return f"{tag}-root" if hold_root else None
        return f"{tag}-{nid}" if child_sessions else None

    def close(nid, why):
        with lock:
            if nid not in sid or nid in closed:
                return
            closed.add(nid)
            held.discard(nid)
            n_open[0] -= 1
        post(base, "/close_session", {"session_id": sid[nid]})
        events.append((time.time(), f"close:{why}", nid))

    def child_started(parent):
        with lock:
            pending[parent] -= 1
            last = pending[parent] == 0
        if last and parent != "root":
            close(parent, "children_admitted")

    def rebalance():
        if not child_sessions:
            return
        with lock:
            if policy == "p2sel":
                def eligible(n):
                    return n in candidates
                ranking = selection_order(nodes, lambda n: known_children[n], N, W,
                                          eligible, len(candidates), shape["c"])
            else:
                ranking = sorted((n for n in candidates if n in when), key=when.get)
            keep = [n for n in ranking if n in held][:k]
            drop = [n for n in held if n not in keep]
        for n in drop:
            close(n, "outranked")

    def step(sim, st, si):
        parent = st["parent"]
        kids = st["children"]
        with lock:
            pending[parent] = len(kids)
        out = [None] * len(kids)

        def one(i):
            nid = kids[i]
            time.sleep(i * SEND_GAP_S)
            prompt = root + path[parent]
            s = session(nid)
            if s is not None:
                with lock:
                    sid[nid] = s
            r = stream_generate(base, prompt, nodes[nid]["len"], s,
                                on_first=lambda: child_started(parent))
            out[i] = (nid, prompt, r)

        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(kids))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if any(o is None for o in out):
            raise RuntimeError(f"step {sim['sim']}.{si}: a child request failed")
        for nid, prompt, r in out:
            outs = r.pop("output_ids")
            cached = r["cached_tokens"] or 0
            records.append({
                "node": nid, "parent": parent, "depth": nodes[nid]["depth"],
                "round": nodes[nid]["round"], "sim": sim["sim"], "step": si,
                "revisit": si == 0 and parent != "root",
                "prompt_len": len(prompt), "output_len": len(outs),
                "recomputed": len(prompt) - cached,
                "full_miss": parent != "root" and cached < root_len - 256,
                **r})
            with lock:
                if nid in sid:
                    n_open[0] += 1
                    n_open[1] = max(n_open)
                path[nid] = path[parent] + outs
                generated.add(nid)
                known_children[parent].append(nid)
        # Decide what to hold among this step's children.
        to_close = []
        with lock:
            for nid in kids:
                if nodes[nid]["terminal"]:
                    to_close.append((nid, "terminal"))
                elif nid != st["follow"]:
                    candidates.add(nid)
                    if nid in sid and nid not in closed:
                        held.add(nid)
        for nid, why in to_close:
            close(nid, why)
        rebalance()

    def simulation(sim, delay):
        time.sleep(delay)
        for si, st in enumerate(sim["steps"]):
            step(sim, st, si)

    post(base, "/flush_cache")
    time.sleep(0.5)
    poller = MetricsPoller(base)
    poller.start()
    t_start = time.time()
    leftover = []
    round_s = []
    errors = []
    try:
        rec = stream_generate(base, root, 1, session("root"))
        rec.pop("output_ids")
        if hold_root:
            sid["root"] = session("root")
            n_open[0] = n_open[1] = 1
        root_rec = rec
        for r, sims in enumerate(rounds):
            t_r = time.time()
            with lock:
                for s in sims:
                    leaf = s["leaf"]
                    sel_log.append({"sim": s["sim"], "leaf": leaf, "round": r,
                                    "held": leaf in held, "leaf_depth": s["leaf_depth"]})
                    selected.add(leaf)
                    candidates.discard(leaf)
                    held.discard(leaf)  # its session closes when its children are admitted
            ths = []
            for i, s in enumerate(sims):
                def target(s=s, i=i):
                    try:
                        simulation(s, i * SEND_GAP_S)
                    except Exception as e:  # report after the round
                        errors.append(repr(e))
                ths.append(threading.Thread(target=target))
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            if errors:
                raise RuntimeError("; ".join(errors))
            with lock:
                for s in sims:
                    n = s["terminal"]
                    while n is not None:
                        N[n] += 1
                        W[n] += s["value"]
                        n = nodes[n]["parent"]
            round_s.append(round(time.time() - t_r, 3))
            rebalance()
        t_end = time.time()
        for n in [n for n in sid if n not in closed]:
            close(n, "tree_done")
    finally:
        leftover = [n for n in sid if n not in closed]
        for n in leftover:
            try:
                close(n, "end")
            except requests.RequestException:
                pass
        poller.stop_flag.set()
        poller.join()

    problems = []
    for r in records:
        want = nodes[r["node"]]["len"]
        if r["completion_tokens"] != want or r["output_len"] != want:
            problems.append(f"{r['node']}: decoded {r['completion_tokens']}/{r['output_len']} of {want}")
        if r["prompt_tokens"] != r["prompt_len"]:
            problems.append(f"{r['node']}: prompt {r['prompt_tokens']} != {r['prompt_len']}")
        if r["num_retractions"]:
            problems.append(f"{r['node']}: {r['num_retractions']} retractions")
    if len(records) != shape["stats"]["requests"]:
        problems.append(f"{len(records)} requests, shape has {shape['stats']['requests']}")
    if leftover:
        problems.append(f"sessions still open after an error: {leftover}")

    rev = [r for r in records if r["revisit"]]
    summary = {
        "completion_s": round(t_end - t_start, 3),
        "root_prefill_s": round(root_rec["client_recv"] - root_rec["client_send"], 3),
        "round_s": round_s,
        "k_candidates": k,
        "recomputed_tokens": sum(r["recomputed"] for r in records),
        "recomputed_revisit": sum(r["recomputed"] for r in rev),
        "recomputed_chain": sum(r["recomputed"] for r in records if not r["revisit"]),
        "full_misses": sum(1 for r in records if r["full_miss"]),
        "full_misses_revisit": sum(1 for r in rev if r["full_miss"]),
        "leaves_held_when_selected": sum(1 for s in sel_log if s["held"]),
        "selections": len(sel_log),
        "open_sessions_peak": n_open[1],
        "t_start": t_start,
        "t_end": t_end,
        "metric_peaks": poller.peaks(),
        "problems": problems,
    }
    return {"summary": summary, "root": root_rec, "records": records,
            "selections": sel_log, "events": events, "metrics": poller.samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--shape", required=True)
    ap.add_argument("--root-len", type=int, required=True)
    ap.add_argument("--policy", choices=POLICIES, required=True)
    ap.add_argument("--budget", type=int, required=True,
                    help="target cached states the server was sized for (8, 16 or 31)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the root's token ids")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    shape = json.load(open(a.shape))
    info = requests.get(a.base + "/get_server_info", timeout=60).json()
    keep = ("version", "context_length", "max_total_num_tokens", "max_running_requests",
            "max_mamba_cache_size", "mamba_radix_cache_strategy", "mamba_track_interval",
            "chunked_prefill_size", "schedule_policy", "disable_overlap_schedule",
            "enable_session_radix_cache", "page_size")
    tag = f"{a.label}-{int(time.time())}"
    result = {"label": a.label, "args": vars(a), "shape_stats": shape["stats"],
              "server_info": {k: info.get(k) for k in keep},
              "started": time.strftime("%F %T")}
    result.update(run_t2(a.base, shape, a.root_len, a.policy, a.seed, tag, a.budget))
    result["finished"] = time.strftime("%F %T")
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    s = result["summary"]
    print(f"{a.label}: completion {s['completion_s']} s, recomputed {s['recomputed_tokens']} "
          f"(revisit {s['recomputed_revisit']}), full misses {s['full_misses']}, "
          f"held at selection {s['leaves_held_when_selected']}/{s['selections']}, k {s['k_candidates']}, "
          f"open sessions peak {s['open_sessions_peak']}"
          f"{'  PROBLEMS: ' + '; '.join(s['problems']) if s['problems'] else ''}", flush=True)
    return 1 if s["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
