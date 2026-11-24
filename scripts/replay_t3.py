"""Run one T3 agent-search tree live against an SGLang server (see tree_shapes.py t3).

The tree program is the same for every policy:

  1. Prefill the root once (max_new_tokens=1).
  2. Run the shape's rounds in order. A round's expansions run at once, each in its
     own thread, 5 ms apart. An expansion of parent p into child c sends two
     requests:
       action  prompt = full(p) + TURN; decodes c's action (`act` tokens, ignore_eos).
       tool    only if c is not terminal: prompt = action prompt + action + c's tool
               output (`obs` random token ids from the seed), max_new_tokens=1.
               Its prompt is full(c), and the state at its end is what c's children
               start from.
     TURN is 2 fixed token ids, as a chat template opens the next assistant turn.
     So every child's prompt runs past its parent's state, and the parent's state
     can be reused exactly. The next round starts when every expansion of this one
     has finished.
  3. At each round's end, back up each new child's score along its path.

Policies differ only in which states they hold with radix-native sessions:

  p0       nothing; the engine's LRU decides.
  p0root   the root, until the tree ends.
  p2sel    the root, plus up to k nodes ranked by the order in which UCT would
           expand them next, from the visit counts known now (tree_shapes.uct_order).
           The search program knows this order; nothing from the future is used.
  poracle  the root, plus up to k nodes ranked by when the shape expands them next.
           It uses the future, so it is only a reference.

Mechanism of p2sel and poracle: every tool request carries its node's session, so
the node's state is referenced from the moment it exists. At each round's end the
held nodes are ranked and those outside the top k are released for good (a
released node is not held again). A node is also released when its last allowed
child is admitted. k = budget - 2 - 1 - parallel: the engine leaves 2 slots of the
pool empty, the root takes 1, and each running expansion adds 1 new held node
before the round's ranking.

Each request's prompt and cached tokens are recorded. A tool request has to
prefill its tool output in any case; an action request should recompute only
TURN when its parent's state is cached.

Usage:
  replay_t3.py --shape shapes/t3-s0.json --root-len 4160 --policy p2sel \
      --budget 16 --label t3-s0-b16-p2sel --out runs/t3/x.json
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
from tree_shapes import uct_order

SEND_GAP_S = 0.005
POLICIES = ("p0", "p0root", "p2sel", "poracle")
TURN = [4242, 4243]


def run_t3(base, shape, root_len, policy, seed, tag, budget):
    nodes = shape["nodes"]
    rounds = shape["rounds"]
    max_children = shape["max_children"]
    max_depth = shape["max_depth"]
    rng = random.Random(f"t3-{shape['seed']}-{root_len}-{seed}")
    root = [rng.randrange(1000, 100000) for _ in range(root_len)]

    def obs_ids(nid):
        r = random.Random(f"t3-obs-{shape['seed']}-{seed}-{nid}")
        return [r.randrange(1000, 100000) for _ in range(nodes[nid]["obs"])]

    hold_root = policy != "p0"
    k = max(0, budget - 3 - shape["parallel"]) if policy in ("p2sel", "poracle") else 0
    node_sessions = k > 0

    # Rounds in which the shape expands each node, for the oracle.
    expands_in = defaultdict(list)
    for r, exps in enumerate(rounds):
        for e in exps:
            expands_in[e["parent"]].append(r)

    lock = threading.Lock()
    sid = {}             # node -> session id, for nodes sent with one
    closed = set()
    events = []          # (time, what, node)
    full = {"root": root}
    view = {"root": dict(nodes["root"], children=[])}  # the tree as the program knows it
    admitted = defaultdict(int)  # parent -> children admitted so far
    held = set()         # nodes other than the root whose session is still open
    N, W = defaultdict(int), defaultdict(float)
    records = []
    sel_log = []
    n_open = [0, 0]
    cur_round = [0]

    def close(nid, why):
        with lock:
            if nid not in sid or nid in closed:
                return
            closed.add(nid)
            held.discard(nid)
            n_open[0] -= 1
        post(base, "/close_session", {"session_id": sid[nid]})
        events.append((time.time(), f"close:{why}", nid))

    def child_admitted(parent):
        with lock:
            admitted[parent] += 1
            last = admitted[parent] >= max_children
        if last and parent != "root":
            close(parent, "full")

    def expandable(n):
        x = view[n]
        return (n != "root" and not x["terminal"] and len(x["children"]) < max_children
                and x["depth"] < max_depth)

    def rebalance():
        if not node_sessions:
            return
        with lock:
            if policy == "p2sel":
                pool = [n for n in view if expandable(n)]
                ranking = uct_order(view, N, W, expandable, len(pool), shape["c"])
            else:
                def next_use(n):
                    # cur_round is the next round to run; its expansions count.
                    later = [r for r in expands_in.get(n, []) if r >= cur_round[0]]
                    return later[0] if later else float("inf")
                ranking = sorted(held, key=lambda n: (next_use(n), view[n]["born"]))
                ranking = [n for n in ranking if next_use(n) != float("inf")]
            keep = [n for n in ranking if n in held][:k]
            drop = [n for n in held if n not in keep]
        for n in drop:
            close(n, "outranked")

    def record(nid, kind, parent, prompt, r, outs):
        cached = r["cached_tokens"] or 0
        rec = {"node": nid, "kind": kind, "parent": parent, "depth": nodes[nid]["depth"],
               "round": nodes[nid]["round"], "prompt_len": len(prompt),
               "output_len": len(outs), "recomputed": len(prompt) - cached,
               "root_miss": parent != "root" and cached < root_len - 256, **r}
        with lock:
            records.append(rec)

    def expansion(e, delay):
        time.sleep(delay)
        p, c = e["parent"], e["child"]
        prompt = full[p] + TURN
        r = stream_generate(base, prompt, nodes[c]["act"], None,
                            on_first=lambda: child_admitted(p))
        act = r.pop("output_ids")
        record(c, "action", p, prompt, r, act)
        if nodes[c]["terminal"]:
            with lock:
                view[c] = dict(nodes[c], children=[])
                view[p]["children"].append(c)
            return
        prompt = prompt + act + obs_ids(c)
        s = f"{tag}-{c}" if node_sessions else None
        if s is not None:
            with lock:
                sid[c] = s
        r = stream_generate(base, prompt, 1, s)
        record(c, "tool", p, prompt, r, r.pop("output_ids"))
        with lock:
            full[c] = prompt
            view[c] = dict(nodes[c], children=[])
            view[p]["children"].append(c)
            if s is not None:
                held.add(c)
                n_open[0] += 1
                n_open[1] = max(n_open)

    post(base, "/flush_cache")
    time.sleep(0.5)
    poller = MetricsPoller(base)
    poller.start()
    t_start = time.time()
    leftover = []
    round_s = []
    errors = []
    try:
        rec = stream_generate(base, root, 1, f"{tag}-root" if hold_root else None)
        rec.pop("output_ids")
        if hold_root:
            sid["root"] = f"{tag}-root"
            n_open[0] = n_open[1] = 1
        root_rec = rec
        for r, exps in enumerate(rounds):
            cur_round[0] = r
            t_r = time.time()
            with lock:
                for e in exps:
                    p = e["parent"]
                    sel_log.append({"it": e["it"], "parent": p, "round": r,
                                    "held": (p == "root" and hold_root) or p in held,
                                    "parent_depth": e["parent_depth"],
                                    "parent_age_nodes": e["parent_age_nodes"]})
            ths = []
            for i, e in enumerate(exps):
                def target(e=e, i=i):
                    try:
                        expansion(e, i * SEND_GAP_S)
                    except Exception as ex:  # report after the round
                        errors.append(repr(ex))
                ths.append(threading.Thread(target=target))
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            if errors:
                raise RuntimeError("; ".join(errors))
            with lock:
                for e in exps:
                    n = e["child"]
                    while n is not None:
                        N[n] += 1
                        W[n] += nodes[e["child"]]["score"]
                        n = nodes[n]["parent"]
            round_s.append(round(time.time() - t_r, 3))
            cur_round[0] = r + 1
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
        want = nodes[r["node"]]["act"] if r["kind"] == "action" else 1
        if r["completion_tokens"] != want or r["output_len"] != want:
            problems.append(f"{r['node']} {r['kind']}: decoded {r['completion_tokens']}/{r['output_len']} of {want}")
        if r["prompt_tokens"] != r["prompt_len"]:
            problems.append(f"{r['node']} {r['kind']}: prompt {r['prompt_tokens']} != {r['prompt_len']}")
        if r["num_retractions"]:
            problems.append(f"{r['node']} {r['kind']}: {r['num_retractions']} retractions")
    if len(records) != shape["stats"]["requests"]:
        problems.append(f"{len(records)} requests, shape has {shape['stats']['requests']}")
    if leftover:
        problems.append(f"sessions still open after an error: {leftover}")

    act = [r for r in records if r["kind"] == "action"]
    tool = [r for r in records if r["kind"] == "tool"]
    # An action request whose parent's state was lost recomputes more than TURN plus
    # one 256-token alignment step.
    lost = [r for r in act if r["recomputed"] > len(TURN) + 256]
    summary = {
        "completion_s": round(t_end - t_start, 3),
        "root_prefill_s": round(root_rec["client_recv"] - root_rec["client_send"], 3),
        "round_s": round_s,
        "k_held": k,
        "recomputed_tokens": sum(r["recomputed"] for r in records),
        "recomputed_action": sum(r["recomputed"] for r in act),
        "recomputed_tool": sum(r["recomputed"] for r in tool),
        "tool_tokens": sum(nodes[r["node"]]["obs"] for r in tool),
        "parent_state_lost": len(lost),
        "parent_state_lost_tokens": sum(r["recomputed"] for r in lost),
        "root_misses": sum(1 for r in records if r["root_miss"]),
        "parents_held_when_selected": sum(1 for s in sel_log if s["held"]),
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
    ap.add_argument("--seed", type=int, default=0, help="seed for the root's and tool outputs' token ids")
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
    result.update(run_t3(a.base, shape, a.root_len, a.policy, a.seed, tag, a.budget))
    result["finished"] = time.strftime("%F %T")
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    s = result["summary"]
    print(f"{a.label}: completion {s['completion_s']} s, recomputed {s['recomputed_tokens']} "
          f"(action {s['recomputed_action']}, tool {s['recomputed_tool']} for {s['tool_tokens']} tool tokens), "
          f"parent state lost {s['parent_state_lost']}, root misses {s['root_misses']}, "
          f"held at selection {s['parents_held_when_selected']}/{s['selections']}, k {s['k_held']}, "
          f"open sessions peak {s['open_sessions_peak']}"
          f"{'  PROBLEMS: ' + '; '.join(s['problems']) if s['problems'] else ''}", flush=True)
    return 1 if s["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
