"""Run one tree shape live against an SGLang server, under policy P0 or P1.

The tree program is the same for both policies:

  1. Prefill the shared root once (max_new_tokens=1).
  2. Expand one level at a time. Every node of the level is sent at once (in node
     order, 5 ms apart), streams its output, and decodes exactly `seg` tokens with
     ignore_eos. A child's prompt is the root plus the outputs its ancestors produced
     in this run.
  3. When the level has finished, prune it to the nodes the shape keeps.

P0 sends nothing else: the engine decides which states to keep.

P1 ("keep pending") uses one radix-native session per node, which references the
state its request ends on. It comes in two forms that differ in when a node is held:

  p1      Hold every node that may be expanded. The root prefill and every node
          below the last level carry their own session id, so each is held from the
          moment it finishes. A pruned node's session is closed as soon as its level
          is pruned.
  p1late  Hold only nodes that will be expanded. Nodes are sent without sessions.
          After a level is pruned, each kept node gets a 1-token request over its own
          path that carries its session id; that request ends on the node's state and
          so references it. The root prefill carries a session in both forms.

p1root is p1 with one change: the root's session stays open until the whole tree
has finished, so the most expensive state to recompute is held throughout.
p0root is P0 plus that one rule: only the root carries a session, held until the
tree has finished; nothing else is held.

In all forms a node's session is closed when its last child returns its first
token, because a child protects the prefix it matched only once the scheduler
admits it. Under a tight budget p1 holds more states than fit (a level's new nodes
plus the parents still waiting for children), and eviction among held states is
plain LRU, so it can lose the parents it needs next; p1late avoids that at the cost
of one short request per kept node.

Each request's prompt and cached tokens are recorded, and for each child the
deepest ancestor whose state it actually reused. /metrics is polled for Mamba pool
occupancy.

Usage:
  replay_tree.py --shape shapes/t1-s0.json --root-len 8000 --policy p1 \
      --label t1-s0-8k-p1 --out runs/trees/x.json
"""

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

SEND_GAP_S = 0.005
METRIC_KEYS = ("mamba", "token_usage", "num_running_reqs", "num_queue_reqs",
               "cache_hit_rate")


def post(base, path, body=None, timeout=60):
    r = requests.post(base + path, json=body or {}, timeout=timeout)
    r.raise_for_status()
    return r


def stream_generate(base, ids, max_new, session_id=None, on_first=None):
    """One streamed /generate call. Returns timings and the final meta_info."""
    body = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": max_new, "ignore_eos": True,
                            "temperature": 1.0},
        "stream": True,
    }
    if session_id is not None:
        body["session_id"] = session_id
    t_send = time.time()
    t_first = None
    last = None
    with requests.post(base + "/generate", json=body, stream=True, timeout=1800) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            obj = json.loads(payload)
            if "error" in obj:
                raise RuntimeError(f"server error: {obj['error']}")
            if t_first is None and obj.get("output_ids"):
                t_first = time.time()
                if on_first is not None:
                    on_first()
            last = obj
    t_recv = time.time()
    if last is None:
        raise RuntimeError("empty stream")
    meta = last.get("meta_info", {})
    return {
        "output_ids": list(last["output_ids"]),
        "prompt_tokens": meta.get("prompt_tokens"),
        "cached_tokens": meta.get("cached_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "finish_reason": (meta.get("finish_reason") or {}).get("type")
        if isinstance(meta.get("finish_reason"), dict) else meta.get("finish_reason"),
        "e2e_latency": meta.get("e2e_latency"),
        "first_token_latency": meta.get("first_token_latency"),
        "queue_time": meta.get("queue_time"),
        "num_retractions": meta.get("num_retractions"),
        # Server-side times (wall clock), present with --enable-metrics: when the
        # request first entered a forward batch, and when its prefill finished.
        "forward_entry_time": meta.get("forward_entry_time"),
        "prefill_finished_time": meta.get("prefill_finished_time"),
        "client_send": t_send,
        "client_first": t_first,
        "client_recv": t_recv,
    }


class MetricsPoller(threading.Thread):
    def __init__(self, base, interval=0.5):
        super().__init__(daemon=True)
        self.base, self.interval = base, interval
        self.samples = []
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            try:
                text = requests.get(self.base + "/metrics", timeout=5).text
                vals = {}
                for ln in text.splitlines():
                    if ln.startswith("#") or not any(k in ln for k in METRIC_KEYS):
                        continue
                    name, _, value = ln.rpartition(" ")
                    name = name.split("{")[0]
                    try:
                        vals[name] = float(value)
                    except ValueError:
                        pass
                self.samples.append((time.time(), vals))
            except requests.RequestException:
                pass
            self.stop_flag.wait(self.interval)

    def peaks(self):
        out = {}
        for _, vals in self.samples:
            for k, v in vals.items():
                out[k] = max(out.get(k, v), v)
        return out


def fallback(shape, nid, cached, root_len):
    """Which ancestor's state did this child reuse?

    Returns (kind, levels_up, fork_at):
      kind       parent | ancestor | root | inside_root | none
      levels_up  1 for the parent, 2 for the grandparent, ...; None if no node's
                 state was reused
      fork_at    the token position where this child forks from its parent
    A reused state sits less than one segment before its node's end.
    """
    seg = shape["seg"]
    nodes = shape["nodes"]
    parent = nodes[nid]["parent"]
    fork_at = root_len + (0 if parent == "root" else nodes[parent]["level"]) * seg
    k, a = 1, parent
    while True:
        end = root_len + (0 if a == "root" else nodes[a]["level"]) * seg
        if cached >= end - seg:
            kind = "parent" if k == 1 else ("root" if a == "root" else "ancestor")
            return kind, k, fork_at
        if a == "root":
            return ("inside_root" if cached > 0 else "none"), None, fork_at
        a = nodes[a]["parent"]
        k += 1


def run_tree(base, shape, root_len, policy, seed, tag, flush=True, poller=None):
    """Run one tree. With flush=False and a shared poller, several trees can run
    at once against one server (see replay_forest.py)."""
    nodes = shape["nodes"]
    seg = shape["seg"]
    depth = shape["depth"]
    rng = random.Random(f"{shape['seed']}-{root_len}-{seed}")
    root = [rng.randrange(1000, 100000) for _ in range(root_len)]

    use_sessions = policy in ("p1", "p1late", "p1root", "p0root")
    hold_root = policy in ("p1root", "p0root")
    sid = {}       # node id -> session id, for nodes that hold one
    closed = set()
    lock = threading.Lock()
    events = []    # (time, what, node)

    def session_for(nid, pin=False):
        if not use_sessions:
            return None
        if nid != "root" and nodes[nid]["level"] >= depth:
            return None  # leaves are never expanded
        if policy == "p1late" and nid != "root" and not pin:
            return None  # held only once it is kept
        if policy == "p0root" and nid != "root":
            return None  # only the root is held
        sid[nid] = f"{tag}-{nid}"
        return sid[nid]

    def close(nid, why):
        with lock:
            if nid not in sid or nid in closed:
                return
            closed.add(nid)
        post(base, "/close_session", {"session_id": sid[nid]})
        events.append((time.time(), f"close:{why}", nid))

    # Children still to be admitted, per parent. Root's children are level 1.
    pending = {"root": len(shape["root_children"])}
    for nid, n in nodes.items():
        if n["kept"]:
            pending[nid] = len(n["children"])

    def child_started(parent):
        with lock:
            pending[parent] -= 1
            last = pending[parent] == 0
        if last and not (parent == "root" and hold_root):
            close(parent, "children_admitted")

    path = {"root": []}  # tokens after the root, per node
    records = {}
    pins = {}      # p1late: the 1-token request that holds each kept node

    if flush:
        post(base, "/flush_cache")
        time.sleep(0.5)
    own_poller = poller is None
    if own_poller:
        poller = MetricsPoller(base)
        poller.start()
    t_start = time.time()

    # Whatever happens, close every session this run opened, so no reference
    # survives into the next run.
    leftover = []
    try:
        rec = stream_generate(base, root, 1, session_for("root"))
        rec.pop("output_ids")
        records["root"] = {"node": "root", "level": 0, **rec}

        for level, ids in enumerate(shape["levels"], 1):
            def one(i_nid):
                i, nid = i_nid
                time.sleep(i * SEND_GAP_S)
                parent = nodes[nid]["parent"]
                prompt = root + path[parent]
                r = stream_generate(base, prompt, seg, session_for(nid),
                                    on_first=lambda p=parent: child_started(p))
                return nid, prompt, r

            with ThreadPoolExecutor(max_workers=len(ids)) as ex:
                results = list(ex.map(one, enumerate(ids)))
            for nid, prompt, r in results:
                parent = nodes[nid]["parent"]
                outs = r.pop("output_ids")
                path[nid] = path[parent] + outs
                kind, up, fork_at = fallback(shape, nid, r["cached_tokens"] or 0, root_len)
                records[nid] = {
                    "node": nid, "level": level, "parent": parent,
                    "kept": nodes[nid]["kept"], "prompt_len": len(prompt),
                    "output_len": len(outs), "fork_at": fork_at,
                    "gap": len(prompt) - 1 - (r["cached_tokens"] or 0),
                    "fallback": kind, "levels_up": up, **r,
                }
            if policy in ("p1", "p1root"):
                for nid in ids:
                    if not nodes[nid]["kept"]:
                        close(nid, "pruned")
            if policy == "p1late":
                kept = [n for n in ids if nodes[n]["kept"]]

                def pin(nid):
                    r = stream_generate(base, root + path[nid], 1, session_for(nid, pin=True))
                    r.pop("output_ids")
                    return nid, r

                if kept:
                    with ThreadPoolExecutor(max_workers=len(kept)) as ex:
                        for nid, r in ex.map(pin, kept):
                            r["pin_gap"] = len(root) + len(path[nid]) - 1 - (r["cached_tokens"] or 0)
                            pins[nid] = r
        t_end = time.time()
        if hold_root:
            close("root", "tree_done")
    finally:
        leftover = [n for n in sid if n not in closed]
        for n in leftover:
            try:
                close(n, "end")
            except requests.RequestException:
                pass
        if own_poller:
            poller.stop_flag.set()
            poller.join()

    node_recs = [r for n, r in records.items() if n != "root"]
    problems = []
    for r in node_recs:
        if r["completion_tokens"] != seg or r["output_len"] != seg:
            problems.append(f"{r['node']}: decoded {r['completion_tokens']}/{r['output_len']}")
        if r["prompt_tokens"] != r["prompt_len"]:
            problems.append(f"{r['node']}: prompt {r['prompt_tokens']} != {r['prompt_len']}")
        if r["num_retractions"]:
            problems.append(f"{r['node']}: {r['num_retractions']} retractions")
    if leftover:
        problems.append(f"sessions still open at the end: {leftover}")

    counts, ups = {}, {}
    for r in node_recs:
        counts[r["fallback"]] = counts.get(r["fallback"], 0) + 1
        ups[str(r["levels_up"])] = ups.get(str(r["levels_up"]), 0) + 1
    level_times = {}
    for level, ids in enumerate(shape["levels"], 1):
        rs = [records[n] for n in ids]
        level_times[level] = round(max(r["client_recv"] for r in rs)
                                   - min(r["client_send"] for r in rs), 3)
    summary = {
        "completion_s": round(t_end - t_start, 3),
        "root_prefill_s": round(records["root"]["client_recv"]
                                - records["root"]["client_send"], 3),
        "level_s": level_times,
        "recomputed_tokens": sum(r["prompt_len"] - (r["cached_tokens"] or 0)
                                 for r in node_recs),
        "gap_tokens": sum(r["gap"] for r in node_recs),
        "fallback_counts": counts,
        "levels_up_counts": ups,
        "pin_requests": len(pins),
        "pins_that_recomputed_a_segment": sum(1 for r in pins.values()
                                              if r["pin_gap"] >= shape["seg"]),
        "t_start": t_start,
        "t_end": t_end,
        "metric_peaks": poller.peaks() if own_poller else None,
        "problems": problems,
    }
    return {
        "summary": summary,
        "records": list(records.values()),
        "pins": pins,
        "events": events,
        "metrics": poller.samples if own_poller else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--shape", required=True)
    ap.add_argument("--root-len", type=int, required=True)
    ap.add_argument("--policy", choices=["p0", "p1", "p1late", "p1root", "p0root"], required=True)
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
    result.update(run_tree(a.base, shape, a.root_len, a.policy, a.seed, tag))
    result["finished"] = time.strftime("%F %T")
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    s = result["summary"]
    print(f"{a.label}: completion {s['completion_s']} s, recomputed {s['recomputed_tokens']}, "
          f"fallback {s['fallback_counts']}, peak mamba "
          f"{ {k: v for k, v in s['metric_peaks'].items() if 'mamba' in k} }"
          f"{'  PROBLEMS: ' + '; '.join(s['problems']) if s['problems'] else ''}",
          flush=True)
    return 1 if s["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
