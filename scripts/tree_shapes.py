"""Seeded tree shapes for the replay runs.

A shape fixes everything about a tree except its text: node lengths, which nodes
are expanded, in what order, and which are pruned. The text is generated live in
each run (see replay_tree.py), so every policy does the same work.

T1 (segment tree, TreePO-like): every node decodes one segment of `seg` tokens.
Each expanded node gets `branch` children. Levels are expanded one at a time; when
a level has more than `width` nodes, `width` of them are kept (chosen by the seed)
and the rest are pruned. Nodes at the last level are leaves.

T2 (search with revisits, after Adaptive Parallel MCTS, arXiv 2604.00510): a node
is one reasoning step of 128-256 tokens with a fixed score, as a PRM would give it.
A simulation selects a childless node from the root by WU-PUCT, then rolls out:
it generates 2 children, follows the higher-scored one, and repeats until a terminal
node. Every generated node stays in the tree, so later simulations can fork from a
sibling left behind long ago. Simulations run in rounds of `parallel`: they are
selected one after another with in-flight counts (WU-UCT), run at once, and backed
up together when the round ends. Scores are drawn from the seed, so the shape does
not depend on the generated text.

T3 (agent search with long tool outputs, after SWE-Search, arXiv 2410.20285, and
LATS, arXiv 2310.04406): a node is one agent step, an action of 100-400 decoded
tokens followed by a tool output of 100-3,000 tokens that the next step reads. Each
iteration picks one expandable node (not terminal, fewer than 3 children, depth
under 20) by UCT over all such nodes and adds one child. Its score, drawn from the
seed as a value model would give it, is backed up along its path. Iterations run in
rounds of `parallel`, picked one after another with in-flight counts and backed up
when the round ends.

Usage: tree_shapes.py t1 --seed 0 --out shapes/t1-s0.json
       tree_shapes.py t2 --seed 0 --out shapes/t2-s0.json
       tree_shapes.py t3 --seed 0 --out shapes/t3-s0.json
"""

import argparse
import json
import math
import random
from collections import defaultdict


def t1(seed, depth=6, branch=2, width=8, seg=256):
    rng = random.Random(seed)
    nodes = {}
    levels = []
    frontier = ["root"]  # nodes whose children form the next level
    for level in range(1, depth + 1):
        ids = []
        for parent in frontier:
            for k in range(branch):
                nid = f"n{level}.{len(ids)}"
                nodes[nid] = {"parent": parent, "level": level, "seg": seg,
                              "kept": False, "children": []}
                if parent != "root":
                    nodes[parent]["children"].append(nid)
                ids.append(nid)
        levels.append(ids)
        if level == depth:
            break  # leaves: nothing is expanded
        kept = ids if len(ids) <= width else sorted(rng.sample(ids, width),
                                                    key=ids.index)
        for nid in kept:
            nodes[nid]["kept"] = True
        frontier = kept
    root_children = levels[0]
    n_fork = 1 + sum(1 for n in nodes.values() if n["kept"])
    return {
        "kind": "t1", "seed": seed, "depth": depth, "branch": branch,
        "width": width, "seg": seg,
        "root_children": root_children,
        "levels": levels,
        "nodes": nodes,
        "stats": {
            "requests": len(nodes),
            "decode_tokens": len(nodes) * seg,
            "fork_states": n_fork,  # root + every expanded node
            "max_level_width": max(len(l) for l in levels),
        },
    }


def ancestors(nodes, nid):
    """nid and every node above it, up to and including the root."""
    out = []
    while nid is not None:
        out.append(nid)
        nid = nodes[nid]["parent"]
    return out


def selection_order(nodes, children, N, W, eligible, m, c=1.0):
    """The childless nodes that m parallel WU-PUCT selections would pick, in order.

    children(n) lists n's children as the program knows them; eligible(n) says
    whether a childless node can be expanded. Each pick adds an in-flight count to
    its path, as WU-UCT does, so the next pick is steered elsewhere. Q of a node not
    yet backed up is its own score. The prior is uniform.
    """
    O = defaultdict(int)
    taken = set()
    memo = {}

    def open_(n):
        if n in memo:
            return memo[n]
        ch = children(n)
        v = (eligible(n) and n not in taken) if not ch else any(open_(k) for k in ch)
        memo[n] = v
        return v

    def q(n):
        return W[n] / N[n] if N[n] else nodes[n]["score"]

    order = []
    while len(order) < m:
        memo.clear()
        if not open_("root"):
            break
        s = "root"
        while children(s):
            ch = [k for k in children(s) if open_(k)]
            prior = 1.0 / len(children(s))
            ns = math.sqrt(max(N[s] + O[s], 1))
            s = max(ch, key=lambda k: q(k) + c * prior * ns / (1 + N[k] + O[k]))
        order.append(s)
        taken.add(s)
        for a in ancestors(nodes, s):
            O[a] += 1
    return order


def t2(seed, sims=16, parallel=4, branch=2, min_len=128, max_len=256,
       min_depth=4, max_depth=10, p_term=0.25, c=1.0):
    rng = random.Random(seed)
    nodes = {"root": {"parent": None, "depth": 0, "len": 0, "score": None,
                      "terminal": False, "children": []}}
    N, W = defaultdict(int), defaultdict(float)
    order = []  # node ids in creation order
    selected = set()

    def new_child(parent, rnd, sim):
        nid = f"n{len(order)}"
        d = nodes[parent]["depth"] + 1
        base = rng.uniform(0.2, 0.9) if parent == "root" else nodes[parent]["score"]
        score = min(1.0, max(0.0, base + rng.gauss(0, 0.15)))
        term = d >= max_depth or (d >= min_depth and rng.random() < p_term)
        nodes[nid] = {"parent": parent, "depth": d, "len": rng.randint(min_len, max_len),
                      "score": round(score, 4), "terminal": term, "children": [],
                      "round": rnd, "sim": sim}
        nodes[parent]["children"].append(nid)
        order.append(nid)
        return nid

    def eligible(n):
        return not nodes[n]["terminal"] and n not in selected

    rounds = []
    done = 0
    while done < sims:
        rnd = len(rounds)
        picks = selection_order(nodes, lambda n: nodes[n]["children"], N, W, eligible,
                                min(parallel, sims - done), c)
        if not picks:
            break
        selected.update(picks)
        this = []
        for leaf in picks:
            created = len(order)
            steps, cur = [], leaf
            while True:
                kids = [new_child(cur, rnd, done + len(this)) for _ in range(branch)]
                best = max(kids, key=lambda k: nodes[k]["score"])
                steps.append({"parent": cur, "children": kids, "follow": best})
                if nodes[best]["terminal"]:
                    break
                cur = best
            path = ancestors(nodes, best)[:-1]  # without the root
            value = min(nodes[n]["score"] for n in path)
            age = None if leaf == "root" else created - order.index(leaf)
            this.append({"sim": done + len(this), "leaf": leaf, "leaf_depth": nodes[leaf]["depth"],
                         "leaf_age_nodes": age, "steps": steps, "terminal": best,
                         "value": round(value, 4)})
        for s in this:
            for a in ancestors(nodes, s["terminal"]):
                N[a] += 1
                W[a] += s["value"]
        rounds.append(this)
        done += len(this)

    sims_all = [s for r in rounds for s in r]
    revisits = [s for s in sims_all if s["leaf"] != "root"]
    return {
        "kind": "t2", "seed": seed, "sims": sims, "parallel": parallel, "branch": branch,
        "len_range": [min_len, max_len], "depth_range": [min_depth, max_depth],
        "p_term": p_term, "c": c,
        "rounds": rounds,
        "nodes": nodes,
        "stats": {
            "requests": len(order),
            "decode_tokens": sum(nodes[n]["len"] for n in order),
            "rounds": [len(r) for r in rounds],
            "steps": sum(len(s["steps"]) for s in sims_all),
            "max_depth": max(nodes[n]["depth"] for n in order),
            "revisit_leaf_depth": sorted(s["leaf_depth"] for s in revisits),
            "revisit_leaf_age_nodes": sorted(s["leaf_age_nodes"] for s in revisits),
            "candidates_left": sum(1 for n in order if not nodes[n]["children"]
                                   and not nodes[n]["terminal"] and n not in selected),
        },
    }


def obs_len(rng):
    """Tool output length: 30% short, 50% medium, 20% at a 3,000-token cap."""
    u = rng.random()
    if u < 0.3:
        return rng.randint(100, 500)
    if u < 0.8:
        return rng.randint(1000, 2000)
    return 3000


def uct_order(nodes, N, W, expandable, m, c=1.0, in_flight=None):
    """The nodes that m parallel UCT picks would expand, in order.

    UCT runs over every expandable node at once, not down from the root:
    Q(n) + c * sqrt(ln(N(parent) + 2) / (N(n) + O(n) + 1)). Q of a node not yet
    backed up is its own score, and the root's is 0.5. Each pick adds an in-flight
    count O to the picked node, and a node is picked at most once per call.
    `in_flight` gives counts for expansions already running.
    """
    O = defaultdict(int, in_flight or {})

    def q(n):
        if N[n]:
            return W[n] / N[n]
        s = nodes[n]["score"]
        return 0.5 if s is None else s

    def uct(n):
        p = nodes[n]["parent"] or n
        return q(n) + c * math.sqrt(math.log(N[p] + 2) / (N[n] + O[n] + 1))

    pool = [n for n in nodes if expandable(n)]
    order = []
    while pool and len(order) < m:
        n = max(pool, key=lambda n: (uct(n), -nodes[n]["born"]))
        order.append(n)
        pool.remove(n)
        O[n] += 1
    return order


def t3(seed, iters=100, parallel=4, max_children=3, act=(100, 400), min_term=5,
       max_depth=20, p_term=0.1, c=1.0):
    rng = random.Random(seed)
    nodes = {"root": {"parent": None, "depth": 0, "act": 0, "obs": 0, "score": None,
                      "terminal": False, "children": [], "round": -1, "born": -1}}
    N, W = defaultdict(int), defaultdict(float)
    order = []
    rounds = []

    def expandable(n):
        x = nodes[n]
        return (not x["terminal"] and len(x["children"]) < max_children
                and x["depth"] < max_depth)

    done = 0
    while done < iters:
        rnd = len(rounds)
        picks = uct_order(nodes, N, W, expandable, min(parallel, iters - done), c)
        if not picks:
            break
        this = []
        for p in picks:
            nid = f"n{len(order)}"
            d = nodes[p]["depth"] + 1
            base = rng.uniform(0.2, 0.9) if p == "root" else nodes[p]["score"]
            score = min(1.0, max(0.0, base + rng.gauss(0, 0.15)))
            term = d >= max_depth or (d >= min_term and rng.random() < p_term)
            nodes[nid] = {"parent": p, "depth": d, "act": rng.randint(*act),
                          "obs": 0 if term else obs_len(rng), "score": round(score, 4),
                          "terminal": term, "children": [], "round": rnd,
                          "born": len(order)}
            age = None if p == "root" else len(order) - nodes[p]["born"] - 1
            nodes[p]["children"].append(nid)
            order.append(nid)
            this.append({"it": done + len(this), "parent": p, "child": nid,
                         "parent_depth": nodes[p]["depth"], "parent_age_nodes": age,
                         "nth_child": len(nodes[p]["children"])})
        for e in this:
            n = e["child"]
            while n is not None:
                N[n] += 1
                W[n] += nodes[e["child"]]["score"]
                n = nodes[n]["parent"]
        rounds.append(this)
        done += len(this)

    exps = [e for r in rounds for e in r]
    ages = [e["parent_age_nodes"] for e in exps if e["parent_age_nodes"] is not None]
    path_tokens = {}
    for n in order:  # tokens below the root on the way to n's children
        p = nodes[n]["parent"]
        path_tokens[n] = (0 if p == "root" else path_tokens[p]) + nodes[n]["act"] + nodes[n]["obs"]
    return {
        "kind": "t3", "seed": seed, "iters": iters, "parallel": parallel,
        "max_children": max_children, "act_range": list(act), "min_term": min_term,
        "max_depth": max_depth, "p_term": p_term, "c": c,
        "rounds": rounds,
        "nodes": nodes,
        "stats": {
            "expansions": len(exps),
            "requests": len(exps) + sum(1 for n in order if not nodes[n]["terminal"]),
            "decode_tokens": sum(nodes[n]["act"] for n in order),
            "obs_tokens": sum(nodes[n]["obs"] for n in order),
            "max_depth": max(nodes[n]["depth"] for n in order),
            "terminals": sum(1 for n in order if nodes[n]["terminal"]),
            "from_previous_round": sum(1 for e in exps if e["parent"] != "root"
                                       and nodes[e["parent"]]["round"] == nodes[e["child"]]["round"] - 1),
            "parent_age_nodes_median": sorted(ages)[len(ages) // 2],
            "parent_age_8_or_more": sum(1 for a in ages if a >= 8),
            "max_path_tokens_below_root": max(path_tokens.values()),
            "expanded_parent_path_tokens_mean": round(sum(
                path_tokens[e["parent"]] for e in exps if e["parent"] != "root")
                / max(1, sum(1 for e in exps if e["parent"] != "root"))),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["t1", "t2", "t3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--branch", type=int, default=2)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--seg", type=int, default=256)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.kind == "t1":
        shape = t1(a.seed, a.depth, a.branch, a.width, a.seg)
    elif a.kind == "t2":
        shape = t2(a.seed, sims=a.sims, parallel=a.parallel, branch=a.branch, c=a.c)
    else:
        shape = t3(a.seed, iters=a.iters, parallel=a.parallel, c=a.c)
    with open(a.out, "w") as f:
        json.dump(shape, f, indent=1)
    print(a.out, json.dumps(shape["stats"]))


if __name__ == "__main__":
    main()
