"""Does a cached hybrid state give the same numbers as recomputing the prefix?

RL frameworks report that prefix caching changes generation logprobs for hybrid
models (NeMo-RL #4622, SkyRL #1981, both on vLLM). This checks SGLang 0.5.23 with
Qwen3.5-9B, at the three kinds of state that T1's children start from:

  chunk  the last 4096-token chunk boundary inside the root (a level-1 child,
         whose prompt is the root itself)
  end    the root's own prefill end state (what P0-root holds)
  decode the state tracked every 256 tokens while the parent decodes (a level-2
         child after its parent)

For each root, with a natural-text prompt R and the parent's output O:

  P1 = R      cold                  flush, generate
              warm-chunk            flush, prefill R, generate
  P2 = R + O  cold, cold-again      flush, generate (twice, to see run-to-run noise)
              warm-end              flush, prefill R, generate
              warm-decode           right after the parent generated O from R

Part A, sampled (as in RL). Every generation samples 256 tokens at temperature 1
and is then scored again from scratch (flush, prefill P + C, read the input
logprobs of C), which plays the trainer's part. Per generation:
|decode logprob - recomputed logprob| per sampled token, summarized as mean, max
and mean exp(|diff|) (a token_mult_prob_error-like number; 1.0 is identical).
SGLang ignores sampling_seed without --enable-deterministic-inference, so the
conditions draw different tokens and are compared only through these numbers.

Part B, teacher-forced (exact). A fixed continuation N (the next 256 corpus tokens
after R) is scored after P twice from scratch and once from each kind of cached
state, and the logprobs of N are compared token by token:
  noise   cold vs cold again (run-to-run noise of a recompute)
  chunk   prefill R, then score R + N        (starts from the 4096 boundary)
  end     prefill R, then score R + O + N    (starts from R's end state)
  decode  generate O' from R, then score R + O' + N (starts from the decode state)

Requests run one at a time, so batch composition does not vary.

Usage:
  cache_fidelity.py --root-len 8000 --roots 8 --out runs/fidelity/r8000.json
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import requests

BASE = "http://127.0.0.1:30000"
SEG = 256
CHUNK = 4096


def post(path, body=None, timeout=600):
    r = requests.post(BASE + path, json=body or {}, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:  # /flush_cache answers with plain text
        return r.text


def flush():
    post("/flush_cache")
    time.sleep(0.3)


def sp(max_new):
    return {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": max_new,
            "ignore_eos": True}


def generate(ids, max_new=SEG):
    out = post("/generate", {"input_ids": ids, "sampling_params": sp(max_new),
                             "return_logprob": True})
    m = out["meta_info"]
    lps = [x[0] for x in m["output_token_logprobs"]]
    toks = [x[1] for x in m["output_token_logprobs"]]
    assert toks == out["output_ids"], "logprob tokens do not match output ids"
    return {"prompt_len": len(ids), "cached": m["cached_tokens"],
            "out": out["output_ids"], "lp": lps}


def prefill(ids):
    out = post("/generate", {"input_ids": ids, "sampling_params": sp(1)})
    return out["meta_info"]["cached_tokens"]


def score(prompt, cont, cold=True):
    """Return the logprob of each token of cont after prompt, and the cached length.

    With cold=True the cache is flushed first, so the whole input is recomputed."""
    if cold:
        flush()
    ids = prompt + cont
    # The row at logprob_start_len itself carries no logprob, so start one token
    # early and drop that row.
    out = post("/generate", {"input_ids": ids, "sampling_params": sp(1),
                             "return_logprob": True, "logprob_start_len": len(prompt) - 1})
    m = out["meta_info"]
    if cold:
        assert m["cached_tokens"] == 0, m["cached_tokens"]
    rows = m["input_token_logprobs"]
    assert len(rows) == len(cont) + 1, (len(rows), len(cont))
    assert rows[0][1] == prompt[-1], "first scored row is not the last prompt token"
    rows = rows[1:]
    assert [r[1] for r in rows] == cont, "scored tokens do not match"
    assert all(r[0] is not None for r in rows), "missing logprob in scored rows"
    return [r[0] for r in rows], m["cached_tokens"]


def diff(x, y):
    d = [abs(a - b) for a, b in zip(x, y)]
    return {"mean_abs": sum(d) / len(d), "max_abs": max(d),
            "mult_err": sum(math.exp(v) for v in d) / len(d),
            "first32_mean": sum(d[:32]) / 32, "last32_mean": sum(d[-32:]) / 32}


def gen_vs_ref(g):
    d = [abs(a - b) for a, b in zip(g["lp"], g["ref"])]
    return {"mean_abs": sum(d) / len(d), "max_abs": max(d),
            "mult_err": sum(math.exp(x) for x in d) / len(d)}


def corpus(tok):
    files = sorted(glob.glob(os.path.expanduser(
        "~/.venvs/sglang0523/lib/python3.11/site-packages/**/*.md"), recursive=True))
    files = [f for f in files if os.path.getsize(f) > 8192]
    text = "\n\n".join(open(f, errors="replace").read() for f in files)
    return tok(text, add_special_tokens=False)["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-len", type=int, required=True)
    ap.add_argument("--roots", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--control-from", help="results of a main run: recompute its cold "
                    "scores on this server (started with another chunk size) and compare")
    ap.add_argument("--split-control-from", help="results of a main run, on a server whose "
                    "chunk size divides the root length: compare the end-state path with a "
                    "cold recompute that splits at the same place")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    ids = corpus(tok)
    L = a.root_len
    span = len(ids) - L - SEG
    assert span > 0, (len(ids), L)
    info = requests.get(BASE + "/get_server_info", timeout=60).json()
    keep = ("version", "max_mamba_cache_size", "mamba_radix_cache_strategy",
            "mamba_track_interval", "chunked_prefill_size", "quantization",
            "disable_overlap_schedule", "random_seed")

    if a.control_from:
        return control(a, ids, info, keep)
    if a.split_control_from:
        return split_control(a, ids, info, keep)

    roots = []
    for k in range(a.roots):
        off = (k * span) // max(a.roots - 1, 1)
        R = ids[off:off + L]
        N = ids[off + L:off + L + SEG]
        g = {}
        flush()
        g["p1_cold"] = generate(R)
        O = g["p1_cold"]["out"]
        P2 = R + O
        g["p2_warm_decode"] = generate(P2)
        flush()
        g["p2_cold"] = generate(P2)
        flush()
        g["p2_cold_again"] = generate(P2)
        flush()
        prefill(R)
        g["p2_warm_end"] = generate(P2)
        flush()
        prefill(R)
        g["p1_warm_chunk"] = generate(R)
        for name, x in g.items():
            prompt = R if name.startswith("p1") else P2
            x["ref"], _ = score(prompt, x["out"])
            x["gen_vs_ref"] = gen_vs_ref(x)

        b = {}
        c1, _ = score(R, N)
        c1b, _ = score(R, N)
        flush()
        prefill(R)
        w1, cw1 = score(R, N, cold=False)
        c2, _ = score(P2, N)
        c2b, _ = score(P2, N)
        flush()
        prefill(R)
        w2, cw2 = score(P2, N, cold=False)
        flush()
        Ob = generate(R)["out"]
        w3, cw3 = score(R + Ob, N, cold=False)
        c3, _ = score(R + Ob, N)
        b["noise_p1"] = {"cached": 0, **diff(c1b, c1)}
        b["noise_p2"] = {"cached": 0, **diff(c2b, c2)}
        b["chunk"] = {"cached": cw1, **diff(w1, c1)}
        b["end"] = {"cached": cw2, **diff(w2, c2)}
        b["decode"] = {"cached": cw3, **diff(w3, c3)}
        # Kept for the chunk-size control (--control-from).
        b["cold_lp"] = {"p1": c1, "p2": c2, "decode": c3}
        b["decode_parent_out"] = Ob

        row = {"root": k, "offset": off, "sampled": g, "teacher_forced": b}
        roots.append(row)
        print(f"root {k} A: " + "; ".join(
            f"{n} cached {x['cached']} mult {x['gen_vs_ref']['mult_err']:.4f} "
            f"max {x['gen_vs_ref']['max_abs']:.3f}" for n, x in g.items()), flush=True)
        print(f"root {k} B: " + "; ".join(
            f"{n} cached {x['cached']} mean {x['mean_abs']:.5f} max {x['max_abs']:.4f}"
            for n, x in b.items() if "mean_abs" in x), flush=True)

    out = {"root_len": L, "roots": a.roots, "corpus_tokens": len(ids),
           "server_info": {k: info.get(k) for k in keep}, "results": roots}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)


def control(a, ids, info, keep):
    """Same cold scores, different chunk boundaries, no cache: pure path noise."""
    main_run = json.load(open(a.control_from))
    L = main_run["root_len"]
    assert L == a.root_len
    rows = []
    for row in main_run["results"]:
        off = row["offset"]
        R = ids[off:off + L]
        N = ids[off + L:off + L + SEG]
        O = row["sampled"]["p1_cold"]["out"]
        Ob = row["teacher_forced"]["decode_parent_out"]
        old = row["teacher_forced"]["cold_lp"]
        new = {"p1": score(R, N)[0], "p2": score(R + O, N)[0], "decode": score(R + Ob, N)[0]}
        d = {k: diff(new[k], old[k]) for k in new}
        rows.append({"root": row["root"], "offset": off, "chunking": d})
        print(f"root {row['root']} control: " + "; ".join(
            f"{k} mean {v['mean_abs']:.5f} max {v['max_abs']:.4f}" for k, v in d.items()),
            flush=True)
    out = {"root_len": L, "control_from": a.control_from,
           "server_info": {k: info.get(k) for k in keep}, "results": rows}
    with open(a.out, "w") as f:
        json.dump(out, f)


def split_control(a, ids, info, keep):
    """Is the cached end state exact, or does only the split at the root's end matter?

    With the chunk size dividing L, a cold recompute of R + O + N also splits at L,
    so it and the end-state path compute the same chunks. Any difference left is
    in the stored state itself."""
    main_run = json.load(open(a.split_control_from))
    L = main_run["root_len"]
    assert L == a.root_len and L % info["chunked_prefill_size"] == 0, info["chunked_prefill_size"]
    rows = []
    for row in main_run["results"]:
        off = row["offset"]
        R = ids[off:off + L]
        N = ids[off + L:off + L + SEG]
        P2 = R + row["sampled"]["p1_cold"]["out"]
        c, _ = score(P2, N)
        flush()
        prefill(R)
        w, cw = score(P2, N, cold=False)
        d = {"end_vs_cold_same_split": {"cached": cw, **diff(w, c)},
             "cold_split_at_L_vs_main_cold": diff(c, row["teacher_forced"]["cold_lp"]["p2"])}
        rows.append({"root": row["root"], "offset": off, **d})
        print(f"root {row['root']} split control: " + "; ".join(
            f"{k} mean {v['mean_abs']:.5f} max {v['max_abs']:.4f}" for k, v in d.items()),
            flush=True)
    out = {"root_len": L, "split_control_from": a.split_control_from,
           "server_info": {k: info.get(k) for k in keep}, "results": rows}
    with open(a.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    sys.exit(main())
