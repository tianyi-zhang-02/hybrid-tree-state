"""How far does an approximately rebuilt hybrid state move the logprobs an RL trainer checks?

Tail-Replay (arXiv 2608.30310) and SuffixReplay (arXiv 2609.33477) keep the attention KV of
a cached prefix but not the Gated DeltaNet (GDN) states. On a hit they rebuild each GDN
group's state from a zero state, by replaying only the last part of the prefix through the
group's layers, starting from the group's exact input hidden states. Both papers judge the
result by task scores. An RL trainer instead compares the generator's per-token logprobs
with its own recompute.

This emulates that rebuild in HF Transformers (torch path, bf16) and measures how far the
logprobs of a sampled continuation move.

Per root (natural text R of n tokens; the cache boundary is m = n - 1):
  1. Prefill R[:m] in chunks into a cache, capturing the input hidden states of each GDN
     group's first layer at every prefix position (kept on the CPU).
  2. Sample a continuation C of SEG tokens at temperature 1 by decoding from the exact
     state, starting with R[m]. These are the generator's own tokens.
  3. Reference: teacher-force [R[m]] + C[:-1] from the exact state in one forward (the
     prefill path), as a trainer's recompute would. Keep the full distributions.
  4. For each replay fraction r, set every GDN layer's conv and recurrent state to the one
     rebuilt from the last k = ceil(r * m) positions, then
       tf   teacher-force the same input in one forward and compare with the reference
            (the effect of the state alone);
       dec  feed the same tokens one at a time through the decode path and compare with
            the reference (what a generator using the rebuilt state would report).
     r = 0 zeroes the states. r = 1 replays the whole prefix with the prefill's own chunk
     boundaries and must reproduce the exact state bit for bit (a check of the emulation).
  Between conditions the attention KV is cropped back to m and the GDN states restored.

The baseline is "exact dec": the decode path from the exact state against the reference,
the gap that exists with no approximation at all.

--rollout runs the direction a replay engine runs in. For each state (exact, then each
fraction) it samples its own continuation from that state, with the same seed, and keeps the
sampling pass's log-probs (the generator's view). It then teacher-forces those tokens from the
exact state (the trainer's view). KL is KL(rollout || trainer), since the tokens come from the
rollout. r = 1 must give the exact state's continuation and numbers bit for bit.

Usage:
  tail_replay_drift.py --root-len 8000 --roots 8 --out runs/tail-replay/r8000.json
  tail_replay_drift.py --rollout --root-len 8000 --roots 8 --out runs/tail-replay-rollout/r8000.json
  tail_replay_drift.py --tiny                      # CPU self-test on a tiny random model
  tail_replay_drift.py --summarize runs/tail-replay/r8000.json runs/tail-replay/r32064.json
"""

import argparse
import glob
import json
import math
import os
import platform
import time

import torch

FRACTIONS = [0.0, 0.01, 0.05, 0.10, 0.20, 1.0]


def corpus(tok):
    """The same natural text as cache_fidelity.py, so the roots are the same."""
    files = sorted(glob.glob(os.path.expanduser(
        "~/.venvs/sglang0523/lib/python3.11/site-packages/**/*.md"), recursive=True))
    files = [f for f in files if os.path.getsize(f) > 8192]
    text = "\n\n".join(open(f, errors="replace").read() for f in files)
    return tok(text, add_special_tokens=False)["input_ids"]


def text_parts(model):
    inner = model.model
    lm = inner.language_model if hasattr(inner, "language_model") else inner
    return lm, model.lm_head


class Runner:
    def __init__(self, model, chunk, device):
        from transformers import DynamicCache
        self.DynamicCache = DynamicCache
        self.lm, self.head = text_parts(model)
        self.cfg = self.lm.config
        self.dev = device
        self.chunk = chunk
        types = self.cfg.layer_types
        self.gdn = [i for i, t in enumerate(types) if t == "linear_attention"]
        self.fa = [i for i, t in enumerate(types) if t == "full_attention"]
        self.groups = []  # runs of consecutive GDN layers
        for i in self.gdn:
            if self.groups and self.groups[-1][-1] == i - 1:
                self.groups[-1].append(i)
            else:
                self.groups.append([i])
        self.capture = None
        for g in self.groups:
            self.lm.layers[g[0]].register_forward_pre_hook(self._hook(g[0]), with_kwargs=True)

    def _hook(self, idx):
        def f(_mod, args, kwargs):
            if self.capture is not None:
                h = args[0] if args else kwargs["hidden_states"]
                self.capture[idx].append(h[0].detach().to("cpu"))
        return f

    def new_cache(self):
        return self.DynamicCache(config=self.cfg)

    @torch.no_grad()
    def prefill(self, ids):
        cache = self.new_cache()
        self.capture = {g[0]: [] for g in self.groups}
        for s in range(0, len(ids), self.chunk):
            x = torch.tensor([ids[s:s + self.chunk]], device=self.dev)
            self.lm(input_ids=x, past_key_values=cache, use_cache=True)
        cap = {k: torch.cat(v, 0) for k, v in self.capture.items()}
        self.capture = None
        return cache, cap

    def states(self, cache):
        return {i: (cache.layers[i].conv_states[0].clone(),
                    cache.layers[i].recurrent_states[0].clone()) for i in self.gdn}

    def load(self, cache, m, states):
        """Crop the attention KV back to m tokens and set the GDN states."""
        for i in self.fa:
            L = cache.layers[i]
            L.keys = L.keys[..., :m, :]
            L.values = L.values[..., :m, :]
        for i, (c, r) in states.items():
            cache.layers[i].conv_states[0].copy_(c)
            cache.layers[i].recurrent_states[0].copy_(r)
        assert cache.get_seq_length() == m

    @torch.no_grad()
    def rebuild(self, cap, m, k, exact):
        """GDN states rebuilt from a zero state over the last k prefix positions, per group,
        from the group's exact input hidden states. Pieces of `chunk` start at m - k."""
        out = {}
        for g in self.groups:
            if k == 0:
                for i in g:
                    out[i] = (torch.zeros_like(exact[i][0]), torch.zeros_like(exact[i][1]))
                continue
            fresh = self.new_cache()
            X = cap[g[0]][m - k:m]
            for s in range(0, k, self.chunk):
                h = X[s:s + self.chunk].to(self.dev)[None]
                for i in g:
                    h = self.lm.layers[i](h, position_embeddings=None, attention_mask=None,
                                          position_ids=None, past_key_values=fresh)
            for i in g:
                out[i] = (fresh.layers[i].conv_states[0].clone(),
                          fresh.layers[i].recurrent_states[0].clone())
        return out

    def logprobs(self, h):
        return torch.log_softmax(self.head(h).float(), -1)

    @torch.no_grad()
    def score_tf(self, cache, inp):
        x = torch.tensor([inp], device=self.dev)
        h = self.lm(input_ids=x, past_key_values=cache, use_cache=True).last_hidden_state
        return self.logprobs(h[0])

    @torch.no_grad()
    def score_dec(self, cache, inp):
        rows = []
        for t in inp:
            x = torch.tensor([[t]], device=self.dev)
            h = self.lm(input_ids=x, past_key_values=cache, use_cache=True).last_hidden_state
            rows.append(self.logprobs(h[0, -1]))
        return torch.stack(rows)

    @torch.no_grad()
    def sample(self, cache, first, n, gen):
        toks, rows, t = [], [], first
        for _ in range(n):
            x = torch.tensor([[t]], device=self.dev)
            h = self.lm(input_ids=x, past_key_values=cache, use_cache=True).last_hidden_state
            lp = self.logprobs(h[0, -1])
            t = int(torch.multinomial(lp.exp(), 1, generator=gen).item())
            rows.append(lp)
            toks.append(t)
        return toks, torch.stack(rows)


def compare(full, ref, tgt):
    """Per-token comparison of two (T, V) log-prob tensors on the target tokens."""
    a = full.gather(1, tgt[:, None])[:, 0]
    b = ref.gather(1, tgt[:, None])[:, 0]
    d = (a - b).abs()
    kl = (ref.exp() * (ref - full)).sum(-1).clamp_min(0)
    n = len(d)
    q = max(n // 8, 1)
    return {
        "mean_abs": d.mean().item(), "max_abs": d.max().item(),
        "mult_err": d.exp().mean().item(),
        "first_mean": d[:q].mean().item(), "last_mean": d[-q:].mean().item(),
        "frac_over_0.2": (d > 0.2).float().mean().item(), "n_over_2": int((d > 2).sum().item()),
        "kl_mean": kl.mean().item(), "kl_max": kl.max().item(),
        "argmax_agree": (full.argmax(-1) == ref.argmax(-1)).float().mean().item(),
        "lp": a.tolist(),
    }


def state_error(rebuilt, exact):
    """Relative Frobenius error of each GDN layer's recurrent state."""
    return {str(i): ((rebuilt[i][1] - exact[i][1]).norm() / exact[i][1].norm().clamp_min(1e-30)).item()
            for i in exact}


def run_root(R, ids, seg, seed, fractions):
    m = len(ids) - 1
    t0 = time.time()
    cache, cap = R.prefill(ids[:m])
    exact = R.states(cache)
    t_prefill = time.time() - t0

    gen = torch.Generator(device=R.dev).manual_seed(seed)
    C, sample_rows = R.sample(cache, ids[m], seg, gen)
    sample_rows = sample_rows.cpu()
    inp = [ids[m]] + C[:-1]
    tgt = torch.tensor(C, device=R.dev)

    R.load(cache, m, exact)
    ref = R.score_tf(cache, inp)
    out = {"m": m, "prefill_s": t_prefill, "cont": C,
           "ref_lp": ref.gather(1, tgt[:, None])[:, 0].tolist(), "conds": {}}

    # Checks of the harness: the decode path recomputed from the restored exact state must
    # equal the sampling pass, bit for bit.
    R.load(cache, m, exact)
    dec = R.score_dec(cache, inp)
    out["check_dec_equals_sampling"] = bool(torch.equal(dec.cpu(), sample_rows))
    out["conds"]["exact"] = {"k": m, "dec": compare(dec, ref, tgt)}
    del dec, sample_rows

    for r in fractions:
        k = math.ceil(r * m)
        t1 = time.time()
        st = R.rebuild(cap, m, k, exact)
        t_rebuild = time.time() - t1
        R.load(cache, m, st)
        tf = R.score_tf(cache, inp)
        tf_cmp = compare(tf, ref, tgt)
        del tf
        R.load(cache, m, st)
        dec = R.score_dec(cache, inp)
        dec_cmp = compare(dec, ref, tgt)
        del dec
        name = f"r{round(r * 100)}"
        out["conds"][name] = {"k": k, "rebuild_s": t_rebuild, "state_err": state_error(st, exact),
                              "states_bit_equal": all(torch.equal(st[i][0], exact[i][0]) and
                                                      torch.equal(st[i][1], exact[i][1]) for i in exact),
                              "tf": tf_cmp, "dec": dec_cmp}
        del st
        print(f"  {name:5s} k={k:6d} tf |d|={tf_cmp['mean_abs']:.4f} kl={tf_cmp['kl_mean']:.4f}  "
              f"dec |d|={dec_cmp['mean_abs']:.4f}", flush=True)

    R.load(cache, m, exact)
    again = R.score_tf(cache, inp)
    out["check_ref_reproduces"] = bool(torch.equal(again, ref))
    del cache, cap, ref, again, exact
    return out


def run_root_rollout(R, ids, seg, seed, fractions):
    """Sample from each state (the generator) and score from the exact state (the trainer)."""
    m = len(ids) - 1
    t0 = time.time()
    cache, cap = R.prefill(ids[:m])
    exact = R.states(cache)
    out = {"m": m, "prefill_s": time.time() - t0, "conds": {}}
    for name, r in [("exact", None)] + [(f"r{round(r * 100)}", r) for r in fractions]:
        k = m if r is None else math.ceil(r * m)
        st = exact if r is None else R.rebuild(cap, m, k, exact)
        R.load(cache, m, st)
        gen = torch.Generator(device=R.dev).manual_seed(seed)
        C, roll = R.sample(cache, ids[m], seg, gen)
        R.load(cache, m, exact)
        train = R.score_tf(cache, [ids[m]] + C[:-1])
        tgt = torch.tensor(C, device=R.dev)
        cmp = compare(train, roll, tgt)  # kl here is KL(rollout || trainer)
        cmp["train_lp"] = cmp.pop("lp")
        cmp["roll_lp"] = roll.gather(1, tgt[:, None])[:, 0].tolist()
        c = {"k": k, "cont": C, "dec": cmp}
        if r is not None:
            c["state_err"] = state_error(st, exact)
            c["states_bit_equal"] = all(torch.equal(st[i][0], exact[i][0]) and
                                        torch.equal(st[i][1], exact[i][1]) for i in exact)
        out["conds"][name] = c
        del roll, train
        print(f"  {name:5s} k={k:6d} |log r|={cmp['mean_abs']:.4f} kl={cmp['kl_mean']:.4f} "
              f"same tokens as exact {sum(a == b for a, b in zip(C, out['conds']['exact']['cont']))}/{seg}",
              flush=True)
    e, f = out["conds"]["exact"], out["conds"].get("r100")
    out["check_r100_same_as_exact"] = bool(f and f["cont"] == e["cont"] and
                                           f["dec"]["roll_lp"] == e["dec"]["roll_lp"] and
                                           f["dec"]["train_lp"] == e["dec"]["train_lp"])
    del cache, cap, exact
    return out


def tiny_selftest():
    """CPU check on a tiny random Qwen3.5 text model, in fp32."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.manual_seed(0)
    types = (["linear_attention"] * 3 + ["full_attention"]) * 2
    cfg = Qwen3_5TextConfig(vocab_size=512, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                            linear_key_head_dim=16, linear_value_head_dim=16, layer_types=types)
    cfg._attn_implementation = "sdpa"
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    R = Runner(model, chunk=64, device="cpu")
    print("groups", R.groups, "fa", R.fa)
    ids = torch.randint(0, 512, (301,)).tolist()
    out = run_root(R, ids, seg=16, seed=0, fractions=FRACTIONS)
    m = out["m"]
    print("check_dec_equals_sampling", out["check_dec_equals_sampling"])
    print("check_ref_reproduces", out["check_ref_reproduces"])
    print("r100 states_bit_equal", out["conds"]["r100"]["states_bit_equal"],
          "tf max", out["conds"]["r100"]["tf"]["max_abs"], "dec max", out["conds"]["r100"]["dec"]["max_abs"])
    # The cached path must agree with one single forward over the whole sequence.
    full = ids[:m] + [ids[m]] + out["cont"][:-1]
    with torch.no_grad():
        h = R.lm(input_ids=torch.tensor([full])).last_hidden_state[0, m:]
        lp = R.logprobs(h)
    tgt = torch.tensor(out["cont"])
    single = lp.gather(1, tgt[:, None])[:, 0]
    dev = (single - torch.tensor(out["ref_lp"])).abs().max().item()
    print("cached vs single forward, max |d|:", dev)
    ok = (out["check_dec_equals_sampling"] and out["check_ref_reproduces"]
          and out["conds"]["r100"]["states_bit_equal"] and out["conds"]["r100"]["tf"]["max_abs"] == 0
          and dev < 1e-3 and out["conds"]["r0"]["tf"]["mean_abs"] > 0)
    for name in ["exact"] + [f"r{round(r * 100)}" for r in FRACTIONS]:
        c = out["conds"][name]
        tf = c.get("tf", {}).get("mean_abs", float("nan"))
        print(f"  {name:5s} k={c['k']:4d} tf {tf:.2e}  dec {c['dec']['mean_abs']:.2e}")
    # Rollout direction: r = 1 must equal the exact state, and the exact state's sample must
    # equal the continuation sampled above with the same seed.
    ro = run_root_rollout(R, ids, seg=16, seed=0, fractions=FRACTIONS)
    same_cont = ro["conds"]["exact"]["cont"] == out["cont"]
    same_ref = ro["conds"]["exact"]["dec"]["train_lp"] == out["ref_lp"]
    print("rollout: r100 same as exact", ro["check_r100_same_as_exact"],
          "| exact sample same as first mode", same_cont, "| trainer lp same", same_ref,
          "| r0 |log r|", f"{ro['conds']['r0']['dec']['mean_abs']:.3e}")
    ok = (ok and ro["check_r100_same_as_exact"] and same_cont and same_ref
          and ro["conds"]["r0"]["dec"]["mean_abs"] > 0)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def summarize(paths):
    rows = []
    for p in paths:
        d = json.load(open(p))
        L = d["meta"]["root_len"]
        roots = d["roots"]
        names = list(roots[0]["conds"].keys())
        checks = all(r.get("check_dec_equals_sampling", True) and r.get("check_ref_reproduces", True)
                     and r.get("check_r100_same_as_exact", True) for r in roots)
        r100 = all(r["conds"]["r100"]["states_bit_equal"] for r in roots if "r100" in r["conds"])
        mode = d["meta"].get("mode", "same tokens")
        print(f"\n### {L} tokens, {len(roots)} roots, {mode} (harness checks {'ok' if checks else 'FAILED'}; "
              f"r100 bit-equal states {'yes' if r100 else 'NO'})\n")
        print("| Condition | k (tokens) | tf mean abs | tf KL | dec mean abs | dec KL | dec >0.2 | "
              "dec argmax agree | dec first/last |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for n in names:
            cs = [r["conds"][n] for r in roots]
            k = sum(c["k"] for c in cs) / len(cs)

            def avg(part, key):
                if part not in cs[0]:
                    return None
                return sum(c[part][key] for c in cs) / len(cs)
            tf_abs, tf_kl = avg("tf", "mean_abs"), avg("tf", "kl_mean")
            f = (lambda x: "–" if x is None else f"{x:.4f}")
            print(f"| {n} | {k:.0f} | {f(tf_abs)} | {f(tf_kl)} | {f(avg('dec', 'mean_abs'))} | "
                  f"{f(avg('dec', 'kl_mean'))} | {avg('dec', 'frac_over_0.2'):.3f} | "
                  f"{avg('dec', 'argmax_agree'):.3f} | {avg('dec', 'first_mean'):.3f}/{avg('dec', 'last_mean'):.3f} |")
            rows.append((L, n, tf_abs, avg("dec", "mean_abs")))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-len", type=int)
    ap.add_argument("--roots", type=int, default=8)
    ap.add_argument("--seg", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--rollout", action="store_true", help="sample from each rebuilt state")
    ap.add_argument("--summarize", nargs="+")
    a = ap.parse_args()
    if a.tiny:
        return tiny_selftest()
    if a.summarize:
        return summarize(a.summarize)

    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = corpus(tok)
    L = a.root_len
    span = len(ids) - L - a.seg
    assert span > 0, (len(ids), L)

    kw = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    try:
        model = AutoModelForImageTextToText.from_pretrained(a.model, device_map="cuda", **kw)
        if hasattr(model.model, "visual"):
            del model.model.visual
    except (ImportError, ValueError) as e:  # device_map may need accelerate
        print(f"device_map load failed ({e}); loading on the CPU", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
        if hasattr(model.model, "visual"):
            del model.model.visual
        model.to("cuda")
    model.eval()
    torch.cuda.empty_cache()
    print(f"model on GPU: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    R = Runner(model, chunk=a.chunk, device="cuda")
    meta = {"root_len": L, "roots": a.roots, "seg": a.seg, "chunk": a.chunk, "model": a.model,
            "dtype": "bfloat16", "attn": "sdpa", "fractions": FRACTIONS,
            "mode": "rollout" if a.rollout else "same tokens",
            "transformers": transformers.__version__, "torch": torch.__version__,
            "gdn_chunk_impl": f"{mq.torch_chunk_gated_delta_rule.__module__}.{mq.torch_chunk_gated_delta_rule.__name__}",
            "gdn_recurrent_impl": f"{mq.torch_recurrent_gated_delta_rule.__module__}.{mq.torch_recurrent_gated_delta_rule.__name__}",
            "conv_impl": f"{mq.causal_conv1d_fn.__module__}.{mq.causal_conv1d_fn.__name__}",
            "groups": R.groups, "full_attention_layers": R.fa, "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0), "corpus_tokens": len(ids)}
    print(json.dumps({k: v for k, v in meta.items() if k != "groups"}), flush=True)
    res = {"meta": meta, "roots": []}
    for k in range(a.roots):
        off = (k * span) // max(a.roots - 1, 1)
        print(f"root {k} off {off}", flush=True)
        t0 = time.time()
        run = run_root_rollout if a.rollout else run_root
        r = run(R, ids[off:off + L], a.seg, seed=1000 + k, fractions=FRACTIONS)
        r["off"] = off
        r["seconds"] = time.time() - t0
        r["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        res["roots"].append(r)
        checks = " ".join(f"{c[6:]}={r[c]}" for c in r if c.startswith("check_"))
        print(f"  {checks} r100 bit-equal={r['conds']['r100']['states_bit_equal']}  {r['seconds']:.0f} s "
              f"peak {r['peak_gb']:.1f} GB", flush=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
