"""cache_fidelity.py for vLLM 0.34.0: does a cached hybrid state change the numbers?

Same roots (same corpus windows), same conditions and the same two parts as the
SGLang version, run in-process with vLLM's offline API, one request at a time.

Differences from SGLang that matter here:
  - Prefix caching for hybrid models is vLLM's "align" mode: Mamba states are
    kept only at multiples of the Mamba block size, so "end" and "decode" land on
    the last block boundary before the root end or inside the parent's output.
    The cached length of every request is recorded.
  - Per-request seeds work, so conditions on one prompt draw the same random
    numbers, and the sampled tokens can be compared directly with the cold run:
    the first token that differs, and the logprob gap before it.
  - Prompt scoring reads the prefix cache only up to prompt_logprob_start, and only
    with skip_reading_prefix_cache=False; cold scoring resets the cache and skips it.

Usage:
  cache_fidelity_vllm.py --root-lens 8000 32064 --roots 8 --out-dir runs/fidelity-vllm
"""

import argparse
import json
import os
import sys

from cache_fidelity import SEG, corpus, diff, gen_vs_ref

MODEL = "Qwen/Qwen3.5-9B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-lens", type=int, nargs="+", required=True)
    ap.add_argument("--roots", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    a = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = {}
    if a.max_num_batched_tokens:
        kw["max_num_batched_tokens"] = a.max_num_batched_tokens
    llm = LLM(model=MODEL, quantization="fp8", max_model_len=36864,
              gpu_memory_utilization=0.80, enable_prefix_caching=True, seed=0,
              max_num_seqs=8, limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    vc = llm.llm_engine.vllm_config
    info = {"vllm": vllm.__version__,
            "mamba_cache_mode": vc.cache_config.mamba_cache_mode,
            "mamba_block_size": vc.cache_config.mamba_block_size,
            "block_size": vc.cache_config.block_size,
            "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
            "enable_chunked_prefill": vc.scheduler_config.enable_chunked_prefill,
            "quantization": vc.model_config.quantization,
            "mamba_cache_dtype": vc.cache_config.mamba_cache_dtype,
            "mamba_ssm_cache_dtype": vc.cache_config.mamba_ssm_cache_dtype}
    print("config", info, flush=True)

    def reset():
        assert llm.reset_prefix_cache(), "prefix cache reset failed"

    def run(ids, sp):
        (out,) = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        return out

    def generate(ids, seed, max_new=SEG):
        sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=0, max_tokens=max_new,
                            ignore_eos=True, seed=seed, logprobs=0)
        out = run(ids, sp)
        o = out.outputs[0]
        toks = list(o.token_ids)
        lps = [o.logprobs[i][t].logprob for i, t in enumerate(toks)]
        return {"prompt_len": len(ids), "cached": out.num_cached_tokens, "out": toks,
                "lp": lps}

    def prefill(ids):
        return run(ids, SamplingParams(max_tokens=1)).num_cached_tokens

    def score(prompt, cont, cold=True):
        if cold:
            reset()
        ids = prompt + cont
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0,
                            prompt_logprob_start=len(prompt) - 1,
                            skip_reading_prefix_cache=cold)
        out = run(ids, sp)
        if cold:
            assert out.num_cached_tokens == 0, out.num_cached_tokens
        pl = out.prompt_logprobs
        assert len(pl) == len(ids), (len(pl), len(ids))
        lps = []
        for j in range(len(prompt), len(ids)):
            assert pl[j] is not None and ids[j] in pl[j], f"no logprob for position {j}"
            lps.append(pl[j][ids[j]].logprob)
        return lps, out.num_cached_tokens

    def vs_cold(g, cold):
        first = next((i for i, (x, y) in enumerate(zip(g["out"], cold["out"])) if x != y),
                     len(g["out"]))
        d = [abs(x - y) for x, y in zip(g["lp"][:first], cold["lp"][:first])]
        return {"first_diff": first,
                "mean_abs_before": sum(d) / len(d) if d else None,
                "max_abs_before": max(d) if d else None,
                "exact_before": all(v == 0 for v in d)}

    from transformers import AutoTokenizer
    ids_all = corpus(AutoTokenizer.from_pretrained(MODEL))
    os.makedirs(a.out_dir, exist_ok=True)

    for L in a.root_lens:
        span = len(ids_all) - L - SEG
        roots = []
        for k in range(a.roots):
            off = (k * span) // max(a.roots - 1, 1)
            R = ids_all[off:off + L]
            N = ids_all[off + L:off + L + SEG]
            s1, s2, s3 = 1000 * k + 1, 1000 * k + 2, 1000 * k + 3
            g = {}
            reset()
            g["p1_cold"] = generate(R, s1)
            O = g["p1_cold"]["out"]
            P2 = R + O
            g["p2_warm_decode"] = generate(P2, s2)
            reset()
            g["p2_cold"] = generate(P2, s2)
            reset()
            g["p2_cold_again"] = generate(P2, s2)
            reset()
            prefill(R)
            g["p2_warm_end"] = generate(P2, s2)
            reset()
            prefill(R)
            g["p1_warm_chunk"] = generate(R, s1)
            reset()
            g["p1_cold_again"] = generate(R, s1)
            for name, x in g.items():
                prompt = R if name.startswith("p1") else P2
                x["ref"], _ = score(prompt, x["out"])
                x["gen_vs_ref"] = gen_vs_ref(x)
                cold = g["p1_cold"] if name.startswith("p1") else g["p2_cold"]
                x["vs_cold"] = vs_cold(x, cold)

            b = {}
            c1, _ = score(R, N)
            c1b, _ = score(R, N)
            reset()
            prefill(R)
            w1, cw1 = score(R, N, cold=False)
            c2, _ = score(P2, N)
            c2b, _ = score(P2, N)
            reset()
            prefill(R)
            w2, cw2 = score(P2, N, cold=False)
            reset()
            Ob = generate(R, s3)["out"]
            w3, cw3 = score(R + Ob, N, cold=False)
            c3, _ = score(R + Ob, N)
            b["noise_p1"] = {"cached": 0, **diff(c1b, c1)}
            b["noise_p2"] = {"cached": 0, **diff(c2b, c2)}
            b["chunk"] = {"cached": cw1, **diff(w1, c1)}
            b["end"] = {"cached": cw2, **diff(w2, c2)}
            b["decode"] = {"cached": cw3, **diff(w3, c3)}
            b["cold_lp"] = {"p1": c1, "p2": c2, "decode": c3}
            b["decode_parent_out"] = Ob

            roots.append({"root": k, "offset": off, "sampled": g, "teacher_forced": b})
            print(f"r{L} root {k} A: " + "; ".join(
                f"{n} cached {x['cached']} mean {x['gen_vs_ref']['mean_abs']:.4f} "
                f"max {x['gen_vs_ref']['max_abs']:.3f} first_diff {x['vs_cold']['first_diff']}"
                for n, x in g.items()), flush=True)
            print(f"r{L} root {k} B: " + "; ".join(
                f"{n} cached {x['cached']} mean {x['mean_abs']:.5f} max {x['max_abs']:.4f}"
                for n, x in b.items() if "mean_abs" in x), flush=True)
        out = {"engine": "vllm", "root_len": L, "roots": a.roots,
               "corpus_tokens": len(ids_all), "server_info": info, "results": roots}
        with open(os.path.join(a.out_dir, f"r{L}.json"), "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    sys.exit(main())
