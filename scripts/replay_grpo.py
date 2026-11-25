"""One GRPO-style rollout step against one SGLang server, as verl sends it.

P prompts, n samples each. Every sample is its own /generate request, as in verl's
agent loop, and all of them are sent at once in prompt-major order after one
/flush_cache. Prompts are random token ids with lengths drawn log-uniformly from
[--min-len, --max-len], seeded; every sample decodes --gen-len tokens.

Policies:
  p0    send the n samples of every prompt at once; the engine's LRU decides.
  warm  first prefill each prompt alone (1 new token), then send its n samples.
  hold  as warm, but the prefill carries a session id, so the prompt's state is
        protected until the session closes. It closes when the last sample of the
        prompt returns its first token, by then every sample has matched the prompt.

--window W (warm and hold): at most W groups are prefilled whose samples have not
all started; the next group is prefilled when one of them has. 0 means no limit,
so every prompt is prefilled at the start. Holding more prompts than the state pool
keeps makes held states evict each other, so hold is run both ways.

Per request it records prompt and cached tokens. A sample's recompute is its prompt
length minus its cached tokens; with the prompt's state found, that is about 0. The
numbers that matter are the step time (first send to last token) and the prompt
tokens recomputed by samples.

Usage:
  replay_grpo.py --prompts 128 --n 8 --policy hold --seed 0 --label g128-hold-s0 \
      --out runs/h100/g128-hold-s0.json
"""

import argparse
import json
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from replay_tree import MetricsPoller, post, stream_generate

MISS_TOKENS = 512  # a sample that recomputes more than this did not find the prompt


def prompt_lengths(seed, p, lo, hi):
    rng = random.Random(f"grpo-len-{seed}")
    return [int(round(math.exp(rng.uniform(math.log(lo), math.log(hi))))) for _ in range(p)]


def prompt_ids(seed, i, length):
    rng = random.Random(f"grpo-ids-{seed}-{i}")
    return [rng.randrange(1000, 100000) for _ in range(length)]


def run(base, a):
    lens = prompt_lengths(a.seed, a.prompts, a.min_len, a.max_len)
    prompts = [prompt_ids(a.seed, i, L) for i, L in enumerate(lens)]
    records, problems = [], []
    lock = threading.Lock()
    started = [0] * a.prompts          # samples of prompt i that returned a first token
    closed = set()

    def close(i):
        with lock:
            if i in closed:
                return
            closed.add(i)
        try:
            post(base, "/close_session", {"session_id": f"{a.label}-g{i}"})
        except requests.RequestException as e:
            with lock:
                problems.append(f"close g{i}: {e}")

    window = threading.Semaphore(a.window) if a.window > 0 and a.policy != "p0" else None

    def on_first(i):
        fired = [False]

        def cb():
            with lock:
                if fired[0]:
                    return
                fired[0] = True
                started[i] += 1
                last = started[i] == a.n
            if last:
                if a.policy == "hold":
                    close(i)
                if window is not None:
                    window.release()
        return cb

    def sample(i, j):
        cb = on_first(i)
        try:
            r = stream_generate(base, prompts[i], a.gen_len, on_first=cb)
        except Exception as e:  # noqa: BLE001 - recorded, the run goes on
            cb()  # a failed sample must not block the window
            with lock:
                problems.append(f"g{i} s{j}: {e}")
            return
        r.pop("output_ids", None)
        r.update(kind="sample", prompt=i, sample=j, prompt_len=len(prompts[i]),
                 recomputed=len(prompts[i]) - (r["cached_tokens"] or 0))
        with lock:
            records.append(r)

    def group(i, pool):
        if a.policy in ("warm", "hold"):
            sid = f"{a.label}-g{i}" if a.policy == "hold" else None
            try:
                r = stream_generate(base, prompts[i], 1, session_id=sid)
                r.pop("output_ids", None)
                r.update(kind="prefill", prompt=i, prompt_len=len(prompts[i]),
                         recomputed=len(prompts[i]) - (r["cached_tokens"] or 0))
                with lock:
                    records.append(r)
            except Exception as e:  # noqa: BLE001
                with lock:
                    problems.append(f"prefill g{i}: {e}")
        return [pool.submit(sample, i, j) for j in range(a.n)]

    post(base, "/flush_cache")
    time.sleep(2)
    poller = MetricsPoller(base)
    poller.start()
    t0 = time.time()
    workers = a.prompts * (a.n + 1) + 8
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if a.policy == "p0":
            futs = []
            for i in range(a.prompts):
                futs += group(i, pool)
                time.sleep(0.001)  # keep prompt-major arrival order
        else:
            gfuts = []
            for i in range(a.prompts):
                if window is not None:
                    window.acquire()  # groups enter in prompt order
                gfuts.append(pool.submit(group, i, pool))
                time.sleep(0.001)
            futs = [f for g in gfuts for f in g.result()]
        for f in futs:
            f.result()
    t1 = time.time()
    poller.stop_flag.set()
    poller.join(timeout=5)
    if a.policy == "hold":
        for i in range(a.prompts):
            if i not in closed:
                problems.append(f"session g{i} still open at the end")
                close(i)

    samples = [r for r in records if r["kind"] == "sample"]
    pre = [r for r in records if r["kind"] == "prefill"]
    miss = [r for r in samples if r["recomputed"] > MISS_TOKENS]
    if len(samples) != a.prompts * a.n:
        problems.append(f"{len(samples)} samples finished of {a.prompts * a.n}")
    summary = {
        "step_s": t1 - t0,
        "prompt_tokens_unique": sum(lens),
        "sample_recomputed": sum(r["recomputed"] for r in samples),
        "prefill_recomputed": sum(r["recomputed"] for r in pre),
        "sample_misses": len(miss),
        "sample_miss_tokens": sum(r["recomputed"] for r in miss),
        "samples": len(samples),
        "decode_tokens": sum(r["completion_tokens"] or 0 for r in samples),
        "retractions": sum(r["num_retractions"] or 0 for r in records),
        "metric_peaks": poller.peaks(),
        "problems": problems,
    }
    return {"label": a.label, "args": vars(a), "prompt_lens": lens,
            "server_info": server_info(base), "summary": summary,
            "records": sorted(records, key=lambda r: r["client_send"]),
            "metrics": poller.samples}


def server_info(base):
    try:
        info = requests.get(base + "/get_server_info", timeout=30).json()
    except requests.RequestException:
        return {}
    keys = ("version", "max_total_num_tokens", "max_running_requests", "max_mamba_cache_size",
            "mamba_radix_cache_strategy", "mamba_track_interval", "chunked_prefill_size",
            "mem_fraction_static", "enable_session_radix_cache", "schedule_policy",
            "page_size", "context_length", "dtype")
    flat = dict(info)
    flat.update(info.get("internal_states", [{}])[0] if info.get("internal_states") else {})
    return {k: flat.get(k) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--prompts", type=int, default=128)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min-len", type=int, default=1024)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--gen-len", type=int, default=1024)
    ap.add_argument("--policy", choices=["p0", "warm", "hold"], required=True)
    ap.add_argument("--window", type=int, default=0,
                    help="warm/hold: groups prefilled but not yet fully started; 0 = no limit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run(a.base, a)
    with open(a.out, "w") as f:
        json.dump(res, f)
    s = res["summary"]
    print(f"{a.label}: step {s['step_s']:.1f} s, samples recomputed "
          f"{s['sample_recomputed'] / 1000:.0f}k of {s['prompt_tokens_unique'] / 1000:.0f}k unique "
          f"prompt tokens, misses {s['sample_misses']}, prefill recomputed "
          f"{s['prefill_recomputed'] / 1000:.0f}k, retractions {s['retractions']}, "
          f"problems {len(s['problems'])}")
    sys.exit(1 if s["problems"] else 0)


if __name__ == "__main__":
    main()
