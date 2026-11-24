"""A fake SGLang server for checking the scripts' control flow without a GPU.

It imitates only the cache behaviour the scripts depend on, as observed on
SGLang 0.5.23 with extra_buffer:

  - a request matches the longest cached state that is a prefix of its prompt,
    shorter than the prompt;
  - each request keeps one state when it ends: the branch point if its prompt left
    an existing path past the matched prefix (at a multiple of 256), otherwise the
    last multiple of 256 it processed (prompt plus all outputs but the last);
  - the state pool has --slots slots. Each running request holds 4 and cached
    states take the rest. Eviction is LRU: states without a session reference go
    first, referenced ones only after those, and states matched by a running
    request are never evicted;
  - at most --max-running requests run; the rest wait in arrival order;
  - a request with a top-level session_id references the state it ends on;
    /close_session drops the session's reference and frees nothing;
  - /generate streams "data: {...}" chunks with cumulative output_ids when
    "stream" is true; /flush_cache drops everything.

Timings are fake: 20 us per new prompt token, 1 ms per decoded token.
"""

import argparse
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GRID = 256
PER_RUN = 4
LOCK = threading.Lock()
STATES = {}   # tuple(token ids) -> {"t": last use, "refs": set(session ids), "locks": int}
PATHS = []    # token sequences seen, for finding branch points
RUNNING = [0]
CFG = {"slots": 64, "max_running": 8}
SEM = None
EVICTIONS = [0]


def match(ids):
    best = 0
    for n in range(GRID, len(ids), GRID):
        if tuple(ids[:n]) in STATES:
            best = n
    return best


def branch_point(ids, prefix):
    best = None
    for p in PATHS:
        n = 0
        for a, b in zip(p, ids):
            if a != b:
                break
            n += 1
        n = n // GRID * GRID
        if prefix < n < len(ids) and n < len(p) and (best is None or n > best):
            best = n
    return best


def evict():
    """Evict until cached states plus running requests fit in the pool."""
    while len(STATES) + PER_RUN * RUNNING[0] > CFG["slots"]:
        free = [(v["t"], k) for k, v in STATES.items() if not v["refs"] and not v["locks"]]
        held = [(v["t"], k) for k, v in STATES.items() if v["refs"] and not v["locks"]]
        pool = free or held
        if not pool:
            return
        del STATES[min(pool)[1]]
        EVICTIONS[0] += 1


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, obj, ctype="application/json"):
        body = (json.dumps(obj) if ctype == "application/json" else obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/get_server_info":
            self.reply({"version": "mock", "context_length": 36864,
                        "max_total_num_tokens": 107074, "schedule_policy": "fcfs",
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "mamba_track_interval": GRID,
                        "max_mamba_cache_size": CFG["slots"],
                        "max_running_requests": CFG["max_running"],
                        "enable_session_radix_cache": True})
        elif self.path == "/metrics":
            with LOCK:
                used = len(STATES) + PER_RUN * RUNNING[0]
                text = (f"sglang:mamba_usage{{m=\"mock\"}} {used / CFG['slots']}\n"
                        f"sglang:num_running_reqs{{m=\"mock\"}} {RUNNING[0]}\n"
                        f"sglang:mock_evictions {EVICTIONS[0]}\n")
            self.reply(text, "text/plain")
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/flush_cache":
            with LOCK:
                STATES.clear()
                PATHS.clear()
            return self.reply("ok")
        if self.path == "/close_session":
            sid = req["session_id"]
            with LOCK:
                for v in STATES.values():
                    v["refs"].discard(sid)
            return self.reply("ok")
        if self.path != "/generate":
            return self.send_error(404)
        self.generate(req)

    def generate(self, req):
        ids = req["input_ids"]
        max_new = req["sampling_params"]["max_new_tokens"]
        sid = req.get("session_id")
        stream = req.get("stream", False)
        t0 = time.time()
        SEM.acquire()
        queue = time.time() - t0
        with LOCK:
            cached = match(ids)
            key = tuple(ids[:cached]) if cached else None
            if key:
                STATES[key]["t"] = time.time()
                STATES[key]["locks"] += 1
            bp = branch_point(ids, cached)
            RUNNING[0] += 1
            evict()
        out = []
        rng = random.Random()
        try:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
            time.sleep(20e-6 * (len(ids) - cached))
            ttft = None
            for _ in range(max_new):
                out.append(rng.randrange(1000, 100000))
                if ttft is None:
                    ttft = time.time() - t0
                if stream:
                    self.wfile.write(b"data: " + json.dumps(
                        {"output_ids": out, "meta_info": {}}).encode() + b"\n\n")
                    self.wfile.flush()
                time.sleep(1e-3)
        finally:
            full = list(ids) + out
            with LOCK:
                if key and key in STATES:
                    STATES[key]["locks"] -= 1
                keep = bp if bp is not None else (len(full) - 1) // GRID * GRID
                if keep:
                    k = tuple(full[:keep])
                    st = STATES.setdefault(k, {"t": 0, "refs": set(), "locks": 0})
                    st["t"] = time.time()
                    if sid is not None:
                        for v in STATES.values():  # a session references one leaf
                            v["refs"].discard(sid)
                        st["refs"].add(sid)
                PATHS.append(full)
                RUNNING[0] -= 1
                evict()
            SEM.release()
        t1 = time.time()
        meta = {"prompt_tokens": len(ids), "cached_tokens": cached,
                "completion_tokens": max_new, "e2e_latency": t1 - t0,
                "first_token_latency": ttft, "queue_time": queue,
                "num_retractions": 0, "finish_reason": {"type": "length"},
                "request_received_ts": t0, "request_finished_ts": t1}
        final = {"output_ids": out, "meta_info": meta}
        if stream:
            self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self.reply(final)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int, nargs="?", default=30123)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--max-running", type=int, default=8)
    a = ap.parse_args()
    CFG.update(slots=a.slots, max_running=a.max_running)
    SEM = threading.Semaphore(a.max_running)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
