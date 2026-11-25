#!/usr/bin/env bash
# Third check (2026-11-25), on a pod with 2 A100 80 GB, TP 2, verl's memory settings.
# 1. Why do p0 samples miss? Same p0 step as run_verl_followup.sh, now with SGLang's
#    server-side forward_entry_time and prefill_finished_time kept per request
#    (replay_tree.stream_generate), so concurrent prefill and eviction can be told apart.
# 2. Does --schedule-policy lpm (SGLang's in-batch prefix check) remove the misses?
#
#   fcfs server -> runs/verl-timing-fcfs: g128 p0 seed 0, then seed 1
#   lpm server  -> runs/verl-timing-lpm:  g128 p0 seed 0
#
# Ready means the server logged "The server is fired up and ready to roll!" (after its
# own warm-up request) and /flush_cache answers 200.
# No run starts after LIMIT_MIN minutes. MOCK=1 uses scripts/mock_server.py.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=${ROOT:-runs}
MODEL=${MODEL:-/models/Qwen3.5-9B}
PY=${PY:-python3}
LIMIT_MIN=${LIMIT_MIN:-30}
MOCK=${MOCK:-}
GEN_LEN=${GEN_LEN:-512}
LOG=$ROOT/verl-check3.log
mkdir -p "$ROOT"
rm -f "$ROOT/verl-check3.done"
ulimit -n 65536 2>/dev/null || ulimit -n 10240
T0=$(date +%s)
SERVER_PID=

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
minutes() { echo $(( ($(date +%s) - T0) / 60 )); }
flushed() { curl -s -m 5 -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$1/flush_cache" | grep -q 200; }

serve() {  # out port, then extra server flags
  local out=$1 port=$2; shift 2
  mkdir -p "$out"
  if [ -n "$MOCK" ]; then
    "$PY" scripts/mock_server.py "$port" --slots 120 --max-running 24 > "$out/serve.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 60); do flushed "$port" && break; sleep 1; done
    echo "The server is fired up and ready to roll! (mock)" >> "$out/serve.log"
  else
    "$PY" -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port "$port" \
      --dtype bfloat16 --tp 2 --mem-fraction-static 0.5 --max-running-requests 256 \
      --context-length 36864 --enable-session-radix-cache --enable-metrics --random-seed 0 \
      "$@" > "$out/serve.log" 2>&1 &
    SERVER_PID=$!
  fi
  for _ in $(seq 1 180); do
    grep -q "The server is fired up and ready to roll" "$out/serve.log" && flushed "$port" && break
    kill -0 "$SERVER_PID" 2>/dev/null || { log "server for $out died; see $out/serve.log"; return 1; }
    sleep 5
  done
  flushed "$port" || { log "server for $out not ready after 15 min"; return 1; }
  curl -s "http://127.0.0.1:$port/get_server_info" > "$out/server_info.json"
  log "$out up: $( { grep -o -E 'max_running_requests is capped to [0-9]+|max_mamba_cache_size: [0-9]+|schedule_policy.: .[a-z-]+' "$out/serve.log" || true; } | sort -u | tr '\n' ' ')"
}
unserve() {
  [ -n "$SERVER_PID" ] || return 0
  kill "$SERVER_PID" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SERVER_PID" 2>/dev/null
  SERVER_PID=
}
finish() { local rc=$?; unserve; log "runner exit $rc after $(minutes) min"; echo "$rc" > "$ROOT/verl-check3.done"; }
trap finish EXIT
trap 'exit 143' TERM INT HUP

p0() {  # out port seed
  local label="g128-s$3-p0"
  if [ -s "$1/$label.json" ]; then log "skip $1/$label"; return 0; fi
  if [ "$(minutes)" -ge "$LIMIT_MIN" ]; then log "time cap: not starting $1/$label"; return 1; fi
  log "start $1/$label"
  "$PY" scripts/replay_grpo.py --base "http://127.0.0.1:$2" --prompts 128 --n 8 \
    --gen-len "$GEN_LEN" --policy p0 --seed "$3" --label "$label" --out "$1/$label.json" \
    2>&1 | tail -2 >> "$LOG"
  log "end $1/$label (exit ${PIPESTATUS[0]})"
}

F=$ROOT/verl-timing-fcfs
L=$ROOT/verl-timing-lpm
if serve "$F" 30000; then p0 "$F" 30000 0; fi
unserve
if serve "$L" 30000 --schedule-policy lpm; then p0 "$L" 30000 0; fi
unserve
if serve "$F" 30000; then p0 "$F" 30000 1; fi
log "plan done"
