#!/usr/bin/env bash
# Follow-up to run_verl_defaults.sh (2026-11-25), on a pod with 2 A100 80 GB.
# Question: are the misses at verl's memory settings fixed by verl's own TP 2, or
# by a larger state pool through SGLang flags, without any change to verl?
#
# Every server: SGLang 0.5.23, Qwen3.5-9B bf16, --mem-fraction-static 0.5,
# --max-running-requests 256, --context-length 36864, session radix cache on.
# Every run: one GRPO step (8 samples, 512 new tokens), p0 and hold --window 32.
#
#   phase 1  TP 2 on both GPUs, SGLang's default pool          -> runs/verl-tp2
#            128 prompts seeds 0 and 1, then 256 prompts seed 0
#   phase 2  TP 1, two servers at once, 128 prompts, seed 0:
#            GPU 0  --mamba-full-memory-ratio 4                 -> runs/verl-ratio4
#            GPU 1  --mamba-ssm-dtype bfloat16                  -> runs/verl-ssm-bf16
#
# A server counts as ready only when /flush_cache answers 200, i.e. after its own
# warm-up request (run_verl_defaults.sh lost its probe and smoke runs to that).
# No run starts after LIMIT_MIN minutes. Rerunning skips runs whose result exists.
# MOCK=1 runs the same plan against scripts/mock_server.py (no GPU).
# Usage on the pod:  setsid nohup bash scripts/run_verl_followup.sh < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=${ROOT:-runs}
MODEL=${MODEL:-/models/Qwen3.5-9B}
PY=${PY:-python3}
LIMIT_MIN=${LIMIT_MIN:-70}
MOCK=${MOCK:-}
GEN_LEN=${GEN_LEN:-512}
LOG=$ROOT/verl-followup.log
mkdir -p "$ROOT"
rm -f "$ROOT/verl-followup.done"
ulimit -n 65536 2>/dev/null || ulimit -n 10240
T0=$(date +%s)
PIDS=()

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
minutes() { echo $(( ($(date +%s) - T0) / 60 )); }
flushed() { curl -s -m 5 -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$1/flush_cache" | grep -q 200; }

serve() {  # out port gpus, then extra server flags; sets SERVER_PID
  local out=$1 port=$2 gpus=$3; shift 3
  mkdir -p "$out"
  if [ -n "$MOCK" ]; then
    "$PY" scripts/mock_server.py "$port" --slots 120 --max-running 24 > "$out/serve.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$gpus "$PY" -m sglang.launch_server --model-path "$MODEL" \
      --host 127.0.0.1 --port "$port" --dtype bfloat16 --mem-fraction-static 0.5 \
      --max-running-requests 256 --context-length 36864 --enable-session-radix-cache \
      --enable-metrics --random-seed 0 "$@" > "$out/serve.log" 2>&1 &
  fi
  SERVER_PID=$!
  PIDS+=("$SERVER_PID")
  for _ in $(seq 1 180); do
    flushed "$port" && break
    kill -0 "$SERVER_PID" 2>/dev/null || { log "server for $out died; see $out/serve.log"; return 1; }
    sleep 5
  done
  flushed "$port" || { log "server for $out not ready after 15 min"; return 1; }
  curl -s "http://127.0.0.1:$port/get_server_info" > "$out/server_info.json"
  log "$out up on port $port: $( { grep -o -E 'max_running_requests is capped to [0-9]+|max_mamba_cache_size: [0-9]+|max_total_num_tokens=[0-9]+' "$out/serve.log" || true; } | sort -u | tr '\n' ' ')"
}
unserve() {  # pid
  kill "$1" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$1" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$1" 2>/dev/null
}
finish() {
  local rc=$?
  for p in "${PIDS[@]}"; do unserve "$p"; done
  [ -n "${GPUMON:-}" ] && kill "$GPUMON" 2>/dev/null
  log "runner exit $rc after $(minutes) min"
  echo "$rc" > "$ROOT/verl-followup.done"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP

grpo() {  # out port prompts seed policy window
  local out=$1 port=$2 p=$3 seed=$4 pol=$5 w=$6
  local label="g$p-s$seed-$pol"; [ "$w" -gt 0 ] && label="$label-w$w"
  if [ -s "$out/$label.json" ]; then log "skip $out/$label (done before)"; return 0; fi
  if [ "$(minutes)" -ge "$LIMIT_MIN" ]; then log "time cap: not starting $out/$label"; return 1; fi
  log "start $out/$label"
  "$PY" scripts/replay_grpo.py --base "http://127.0.0.1:$port" --prompts "$p" --n 8 \
    --gen-len "$GEN_LEN" --policy "$pol" --window "$w" --seed "$seed" \
    --label "$label" --out "$out/$label.json" 2>&1 | tail -2 >> "$LOG"
  log "end $out/$label (exit ${PIPESTATUS[0]})"
}
pair() {  # out port prompts seed: p0 then hold, or the reverse for odd seeds
  if [ $(( $4 % 2 )) -eq 0 ]; then
    grpo "$1" "$2" "$3" "$4" p0 0 && grpo "$1" "$2" "$3" "$4" hold 32
  else
    grpo "$1" "$2" "$3" "$4" hold 32 && grpo "$1" "$2" "$3" "$4" p0 0
  fi
}

if [ -z "$MOCK" ]; then
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 5 \
    > "$ROOT/verl-followup-gpu.csv" 2>/dev/null &
  GPUMON=$!
fi

# Phase 1: TP 2, default pool.
A=$ROOT/verl-tp2
if serve "$A" 30000 0,1 --tp 2; then
  pair "$A" 30000 128 0 && pair "$A" 30000 128 1 && pair "$A" 30000 256 0
  unserve "$SERVER_PID"
fi

# Phase 2: TP 1, larger pool two ways, one server per GPU at the same time.
B=$ROOT/verl-ratio4
C=$ROOT/verl-ssm-bf16
( trap - EXIT; serve "$B" 30001 0 --mamba-full-memory-ratio 4 && pair "$B" 30001 128 0; unserve "$SERVER_PID" ) &
JB=$!
( trap - EXIT; serve "$C" 30002 1 --mamba-ssm-dtype bfloat16 && pair "$C" 30002 128 0; unserve "$SERVER_PID" ) &
JC=$!
wait "$JB" "$JC"
log "plan done"
