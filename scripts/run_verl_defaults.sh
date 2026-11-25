#!/usr/bin/env bash
# Check at verl's rollout defaults on one 80 GB GPU (runs on the Runpod pod).
#
# Server: SGLang 0.5.23, Qwen3.5-9B in bf16, mem_fraction_static 0.5 and
# max_running_requests 256 (verl's gpu_memory_utilization and max_num_seqs), the
# session radix cache on, and everything else, including the Mamba pool size, left
# at SGLang's defaults. If the server will not start with 256, it is restarted
# without the flag and the engine's own limit is recorded.
#
# Runs, in priority order so that a time cap still leaves complete seeds:
#   smoke      GRPO step with 16 prompts, p0 and hold, to see sessions work
#   per seed   GRPO step (128 prompts x 8 samples, 1k-16k prompts, 512 new tokens):
#              p0, warm, hold, warm --window 32, hold --window 32;
#              forest of 16 trees with 8k roots: p0, p0root
# No run starts after LIMIT_MIN minutes. Rerunning skips runs whose result exists.
#
# Usage on the pod:  setsid nohup bash scripts/run_verl_defaults.sh < /dev/null &
# MOCK=1 runs the same sweep against scripts/mock_server.py (no GPU), for checking.
# Writes $OUT/runner.done holding the exit status when it ends.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/verl-defaults}
MODEL=${MODEL:-/models/Qwen3.5-9B}
PY=${PY:-python3}
LIMIT_MIN=${LIMIT_MIN:-110}
SEEDS=${SEEDS:-"0 1 2"}
GRPO_PROMPTS=${GRPO_PROMPTS:-128}
GEN_LEN=${GEN_LEN:-512}
TREES=${TREES:-16}
MOCK=${MOCK:-}
BASE=http://127.0.0.1:30000
mkdir -p "$OUT"
rm -f "$OUT/runner.done"
T0=$(date +%s)
SERVER=

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }
minutes() { echo $(( ($(date +%s) - T0) / 60 )); }
up() { curl -s -m 3 -o /dev/null -w '%{http_code}' "$BASE/get_server_info" | grep -q 200; }

start_server() {  # extra server flags
  if [ -n "$MOCK" ]; then
    "$PY" scripts/mock_server.py 30000 --slots 120 --max-running 24 > "$OUT/serve.log" 2>&1 &
  else
    "$PY" -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port 30000 \
      --dtype bfloat16 --mem-fraction-static 0.5 --context-length 36864 \
      --enable-session-radix-cache --enable-metrics --random-seed 0 "$@" \
      > "$OUT/serve.log" 2>&1 &
  fi
  SERVER=$!
  for _ in $(seq 1 120); do
    up && return 0
    kill -0 "$SERVER" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}
stop_server() {
  [ -n "$SERVER" ] || return 0
  kill "$SERVER" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$SERVER" 2>/dev/null || break; sleep 1; done
  kill -9 "$SERVER" 2>/dev/null
  SERVER=
}
GPUMON=
finish() { local rc=$?; [ -n "$GPUMON" ] && kill "$GPUMON" 2>/dev/null; stop_server; log "runner exit $rc after $(minutes) min"; echo "$rc" > "$OUT/runner.done"; }
trap finish EXIT
trap 'exit 143' TERM INT HUP

# 1. Server at verl's defaults; fall back to the engine's own running limit.
if start_server --max-running-requests 256; then
  log "server up with --max-running-requests 256"
else
  log "server did not start with --max-running-requests 256; see serve-256.log. Retrying without it"
  cp "$OUT/serve.log" "$OUT/serve-256.log"
  stop_server
  start_server || { log "server did not start; see serve.log"; exit 2; }
  log "server up with the engine's own running limit"
fi
curl -s "$BASE/get_server_info" > "$OUT/server_info.json"
[ -z "$MOCK" ] && grep -o -E 'max_total_num_tokens=[0-9]+|max_mamba_cache_size[=:] ?[0-9]+|max_running_requests[=:] ?[0-9]+|#tokens: [0-9]+' \
  "$OUT/serve.log" | sort -u | tr '\n' ' ' | sed 's/^/resolved: /' | tee -a "$OUT/runner.log"; echo
"$PY" scripts/probe_capacity.py --base "$BASE" --n 600 --check 32 --out "$OUT/probe.json" 2>&1 | tail -2 | tee -a "$OUT/runner.log"
if [ -z "$MOCK" ]; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 5 > "$OUT/gpu.csv" 2>/dev/null &
  GPUMON=$!
fi

run() {  # label, then the command
  local label=$1; shift
  if [ -s "$OUT/$label.json" ]; then log "skip $label (done before)"; return 0; fi
  if [ "$(minutes)" -ge "$LIMIT_MIN" ]; then log "time cap: not starting $label"; return 1; fi
  log "start $label"
  "$@" --label "$label" --out "$OUT/$label.json" 2>&1 | tail -3 | tee -a "$OUT/runner.log"
  log "end $label (exit ${PIPESTATUS[0]})"
}
grpo() {  # seed policy window
  local tag="g${GRPO_PROMPTS}-s$1-$2"; [ "$3" -gt 0 ] && tag="$tag-w$3"
  run "$tag" "$PY" scripts/replay_grpo.py --base "$BASE" --prompts "$GRPO_PROMPTS" --n 8 \
    --gen-len "$GEN_LEN" --policy "$2" --window "$3" --seed "$1"
}
forest() {  # seed policy
  run "f$TREES-r8000-s$1-$2" "$PY" scripts/replay_forest.py --base "$BASE" --trees "$TREES" \
    --root-len 8000 --policy "$2" --seed "$1"
}

# 2. Smoke: sessions must keep the prompt; if hold is no better than p0 here, stop.
run smoke-p0 "$PY" scripts/replay_grpo.py --base "$BASE" --prompts 16 --n 8 --gen-len 128 --policy p0 --seed 9
run smoke-hold "$PY" scripts/replay_grpo.py --base "$BASE" --prompts 16 --n 8 --gen-len 128 --policy hold --seed 9

# 3. The sweep. The policy order rotates with the seed.
POLS=("p0 0" "warm 0" "hold 0" "warm 32" "hold 32")
for seed in $SEEDS; do
  for k in 0 1 2 3 4; do
    set -- ${POLS[$(((k + seed) % 5))]}
    grpo "$seed" "$1" "$2" || break 2
  done
  if [ $((seed % 2)) -eq 0 ]; then
    forest "$seed" p0 || break; forest "$seed" p0root || break
  else
    forest "$seed" p0root || break; forest "$seed" p0 || break
  fi
done
log "sweep done"
