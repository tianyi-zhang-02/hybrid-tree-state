#!/usr/bin/env bash
# T3 (agent search with long tool outputs) on the 4090 (WSL): one tree per run under
# P0, P0-root, P2-sel and the oracle.
#
# Waits until the Windows desktop is idle, then for each budget starts a server,
# probes its state capacity and runs every tree. The first run is a check: if it
# reports a problem, the sweep stops. GPU use and the desktop apps that would
# disturb timing are logged throughout.
#
#   strategy:  extra_buffer, 4 slots per running request, 8 running requests
#   budgets:   8, 16 and 31 cached states; P2-sel and the oracle hold up to 1, 9
#              and 24 nodes besides the root. Every policy runs at every budget.
#   root:      4,160 tokens
#   seeds:     0, 1, 2 (shape t3-s{seed}; root and tool-output ids from the seed)
#
# Usage: setsid nohup bash scripts/run_t3.sh [max_wait_minutes] < /dev/null &
# Writes runs/t3/runner.done (holding the exit status) when it ends.
# Rerunning resumes: runs whose result file exists are skipped.
# DRY_RUN=1 ROOT_DIR=<project> prints the commands instead of running anything.
set -uo pipefail
DRY=${DRY_RUN:-}
cd "${ROOT_DIR:-$HOME/hybrid-tree-state}"
OUT=runs/t3
mkdir -p "$OUT" logs
echo $$ > logs/runner.pid
rm -f "$OUT/runner.done"

MAX_WAIT=${1:-480}
RUNNING=8
TARGETS=(8 16 31)
ROOTS=(4160)
SEEDS=(0 1 2)
PY=~/.venvs/sglang0523/bin/python
TASKLIST=/mnt/c/Windows/System32/tasklist.exe
# Windows process names (a regex over tasklist output) that load the GPU and
# block the start, and names that are only logged. Empty means none.
GATE_RE=${GATE_RE:-}
WATCH_RE=${WATCH_RE:-}
IDLE_UTIL=15

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }
apps_matching() { [ -n "$1" ] || return 0; timeout 30 "$TASKLIST" /FO CSV /NH 2>/dev/null | grep -i -E -o "$1" | sort -u | tr '\n' ' '; }
gpu_util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' '; }
# Other model jobs on this machine (another session's vLLM or SGLang runs). Two
# models do not fit in 24 GB, so the sweep waits until none is running.
other_jobs() { pgrep -f '\.venvs/[^/]+/bin/python' | grep -v -x "$$" | wc -l | tr -d ' '; }
# A note for other sessions on this machine while the sweep holds the GPU.
BUSY=$HOME/GPU_BUSY
OWN_BUSY=0

run() { if [ -n "$DRY" ]; then echo "DRY: $*"; return 0; fi; "$@"; }

server_pid() { pgrep -f '^[^ ]*python -m sglang\.launch_server' | head -1; }
stop_server() {
  [ -n "$DRY" ] && { echo "DRY: stop server"; return 0; }
  local p; p=$(server_pid)
  [ -n "$p" ] || return 0
  kill "$p"
  for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
  sleep 5
}
start_server() {  # name strategy mamba_slots
  local name=$1 strategy=$2 slots=$3
  [ -n "$DRY" ] && { echo "DRY: start server $name: strategy=$strategy slots=$slots running=$RUNNING"; return 0; }
  CONTEXT_LEN=36864 MAX_RUNNING=$RUNNING setsid nohup bash scripts/serve_sglang.sh \
    "$strategy" "$slots" > "$OUT/serve-$name.log" 2>&1 < /dev/null &
  for i in $(seq 1 90); do
    if [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health)" = 200 ]; then
      log "server $name up after $((i * 5)) s: $(grep -o -E '#tokens: [0-9]+|max_mamba_cache_size: [0-9]+|max_running_requests=[0-9]+' "$OUT/serve-$name.log" | sort -u | tr '\n' ' ')"
      return 0
    fi
    sleep 5
  done
  log "server $name did not come up; see $OUT/serve-$name.log"
  return 1
}

MON=()
finish() {
  local rc=$?
  for p in ${MON[@]+"${MON[@]}"}; do kill "$p" 2>/dev/null; done
  stop_server
  [ "$OWN_BUSY" -eq 1 ] && rm -f "$BUSY"
  log "runner exit $rc"
  echo "$rc" > "$OUT/runner.done"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP

# 1. Wait for an idle desktop: no gating app open and GPU use under IDLE_UTIL
#    percent, three checks in a row, one minute apart.
log "waiting for an idle desktop (up to $MAX_WAIT min)"
quiet=0
[ -n "$DRY" ] && quiet=3 && MAX_WAIT=0
for ((m = 0; m < MAX_WAIT; m++)); do
  apps=$(apps_matching "$GATE_RE"); util=$(gpu_util); jobs_=$(other_jobs)
  [ -e "$BUSY" ] && jobs_=$((jobs_ + 1))  # another job announced itself
  if [ -z "$apps" ] && [ "$util" -lt "$IDLE_UTIL" ] && [ "$jobs_" -eq 0 ]; then
    quiet=$((quiet + 1)); else quiet=0; fi
  [ $((m % 15)) -eq 0 ] && log "check: util=${util}% apps=[${apps}] other model jobs=$jobs_ quiet=$quiet"
  [ "$quiet" -ge 3 ] && break
  sleep 60
done
if [ "$quiet" -lt 3 ]; then
  log "desktop not idle within $MAX_WAIT min; nothing run"
  exit 2
fi
log "desktop idle; starting"
if [ -z "$DRY" ]; then
  echo "$(date '+%F %T') T3 sweep (~/hybrid-tree-state/scripts/run_t3.sh, pid $$) is using the GPU for about 90 minutes. Please do not start model jobs until this file is gone." > "$BUSY"
  OWN_BUSY=1
fi

# 2. Monitors for the whole timed part.
[ -z "$DRY" ] && nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 2 \
  > "$OUT/gpu.csv" 2>/dev/null &
[ -z "$DRY" ] && MON+=($!)
[ -z "$DRY" ] && ( while true; do echo "$(date +%T) [$(apps_matching "$WATCH_RE")] vllm_jobs=$(pgrep -f '\.venvs/vllm[^/]*/bin/python' | wc -l | tr -d ' ')"; sleep 30; done ) > "$OUT/desktop.log" &
[ -z "$DRY" ] && MON+=($!)

# 3. The sweep.
first=1
failed=0
per_req=4
stop_server
for target in "${TARGETS[@]}"; do
  slots=$((per_req * RUNNING + target))
  name="extra_buffer-b$target"
  pols=(p0 p0root p2sel poracle)
  start_server "$name" extra_buffer "$slots" || exit 1
  run "$PY" scripts/probe_capacity.py --out "$OUT/probe-$name.json" 2>&1 | tee -a "$OUT/runner.log"
  for seed in "${SEEDS[@]}"; do
    # Rotate the policy order by seed so drift does not favour one policy.
    n=${#pols[@]}; order=()
    for ((k = 0; k < n; k++)); do order+=("${pols[$(((k + seed) % n))]}"); done
    for root in "${ROOTS[@]}"; do
      for pol in "${order[@]}"; do
        label="t3-s$seed-b$target-$pol"
        # Resume: a run whose result file exists was finished; do not repeat it.
        if [ -s "$OUT/$label.json" ]; then log "skip $label (done before)"; first=0; continue; fi
        run "$PY" scripts/replay_t3.py --shape "shapes/t3-s$seed.json" --root-len "$root" \
          --policy "$pol" --budget "$target" --seed "$seed" --label "$label" \
          --out "$OUT/$label.json" 2>&1 | tee -a "$OUT/runner.log"
        rc=${PIPESTATUS[0]}
        if [ "$rc" -ne 0 ]; then
          failed=$((failed + 1))
          log "run $label exit $rc"
          if [ "$first" -eq 1 ]; then log "first run failed; stopping the sweep"; exit 3; fi
        fi
        first=0
      done
    done
  done
  stop_server
done
log "sweep done; $failed runs reported problems"
