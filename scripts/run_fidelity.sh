#!/usr/bin/env bash
# Cached vs recomputed logprobs (scripts/cache_fidelity.py), then the chunk-size
# control. Starts and stops its own servers. Timing is not measured, so there is
# no idle-desktop gate.
#   1. default chunked prefill (4096): main runs at 8,000 and 32,064 tokens, 8 roots each
#   2. chunked prefill 2048: recompute the same cold scores, no cache
set -uo pipefail
cd ~/hybrid-tree-state
OUT=${OUT:-runs/fidelity}
ROOTS=${ROOTS:-8}
mkdir -p "$OUT"
PY=~/.venvs/sglang0523/bin/python
export HF_HUB_OFFLINE=1
PAT='^[^ ]*python -m sglang\.launch_server'

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/runner.log"; }

stop_server() {
  pkill -f "$PAT" || true
  for _ in $(seq 60); do pgrep -f "$PAT" >/dev/null || return 0; sleep 2; done
  pkill -9 -f "$PAT" || true
  sleep 5
}

start_server() {
  local tag=$1; shift
  env "$@" CONTEXT_LEN=36864 MAX_RUNNING=8 setsid nohup bash scripts/serve_sglang.sh \
    extra_buffer 63 > "$OUT/serve-$tag.log" 2>&1 < /dev/null &
  for _ in $(seq 120); do
    if curl -s -m 5 http://127.0.0.1:30000/get_server_info >/dev/null; then
      log "server $tag up"; return 0
    fi
    sleep 5
  done
  log "server $tag did not come up"; return 1
}

rc=0
stop_server
if start_server c4096; then
  for L in 8000 32064; do
    log "main r$L"
    "$PY" scripts/cache_fidelity.py --root-len "$L" --roots "$ROOTS" --out "$OUT/r$L.json" \
      2>&1 | grep -v -i warn | tee -a "$OUT/runner.log" || rc=1
  done
else
  rc=1
fi
stop_server
if [[ $rc == 0 ]] && start_server c2048 CHUNKED=2048; then
  for L in 8000 32064; do
    log "control r$L"
    "$PY" scripts/cache_fidelity.py --root-len "$L" --out "$OUT/control-r$L.json" \
      --control-from "$OUT/r$L.json" 2>&1 | grep -v -i warn | tee -a "$OUT/runner.log" || rc=1
  done
else
  rc=1
fi
stop_server
log "runner exit $rc"
echo "$rc" > "$OUT/runner.done"
