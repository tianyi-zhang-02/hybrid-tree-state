#!/usr/bin/env bash
# Split control for cache_fidelity.py: chunk size equal to (or dividing) the root
# length, so a cold recompute splits where the cached end state sits.
#   8,000-token roots with chunks of 8,000; 32,064-token roots with chunks of 10,688.
set -uo pipefail
cd ~/hybrid-tree-state
OUT=${OUT:-runs/fidelity}
PY=~/.venvs/sglang0523/bin/python
export HF_HUB_OFFLINE=1
PAT='^[^ ]*python -m sglang\.launch_server'
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/split.log"; }
stop_server() {
  pkill -f "$PAT" || true
  for _ in $(seq 60); do pgrep -f "$PAT" >/dev/null || return 0; sleep 2; done
  pkill -9 -f "$PAT" || true; sleep 5
}
start_server() {
  local tag=$1; shift
  env "$@" CONTEXT_LEN=36864 MAX_RUNNING=8 setsid nohup bash scripts/serve_sglang.sh \
    extra_buffer 63 > "$OUT/serve-$tag.log" 2>&1 < /dev/null &
  for _ in $(seq 120); do
    curl -s -m 5 http://127.0.0.1:30000/get_server_info >/dev/null && { log "server $tag up"; return 0; }
    sleep 5
  done
  log "server $tag did not come up"; return 1
}
rc=0
for pair in "8000 8000" "32064 10688"; do
  set -- $pair; L=$1; C=$2
  stop_server
  if start_server "c$C" CHUNKED="$C"; then
    log "split control r$L chunk $C"
    "$PY" scripts/cache_fidelity.py --root-len "$L" --out "$OUT/split-r$L.json" \
      --split-control-from "$OUT/r$L.json" 2>&1 | grep -v -i warn | tee -a "$OUT/split.log" || rc=1
  else
    rc=1
  fi
done
stop_server
log "split exit $rc"
echo "$rc" > "$OUT/split.done"
