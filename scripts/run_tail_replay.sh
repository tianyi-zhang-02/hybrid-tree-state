#!/usr/bin/env bash
# tail_replay_drift.py at 8,000 and 32,064 tokens, 8 roots each, one process per length.
# Not a timed run, but it needs most of the GPU's memory: it does not start while an
# SGLang or vLLM server is running.
#
# Usage: setsid nohup bash scripts/run_tail_replay.sh < /dev/null > /dev/null 2>&1 &
#   ROOTS=1 OUT=runs/tail-replay-smoke for a smoke test.
#   EXTRA=--rollout OUT=runs/tail-replay-rollout samples from each rebuilt state.
# Writes $OUT/runner.done (holding the exit status) when it ends.
set -uo pipefail
cd ~/hybrid-tree-state
OUT=${OUT:-runs/tail-replay}
mkdir -p "$OUT"
rm -f "$OUT/runner.done"
echo $$ > "$OUT/runner.pid"
export HF_HUB_OFFLINE=1
PY=~/.venvs/vllm034/bin/python
if pgrep -f 'sglang\.launch_server|vllm serve|vllm\.entrypoints' > /dev/null; then
  echo "[$(date '+%F %T')] a server is running; nothing started" | tee -a "$OUT/runner.log"
  echo 9 > "$OUT/runner.done"
  exit 9
fi
rc=0
for L in ${LENS:-8000 32064}; do
  echo "[$(date '+%F %T')] start $L" >> "$OUT/runner.log"
  "$PY" scripts/tail_replay_drift.py ${EXTRA:-} --root-len "$L" --roots "${ROOTS:-8}" \
    --out "$OUT/r$L.json" > "$OUT/r$L.log" 2>&1 || rc=$?
  echo "[$(date '+%F %T')] end $L rc=$rc" >> "$OUT/runner.log"
done
"$PY" scripts/tail_replay_drift.py --summarize "$OUT"/r*.json > "$OUT/summary.md" 2>&1
echo "$rc" > "$OUT/runner.done"
