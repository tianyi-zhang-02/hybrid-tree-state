#!/usr/bin/env bash
# cache_fidelity_vllm.py at 8,000 and 32,064 tokens, 8 roots each, one process.
set -uo pipefail
cd ~/hybrid-tree-state/scripts
OUT=${OUT:-../runs/fidelity-vllm}
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1
~/.venvs/vllm034/bin/python cache_fidelity_vllm.py --root-lens 8000 32064 --roots "${ROOTS:-8}" \
  --out-dir "$OUT" > "$OUT/out.log" 2>&1
rc=$?
echo "$rc" > "$OUT/runner.done"
