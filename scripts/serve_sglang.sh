#!/usr/bin/env bash
# Start SGLang 0.5.23 for the hybrid tree-state smoke test.
# Usage: serve_sglang.sh <auto|extra_buffer|no_buffer> [mamba_slots]
# Optional env: CONTEXT_LEN (default 8192), MAX_RUNNING (engine default if unset),
# CHUNKED (chunked prefill size; engine default if unset).
set -euo pipefail

STRATEGY=${1:?strategy}
MAMBA_SLOTS=${2:-24}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
PORT=${PORT:-30000}
CONTEXT_LEN=${CONTEXT_LEN:-8192}

extra=()
if [[ "$STRATEGY" == "no_buffer" ]]; then
  extra+=(--disable-overlap-schedule)
fi
if [[ -n "${CHUNKED:-}" ]]; then
  extra+=(--chunked-prefill-size "$CHUNKED")
fi
if [[ -n "${MAX_RUNNING:-}" ]]; then
  extra+=(--max-running-requests "$MAX_RUNNING")
fi

VENV=~/.venvs/sglang0523
# No system CUDA toolkit on this host. SGLang's JIT kernels need ninja and nvcc;
# both come from the venv (ninja wheel, nvidia cu13 wheel with nvcc and headers).
export PATH="$VENV/bin:$PATH"
export CUDA_HOME="$VENV/lib/python3.11/site-packages/nvidia/cu13"
# The wheel ships only libcudart.so.13, so `-lcudart` finds nothing. Link
# against an unversioned symlink kept outside the venv; libcuda comes from WSL.
LINKDIR=~/hybrid-tree-state/cuda-link
mkdir -p "$LINKDIR"
ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$LINKDIR/libcudart.so"
export LIBRARY_PATH="$LINKDIR:/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
# The wheel's nvcc is 13.5 but its runtime headers are 13.0. FlashInfer's
# bundled CCCL refuses that pairing; kernels are built as sm_89 SASS only.
export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export HF_HUB_OFFLINE=1
exec "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --quantization fp8 \
  --context-length "$CONTEXT_LEN" \
  --mem-fraction-static 0.80 \
  --enable-session-radix-cache \
  --mamba-radix-cache-strategy "$STRATEGY" \
  --max-mamba-cache-size "$MAMBA_SLOTS" \
  --enable-metrics \
  --random-seed 0 \
  "${extra[@]}"
