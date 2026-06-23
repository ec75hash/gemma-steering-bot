#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if ! command -v llama-cli >/dev/null 2>&1; then
  echo "llama-cli not found on PATH." >&2
  echo "Install llama.cpp, then make sure llama-cli is on PATH." >&2
  exit 1
fi

HF_REPO="${HF_REPO:-google/gemma-4-12B-it-qat-q4_0-gguf}"
HF_FILE="${HF_FILE:-gemma-4-12b-it-qat-q4_0.gguf}"
LOCAL_MODEL="${LOCAL_MODEL:-models/gemma-4-12B-it-qat-q4_0-gguf/${HF_FILE}}"

CTX_SIZE="${CTX_SIZE:-2048}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-128}"
TEMP="${TEMP:-0.7}"
TOP_K="${TOP_K:-64}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:--1}"
N_PREDICT="${N_PREDICT:-512}"
GPU_LAYERS="${GPU_LAYERS:-auto}"
REASONING="${REASONING:-off}"
MULTILINE="${MULTILINE:-0}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"

model_args=()
if [ -f "$LOCAL_MODEL" ]; then
  model_args=(-m "$LOCAL_MODEL")
else
  echo "Local GGUF not found: $LOCAL_MODEL" >&2
  echo "Using llama.cpp Hugging Face download: ${HF_REPO} / ${HF_FILE}" >&2
  model_args=(-hf "$HF_REPO" --hf-file "$HF_FILE" --no-mmproj)
fi

args=(
  "${model_args[@]}"
  -ngl "$GPU_LAYERS"
  -c "$CTX_SIZE"
  -b "$BATCH_SIZE"
  -ub "$UBATCH_SIZE"
  -n "$N_PREDICT"
  --temp "$TEMP"
  --top-k "$TOP_K"
  --top-p "$TOP_P"
  --seed "$SEED"
  --reasoning "$REASONING"
  -cnv
  --no-display-prompt
  --no-show-timings
)

if [ "$MULTILINE" = "1" ]; then
  args+=(-mli)
fi

if [ -t 0 ]; then
  args+=(--color on)
else
  args+=(--color off --simple-io)
fi

if [ -n "$SYSTEM_PROMPT" ]; then
  args+=(-sys "$SYSTEM_PROMPT")
fi

exec llama-cli "${args[@]}" "$@"
