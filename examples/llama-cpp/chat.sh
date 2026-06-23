#!/usr/bin/env zsh
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-models/gemma-3-4b-it-Q8_0.gguf}"
TEMPLATE="${TEMPLATE:-examples/llama-cpp/chat-template.jinja}"
CTX_SIZE="${CTX_SIZE:-4096}"
TEMP="${TEMP:-0.9}"
TOP_K="${TOP_K:-40}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:--1}"
N_PREDICT="${N_PREDICT:-512}"
GPU_LAYERS="${GPU_LAYERS:-999}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

if ! command -v llama-cli >/dev/null 2>&1; then
  echo "llama-cli not found on PATH" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  exit 1
fi

if [[ ! -f "$TEMPLATE" ]]; then
  echo "Chat template not found: $TEMPLATE" >&2
  exit 1
fi

args=(
  -m "$MODEL"
  -ngl "$GPU_LAYERS"
  -c "$CTX_SIZE"
  --temp "$TEMP"
  --top-k "$TOP_K"
  --top-p "$TOP_P"
  --seed "$SEED"
  --chat-template-file "$TEMPLATE"
  --color on
  --no-display-prompt
  -n "$N_PREDICT"
)

if [[ -n "$SYSTEM_PROMPT" ]]; then
  args+=(-sys "$SYSTEM_PROMPT")
fi

exec llama-cli "${args[@]}"
