#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GEMMA_REPO="google/gemma-3-4b-it"
GEMMA_REV="093f9f388b31de276ce2de164bdc2081324b9767"
SAE_REPO="google/gemma-scope-2-4b-it"
SAE_REV="3e94b68be95290aada5b7525cf431d3040f81bb1"
BASE_REPO="google/gemma-3-4b-pt"
BASE_REV="cc012e0a6d0787b4adcc0fa2c4da74402494554d"
BASE_SAE_REPO="google/gemma-scope-2-4b-pt"
BASE_SAE_REV="a0ffd6132a985bc84077a66d1a1033e10b604fa8"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if command -v hf >/dev/null 2>&1; then
  HF=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli)
else
  echo "Hugging Face CLI not found. Install with:"
  echo '  python3 -m pip install -U "huggingface_hub[cli]"'
  exit 1
fi

mkdir -p models

echo "Downloading ${GEMMA_REPO} -> models/gemma-3-4b-it-hf"
echo "If this fails, log in to Hugging Face and accept the Gemma license terms first."
"${HF[@]}" download "$GEMMA_REPO" \
  --revision "$GEMMA_REV" \
  --local-dir models/gemma-3-4b-it-hf \
  --include "*.json" --include "*.model" --include "*.safetensors"

echo "Downloading selected GemmaScope 2 RES-16K SAE layers -> models/gemma-scope-2-4b-it"
"${HF[@]}" download "$SAE_REPO" \
  --revision "$SAE_REV" \
  --local-dir models/gemma-scope-2-4b-it \
  --include "resid_post/layer_9_width_16k_l0_medium/*" \
  --include "resid_post/layer_17_width_16k_l0_medium/*" \
  --include "resid_post/layer_22_width_16k_l0_medium/*" \
  --include "resid_post/layer_29_width_16k_l0_medium/*"

echo "Downloading ${BASE_REPO} -> models/gemma-3-4b-pt-hf (looking-glass base-model phase)"
echo "If this fails, log in to Hugging Face and accept the Gemma license terms first."
"${HF[@]}" download "$BASE_REPO" \
  --revision "$BASE_REV" \
  --local-dir models/gemma-3-4b-pt-hf \
  --include "*.json" --include "*.model" --include "*.safetensors"

echo "Downloading selected base GemmaScope 2 RES-16K SAE layers -> models/gemma-scope-2-4b-pt"
"${HF[@]}" download "$BASE_SAE_REPO" \
  --revision "$BASE_SAE_REV" \
  --local-dir models/gemma-scope-2-4b-pt \
  --include "resid_post/layer_9_width_16k_l0_medium/*" \
  --include "resid_post/layer_17_width_16k_l0_medium/*" \
  --include "resid_post/layer_22_width_16k_l0_medium/*" \
  --include "resid_post/layer_29_width_16k_l0_medium/*"

echo
echo "Done. Expected files:"
test -f models/gemma-3-4b-it-hf/model.safetensors.index.json
test -f models/gemma-scope-2-4b-it/resid_post/layer_9_width_16k_l0_medium/params.safetensors
test -f models/gemma-scope-2-4b-it/resid_post/layer_17_width_16k_l0_medium/params.safetensors
test -f models/gemma-scope-2-4b-it/resid_post/layer_22_width_16k_l0_medium/params.safetensors
test -f models/gemma-scope-2-4b-it/resid_post/layer_29_width_16k_l0_medium/params.safetensors
test -f models/gemma-3-4b-pt-hf/model.safetensors.index.json
test -f models/gemma-scope-2-4b-pt/resid_post/layer_9_width_16k_l0_medium/params.safetensors
test -f models/gemma-scope-2-4b-pt/resid_post/layer_17_width_16k_l0_medium/params.safetensors
test -f models/gemma-scope-2-4b-pt/resid_post/layer_22_width_16k_l0_medium/params.safetensors
test -f models/gemma-scope-2-4b-pt/resid_post/layer_29_width_16k_l0_medium/params.safetensors
echo "Weight layout verified."
