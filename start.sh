#!/usr/bin/env bash
#
# Gemma's Test Kitchen — one-command launcher.
#
# Installs what's missing, checks the model is downloaded (guides you if not),
# builds the web UI, starts the server, and opens your browser. Safe to run again
# anytime — it skips steps that are already done.
#
#   ./start.sh
#
# Env: KITCHEN_PORT (default 8099), KITCHEN_HOST, KITCHEN_MODEL (it|base).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PORT="${KITCHEN_PORT:-8099}"
PY=python3

echo "🍳  Gemma's Test Kitchen — starting up"

# 1) Python environment ------------------------------------------------------
# Fast path: if the current python3 already has everything, just use it.
if python3 -c "import fastapi, uvicorn, torch, transformers" >/dev/null 2>&1; then
  echo "→  Python packages already present."
else
  echo "→  Setting up a Python virtual environment (.venv) — first run only…"
  [ -d .venv ] || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY=python
  python -m pip install -U pip >/dev/null
  echo "   installing packages (this can take a few minutes the first time)…"
  python -m pip install -r requirements.txt
fi

# 2) Model weights -----------------------------------------------------------
if [ ! -f models/gemma-3-4b-it-hf/model.safetensors.index.json ]; then
  cat <<'MSG'

⚠   Gemma's weights aren't downloaded yet.
    This is a one-time ~8 GB download from Hugging Face and needs a free account:

      1. Make a free account at  https://huggingface.co
      2. Accept the license at   https://huggingface.co/google/gemma-3-4b-it
      3. Log in:   hf auth login      (paste a token from huggingface.co/settings/tokens)

MSG
  read -r -p "Done with steps 1–3 and want to download now? [y/N] " ans
  if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
    scripts/download_weights.sh
  else
    echo "No problem — run ./start.sh again once you're ready."
    exit 0
  fi
fi

# 3) Web UI ------------------------------------------------------------------
if [ ! -f web/dist/index.html ]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "✗  The web UI needs Node.js (npm). Install Node 20+ from https://nodejs.org and re-run."
    exit 1
  fi
  echo "→  Building the web UI (first run)…"
  ( cd web && npm install && npm run build )
fi

# 4) Launch ------------------------------------------------------------------
URL="http://127.0.0.1:${PORT}"
echo
echo "→  Opening Gemma's Test Kitchen at ${URL}"
echo "   The first chat loads the model on your GPU (~20s). Press Ctrl+C to stop."
# Pop the browser once the server is up (macOS `open`; ignored elsewhere).
( sleep 3; command -v open >/dev/null 2>&1 && open "$URL" >/dev/null 2>&1 || true ) &
exec "$PY" steer_server.py
