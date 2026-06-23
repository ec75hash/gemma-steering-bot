# Gemma's Test Kitchen — web UI

A kitchen-themed web front-end that does **everything `chat_steer.py` does**: live SAE
steering of gemma-3-4b-it (gemma-scope), streaming chat, and next-token logit analysis.

## Architecture

```
web/ (React 19 + Vite + Tailwind v4 + GSAP)   ──/api──▶   steer_server.py (FastAPI)
                                                              └── steer_engine.py (SteerEngine)
                                                                    └── gemma-3-4b-it + gemma-scope SAEs (PyTorch / MPS)
```

- `steer_engine.py` — a side-effect-free port of `chat_steer.py`'s engine (catalogs,
  alias resolution, hook math, generation, logits) as a reusable `SteerEngine` class.
  `chat_steer.py` is untouched; the CLI still works.
- `steer_server.py` — FastAPI REST + SSE API over the engine; also serves the built
  `web/dist` in production.
- `web/` — the kitchen UI (relocated from the matrix reference app).

## Run

**Production (one process — API + UI on :8099):**
```bash
cd web && npm install && npm run build && cd ..
python3 steer_server.py            # http://127.0.0.1:8099
```

**Dev (hot-reload UI + API):**
```bash
python3 steer_server.py            # terminal 1 — API on :8099
cd web && npm run dev              # terminal 2 — UI on :5179 (proxies /api → :8099)
```

Open the UI, pick the model in **The Range**, click **Fire Up The Range** (loads ~8 GB
on MPS, ~20 s), then cook.

Env: `KITCHEN_PORT` (default 8099), `KITCHEN_HOST`, `KITCHEN_MODEL` (`it`|`base`),
`KITCHEN_PROMPT_MODE`. Dev proxy target: `KITCHEN_API` for the Vite server.

## What maps to what (chat_steer.py → UI)

| chat_steer.py | UI |
|---|---|
| chat / `/hum` | **The Pass** chat + **Hum** button (SSE streaming) |
| `/temp /seed /tokens [soft] /context /phase` | **Seasoning** panel |
| `/set /add /clear`, configs (`/no-ai` …) | **Pantry → Recipes / Spice Rack**, **On The Pass** list |
| `/dim /boost`, `/no-hedge`, `/inject`, `/injectvec` | **Pantry → Hand-Mix** |
| `/logits [hum] [save]` + snapshots | **Pantry → Tasting** |
| `/status /reset` | header status chip + **New** |
| `--model it|base`, `--prompt-mode` | **The Range** |

## Dependencies

Python (installed with `--break-system-packages` to match the existing torch install):
`fastapi`, `uvicorn`, `python-multipart` (plus the existing `torch`, `transformers`,
`safetensors`). Frontend deps are in `web/package.json`.

> Note: `steer_engine.py` copies the steering catalogs/math from `chat_steer.py`. If you
> change presets or the hook math in `chat_steer.py`, mirror it in `steer_engine.py`.
