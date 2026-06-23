# Gemma's Test Kitchen — web UI

The kitchen-themed front-end for live SAE steering of `gemma-3-4b-it`. React 19 + Vite +
Tailwind v4 + GSAP, talking to `steer_server.py` (FastAPI) over `/api`.

```
web/ (this app)  ──/api──▶  ../steer_server.py (FastAPI)  ──▶  ../steer_engine.py  ──▶  gemma-3-4b-it + GemmaScope SAEs (PyTorch/MPS)
```

## Two modes (toggle, top-right of the chat header)

- **Vibes** (default, beginner): one-click personas from `src/lib/vibes.js`. Each card maps
  a plain-language name onto a real preset in `../steer_engine.py` and shows an "under the
  hood" reveal. Built in `src/Beginner.jsx`.
- **Lab** (advanced): the full surface — The Range (model load), Seasoning (temp/seed/
  tokens/phase), the Pantry (Recipes / Spice Rack / Hand-Mix raw inject·dim / Tasting logit
  lens), and On The Pass (active steering). Lives in `src/App.jsx`.

Both modes share the same chat column (`ChatColumn` in `src/App.jsx`) and the API client in
`src/lib/api.js`.

## Run

Don't run this directly for normal use — `../start.sh` (or `python3 ../steer_server.py`)
serves the built app for you. Use these only when hacking on the UI:

```bash
# Production build (served by steer_server.py at :8099)
npm install && npm run build

# Hot-reload dev: API in one terminal, Vite in another (Vite proxies /api → :8099)
python3 ../steer_server.py     # terminal 1
npm run dev                    # terminal 2 (UI on :5179)
```

Env for the dev proxy target: `KITCHEN_API` (defaults to the server on :8099).

## Theming

Warm kitchen palette via CSS custom properties in `src/index.css` (`--primary` flame,
`--amber` saffron, `--basil`, `--copper`). The decorative scene is `KitchenScene` in
`src/App.jsx` over `public/scene/kitchen.jpg`. Components are shadcn-style primitives over
Radix in `src/components/ui/*`.
