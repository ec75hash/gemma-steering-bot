# Neuronpedia API Workflow

This project can use `NEURONPEDIA_API_KEY` from `.env`, but basic feature/search
workflows should be tried without a key first. If you do use a key, do not print
it; the local client sends it as the `x-api-key` header.

## What Works For Gemma 3 4B IT

Working:

- Fetch a feature dashboard:
  - `GET /api/feature/{modelId}/{layer}/{index}`
- Search feature explanations:
  - `POST /api/explanation/search-model`
  - `POST /api/explanation/search`

Currently not useful for `gemma-3-4b-it/gemmascope-2-res-16k`:

- `POST /api/search-all`
- `POST /api/activation/new`

Neuronpedia lists this source set with `allowInferenceSearch: false`, so use
explanation search plus dashboard fetches for candidate discovery.

## Commands

Search RES-16K golf candidates:

```bash
./scripts/neuronpedia.py search golf --res16k --limit 20 \
  --out data/neuronpedia-golf-res16k.json
```

Fetch the strongest golf feature we found:

```bash
./scripts/neuronpedia.py feature 29-gemmascope-2-res-16k 3959 \
  --out data/neuronpedia-feature-29-res16k-3959.json
```

Fetch the full raw dashboard JSON if needed:

```bash
./scripts/neuronpedia.py feature 29-gemmascope-2-res-16k 3959 --full \
  --out data/neuronpedia-feature-29-res16k-3959-full.json
```

## Useful Gemma 3 4B IT IDs

- Model ID: `gemma-3-4b-it`
- RES-16K source set: `gemmascope-2-res-16k`
- RES-16K layers: `9-gemmascope-2-res-16k`, `17-gemmascope-2-res-16k`,
  `22-gemmascope-2-res-16k`, `29-gemmascope-2-res-16k`
- Transcoder 262K source set: `gemmascope-2-transcoder-262k`

## Steering Use

Use Neuronpedia to rank and inspect candidate features. Use local GemmaScope SAE
weights and Python/Transformers for actual intervention runs.

Simple first steering candidates:

- `17-gemmascope-2-res-16k / 4271` - Buddhist concepts
- `17-gemmascope-2-res-16k / 42` - soft sounds
- `29-gemmascope-2-res-16k / 3959` - golf
