# Weight Setup

This repo does not commit model weights, SAE weights, GGUFs, or tensor capture
files. Clone the repo, then run one script to download the required Hugging Face
assets into the expected local paths.

This keeps normal git usable without Git LFS while still making the project
reproducible for someone with enough local GPU/unified-memory headroom.

This repo intentionally uses bf16 Hugging Face/Transformers weights rather than
GGUF/Ollama for the steering runs. The scripts need PyTorch forward hooks into
layer hidden states. GGUF is fine for ordinary chat, but it is not enough to
reproduce the SAE interventions here.

## Required Sources

Gemma 3 4B instruction-tuned bf16 weights:

```text
repo: google/gemma-3-4b-it
revision: 093f9f388b31de276ce2de164bdc2081324b9767
local path: models/gemma-3-4b-it-hf
official page: https://huggingface.co/google/gemma-3-4b-it
```

GemmaScope 2 4B-IT RES-16K SAE weights:

```text
repo: google/gemma-scope-2-4b-it
revision: 3e94b68be95290aada5b7525cf431d3040f81bb1
local path: models/gemma-scope-2-4b-it
official page: https://huggingface.co/google/gemma-scope-2-4b-it
landing page: https://huggingface.co/google/gemma-scope-2
```

Only these SAE folders are needed for the current scripts:

```text
resid_post/layer_9_width_16k_l0_medium
resid_post/layer_17_width_16k_l0_medium
resid_post/layer_22_width_16k_l0_medium
resid_post/layer_29_width_16k_l0_medium
```

## Simple Download

From a fresh clone:

```bash
git clone https://github.com/jeffreywilliamportfolio/gemma-3-4b-it-sae-demo.git
cd gemma-3-4b-it-sae-demo
python3 -m pip install -r requirements.txt
scripts/download_weights.sh
python3 scripts/check_setup.py
```

If needed, create `.env` from `.env.example` and set `HF_TOKEN`. Do not commit
`.env`.

Gemma is gated on Hugging Face. Log in and accept the model terms for
`google/gemma-3-4b-it` before downloading.

If `scripts/check_setup.py` passes after the download, the repo should be ready
to run with `python3 chat_steer.py`.

## Expected Layout

After download:

```text
models/gemma-3-4b-it-hf/
  config.json
  generation_config.json
  model.safetensors.index.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  tokenizer.json
  tokenizer.model
  tokenizer_config.json

models/gemma-scope-2-4b-it/resid_post/
  layer_9_width_16k_l0_medium/params.safetensors
  layer_17_width_16k_l0_medium/params.safetensors
  layer_22_width_16k_l0_medium/params.safetensors
  layer_29_width_16k_l0_medium/params.safetensors
```

The download script also fetches small config/example files in those SAE
directories when present.

## Why We Do Not Commit Weights

Normal git is the wrong transport for multi-GB model files. This repo is intended
to be cloned, then hydrated locally from the pinned Hugging Face sources above.

No Git LFS is required.
