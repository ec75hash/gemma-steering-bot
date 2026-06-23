# Is-Ness Feature Workflow

Goal: compare a Neuronpedia 262k feature against the matching local 16k SAE at
the same residual layer, then optionally steer with the exact extracted 262k
decoder vector.

The original seed feature was:

```text
gemma-3-4b-it / 17-gemmascope-2-res-262k / 61553
```

For layer 29, first pick the target feature from:

```text
https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-262k
```

The 262k layer-17 `params.safetensors` file is about 5 GiB on disk. Do not load
the whole tensor during chat. The search script reads one target row and scans
the local 16k decoder in chunks.

## Download Only The Needed Layer-29 262k Params

```bash
hf download google/gemma-scope-2-4b-it \
  resid_post/layer_29_width_262k_l0_medium/config.json \
  resid_post/layer_29_width_262k_l0_medium/params.safetensors \
  --local-dir models/gemma-scope-2-4b-it
```

## Find The Closest Local 16k Features

```bash
python3 scripts/find_sae_analog.py \
  --layer 29 \
  --target-index FEATURE_ID \
  --top-k 25 \
  --chunk-size 512 \
  --enrich \
  --save-target-vector vectors/gemma3-4b-it-l29-res262k-fFEATURE_ID.safetensors \
  --out scratch/ness-l29-262k-to-16k.json
```

The first-ranked candidates are decoder-cosine neighbors. Treat them as
candidates, not proof. Validate them with activation tests and steering behavior.

## Probe Activation Behavior

```bash
python3 scripts/neuronpedia.py activate 17-gemmascope-2-res-262k 61553 \
  "consciousness stillness loneliness mindfulness fearlessness faithfulness"
```

Layer-29 equivalent:

```bash
python3 scripts/neuronpedia.py activate 29-gemmascope-2-res-262k FEATURE_ID \
  "consciousness stillness loneliness mindfulness fearlessness faithfulness"
```

Then repeat for the top layer-29 16k candidates:

```bash
python3 scripts/neuronpedia.py activate 29-gemmascope-2-res-16k FEATURE_ID \
  "consciousness stillness loneliness mindfulness fearlessness faithfulness"
```

## Direct 262k Vector Steering

After `--save-target-vector`, start the local 4B steering chat:

```bash
python3 chat_steer.py
```

Then inject the extracted vector:

```text
/injectvec 17 200 vectors/gemma3-4b-it-l17-res262k-f61553.safetensors
```

For a layer-29 vector:

```text
/injectvec 29 200 vectors/gemma3-4b-it-l29-res262k-fFEATURE_ID.safetensors
```

Start low and step up. Use the feature page's `maxActApprox` as the dose guide.
