# Feature Workflow

This is the simple loop used by the demo:

```text
Neuronpedia -> pick feature -> local chat -> inject/dim -> compare output
```

You do not need to understand mech-interp terminology to use the repo. Treat a
GemmaScope feature as a named hidden-state direction you can test.

## 1. Search Neuronpedia

Open:

```text
https://www.neuronpedia.org/gemma-3-4b-it
```

Or use the local helper:

```bash
./scripts/neuronpedia.py search "Buddhist concepts" --res16k --limit 10
./scripts/neuronpedia.py search "soft sounds" --res16k --limit 10
./scripts/neuronpedia.py search "golf" --res16k --limit 10
```

This repo uses these GemmaScope RES-16K sources:

```text
9-gemmascope-2-res-16k
17-gemmascope-2-res-16k
22-gemmascope-2-res-16k
29-gemmascope-2-res-16k
```

Neuronpedia IDs map directly to chat commands:

```text
17-gemmascope-2-res-16k / 4271 -> /inject 17:4271:STRENGTH
29-gemmascope-2-res-16k / 3959 -> /inject 29:3959:STRENGTH
```

## 2. Inspect One Feature

```bash
./scripts/neuronpedia.py feature 17-gemmascope-2-res-16k 4271
```

Look at:

```text
description
maxActApprox
top activating examples on the Neuronpedia page
positive/negative logits if shown
```

The label is a hint, not proof. The test is whether steering it changes the
model in the predicted direction.

## 3. Start Chat

```bash
python3 chat_steer.py
```

Baseline:

```text
/clear
/reset
/temp 0.7
/seed 1
Write a short paragraph about silence.
```

Steered run:

```text
/clear
/reset
/temp 0.7
/seed 1
/inject 17:4271:1190
Write a short paragraph about silence.
```

Compare the two answers.

## 4. Pick A Strength

`/inject` strength is a raw activation scale on the SAE decoder direction.

Practical rule:

```text
start small
double slowly
stop when the model gets repetitive, incoherent, or overpowered
```

If Neuronpedia shows `maxActApprox`, try rough fractions of it:

```text
0.1x maxActApprox  = light nudge
0.5x maxActApprox  = obvious nudge
1.0x maxActApprox  = strong
2.0x+ maxActApprox = may capture/break style
```

Known working examples in this repo:

```text
/inject 17:4271:1190   # Buddhist-concepts register
/inject 17:42:2200     # soft-sounds register
/inject 17:15728:1400  # Christ/witness register
```

These are examples for learning the tool. Use Neuronpedia to find your own.

## 5. Dim Instead Of Inject

Injection asks:

```text
What happens if this feature is pushed into the stream?
```

Dimming asks:

```text
What happens if this live feature contribution is reduced?
```

Example:

```text
/dim 17:4271:0.8
```

The included multi-layer carrier shortcut:

```text
/dim carriers 0.8
```

Avoid extreme values at first:

```text
/dim carriers 0     # full ablation, often breaks generation
/dim carriers 2     # strong boost, often breaks generation
```

## 6. Keep Runs Clean

Before each comparison:

```text
/clear
/reset
/seed 1
/temp 0.7
```

Remember:

```text
/reset clears history only
/clear removes steering
/inject and /dim stack until /clear
```

## 7. Save What Worked

When you find a good feature, write down:

```text
Neuronpedia URL
layer/source
feature index
strength or dim scale
prompt
seed/temp/tokens
baseline output
steered output
```

That is enough for someone else to reproduce the behavior locally.
