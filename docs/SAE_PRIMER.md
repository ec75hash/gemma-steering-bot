# SAE Primer For Local Model People

This repo assumes you can run local models and understand basic inference
settings. It does not assume you know mechanistic interpretability.

## What An SAE Is Here

An SAE is a frozen sparse autoencoder trained on a model's hidden states. In this
repo, the SAEs are Google's GemmaScope 2 files for `gemma-3-4b-it`.

For a transformer layer, the model has a hidden vector for each token. The SAE
tries to decompose that hidden vector into a sparse set of feature activations:

```text
hidden state -> SAE encoder -> feature activations -> SAE decoder directions
```

You can read a feature loosely as:

```text
"a learned direction in hidden-state space that tends to activate on some pattern"
```

The label or Neuronpedia description is not the proof. The proof is:

```text
1. when the feature activates
2. what happens when you add it
3. what happens when you suppress it
4. whether matched controls do the same thing
```

## How This Differs From Fine-Tuning

No weights are trained or updated during these runs.

Fine-tuning changes the model's parameters. This repo changes the hidden state
temporarily during one forward pass.

```text
fine-tuning / LoRA:
  update weights, persistent behavior change

SAE steering:
  leave weights frozen, modify live activations during inference
```

Think of it as an inference-time hook, not a training method.

## How This Differs From Prompting

Prompting changes the tokens the model reads.

SAE steering changes internal vectors while the model reads or writes tokens,
without putting the steered concept into the text.

For example:

```text
/inject 17:4271:1190
```

adds one SAE decoder direction at layer 17 during generation. The prompt does not
say "Buddhist concepts", but the model's answer may shift toward that register.

## What Dimming Means

`/dim` does not delete a model neuron. It rescales the live contribution of named
SAE features.

For a dim operation:

```text
1. read the current hidden state
2. estimate selected SAE feature activations
3. reconstruct just those selected decoder contributions
4. add back a scaled delta
```

At `0.8x`, the selected live feature contribution is reduced by 20%.

At `0x`, the selected live feature contribution is fully ablated. That is a
strong intervention and can break generation.

## What "Carrier" Means In The Included Demo

The carrier bundle is a set of 16 SAE features that behaved like tonic/background
features in the local captures:

```text
L9:  16316, 14635, 16367, 1324
L17: 14191, 15391, 16361, 15012
L22: 14375, 14010, 13916, 13958
L29: 1062, 135, 509, 171
```

They are called carriers because dimming them changed the hum-report behavior in
the n=12 run, while matched controls did not.

That name is operational. It means "this bundle appears to carry part of this
behavior under this intervention." It does not mean the model is conscious, and
it does not mean the SAE feature label is automatically correct.

## Why This Requires BF16 Transformers, Not Ollama

Ollama/GGUF is better for efficient local chat, but it does not expose the
per-layer hidden states and forward hooks used by this repo.

This repo uses:

```text
Transformers + PyTorch + bf16 + MPS
```

because `chat_steer.py` needs to hook layer outputs and modify residual-stream
activations. A GGUF/Ollama runtime can reproduce ordinary prompts, but not these
SAE interventions without a custom backend.

## How To Read The Included Prior Runs

The included prior-run documents are examples of how to use this tool. The repo's
main purpose is not to argue for those results; it is to help you run the same
style of feature steering yourself.

The prior-run claim is not:

```text
Gemma is experiencing a hum.
```

The main claim is:

```text
On a specific self-processing prompt, dimming a specific always-on SAE feature
bundle to 0.8x changed the answer basin in 9/12 seeds, while matched controls
did not.
```

Interpret the outputs carefully. The useful part for new users is the method:
same model, same prompt, same sampling settings, selected hidden-state directions
changed during inference.
