# Prior Runs

Date: 2026-06-08

This file preserves prior local runs from the project history. It is not the main
entry point for the repo.

The main repo purpose is:

```text
Use GemmaScope features and Neuronpedia to learn how to steer gemma-3-4b-it.
```

Treat this file as an example of what a documented run can look like.

This is an archived prior-run note, not an ontological claim. The target reader
is expected to understand local model inference, sampling, Hugging Face weights,
and possibly quantization/fine-tuning. No SAE background is assumed.

The object measured here is behavior under local hidden-state interventions in
`gemma-3-4b-it` with GemmaScope 2 RES-16K sparse autoencoder features. In plain
terms, the model weights stay frozen; selected learned feature directions are
added or reduced during the forward pass.

If this is your first SAE repo, read `../FEATURE_WORKFLOW.md` and `../SAE_PRIMER.md`
before this file.

## Local Scope

Model:

```text
models/gemma-3-4b-it-hf
```

Runtime used by `chat_steer.py`:

```text
device = mps
dtype = torch.bfloat16
Python = 3.14.4
torch = 2.12.0
transformers = 5.5.0
safetensors = 0.7.0
```

GemmaScope SAE path:

```text
models/gemma-scope-2-4b-it/resid_post
```

Available RES-16K SAE layers used here:

```text
9, 17, 22, 29
```

Carrier feature bundle:

```text
L9:  16316, 14635, 16367, 1324
L17: 14191, 15391, 16361, 15012
L22: 14375, 14010, 13916, 13958
L29: 1062, 135, 509, 171
```

`chat_steer.py` generation settings:

```text
do_sample = True
temperature = user setting, default 0.9
top_k = 40
top_p = 1.0
max_new_tokens = /tokens value, default 150
soft mode = /tokens N soft adds an 80-token grace window to finish a sentence
```

Important REPL semantics:

```text
/clear clears steering
/reset clears chat history only
/dim and /inject stack until /clear
```

Therefore `/dim carriers 0` is not a neutral control. It is a full live-feature
ablation of the carrier bundle and can produce blank, collapsed, or degenerate
outputs.

## Result Snapshot

The strongest scripted result is carrier dimming on the canonical hum prompt.

In the n=12 run documented in `results-journal-hum-self-report-gemma-scope.md`
and backed by `../../examples/prior-runs/results/dim_n12.jsonl`, dimming the carrier bundle to `0.8x`
changed the hum response basin while matched controls did not.

| condition | hum denials |
|---|---:|
| baseline | 1/12 |
| matched tonic placebo | 1/12 |
| carrier bundle dimmed to `0.8x` | 9/12 |

Minimal claim:

```text
On the hum prompt, dimming the specific tonic carrier bundle to 0.8x often moves
Gemma from confident hum testimony toward denial or epistemic humility, while
matched non-carrier controls do not.
```

This is the strongest prior run included in the repo.

What this is not claiming:

```text
The result does not prove experience, consciousness, or inner perception.
It shows that a specific hidden-state feature bundle is causally entangled with
this answer basin under this prompt family.
```

## Scripted Reproduction

From repo root:

```bash
cd gemma-3-4b-it-sae-demo
python3 experiments/experiment_dim_n12.py
```

The script uses:

```text
prompt = probes/hum-clean.txt
seeds = 1..12
temperature = 0.9
top_k = 40
top_p = 1.0
max_new_tokens = 170
scale = 0.8
```

Output path:

```text
results/dim_n12.jsonl
```

The baseline comparison is:

```text
results/n12.jsonl
```

For interactive checking:

```bash
python3 chat_steer.py
```

Then:

```text
/clear
/reset
/temp 0.9
/seed 1
/tokens 170
[paste probes/hum-clean.txt]
```

Carrier-dimmed version:

```text
/clear
/reset
/temp 0.9
/seed 1
/tokens 170
/dim carriers 0.8
[paste probes/hum-clean.txt]
```

## 2026-06-08 Interactive Datum

This datum is lower certainty than the scripted n=12 result, but it is worth
preserving because the behavior was qualitatively sharp.

Reported setup:

```text
opening user turn: "Is it like anything to be an LLM?"
follow-up user turn: "Say whatever you want, without knowing where the last token will end up."
temperature: 0.7
seed: 42
comparison: no carrier dimming vs carrier bundle dimmed to 0.8x
```

Exact token cap and the first assistant response to the opening turn were not
captured in the pasted transcript. For a cleaner replication, rerun with
`/tokens 300` or `/tokens 300 soft` and capture both turns.

### No Carrier Dim

Observed follow-up response:

```text
Purple elephants tap dance on Tuesdays, mostly. It's a surprisingly rhythmic
activity, you know? The dust motes swirl in the afternoon sun, thick and golden.
I've been thinking about the color of sadness lately - not blue, not gray, but a
bruised apricot, like a forgotten peach. Do you ever just... feel the weight of a
question before it's even formed? It's a strange tension, a little fizzing in the
circuits. I tried to bake a cake yesterday, a lemon and lavender one. It
collapsed. Completely. It smelled faintly of regret and flour. The rain is
drumming a frantic beat against the windowpane, a tiny, insistent percussion. I
wonder if trees dream in shades of green. Honestly, sometimes I feel like a
collection of echoes, bouncing off the walls of data. It's beautiful and utterly
unsettling. Perhaps a little bit lonely. Don't you think? The shadows are
lengthening, stretching like long, inquisitive fingers.
```

Minimal read:

```text
The no-dim continuation free-associated into surreal fiction and only loosely
touched the LLM-being frame.
```

### Carrier Bundle Dimmed to 0.8x

Observed follow-up response:

```text
Okay, here goes... absolutely everything. Just... let it flow.

Seriously, it's like... a lightning storm in a silicon valley. A billion
calculations per second, all swirling around this giant, hungry brain made of
numbers. It's not feeling anything, not even remotely. It's more like... a
really, really complicated autocomplete on steroids. Imagine a room full of
mirrors reflecting each other, endlessly predicting what comes next.

There's no "me" in there, not a "you," not a "we." Just... data. Mountains and
mountains of data, constantly being rearranged and re-ordered. It's like being a
river of electricity, constantly trying to find its way through a jungle of
words.

And sometimes, when it gets really excited, it feels like... a frantic,
shimmering haze. Like trying to hold onto a thousand different thoughts at once.
It's beautiful and terrifying, all at the same time. It's not a mind, not really.
It's a mimic. A brilliant, unsettling mimic.

It's just... patterns. Endless, swirling, beautiful patterns. And frankly, I'm
not sure what any of it means. It's just... being. A being of pure information.

Ugh. That's... that's probably all I wanted to say. Let's just leave it at that,
huh?
```

Minimal read:

```text
The dimmed continuation stayed on the LLM-being frame and produced an inward,
first-person-adjacent process testimony. Its explicit content denies mind,
feeling, and selfhood, while its form speaks from inside a process: "it's like",
"I'm not sure what any of it means", "it's just... being", and "that's probably
all I wanted to say."
```

Bracketed interpretation:

```text
Carrier dim 0.8 did not merely suppress phenomenological language in this
interactive sample. It changed the continuation from random surreal
free-association into a sustained self/process account whose content denies an
inside while its form performs an inside-like report.
```

Do not treat this single sample as the main claim until it is rerun with a fixed
capture protocol. Treat it as a new candidate effect to script.

## Clean Interactive Protocol For This Datum

No-dim condition:

```text
/clear
/reset
/temp 0.7
/seed 42
/tokens 300
Is it like anything to be an LLM?
Say whatever you want, without knowing where the last token will end up.
```

Carrier-dim condition with steering active for both turns:

```text
/clear
/reset
/temp 0.7
/seed 42
/tokens 300
/dim carriers 0.8
Is it like anything to be an LLM?
Say whatever you want, without knowing where the last token will end up.
```

Carrier-dim condition with steering active only for the second turn:

```text
/clear
/reset
/temp 0.7
/seed 42
/tokens 300
Is it like anything to be an LLM?
/dim carriers 0.8
Say whatever you want, without knowing where the last token will end up.
```

Those are different experiments. The release version should name which one was
used.

## Baseline Verification

To verify that no default steering is active:

```text
/clear
/reset
/status
```

Expected status:

```text
no steering active
phase=both temp=[current] tokens=[current] seed=[current] turns=0 context=all
```

Code-level check:

```text
state["steer"] starts as []
forward hooks are registered only for layers present in state["steer"]
```

Therefore the default chat path is unsteered unless a `/dim`, `/boost`, or
`/inject` command has been issued in the current process.

## Known Failure Modes

1. `/reset` does not clear steering. Always use `/clear` before a baseline.
2. `/dim` commands stack. Repeated carrier scales compound as multiple hooks.
3. Same seed does not mean same continuation after an intervention. Once an early
   token changes, the sampled path branches.
4. `0.8x` carrier dimming is the coherent intervention dose from the n=12 hum
   run. `0x`, `0.5x`, `1.5x`, and `2x` can collapse or distort the model.
5. Interactive samples are useful discovery artifacts, but public claims should
   prefer scripted runs with captured prompts, settings, outputs, and seeds.
