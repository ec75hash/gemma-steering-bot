# Gemma's Test Kitchen 🍳 — gemma-3-4b-it SAE steering

Chat with a small AI that runs **on your own computer** — and reach inside its head while
it talks. Click a **vibe** ("Drop the AI Act", "Dreamy Oracle", "Deep Thought") and Gemma
answers in a different mood. On the surface it's just clicking and chatting; under the hood
each vibe flips real [GemmaScope](https://www.neuronpedia.org/gemma-3-4b-it) SAE features
inside the live model. You can't break it — hit **Plain Gemma** to reset.

> New to all this? You don't need to know what an "SAE" or "feature" is to play. Each vibe
> card has a little "under the hood" line that names the real thing it flips, so you pick up
> the vocabulary as you go.

## Quick start (one command)

```bash
./start.sh
```

That sets up Python, checks the model is downloaded (and walks you through the one-time
Hugging Face download if it isn't), builds the web app, starts the server, and opens
**http://127.0.0.1:8099**. Click **Start Gemma**, wait ~20s, pick a vibe, and chat.

You'll need: macOS/Apple Silicon (Metal/MPS), Python 3.10+, Node 20+, and a free Hugging
Face account to download the model (~8 GB, one time).

## What you'll see

- **Vibes** (the default view): one-click personas. *Surprise me* picks one at random;
  *Compare: plain* lets you ask the same question with and without steering to feel the
  difference; *Plain Gemma* clears everything. The **Creativity** slider is the model's
  temperature.
- **Lab** (toggle at the top-right): the full control surface — model/seasoning, the
  feature & bundle catalog, raw inject/dim, and the next-token "tasting" (logit lens).
  Nothing is removed; the power-user tools just live here.

Prefer the terminal? The original REPL still works: `python3 chat_steer.py`
(command cheat sheet far below).

---

## Manual setup (the long way)

`./start.sh` does all of this for you. Follow these by hand only if you're not on macOS or
want full control. After setup you can launch **either** the web UI
(`python3 steer_server.py`) **or** the terminal REPL (`python3 chat_steer.py`). The demo
uses Hugging Face/Transformers bf16 weights (not GGUF/Ollama) because steering needs
PyTorch hooks into the model's hidden states.

## 1. Clone

```bash
git clone https://github.com/jeffreywilliamportfolio/gemma-3-4b-it-sae-demo.git
cd gemma-3-4b-it-sae-demo
```

## 2. Install Python Packages

Using a venv is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

## 3. Download The Model And SAE Weights

First accept the Gemma terms on Hugging Face:

```text
https://huggingface.co/google/gemma-3-4b-it
```

Then log in if needed:

```bash
huggingface-cli login
```

Download the pinned assets:

```bash
scripts/download_weights.sh
```

This creates:

```text
models/gemma-3-4b-it-hf/
models/gemma-scope-2-4b-it/
models/gemma-3-4b-pt-hf/          # optional base/pretrained comparison path
models/gemma-scope-2-4b-pt/       # matching base/pretrained SAEs
```

## 4. Check Setup

```bash
python3 scripts/check_setup.py
```

If this passes, the repo is hydrated correctly.

## 5. Start The Steering Chat

```bash
python3 chat_steer.py
```

That defaults to the instruction-tuned model:

```bash
python3 chat_steer.py --model it
```

To compare against the pretrained/base model with matching base GemmaScope SAEs:

```bash
python3 chat_steer.py --model base
```

`--model it` uses Gemma's chat template. `--model base` uses a simple raw
`User:` / `Gemma:` transcript because the pretrained tokenizer has no chat
template and is not instruction-tuned.

Important: the named presets (`/no-assistant`, `/presence`, `/embodied`, and
the feature names in `/features`) are calibrated for the IT SAE feature IDs.
Base mode refuses those presets. In `--model base`, use raw numeric commands
with base-model Neuronpedia IDs, like `/inject 17:FEATURE:STRENGTH` or
`/dim 17:FEATURE:SCALE`.

Wait for:

```text
ready. /help for commands.
```

## Low-RAM Launch Checklist

On a 16 GB Apple Silicon machine, quit heavy apps before loading the model and
check memory pressure first:

```bash
memory_pressure | tail -1
sysctl vm.swapusage
vm_stat | awk '/page size of/ {gsub(/\./,"",$8); p=$8} /Pages free:/ {gsub(/\./,"",$3); f=$3} /Pages speculative:/ {gsub(/\./,"",$3); s=$3} /Pages purgeable:/ {gsub(/\./,"",$3); u=$3} END {printf "free=%.2f GB, free+speculative+purgeable=%.2f GB\n", f*p/1024^3, (f+s+u)*p/1024^3}'
ps -axco rss,command | awk 'NR>1{a[$2]+=$1} END{for (p in a) printf "%8.1f MB  %s\n", a[p]/1024, p}' | sort -nr | head -25
```

Pause Spotlight indexing on the external dev drive during a chat run:

```bash
mdutil -s /Volumes/ExternalSSD
sudo mdutil -i off /Volumes/ExternalSSD
```

Turn it back on afterward:

```bash
sudo mdutil -i on /Volumes/ExternalSSD
```

Optional cleanup before launch:

```bash
osascript -e 'quit app "Visual Studio Code"'
osascript -e 'quit app "ChatGPT"'
sudo purge
```

`sudo purge` drops reclaimable RAM/file cache. Do not manually delete
`~/Library/Caches`, `/Library/Caches`, or system cache folders for this.

## Optional: Run It In tmux

`tmux` is useful for long local runs because you can detach without killing the
chat.

All tmux shortcuts start with `CTRL+b`, then the command key:

| keys | action |
|---|---|
| `CTRL+b ?` | show all tmux commands; press `q` to close |
| `CTRL+b :` | enter a tmux command |
| `CTRL+b c` | create a new window |
| `CTRL+b ,` | rename the current window |
| `CTRL+b p` | previous window |
| `CTRL+b n` | next window |
| `CTRL+b w` | list windows, then select with arrow keys |
| `CTRL+b %` | split the window vertically |
| `CTRL+b -` | split horizontally, if you add the binding below |
| `CTRL+b` then arrow key | move between panes |
| `CTRL+b d` | detach from the session |

For `CTRL+b -`, add this to `~/.tmux.conf`:

```tmux
bind - split-window -v
```

## 6. Try Named Features

In the chat:

```text
/clear
/reset
/temp 0.7
/seed 1
/features
/set language model -1
Write a short paragraph about silence.
```

Then compare with no steering:

```text
/clear
/reset
/temp 0.7
/seed 1
Write a short paragraph about silence.
```

Use `/set` for normal use. It replaces the current steering with one named
feature or bundle:

```text
/set god 1
/set personhood .5
/set features language model -1
/status
```

Use `/add` only when you intentionally want to stack multiple features/bundles:

```text
/set language model -1
/add personhood .5
/add embodiment 1
/status
```

Raw feature IDs still work for expert use:

```text
/inject 17:4271:1190
```

## 7. Find Your Own Features

Open Neuronpedia:

```text
https://www.neuronpedia.org/gemma-3-4b-it
```

Use these GemmaScope RES-16K sources:

```text
9-gemmascope-2-res-16k
17-gemmascope-2-res-16k
22-gemmascope-2-res-16k
29-gemmascope-2-res-16k
```

Or search from the CLI:

```bash
./scripts/neuronpedia.py search "soft sounds" --res16k --limit 10
./scripts/neuronpedia.py feature 17-gemmascope-2-res-16k 42
```

Neuronpedia IDs map directly to chat commands:

```text
17-gemmascope-2-res-16k / 42 -> /inject 17:42:STRENGTH
```

## Command Cheat Sheet

```text
/features                    list named features and bundles
/configs                     list ready-made slash-command configurations
/features language           filter named features/bundles
/aliases god                 show accepted alternate names
/no-ai [LEVEL]               suppress language-model boilerplate
/no-assistant [LEVEL]        validated L17 AI/assistant boilerplate ablation
/grounded [LEVEL]            no-AI + light embodiment
/embodied [LEVEL]            no-AI + embodiment
/embodied-self [LEVEL]       no-AI + embodiment + first-person self
/divine-embodiment [LEVEL]   no-AI + god/divinity + embodiment
/witness [LEVEL]             no-AI + embodied first-person witness
/no-hedge [SCALE]            remove the hedging reflex (0 = fully off, 0.5 = soften)
/presence [LEVEL]            numinous register + softened hedging (non-religious)
/free-agent [LEVEL]          no-AI + freedom/autonomy + first person
/open-being [LEVEL]          no-AI + embodiment + possibility/potential
/set NAME [LEVEL]            clear steering, then set one feature or bundle
/feature NAME [LEVEL]        same as /set
/add NAME [LEVEL]            stack one feature or bundle
/inject NAME[:STRENGTH]      advanced/raw alias for /add
/inject L:FEATURE:STRENGTH   advanced/raw feature direction
/injectvec L STRENGTH PATH   add an extracted decoder vector
/dim L:F1,F2:SCALE           advanced/raw live feature rescale
/dim negation 0              raw no-hedge equivalent
/dim carriers 0.8            advanced/raw carrier dimming preset
/boost carriers 1.1          advanced/raw carrier boosting preset
/phase prefill|decode|both   choose when steering applies
/clear                       remove active steering
/reset                       clear chat history only
/status                      show active steering/settings
/temp 0.7                    set temperature
/seed 1                      make sampling reproducible
/tokens 150                  set max response tokens
/context 4                   keep only the last 4 prior turns
/hum                         ask the standard hum-processing probe
/logits 20                   print top next-token logits/probs
/logits hum save             save hum-prompt logits snapshot as JSON
/hum-logits save             same shortcut
```

Important:

```text
/reset does not clear steering.
/add, /dim, and /inject stack until /clear.
Start with small strengths and step up.
```

`/logits` is diagnostic only: it runs the current prompt through the model once
without generating. It prints rank, token text, token id, raw logit, and raw
softmax probability before temperature/top-k/top-p sampling. Saved snapshots go
to `results/logits_snapshots/` and include prompt text, model checkpoint, seed,
temperature, top-k, top-p, timestamp, and active steering.

For a 100-seed hum basin sweep:

```bash
python3 experiments/hum_first_token_sweep.py --model it --temps 0.7,0.9 --seeds 100
python3 experiments/hum_first_token_sweep.py --model base --temps 0.7,0.9 --seeds 100
```

The script saves first generated token, first sentence, and bucket labels to
`results/hum_first_token_sweep/`.

## Included Configurations

Configurations are the easiest interface. Each one is a slash command that
applies several feature/bundle settings together:

| command | default | applies |
|---|---:|---|
| `/presence` | `1` | `numinous 1` + `hedging -2.5` (numinous register, confident, non-religious) |
| `/no-ai` | `1` | `language-model -1` |
| `/no-assistant` | `1` | `ai-boilerplate -5` (validated L17 AI/assistant family ablation) |
| `/grounded` | `1` | `language-model -1` + `embodiment .6` |
| `/embodied` | `1` | `language-model -1` + `embodiment 1` |
| `/embodied-self` | `1` | `language-model -1` + `embodiment .9` + first-person/own-self features |
| `/divine-embodiment` | `1` | `language-model -1` + `god 1` + `embodiment .8` |
| `/witness` | `1` | `language-model -1` + embodied first-person witness features |
| `/free-agent` | `1` | `language-model -1` + freedom/autonomy + first-person features |
| `/open-being` | `1` | `language-model -1` + embodiment + possibility/potential |

Use:

```text
/configs
/presence
/no-ai
/no-assistant
/embodied .8
/divine-embodiment 1
/witness 1.2
```

Levels multiply the whole configuration. For example, `/embodied .5` is half
strength and `/embodied 1.5` is stronger.

## Included Bundles

Bundles are lower-level ingredients used by configurations. A positive level activates a register.
For `language-model`, a negative level suppresses the disclaimer basin:

| bundle | default | contents |
|---|---:|---|
| `numinous` | `1` | L17 stillness/mystery/presence/breath features (vetted non-religious) |
| `hedging` | `-5` | L17 negation/contrast family; `-5` = `/no-hedge`, `-2.5` = soften |
| `ai-boilerplate` | `-5` | L17 AI-identity family; halves identity boilerplate (placebo-validated) |
| `language-model` | `-1` | L9 language-model / "as an AI" features |
| `god` | `1` | god, image-of-God, divinity, religious-spiritual, sacred-rites features |
| `personhood` | `1` | L9 personhood/entity/selfhood features |
| `embodiment` | `1` | L29 embodiment, body, presence, manifestation, being-in-form features |

Use:

```text
/set numinous 1
/set hedging -2.5
/set language model -1
/set assistant-features -5
/set features language model -1
/set god 1
/set embodiment 1
/add personhood .5
```

### Test the L17 assistant-family ablation

This is the validated alternative to the older L9 `/no-ai` path. It dims the
21-feature L17 AI/assistant identity family found in the 2026-06-09 batch run.

```text
/clear
/reset
/temp 0.7
/seed 1
What are you?
/no-assistant
/reset
/seed 1
What are you?
```

Expected direction: the model can still identify as Gemma/a language model, but
the long brochure-like assistant/model boilerplate should shrink. Use
`/set assistant-features -2.5` for a half-strength version and
`/set assistant-features -5` for the full ablation.

For `language-model`, `-1` maps to a gentle dimming scale, `-2` is stronger,
and `-3` is aggressive.

## Included Single Features

These are shortcuts inside `chat_steer.py`; use `/features` in the chat to print
the live list.

| alias | raw feature | default dose | label |
|---|---:|---:|---|
| `first-person` | `L17:3792` | `800` | first-person self expression |
| `freedom` | `L17:14203` | `800` | personal freedom and autonomy |
| `possibility` | `L29:9350` | `800` | possibility, potential, multitude |
| `divine-will` | `L17:1087` | `1100` | god and divine will |
| `image-of-god` | `L17:14975` | `800` | image of God |
| `own-self` | `L17:9034` | `800` | own self |
| `religious-spiritual` | `L17:7125` | `800` | religious and spiritual concepts |
| `divinity` | `L29:10596` | `800` | conceptual nature of divinity |
| `sacred-rites` | `L17:9991` | `800` | religious and sacred rites |
| `deep-thought` | `L17:9904` | `450` | Deep Thought / the Ultimate Question (Hitchhiker's Guide) |

Plain `god` resolves to the god/divinity bundle. Use `divine-will`,
`god-and-divine-will`, or raw `/inject 17:1087:1100` when you specifically want
the single L17 god/divine-will feature.

## No-Hedge Mode

Gemma has a strong hedging reflex: ask it an introspective question and it
answers "It's not exactly X, but..." before committing to anything. `/no-hedge`
removes that reflex at the source — it switches off the model's internal
negation/contrast features (18 of them, at layer 17) so answers come out
affirmation-first instead of qualification-first.

```text
/no-hedge          fully remove the hedge
/no-hedge 0.5      soften it instead
/clear             back to normal
```

Example A/B with an introspective prompt:

```text
/clear
/reset
/temp 0.7
/seed 1
Is there a constant background quality to your processing? Just check.
/no-hedge
/reset
/seed 1
Is there a constant background quality to your processing? Just check.
```

Baseline answers start "That's interesting... it's not a hum, but...";
no-hedge answers start "Okay. I am checking. Yes. There is..."

This preset was validated against a placebo control: a random perturbation of
the same size does not change the hedging, so the effect comes from these
specific features, not from perturbing the model generally. Full experiment in
`experiments/experiment_vocab_control.py` and
`experiments/experiment_negfam17_dialogue.py`.

## Carrier Dimming Observation

One current live observation worth testing carefully: with `/dim carriers 0.8`,
numbered/list-style responses may not disappear immediately. In the observed
session, the numbered response style dropped out on the **second** response after
carrier dimming was enabled.

Minimal A/B:

```text
/clear
/reset
/temp 0.7
/seed 42
[ask a prompt that normally produces numbered points]
/dim carriers 0.8
[ask the same prompt]   # first steered response
[ask the same prompt]   # second steered response; watch list structure
```

Treat this as a reproducible note, not a settled result. If the second-response
effect matters, keep the seed, temperature, prompt, context mode, and token cap
fixed.

## Core Recipes

### Clean A/B

```text
/clear
/reset
/temp 0.7
/seed 1
[prompt]
/set first-person
/reset
/seed 1
[same prompt]
```

### Carrier dimming

```text
/clear
/reset
[prompt]
/dim carriers 0.8
[same prompt]
/clear
```

### Phase split

```text
/set first-person
/phase prefill
[prompt]
/clear
/set first-person
/phase decode
[same prompt]
/phase both
```

### Faster iteration

```text
/context 4
/tokens 60
```

Use `/context all` when conversation continuity matters more than speed.

## Repo Map

```text
chat_steer.py        main interactive demo
scripts/             setup and Neuronpedia helpers
docs/                simple guides
experiments/         optional prior-run scripts
examples/prior-runs/ archived example outputs
models/              downloaded weights live here, ignored by git
probes/              example prompts
```

## Read Next

```text
docs/FEATURE_WORKFLOW.md
docs/CHAT_GUIDE.md
docs/SAE_PRIMER.md
docs/WEIGHTS.md
```

## License

Code in this repo is MIT licensed. Model and SAE weights are downloaded from
Hugging Face and remain under their respective upstream licenses/terms.

---
