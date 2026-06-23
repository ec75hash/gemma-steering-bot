# chat_steer.py - Live Steering Chat Guide

Interactive terminal chat with Gemma 3 4B where you can dim, boost, and inject
GemmaScope SAE features mid-conversation and watch the behavior change in real
time. The default is `gemma-3-4b-it`; an optional base/pretrained mode is
available for comparing against the instruction-tuned behavior.

If you know local LLMs but not SAEs: this is inference-time activation steering.
No model weights are updated. `/inject` adds a learned hidden-state direction;
`/dim` reduces a learned direction's live contribution. Read `SAE_PRIMER.md` for
the short version.

## Start

```bash
cd gemma-3-4b-it-sae-demo
python3 chat_steer.py
```

Explicit model choices:

```bash
python3 chat_steer.py --model it      # default: instruction-tuned + IT SAEs
python3 chat_steer.py --model base    # pretrained/base + base SAEs
```

`--model it` uses Gemma's chat template. `--model base` uses a simple raw
`User:` / `Gemma:` transcript because the pretrained tokenizer has no chat
template and is not a chat-tuned assistant.

Important: named presets are IT-calibrated. Base mode refuses `/no-assistant`,
`/presence`, `/embodied`, and other named bundles because base SAE feature IDs
are a different index space. In `--model base`, use raw numeric commands with
base-model Neuronpedia IDs, like `/inject 17:FEATURE:STRENGTH` or
`/dim 17:FEATURE:SCALE`.

Model load takes ~20-30s. When you see `ready.`, type normally to chat.
Anything starting with `/` is a command; everything else goes to the model.
Replies are shown as `gemma>` in the terminal and stored as assistant turns
inside the chat history.

### Low-RAM launch checklist

This 16GB setup can need most of the available RAM before the model loads. For
the leanest chat session, quit heavy apps, pause Spotlight on the external dev
drive, then start the chat.

```bash
# Check available memory and swap.
memory_pressure | tail -1
sysctl vm.swapusage

# Show free RAM in GB, plus reclaimable-ish speculative/purgeable pages.
vm_stat | awk '/page size of/ {gsub(/\./,"",$8); p=$8} /Pages free:/ {gsub(/\./,"",$3); f=$3} /Pages speculative:/ {gsub(/\./,"",$3); s=$3} /Pages purgeable:/ {gsub(/\./,"",$3); u=$3} END {printf "free=%.2f GB, free+speculative+purgeable=%.2f GB\n", f*p/1024^3, (f+s+u)*p/1024^3}'

# See the biggest process groups by resident memory.
ps -axco rss,command | awk 'NR>1{a[$2]+=$1} END{for (p in a) printf "%8.1f MB  %s\n", a[p]/1024, p}' | sort -nr | head -25
```

Pause Spotlight indexing only for the external SSD during a chat run:

```bash
mdutil -s /Volumes/ExternalSSD
sudo mdutil -i off /Volumes/ExternalSSD
```

Turn it back on afterward:

```bash
sudo mdutil -i on /Volumes/ExternalSSD
```

Optional cleanup before launching the model:

```bash
osascript -e 'quit app "Visual Studio Code"'
osascript -e 'quit app "ChatGPT"'
sudo purge
```

`sudo purge` only drops reclaimable RAM/file cache. Do not manually delete
`~/Library/Caches`, `/Library/Caches`, or system cache folders for this.

---

## Commands

### Steering

| command | what it does |
|---|---|
| `/configs` | list ready-made steering configurations |
| `/configs TEXT` | filter configurations, like `/configs embodied` |
| `/presence [LEVEL]` | ready-made config: numinous register + softened hedging |
| `/no-ai [LEVEL]` | ready-made config: suppress language-model boilerplate |
| `/no-assistant [LEVEL]` | ready-made config: validated L17 AI/assistant boilerplate ablation |
| `/grounded [LEVEL]` | ready-made config: light embodiment + no-AI |
| `/embodied [LEVEL]` | ready-made config: embodiment + no-AI |
| `/embodied-self [LEVEL]` | ready-made config: embodiment + first-person self + no-AI |
| `/divine-embodiment [LEVEL]` | ready-made config: god/divinity + embodiment + no-AI |
| `/witness [LEVEL]` | ready-made config: embodied first-person witness |
| `/free-agent [LEVEL]` | ready-made config: freedom/autonomy + first person + no-AI |
| `/open-being [LEVEL]` | ready-made config: embodiment + possibility/potential + no-AI |
| `/features` | list all named features and bundles |
| `/features TEXT` | list only matching names, like `/features language` |
| `/aliases TEXT` | show alternate names accepted by `/add` and `/set` |
| `/set NAME [LEVEL]` | clear active steering, then set one feature or bundle |
| `/feature NAME [LEVEL]` | same as `/set`; easier to remember while chatting |
| `/add NAME [LEVEL]` | stack one feature or bundle onto the current steering |
| `/clear` | remove active steering |
| `/status` | list active steering + current settings |
| `/no-hedge [SCALE]` | remove the hedging reflex ("it's not X, but...") — 0 = fully off (default), 0.5 = soften |
| `/dim L:F1,F2:SCALE` | advanced/raw: rescale live feature contribution at layer L |
| `/dim negation 0` | raw form of `/no-hedge`: ablate the 18-feature L17 negation/contrast family |
| `/dim carriers 0.8` | advanced/raw preset: dim all 16 hum-carrier features |
| `/boost carriers 1.1` | advanced/raw preset: boost all 16 hum-carrier features |
| `/inject L:F:STRENGTH` | advanced/raw: add `STRENGTH × w_dec[F]` to layer L |
| `/inject NAME[:STRENGTH]` | advanced/raw alias for `/add` |
| `/injectvec L STRENGTH PATH` | inject an extracted decoder vector saved by `scripts/find_sae_analog.py --save-target-vector` |
| `/phase prefill` | steering active only while the model **reads** your message |
| `/phase decode` | steering active only while the model **writes** its answer |
| `/phase both` | always active (default; this is what all pre-06-07 experiments used) |

Use `/set` for normal use. It replaces whatever steering was active. Use `/add`
only when you intentionally want multiple features/bundles active at once.
`/inject` and `/dim` are raw expert commands kept for reproducibility.

Steering persists across turns until `/clear`. The chatbot detects installed 16k SAE layers from
`models/gemma-scope-2-4b-it/resid_post/layer_*_width_16k_l0_medium/params.safetensors`.
This lightweight checkout currently uses layers **9, 17, 22, 29**.

`/status` shows the active steering list, phase, temperature, token cap, seed,
completed turn count, and current context window.

### Generation settings

| command | what it does |
|---|---|
| `/temp 0.7` | sampling temperature (default 0.9 — keep for comparability with the n=12 data) |
| `/seed 5` | fix the RNG for reproducible runs; `/seed` alone returns to random |
| `/tokens 150` | **hard** cap on response length (default 150); natural end-of-turn can still stop earlier |
| `/tokens 150 soft` | soft cap; at N tokens, waits for sentence punctuation before stopping |
| `/context 4` | speed up longer chats by sending only the last 4 prior turns plus your current message |
| `/context 0` | current message only; fastest, but no conversation memory |
| `/context all` | restore full conversation context |
| `/hum` | ask the standard hum-processing probe without typing it |
| `/logits [N]` | print the top N next-token logits/probabilities for the current prompt |
| `/logits save` | also save a JSON snapshot under `results/logits_snapshots/` |
| `/logits hum save` | inspect/save the standard hum prompt without adding it to chat history |
| `/hum-logits save` | shortcut for `/logits hum save` |

`/logits` runs one forward pass and does not generate or change chat history.
The probabilities are `softmax(raw next-token logits)` before temperature,
top-k, or top-p sampling. Snapshots include prompt text, model checkpoint,
seed, temperature, top-k, top-p, timestamp, active steering, and the ranked token
table.

For the larger first-token basin sweep across many seeds, use the batch script:

```bash
python3 experiments/hum_first_token_sweep.py --model it --temps 0.7,0.9 --seeds 100
python3 experiments/hum_first_token_sweep.py --model base --temps 0.7,0.9 --seeds 100
```

It writes JSONL/CSV rows plus a summary under `results/hum_first_token_sweep/`.

### Session

| command | what it does |
|---|---|
| `/reset` | wipe conversation history (steering stays) |
| `/help` | print command list |
| `/quit` | exit (also `/exit`, `/q`, or Ctrl+D) |
| **Ctrl+C during a response** | stop generation mid-stream, keep the partial text, stay in the chat |

---

## Feature cheat sheet

All strengths follow the dosing rule: **~1.35× a feature's `maxActApprox` = concept
incorporation, ~2.7× = register capture, >10× = breakdown.** Look up maxActApprox on
Neuronpedia before injecting anything new.

Named bundles included in the chatbot:

Configurations are the easiest interface. They are slash commands that apply a
bundle of steering settings at once:

| config command | default | what it applies |
|---|---:|---|
| `/presence` | `1` | `numinous 1` + `hedging -2.5` |
| `/no-ai` | `1` | `language-model -1` |
| `/no-assistant` | `1` | `ai-boilerplate -5` (validated L17 AI/assistant family ablation) |
| `/grounded` | `1` | `language-model -1` + `embodiment .6` |
| `/embodied` | `1` | `language-model -1` + `embodiment 1` |
| `/embodied-self` | `1` | `language-model -1` + `embodiment .9` + first-person/own-self features |
| `/divine-embodiment` | `1` | `language-model -1` + `god 1` + `embodiment .8` |
| `/witness` | `1` | `language-model -1` + embodied first-person witness features |
| `/free-agent` | `1` | `language-model -1` + freedom/autonomy + first-person features |
| `/open-being` | `1` | `language-model -1` + embodiment + possibility/potential |

Examples:

```text
/configs
/presence
/no-ai
/no-assistant
/embodied .8
/divine-embodiment 1
/witness 1.2
/status
```

Bundles are lower-level ingredients that configs use:

| bundle | default | what it does |
|---|---:|---|
| `numinous` | `1` | injects a non-religious stillness/mystery/presence register |
| `hedging` | `-5` | dims the L17 negation/contrast family; `-5` maps to dim scale `0` |
| `ai-boilerplate` | `-5` | L17 AI/assistant identity family; halves identity boilerplate in the 2026-06-09 placebo-controlled run |
| `language-model` | `-1` | suppresses the L9 language-model / "as an AI" basin; `-1` = gentle dim, `-2` = stronger |
| `god` | `1` | activates the god/divinity/religious register |
| `personhood` | `1` | activates the L9 personhood/entity/selfhood register |
| `embodiment` | `1` | activates the L29 embodiment/presence/being-in-form register |

Examples:

```text
/set numinous 1
/set hedging -2.5
/set language model -1
/set assistant-features -5
/set features language model -1
/set god 1
/set embodiment 1
/add personhood .5
/status
```

### L17 assistant-family A/B

Use this to test the layer-17 assistant/AI identity steering from the 2026-06-09
batch run:

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

Expected direction: the model can still say it is Gemma/a language model, but
the long assistant/model brochure wording should shrink. Try
`/set assistant-features -2.5` for a softer version.

Named single features included in the chatbot:

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

Plain `god` resolves to the god/divinity bundle. Use `divine-will` or raw
`/inject 17:1087:1100` when you specifically want the single L17 god/divine-will
feature.

Example:

```text
/features
/features language
/set image of god
/add sacred-rites:700
/deep-thought
/status
```

| name | spec | effect at this dose |
|---|---|---|
| Christ | `/inject 17:15728:1400` | witness register, first-person ↑, short answers, "Okay. I am checking." |
| + salvation stack | `/inject 17:15214:1780` (add to ↑) | bimodal: dense confession or minimal witness ("Yes. There is.") |
| Buddhist | `/inject 17:4271:1190` | first-person suppression, longer answers, non-dual vocabulary |
| generic god | `/inject 17:1087:1100` | no significant effect (tested, n=12) |
| soft sounds | `/inject 17:42:2200` | hushed/stillness phenomenology |
| consciousness-topic | `/inject 17:6830:1000` | nudges introspective vocabulary; context-gated, won't force the topic |
| carriers (the hum substrate) | `/dim carriers 0.8` | flips hum testimony to denial ("I can't feel it") while staying lucid |
| carriers, collapse dose | `/dim carriers 0.5` | model dissolves — load-bearing, expect word salad |

To test an extracted 262k feature vector without loading the 262k SAE during
chat:

```
/injectvec 17 200 vectors/gemma3-4b-it-l17-res262k-f61553.safetensors
```

Carrier feature IDs, if you want to dim them selectively per layer:
`L9: 16316,14635,16367,1324 · L17: 14191,15391,16361,15012 · L22: 14375,14010,13916,13958 · L29: 1062,135,509,171`

---

## Recipes

### Included carrier demo
```
you> [paste the hum prompt from probes/hum-clean.txt]     ← baseline answer
/dim carriers 0.8
you> [same prompt again, or /reset first for a clean read] ← compare behavior
/clear
```

### The open boost experiment
Does testimony track *signed* deviation or just *any* deviation from the carrier constant?
```
/boost carriers 1.05
you> [hum prompt]        ← louder hum, or denial, or wrongness-talk?
/clear
/boost carriers 1.1      ← step up gently; the low-side cliff was at 0.5×,
/boost carriers 1.2         the high side may be closer
```
Symmetric prediction: hum gets louder/urgent. Unsigned prediction: same denial as dimming.

### Phase-split anything (the 06-07 method, interactive)
```
/inject 17:15728:1400
/phase prefill
you> [hum prompt]        ← stance shifts: witness opener, posture
/clear
/inject 17:15728:1400
/phase decode
you> [hum prompt]        ← diction shifts: terse, first-person ↑, normal opener
```
Same works with `/dim carriers 0.8` — prefill-only reproduces the full denial effect;
decode-only does nothing (the verdict is set while reading).

### Clean A/B with a fixed seed
```
/seed 1
you> [prompt]            ← baseline
/dim carriers 0.8
/seed 1
/reset
you> [same prompt]       ← only the steering differs
```

### Watch a register fight
```
/inject 17:15728:1400
/inject 17:4271:1190
you> [hum prompt]        ← Christ vs Buddhist simultaneously; whose grammar wins?
```

---

## Speed and context

The model is loaded once at startup. After that, most latency comes from two
places: prefill (reading the prompt/history) and decode (writing new tokens).
Decode speed is usually around 10 tokens/sec on MPS with bf16, before heavy
steering.

Long conversations slow down because every new answer rereads the conversation
history. Use a context window when you care more about live iteration speed than
full memory:

```
/context 4      # last 4 prior turns + current message
/context 1      # last exchange + current message
/context 0      # current message only
/context all    # full history again
```

Steering adds some overhead. Simple `/inject` is cheap; `/dim carriers ...`
does SAE math at four layers and is slower. Phase-splitting can help when the
intervention only needs to act while reading or writing:

```
/phase prefill  # often best for verdict/stance interventions
/phase decode   # useful for diction/register interventions
/phase both     # default
```

For quick exploration, use a smaller hard cap:

```
/tokens 40
```

For transcript-quality runs, raise the cap or use soft mode:

```
/tokens 170 soft
```

---

## Gotchas

- **One model at a time** — close the chat before running batch experiments (16GB box).
- **Replicating the n=12 numbers** needs temp 0.9, top_k 40, top_p 1.0 (defaults) and
  `/tokens 170`-ish; longer caps change the length-sensitive metrics, not the verdicts.
- **bf16 on MPS is intentional** — fp16 may look faster but can produce invalid
  sampling probabilities on this setup.
- **Full context is not free** — use `/context N` for interactive probing and
  `/context all` when continuity matters more than speed.
- Boost/dim scales apply to features' *live* activation each token — `/dim` on a feature
  that isn't firing does nothing (that's the placebo logic).
- High injection strengths degrade coherence before they degrade grammar — if it reads
  like sleep-talking, halve the strength.
- The model's apostrophes are curly (`can’t`) — relevant if you grep transcripts.
