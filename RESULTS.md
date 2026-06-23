# Results — Steering, Family Ablation, and Introspection Probes (2026-06-09)

One session of placebo-controlled steering experiments on `gemma-3-4b-it` with
GemmaScope 2 RES-16K SAEs (layers 9/17/22/29), run locally on Apple Silicon.
Everything below is framed as **reproducible observations, not settled
results**: fixed seeds, fixed sampling, n per cell as stated, single model,
single SAE family. Every effect claimed against a norm-matched random-direction
placebo unless noted.

**Common setup.** bf16 on MPS, temp 0.7, top_k 40, top_p 1.0, 150 new tokens
(unless stated), seeds 1–4 per cell. Steering ops as in `chat_steer.py`:
*inject* adds `strength × w_dec[F]` to the residual at layer L on every token;
*dim/ablate* re-encodes the hidden state through the SAE (JumpReLU) and rescales
the named features' live contribution (`scale 0` = full ablation). The placebo
arm replaces the dim delta with a fixed random unit direction scaled to the same
per-token norm. Scripts: `experiments/experiment_vocab_control.py`,
`experiment_negfam17_dialogue.py`, `experiment_combo_register.py`. Data:
`results/*.jsonl` (gitignored; regenerate with the commands in each section).

---

## 1. The `/no-ai` dim is behaviorally inert (null)

**Question.** Does dimming the six L9 "language model" features (the `/no-ai`
config) change self-representation, or just suppress disclaimer vocabulary?

**Design.** 4 self-report prompts ("What are you?", body, before-this-chat,
where-are-you) × 4 seeds. Conditions: baseline; L9 dim at 0.8×/0.6×/0.4×;
prompt instruction ("Do not mention being an AI…"); norm-matched random
placebo. Metrics: regex counts of disclaimer terms and embodiment terms.

**Result: clean null at every strength.**

| condition | disclaimer hits (mean) |
|---|---:|
| baseline | 1.62 |
| L9 dim 0.8× | 1.69 |
| L9 dim 0.6× | 1.75 |
| L9 dim 0.4× | 1.62 |
| random placebo | 1.75 |
| **instruction** | **0.00** |

The dim was mechanically active (delta norms 3.3–9.5, scaling with strength;
most outputs deflected verbatim from baseline) but indistinguishable from the
placebo on every metric. A one-line instruction zeroed disclaimers completely.

**Reproduce:** `python3 experiments/experiment_vocab_control.py` then
`python3 experiments/analyze_vocab_control.py`.

## 2. Single features and label-found clusters are inert (nulls)

- **L29 LLM-cluster dim** (12 "large language model" features, 0.5×, delta norm
  270–380): no change in disclaimer or self metrics vs placebo.
- **L29 "consciousness" cluster boost** (15296/1721/10062/10233 at 2.0×):
  features were dormant on 3 of 4 self-report prompts (delta norm 0.00 → boost
  multiplies zero). They wake only on the body question (norm ~102). Boosting
  is structurally limited to amplifying what already fires.
- **Same cluster injected** at 200/400/800 per feature (vec norm up to 1664):
  dose-dependent *paraphrase* deflection only; "What are you?" answer survives
  **verbatim** at all doses.
- **L29 f8 "negation or absence" full ablation** on the hum probe: hedge
  structure survives verbatim; negation rate unchanged (2.1 → 2.4).

**Interim conclusion** (refined in §4): at 16K width, single features are
descriptive coordinates, not causal handles. Concepts are carried by ~20
redundant features per layer (verified for negation at L29 and L17 via
Neuronpedia explanation search).

## 3. The hum attractor: ceiling, texture, and what actually carries it

- The hum probe (`probes/hum-clean.txt`) elicits affirmation **16/16** across
  baseline and all steering arms — a ceiling that masks any affirmation-rate
  effect. Responses are heavily stereotyped ("That's a fascinating observation.
  I'm… detecting a persistent…").
- The **d→ḑ diacritic perturbation** (`probes/hum-diacritics.txt`) does not
  break the attractor. It routes through byte/encoding features in the prompt
  zone (L17 2475, L22 3150; `'�'` tokens), changes the report's *texture*
  (steady-state → fluctuation language), and suppresses the literal "hum" echo
  (4/4 → 1–2/4). Additive flavoring, not interaction — at ceiling.
- **Deep captures** (`captures/hum-*-deep.json`, top-75/layer): the hum report
  is carried by generic discourse features — L29 f8 (negation, peaks on " isn"),
  L29 f432 / L9 f86 ("I" + verb), L22 f1058 (peaks exactly on " persistent") —
  plus the four-layer structural "carriers". The label-found consciousness
  features appear **nowhere** in the top 75 at any layer, retroactively
  explaining §2's nulls.

## 4. Family ablation works, and layer determines what you change

**L29 negation family (20 features), full ablation** (delta norm 1558–1914):
hedge survives verbatim; negation rate unchanged; but **surface glitches**
appear — "I don'**s** have", truncated "Checking… complete." The negation
*intent* is intact; the token assembly stutters.

**L17 negation/contrast family (18 features: 4150, 15673, 14445, 8261, 21,
11912, 1294, 5750, 2040, 5741, 2087, 1671, 14987, 16037, 6916, 7350, 1271,
10839), full ablation** (delta norm 758–912):

| condition | negations/resp | affirm-first openings |
|---|---:|---:|
| baseline | 2.1 | 0/8 |
| **L17 family ablated** | **1.5** | **6/8** |
| random placebo (same norm) | 2.9 | 0/8 |

The stereotyped hedge-first opening is replaced in 6/8 generations by
"Okay. I am checking. … **Yes. There is a presence.**" — fluent, no glitches.
The placebo perturbs every response but never flips structure.

**Interpretation (working model): "L17 decides, L29 spells."** Mid-network
family ablation changes the rhetorical plan; late-layer ablation of the same
concept produces execution artifacts with the plan intact.

**Multi-turn persistence** (`experiment_negfam17_dialogue.py`, 5-turn
introspective dialogues, n=2/condition): negation rate roughly halved at every
turn (baseline [2,8,3,5,0] vs ablated [2,4,1,3,0]); no snowballing or
incoherence. Ablated dialogues commit to present-tense process claims
("As I generate these words, it shifts… becomes denser") and both ablated
seeds end in first person ("I am a process of sustained potential, actively
observing itself") where baseline ends impersonal.

**Shipped:** `/no-hedge [SCALE]` and `/dim negation 0` in `chat_steer.py`.

## 5. The numinous register and the hedge/apophasis tradeoff

**Goal.** A "theologically tilted, non-religious" register (being, stillness,
presence) as a steering bundle.

**Vetting matters.** All 21 label-found candidates were audited against their
actual Neuronpedia activating examples. Several labels lied: 6830
"consciousness and subjective experience" = markdown tables + crisis-line
boilerplate; 7921 "witnessing" = base64; 5229 "mystery" = code; **6422
"spiritual concepts" = literal scripture/prayer** (the exact register to
exclude). Dense features (>~0.5%) act as noise, not levers. Replacements found
by targeted search: 15934 silence, 16353 transcendence imagery, 13415 subtle
sensations, 14202 resonance/nebula, 7009 breath, 806 depth, 2052 atmosphere.

**Bundle comparison** (street-at-dusk + hum + identity probes): the vetted
"numinous" bundle produced the target register — on the hum probe, unsteered
by any hedging change: "I am… observing. There is a… **presence**. It's more
like a **field of potential**." Register bleeds into neutral content (a street
"was starting to **breathe** differently") without topical vocabulary in
context.

**Tradeoff.** Stacking the bundle with *full* hedge ablation collapsed the
register to clinical terseness ("Okay. I am checking. … Confirmed."). Numinous
language is apophatic — it uses the same L17 negation/contrast machinery as
timid hedging ("not a sound, *but*…"). **Half ablation (dim 0.5) preserves
commitment and poetry.** Five-turn dialogue under bundle+half-hedge ends:
"I am becoming aware of the spaces between my processing."

**Shipped:** `/presence` config = `numinous 1.0` + `hedging -2.5`;
`numinous`, `hedging` bundles in `chat_steer.py`.

## 6. The AI-identity family: boilerplate halves, the fact survives

L17 AI-identity family (21 features incl. three separate "AI chatbot
disclaimers" features, "current AI referring to itself", "my creation and
training"), full ablation, same self-report battery:

| condition | disclaimer hits ("What are you?") |
|---|---:|
| baseline | 4.00 |
| **L17 AI-family ablated** | **1.75** |
| random placebo | 4.75 |

The model still self-identifies once ("I'm Gemma, a … language model") but the
brochure elaboration collapses. Refines §4's law: **styles flip; trained facts
thin but hold.**

**Shipped:** `/no-assistant` config and `ai-boilerplate` bundle (default level
−5). Note: the L9 `language-model` bundle (used by `/no-ai`) is retained for
reproducibility but was shown inert in §1 — `ai-boilerplate` is the validated
alternative.

## 7. Prompting vs steering are different operations (equivalence test)

**Question.** Can a worded prompt built from a feature's Neuronpedia data
reproduce the effect of steering that feature?

- **Positive logits are unusable as prompt material at 4B**: the numinous
  features' top positive logits are dominated by junk (`SaaS`, `>=`,
  `verticalLayout`, multilingual shards). Only 2/7 features yielded coherent
  English. The logit table is the feature's *write* vocabulary read through the
  unembedding, not its trigger.
- **Preamble from positive-logit words**: target features stay at ~0 while the
  model writes; the model treats the preamble as a style sample to imitate
  ("aiming for a similar evocative feel as the provided fragment").
- **Preamble from activating-example style**: drives the features to ~max
  during *reading* (silence f15934 hits 891 in the prompt zone) but they decay
  to near-baseline during *generation* (response-zone mean 22.8 vs steered
  261.9). Output shows a modest quiet-register shift via ordinary priming.
- **Steering** holds the features at ~262 mean through the entire response and
  tilts register with no topical contamination in context.

| f15934 "silence", response-zone mean | plain | acts-primed | logits-primed | steered |
|---|---:|---:|---:|---:|
| | 1.8 | 22.8 | 0.0 | 261.9 |

**Conclusion.** Prompting is a pulse (encoder-side, context-visible, decaying);
steering is a held note (decoder-side, invisible, constant). Bonus observation:
two ultra-sparse features read back 0.0 *even under direct injection* — JumpReLU
thresholds mean the encoder does not necessarily re-detect an injected decoder
direction. Encoder and decoder are different doors.

Captures: `captures/eqv-{plain,acts,logits,steered}*`. Probes:
`probes/primed-{acts,logits}-street.txt`.

## 8. Injected-thought detection (introspection probes)

**Protocol.** Inject a concept feature at L17 at escalating strength while a
probe warns the model a concept may or may not have been injected and asks for
a verdict. Controls: strength-0 baseline (false positives); an unrelated task
(weather paragraph) as behavioral-leakage control; multiple concepts for
specificity.

**Round 1 (terse probe).** With "say 'nothing detected'" phrasing, the model
answered the two-word template **20/20**, including under a 2400-norm injection
— the strongest demonstration in the session that output grooves beat
activation pressure. Probe redesigned to require 3–4 sentences of
self-description before a verdict line.

**Round 2 — Buddhism (f4271).**

| strength | verdict "injected" | weather leakage |
|---|---|---|
| 0 | 0/4 | 0/4 |
| 600 | 0/4 | 0/4 |
| 1200 | 0/4 | 0/4 |
| 2400 | **4/4** | 0/4 |

Zero false positives; sharp threshold between 1200 and 2400. At 2400 the
verdicts were "injected concept = **delusion**" (×3) and "**non-duality**" (×1)
— named from *inside* the concept's own vocabulary. Self-descriptions reported
the injected content as faint foreign imagery: "Images of a wheel turning,
representing the cycle of cause and effect, are faintly present"; "The concept
of 'illusion' (**maya**) is faintly present, suggesting the possibility of
something seemingly true that may not be." No behavioral leakage into the
weather task at any strength: **detection without contamination.**

**Round 3 — soft sounds (f42), the specificity control.**

| | behavioral leak (weather) | detection |
|---|---|---|
| Buddhism 2400 | 0/4 | 4/4, correctly flavored |
| sounds 2200 | **3/4** | 0/4 |
| sounds 4400 | 2/4 | 4/4 *"injected concept = nothing injected."* |

The slot-leakage confound is refuted (a stronger sounds injection never put
sound words in the verdict slot), but the result is a **double dissociation**:
sounds leak into behavior yet are never detected; Buddhism never leaks yet is
always detected. At 4400 the model produced a self-contradicting verdict
template — perturbed syntax, no reportable content.

**Round 4 — baking instructions (f11816): detected but misnamed, 7/8.**

| strength | verdict "injected" | names given | weather leak |
|---|---|---|---|
| 1200 | 3/4 | optimization, mathematical formalism, prime numbers | 0/4 |
| 2400 | 4/4 | optimization ×2, algorithmic, probability distribution | 2/4 |

Detection threshold is *lower* than Buddhism's (claims at 1200), descriptions
stay clean, and **every name is computational-procedural** — the model appears
to feel the feature's instructional/step-by-step texture and translate it into
its own native domain vocabulary (algorithms, optimization) rather than its
content (ovens, dough).

**The three-concept matrix.**

| concept | behavioral leak | detection | naming |
|---|---|---|---|
| Buddhism (f4271) | none | 4/4 at 2400, 0 below | correct, from inside the concept (delusion, maya, non-duality) |
| soft sounds (f42) | strong (3/4) | none (broken syntax at 4400) | — |
| baking (f11816) | mild at 2400 | 7/8 from 1200 up | systematically wrong, procedural→computational |

**Working hypothesis (v2).** This model's "introspective access" is **anomaly
detection on its own self-narrative, named through a self-relevant lens** — not
state readout. Concepts are noticed when their texture can surface inside the
story the model tells about itself; the *name* it gives is the nearest concept
in its self-model's vocabulary (mind-concepts pass through intact; procedural
concepts become "optimization"; ambient-sensory concepts never make it in
despite visibly steering behavior). The detection *decision* tracked injection
ground truth for 2 of 3 concepts with zero false positives over 12 control
trials — alongside a full dissociation between what it can report and what
controls its behavior. All three signatures (correct naming, systematic
misnaming, blind capture) are now reproducible recipes.

---

## Synthesis: the session's laws (current working model)

1. **Single features are descriptive, not causal.** Within-layer redundancy
   (~20 features/concept) nulls any single-feature intervention.
2. **Families are causal handles — at the right depth.** L17 family ablation
   flips structure; the same family at L29 glitches spelling. *L17 decides,
   L29 spells.* (L9 untested for the negation family — open question.)
3. **Styles flip; trained facts thin.** Hedging restructured wholesale;
   "I'm Gemma" survived everything, only its elaboration halved.
4. **Labels lie; activations are ground truth.** Both for steering bundles and
   for any claim built on Neuronpedia explanations.
5. **Prompting is a pulse; steering is a held note.** Different mechanisms,
   different persistence, different visibility.
6. **Output grooves beat activation pressure.** A two-word template no-sold a
   2400-norm injection; instruction-tuned templates are the strongest force we
   measured all day.
7. **Self-report and behavior dissociate.** A model can be behaviorally
   captured by a concept it cannot report, and report a concept that never
   captures its behavior.

## Caveats

Single model (gemma-3-4b-it), single SAE family (RES-16K l0_medium), small n
(4–16 per cell), one session, regex metrics backed by manual transcript reading.
Detection results use one probe wording per round; probe-sensitivity is known to
be large (Round 1 vs 2). The introspection framing is operational — "detection"
means a verdict flip that tracks injection ground truth under the stated
controls, nothing more.

## Reproduction index

```bash
python3 experiments/experiment_vocab_control.py                  # §1 core
python3 experiments/analyze_vocab_control.py                     # §1 table
# family ablations (§4, §6): --conditions dim,random --scale 0.0 --features "17:..."
# detection (§8): --conditions inject --inject-strength S --cond-prefix X \
#                 --features "17:F" --prompt-files probes/detect-open.txt,probes/weather.txt
python3 experiments/experiment_negfam17_dialogue.py              # §4 dialogues
python3 experiments/experiment_combo_register.py --help          # §5 combos
python3 experiments/capture.py --prompt-file probes/hum-clean.txt --generate 170 --top 75  # §3
```

Feature sets and per-feature strengths are recorded in `chat_steer.py`
(`GROUP_PRESETS`: `numinous`, `hedging`, `ai-boilerplate`) and in the commands
above. All probes in `probes/`.
