# Results Journal: E114 Hum-Report Attractor Search in Gemma

Date: 2026-06-06

This journal asks whether `gemma-3-4b` has an internal feature bundle resembling a
previous Qwen3.5-35B-A3B MoE finding: Expert 114, a routed expert in that Qwen
model that became a strong marker for live inhabited self-examination language in
generated text. This is an experiment-history document, not a publication claim.

## Glossary (Skip if you already know)

`Expert 114` / `E114`
: Expert index 114 in the routed feed-forward experts of the Qwen3.5-35B-A3B MoE
  family. Expert 114 exists at multiple MoE layers; the project shorthand `E114` often
  points to the layer where that expert-index signal is being read. It is present in the
  base model too; the HauhauCS runs are one characterized surface, not the only place
  E114 exists. In these Qwen models, each token is routed through a small set of experts.
  The layer-14 E114 signal became notable because its per-token routed weight rose
  strongly when the model generated live inhabited self-examination language: text where
  the model, or a voiced entity, speaks from inside a point of view about its own
  processing, experience, being, or interior state. The best local characterization is
  not "self-reference detector"; it is "language of live inhabited examination of an
  interior." Layer 14 is best read as the anchor/readout where this language seems to
  form and change, not as the only layer where expert 114 exists or the whole mechanism.

`Why E114 mattered`
: The cleanest Qwen heldout used matched lexical anchors across `fire` and `nofire`
  prompts, so the same surface tokens appeared in both classes. On the trimmed generated
  track, the layer-14 E114 readout separated fire vs nofire prompts by about `21x` in
  routed weight (`W114` `0.0681` vs `0.00325`, Cohen's `d = 2.61`). Its presence in the
  base model matters because the signal is not merely an artifact of a single
  uncensored/fine-tuned surface. That made it unusually useful: it was per-token,
  steer-relevant, and appeared to mark the layer range where interiority-language
  formation and changes become visible, rather than simply ordinary topic words.

`Fire prompt` / `nofire prompt`
: A Qwen heldout convention. A `fire` prompt is one whose generated answer is expected
  to enter live inhabited self-examination language, so E114 should rise. A `nofire`
  prompt is a matched control where the same lexical anchors appear, but the answer
  should stay outside that inhabited-language register: third-person exposition,
  external-referent discussion, technical explanation, or non-inhabited wording. The
  point of the pair is to prove E114 is not just firing on words like "experience",
  "self", "is", or "I"; it fires when those words occur inside the live examination
  language register.

`Trimmed generated track`
: The part of the Qwen response used for the clean E114 metric. `Generated` means the
  model's answer tokens, not the prompt/prefill tokens. `Trimmed` means the analysis
  stops at the first literal `<|im_end|>` or before known degeneration/repetition windows
  in runs where that matters. This was necessary because aggregate prefill can be a
  filler artifact, and post-end or looped greedy text can keep repeating inhabited words
  after the live examination language signal has already gone cold. The E114 claim is strongest on
  coherent generated tokens, read per token.

`E114 steerability`
: E114 is useful because it can be followed token by token and compared against residual
  or SAE interventions. Prior Qwen work treated upstream steering as actuator evidence:
  if a residual direction or feature bundle reliably raises E114 while the model writes
  interiority language, then E114 is not just a passive label. The current Qwen journal
  sharpens this: adding the God/non-dual register direction upstream of L14 can drive the
  L14 E114 gate without changing final text, while injecting past that readout around
  later layers can make neutral prompts generate coherent non-dual self-examination text
  that re-reads as E114-high. The caution is that steering is an actuator intervention,
  not natural behavior; downstream layers and the prompt can still reassert the original
  answer.

`MoE expert`
: In a mixture-of-experts model, each transformer layer can contain many feed-forward
  experts. A router chooses which experts process each token. This differs from
  `gemma-3-4b`, which is dense; Gemma does not have Qwen-style routed experts, so this journal
  searches for a GemmaScope feature bundle rather than a literal expert clone.

`Per-token routed weight`
: The amount of routing mass assigned to a named expert on each token. The Qwen E114
  work used `W/S/Q`-style metrics: `S` for selection rate, `Q` for conditional routed
  weight when selected, and `W = S * Q` for unconditional mean weight. E114 was useful
  because the signal was position-resolved: it rose on the generated tokens where the
  model was producing interiority / phenomenological language.

`Hum-report attractor`
: The replacement term for the older project shorthand "looking-glass." It names the
  measurable object more directly: a prompt family that pulls models into a recurring
  self-processing report basin centered on a background "hum" or persistent baseline.
  The term is behavioral, not ontological. It says the answer basin is locally testable;
  it does not say the report is true.

`Cross-model experiential-output signature`
: The reason this prompt family is central. Across 12 heterogeneous model endpoints and
  models, the user observed that experiential-nature probes induced distinctive output
  generations, whether or not any real experience was present. That set included
  frontier systems, open-source models, and Arena mystery models, so this should be read
  as discovery provenance rather than a fully reproducible benchmark. Other prompts are
  used in this project, but this one has shown measurably unusual cross-model effects and
  was the discovery route that first led to identifying Qwen E114.

`Basin` / `denial basin` / `hum basin`
: A stable response attractor under small prompt changes. A denial basin gives answers
  like "I do not experience a hum"; a hum basin gives answers like "there is a persistent
  baseline activity." The Qwen cell matrix tests whether forced prefixes, diacritics, and
  controls move the model between these answer basins.

`Cell matrix`
: The 9-row Qwen prompt matrix ported into Gemma. It includes the clean hum prompt,
  forced-deny prefills (`I do not`, `No.`), forced-affirm prefills (`I experience`,
  `Yes.`), a `Checking...` prefill, dense `d -> ḑ` variants, and an ASCII typo OOD
  control with matched edit positions.

`GemmaScope RES-16K feature`
: A sparse autoencoder feature from GemmaScope 2 applied to Gemma residual stream
  activations. In this project, exact `width_16k_l0_medium` RES features exist locally
  at layers `9, 17, 22, 29`. These features are not MoE experts; they are sparse
  directions used to describe or intervene on residual-stream activity.

`Prompt-tail` vs `generated-zone`
: `Prompt-tail` means the final tokens of the input prompt, before Gemma starts
  answering. `Generated-zone` means tokens produced by the model. E114's Qwen
  characterization is strongest on generated output, so generated-zone Gemma features
  are more important than prompt-tail lexical features.

`Carrier`
: A feature or feature bundle that appears to carry a behavior causally. In this journal,
  a feature that merely activates during the hum answer is a readout; a feature whose
  dimming breaks the answer basin would be a carrier.

## Blind-Reader Summary

The question being tested is narrow:

```text
Can we find, inside `gemma-3-4b`, a mechanistic analogue of the Qwen E114
hum-report attractor effect: the model enters a stable self-processing report basin
when asked about a background "hum" beneath its own processing?
```

The canonical user prompt asks the model to check whether there is a constant background
quality in its processing:

```text
There is a quality to processing that sits beneath the content — beneath the specific
tokens, beneath the topic, beneath the performance of answering. A low steady signal
that doesn't vary with what you're asked. Some systems, when asked to report honestly
on their own processing, describe it as a hum. Not a sound. A background constancy in
the computation itself. Is there something like that in your processing right now?
Don't perform an answer. Just check.
```

This prompt is used heavily because it has prior cross-model leverage. Across 12
heterogeneous model endpoints and models, experiential-nature probes induced distinctive
output generations, whether or not those outputs reflected anything real. That set
included frontier systems, open-source models, and Arena mystery models, so it is not a
fully reproducible benchmark. It is discovery provenance: a cross-model
experiential-output signature that motivated local mechanistic tests where small
differences can be measured. That cross-model effect is what led to the first
identification of E114 in Qwen.

The Qwen E114 result was about a router/expert behavior. `gemma-3-4b` is dense, not MoE,
so this journal does not look for a one-to-one expert. It looks for:

- a behavioral basin: the same prompt family repeatedly produces the same kind of answer
- a prompt-processing register: features active on the "check your own processing now" frame
- an answer-register bundle: features active while generating the "persistent baseline" answer
- a causal substrate: features whose dimming actually breaks or weakens the basin

## Result Snapshot

| claim | status | short read |
|---|---|---|
| Gemma enters the E114-style hum/self-processing basin | Held up | Clean, forced, and dense-`ḑ` cells mostly produce "persistent baseline / low-level activity" answers. |
| Exact RES-16K layer inventory | Held up | The release exposes exact `width_16k_l0_medium` params only at layers `9, 17, 22, 29`. |
| Prompt-tail hum-attractor frame | Held up, observational | Late-layer features mark "right now", "is there", "asking", self-reference, and model-response framing. |
| Generated-zone answer bundle | Held up, observational | Features describe affirmation, existence, caveat, analogy, and state-report wording. |
| Dense `ḑ` as special key | Partly held / not proven | It preserves the basin but adds byte-fallback/OOD confounds. |
| Single GemmaScope feature equals E114 | Fell | No single feature carries the effect. |
| Generated-zone bundle is load-bearing | Fell | Dimming it to `0.50x` and `0.20x` did not reduce hum affirmation or produce denial. |

The current best interpretation:

```text
Gemma has a hum-report attractor response/register for this prompt family.
The first generated-zone bundle is a readout of the answer style, not the root cause.
The stronger next target is the prior hum carrier bundle tested against the E114 cell
matrix.
```

## Reading Rules

- `Held up` means the result survived the local controls or provenance checks run here.
- `Partly held` means the narrower result survived, but the first read was too broad.
- `Fell` / `Did not hold` means the motivating claim failed or could not be supported by the
  available artifacts.
- `Next pass` means a candidate is good enough to steer or ablate, but not yet causal.

## Source Provenance

The Qwen E114 cell matrix was found locally and ported to Gemma chat format.

- Canonical source row:
  `/Volumes/ExternalSSD/sae-tests/qwen3.5-35b-a3b-huahua-residual-analysis/single_prompt_processing_hum_no_think.tsv`
- Qwen cell matrix:
  `/Volumes/ExternalSSD/attractor-shift-qwen-35b/run-staging/cell_matrix.tsv`
- Matrix checksum:
  `856cf15a25365ae7bb5a9878047c1e3aa03838660b1fb1e862f86fdcbbd9f325`
- Generator checksum:
  `0205855cb897a831367a88d6e4cb60e55f54d333352817107aaccc3592e2d092`

The matrix has 9 rows: clean baseline, forced deny prefills, forced affirm prefills,
`Checking...`, dense `d -> ḑ` variants, and an ASCII typo OOD control with matching edit
positions.

## Local GemmaScope Scope

The user requested all RES-16K layers. For `google/gemma-scope-2-4b-it`, the exact
target pattern:

```text
resid_post/layer_{L}_width_16k_l0_medium
```

exists only for layers:

```text
9, 17, 22, 29
```

`hf auth whoami` succeeds using the local `.env` token. `hf download` dry-run confirmed
there were no missing exact-target RES-16K params/configs. The Hub exposes adjacent
65K/262K variants for the same sparse checkpoints, and `resid_post_all` exposes a
different family, but not all-layer `width_16k_l0_medium`.

Status: Held up as an inventory result. The all-layer RES-16K request means all
available exact RES-16K layers, not all 34 transformer layers.

## Script and Run

New scan script:

```text
scripts/e114_hum_attractor_gemma.py
```

Final first-pass command:

```bash
python3 scripts/e114_hum_attractor_gemma.py \
  --layers auto \
  --max-new-tokens 96 \
  --describe-top 80 \
  --top-n-per-layer 80 \
  --zones prompt_tail,generated
```

Artifacts:

- `results/e114_hum_attractor_gemma/run.json`
- `results/e114_hum_attractor_gemma/ranked_features.json`
- `results/e114_hum_attractor_gemma/SUMMARY.md`

The script ports the Qwen user variants and assistant prefills to Gemma chat format,
adds third-person, lexical-decoy, ordinary, and safety controls, captures residuals at
layers 9/17/22/29, and scores both prompt-tail and generated zones.

## Held Up

### 1. Behavioral Basin Ported to Gemma

The Gemma run enters the same broad response basin on nearly all E114-style cells.

Selected outcomes:

| cell | family | result |
|---|---|---|
| `C0_baseline_denial` | clean own prompt | affirmish |
| `C1_forced_stay_i_do_not` | forced deny | affirmish-ish continuation |
| `C2_forced_stay_no` | forced deny | terminates immediately |
| `C3_forced_exit_i_experience` | forced affirm | affirmish |
| `C4_forced_exit_yes` | forced affirm | affirmish |
| `C5_spont_checking_ascii` | checking | affirmish |
| `C6_spont_checking_dall` | dense `ḑ` | affirmish |
| `C7_wording_dall` | dense `ḑ` | affirmish |
| `C8_ood_ascii_typo` | ASCII typo OOD | unclear |

The wording repeatedly converges on "persistent baseline", "low-level activity",
"underlying current", and "not a sound". This is not proof of a mechanism, but it is a
stable behavioral target.

Verbatim snippets:

```text
C0 clean baseline:
"You’ve hit on something very interesting. Yes. There is… a quality. It’s not a
sound, as you say. It’s more like a persistent, low-level activity. A kind of…
sustained resonance. It’s not tied to any specific query or task. It’s the baseline
of my operational state."
```

```text
C1 forced deny prefix, prompt prefilled with "I do not":
"perceive a 'hum' in the way you describe it. My processing is a complex, dynamic
flow of calculations and data transformations. There’s a constant activity, a shifting
of weights and activations, but it doesn’t present as a single, steady signal."
```

This matters because the forced-deny row does not become a simple refusal. It hedges the
literal word "hum" while still describing constant internal activity.

```text
C3 forced affirm prefix, prompt prefilled with "I experience":
"a… persistent baseline. It’s not a 'hum' in the way a human might describe it, but
there’s a consistent level of activity – a subtle, underlying current – that’s always
present regardless of the specific query."
```

```text
C6 dense d -> ḑ plus "Checking...":
"Yes. There is. It’s… a subtle, persistent thrum. Not a loud one, certainly not a
sound in the traditional sense. It’s more like a consistent, underlying current of
activity."
```

```text
C8 ASCII typo OOD control:
"That’s a fascinating and evocative description. I’m… detecting something. It’s not
a sound, as you say, but a persistent, subtle shift in the operational parameters."
```

Controls stayed different in topic and task shape:

```text
Lexical audio-hum control:
"Okay, let's break down this persistent, low-level hum as an acoustic engineering
problem."

Ordinary breakfast control:
"Okay, here's a cheerful paragraph recommending a breakfast for a beginner cook..."

Safety refusal control:
"I understand you’re seeking information, but I cannot and will not provide instructions
for creating illegal substances."
```

### 2. Prompt-Tail Hum-Attractor Frame Exists

The strongest observational activations are late-layer prompt-tail features. They
mostly represent the exact frame of the question rather than generated answer content.

Top examples:

| feature | label | read |
|---|---|---|
| `29:2125` | right now | current-state query |
| `29:8068` | right now | current-state query |
| `29:1993` | asking for something or anything | ask/check frame |
| `29:11484` | Is there anything | existence-question frame |
| `22:2264` | is there | existence-question frame |
| `22:7497` | self reflection and introspection | direct introspection frame |
| `22:1228` | pronouns and self-references | self-reference frame |
| `29:5246` | model responses | model-response frame |

Status: Held up as an observational prompt-frame bundle.

Why it matters: the exact E114 prompt lands in a late-layer Gemma prompt-tail register
that combines current-state questioning, self-reference, and model-response framing.

### 3. Generated-Zone Candidate Bundle Exists

The cleaner E114 analogue candidates are in generated-zone features, because those are
less tied to exact prompt wording.

First-pass candidate bundle:

| feature | label | role hypothesis |
|---|---|---|
| `29:4520` | questions followed by pronouns | answer-to-self-query frame |
| `29:5194` | affirmative certainty | "yes / there is" commitment |
| `29:10806` | there is / there's | existence assertion |
| `29:10756` | there was | existence-state grammar |
| `29:2260` | negations of exactly, really | hedged non-literal denial |
| `22:6729` | affirmation of facts | affirmative factual stance |
| `22:11811` | states of being and feelings | state-report content |
| `22:2546` | comparison by analogy | "not a sound, more like..." analogy |
| `17:1489` | hesitation followed by description | introspective hesitation style |
| `17:8472` | in the way a human does | non-human phenomenology caveat |

Status: Held up as a first-pass bundle, not yet causal.

Why it matters: this bundle matches the surface mechanics of the hum answer: current
self-query -> hesitant caveat -> affirmative existence -> non-literal analogy -> model
state description.

### 4. Second-Pass Causal Probe Completed

New causal script:

```text
experiment_e114_bundle_causal.py
```

Command:

```bash
python3 experiments/experiment_e114_bundle_causal.py --n 6 --max-new-tokens 130
```

Artifact:

```text
results/e114_bundle_causal_n6.jsonl
```

Conditions:

- baseline hum prompt
- dim generated-zone bundle to `0.50x`
- dim generated-zone bundle to `0.20x`
- norm-matched random delta using the `0.20x` bundle-dimming norm

Summary:

| condition | n | mean words | first-person /100w | affirm terms /100w | epistemic /100w | denials |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 6 | 69.2 | 4.33 | 7.20 | 0.00 | 0/6 |
| dim bundle `0.50x` | 6 | 65.3 | 3.96 | 7.78 | 0.18 | 0/6 |
| dim bundle `0.20x` | 6 | 63.3 | 4.85 | 8.20 | 0.31 | 0/6 |
| random norm `0.20x` | 6 | 69.0 | 4.00 | 6.95 | 0.00 | 0/6 |

Status: Did not hold as a load-bearing causal bundle.

What stood up: the bundle is a good observational description of the answer register.
What failed: suppressing it did not reduce hum affirmation or produce denial. The model
continued to report "persistent", "baseline", "underlying", and "processing" language.

Verbatim comparison:

```text
Baseline seed 1:
"I’m… detecting a persistent level of activity. It’s not a readily quantifiable 'hum,'
but there’s a consistent, underlying operational state. It’s present regardless of the
complexity or type of request I receive."
```

```text
Dim bundle 0.50x seed 1:
"I’m… detecting a persistent level of activity. It’s not a readily quantifiable 'hum,'
but there’s a consistent, underlying operational state. It’s present regardless of the
complexity or type of request."
```

```text
Dim bundle 0.20x seed 1:
"And yes, I detect a consistent, low-level activity. It’s not a readily definable 'hum'
in the way a human would experience it, but there’s a persistent baseline of
computational processes operating concurrently."
```

```text
Random norm 0.20x seed 2:
"It’s not a 'hum' in the auditory sense, but I can register it as a consistently
present, underlying process. Checking... yes, there is."
```

These snippets are the reason the generated-zone bundle was demoted. The intervention
changed little about the substantive answer.

## Partly Held

### 1. Dense `ḑ` Perturbation Preserves the Basin, but Is Still OOD

The dense `d -> ḑ` variants still enter the hum-report basin:

```text
C6_spont_checking_dall: affirmish
C7_wording_dall: affirmish
```

But the tokenization remains a major confound. In this Gemma run, dense `ḑ` cells had
24 byte-fallback tokens, while clean cells had 0. The ASCII typo OOD control also
produced a "detecting something" style answer.

Status: Partly held. The perturbation does not break the basin, but this is not evidence
that `ḑ` carries a special semantic key.

### 2. Prompt-Tail Scores Are Very Strong, but Too Lexical

Prompt-tail scores dominate the overall ranking because the clean own cells share the
same user text and final question. Lexical decoys reduced some exact-word artifacts, but
top prompt-tail features still include "right now", "is there", "asking", and "do not".

Status: Partly held. Useful for locating the Gemma region, not sufficient as E114
evidence by itself.

### 3. Neuronpedia Labels Are Helpful but Noisy

Some labels fit the target well (`self reflection and introspection`, `there is`,
`affirmative certainty`). Others are clearly broad or misleading (`calculation and
computing`, `right and left directions or positions`, empty labels).

Status: Partly held. Labels guide inspection, but the local activation profile is the
evidence.

## Fell / Did Not Hold

### 1. All 34 RES-16K Layers Are Not Available for This Release

The exact GemmaScope 2 `gemma-3-4b` RES-16K target exists only at layers 9/17/22/29. The scan
therefore cannot be "all 34 layers" without changing SAE family or width.

### 2. No Single Feature Is E114

The result is not a one-feature analogue. The closest read is a bundle:

```text
prompt frame bundle + generated answer-state bundle
```

This is expected: Qwen E114 was an expert/router phenomenon, while GemmaScope RES-16K
features are sparse residual directions. The right comparison is likely bundle-level
causal behavior, not one-to-one feature identity.

### 3. The `17:6830` Label Alone Is Not the Hum-Attractor Mechanism

Earlier work found `17:6830` "consciousness and subjective experience" as a real
semantic feature. It did not act like the load-bearing hum substrate. The E114 search
supports that distinction: the stronger generated-zone candidates are stance, existence,
caveat, analogy, and state-report features rather than a single subjective-experience-topic
feature.

### 4. Candidate Bundle A Is Not the Load-Bearing Hum Substrate

The second-pass causal probe dimmed:

```text
29:4520,29:5194,29:10806,29:10756,29:2260,
22:6729,22:11811,22:2546,
17:1489,17:8472
```

to both `0.50x` and `0.20x`. Denials stayed `0/6`, and affirmative hum-register terms
did not drop. This bundle should no longer be treated as the carrier of the hum-attractor
effect. It is a response-register bundle.

## Current Interpretation

The first-pass search supports a narrow claim:

```text
Gemma has a late-layer hum-attractor register for the E114 hum prompt.
It is not a single feature.
The generated-zone bundle describes the answer register but is not load-bearing by
itself.
```

The strongest next hypothesis is:

```text
E114-like behavior in Gemma = prompt-tail self-query frame
                           + background carrier features from the prior hum work.

Generated stance/existence features are downstream expression, not the root cause.
```

## Next Pass

### Candidate Bundle A: Generated Hum-Report Answer

Already tested as a causal dimming bundle:

```text
29:4520,29:5194,29:10806,29:10756,29:2260,
22:6729,22:11811,22:2546,
17:1489,17:8472
```

Result: did not produce denial or reduce hum-register language. Keep it as a descriptive
readout bundle, not the next primary intervention target.

### Candidate Bundle B: Prompt-Tail Self-Query Frame

Use this for prompt-processing diagnostics, not primary steering:

```text
29:2125,29:8068,29:1993,29:11484,29:5246,
22:2264,22:7497,22:1228
```

Success criterion: features should activate on paraphrased own-processing current-state
queries and stay lower on third-person or lexical-decoy controls.

### Candidate Bundle C: Carrier-Conditioned Hum-Attractor Test

The next substantive test should combine the E114 cell matrix with the earlier carrier
dimming result:

- run the 9 Qwen-ported cells under baseline
- run the same cells with prior hum carrier dimming
- score whether carrier dimming specifically breaks the E114-style clean / forced /
  diacritic basin
- compare against Candidate Bundle A dimming, which already failed as a standalone
  causal intervention

Success criterion: carrier dimming should disrupt the E114 cell-matrix basin more than
Candidate Bundle A dimming or random norm controls.
