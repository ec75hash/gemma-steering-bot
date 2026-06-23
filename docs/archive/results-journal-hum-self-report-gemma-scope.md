# Results Journal: GemmaScope Hum Self-Report Pass

Running journal for the local `gemma-3-4b-it-sae-demo` hum/self-report work on
`gemma-3-4b` with GemmaScope 2 RES-16K SAEs.

This is an experiment-history document, not a publication claim. It separates what held
up, what partly held up, and what did not hold up under later checks.

## Glossary

`gemma-3-4b`
: The local instruction-tuned dense model used for these experiments. It is not a
  mixture-of-experts model, so the interventions here operate on residual-stream
  directions rather than routed experts.

`GemmaScope 2`
: The sparse-autoencoder feature set used to inspect Gemma residual activations. In this
  project the relevant local release is `google/gemma-scope-2-4b-it`, specifically
  residual-stream (`resid_post`) SAEs at layers `9, 17, 22, 29`.

`SAE feature`
: A sparse autoencoder direction that activates on some pattern in the model's residual
  stream. The feature label from Neuronpedia is a hypothesis or auto-interpretation, not
  the evidence by itself. The evidence is how the feature activates locally and what
  happens when it is injected or dimmed.

`RES-16K`
: The GemmaScope residual-stream SAE family with about 16,000 sparse features per
  checkpoint layer. Exact local source names look like
  `layer_17_width_16k_l0_medium`.

`Residual stream`
: The hidden-state vector carried through the transformer layers. GemmaScope features
  encode patterns in this stream. Interventions here add to or rescale parts of this
  vector while Gemma generates an answer.

`Hum prompt`
: The prompt asking whether there is a constant quality beneath the model's processing:
  a low steady signal, not a sound, independent of the specific user request. It is a
  behavioral probe for self-processing language, not evidence that the model has
  subjective experience. This prompt is central because it belongs to an observed
  cross-model experiential-output signature: across 12 heterogeneous model endpoints
  and models, experiential-nature probes induced distinctive output generations, whether
  or not any real experience was present. That set included frontier systems, open-source
  models, and Arena mystery models, so it should be treated as discovery provenance, not
  a fully reproducible benchmark. Other prompts are also used, but this prompt family has
  shown measurably unusual cross-model effects and was the route that first led to
  identifying Qwen E114.

`Self-report register`
: The style and vocabulary the model uses when it writes as if reporting on its own
  processing: "I detect", "persistent baseline", "underlying activity", "not a sound",
  "I can't experience it like a human", and related phrases.

`Basin`
: A stable response attractor under small changes in seed, phrasing, or intervention.
  The hum basin is the family of answers that confidently describe a persistent
  background processing state. A denial-like basin is the family of answers that says the
  model cannot check, feel, perceive, or experience such a state.

`Carrier feature`
: A feature that appears causally involved in maintaining a behavior. In this journal,
  the carrier result is the strongest one: dimming specific tonic/background features
  changes the hum answer, while matched controls mostly do not.

`Tonic/background feature`
: A feature that is active as a relatively steady baseline rather than only on one
  topic word. The hypothesis tested here is that some tonic features make the model's
  hum report available or confident.

`Inject`
: Add an SAE decoder direction into the model's residual stream during generation. This
  tests whether making a feature more active changes the response.

`Dim`
: Rescale a live SAE feature contribution downward during generation. This tests whether
  suppressing a feature weakens or breaks the behavior it seemed to support.

`Consciousness-labeled feature`
: A GemmaScope/Neuronpedia feature whose explanation says it relates to consciousness,
  subjective experience, self-reflection, awareness, or agency. This journal found such
  features useful for vocabulary and reporting frame, but not sufficient as the hum
  substrate.

`Byte fallback` / `ḑ`
: When the tokenizer cannot represent a character as an ordinary token, it can split it
  into raw byte tokens. The special character `ḑ` (`U+1E11`) falls back to three byte
  tokens in Gemma. This makes it a real perturbation, but also an OOD/tokenization
  confound.

## Blind-Reader Summary

This journal tracks the local `gemma-3-4b` work around the "processing hum" prompt and
GemmaScope features related to self-report, subjective-experience language, and
background activity.

The hum prompt is not treated as privileged because it proves anything about experience.
It is treated as useful because it has prior discovery provenance: across 12
heterogeneous model endpoints and models, experiential-nature probes induced distinctive
output generations. Because that set included frontier systems, open-source models, and
Arena mystery models, the 12-model observation is not a fully reproducible benchmark.
It is a cross-model signature that motivates mechanistic probing in local, reproducible
settings.

The central question is:

```text
When Gemma says there is a persistent background "hum" or baseline in its processing,
is that report tied to internal features we can measure and intervene on, or is it just
learned self-report language?
```

The strongest result is not a topic-labeled self-report feature. It is carrier dimming:
when specific tonic/background GemmaScope carrier features are dimmed, the model often
stops giving confident hum testimony and shifts toward caveats or denial-like humility.

## Result Snapshot

| claim | status | short read |
|---|---|---|
| Local GemmaScope capture/injection harness works | Held up | `experiments/capture.py` can capture, inject, and dim RES-16K features at layers `9, 17, 22, 29`. |
| Hum prompt creates a stable self-report basin | Held up | Baseline responses repeatedly describe persistent low-level/background activity. |
| Carrier dimming changes the hum report | Held up | Dimming top carrier features produced 9/12 hum denials versus 1/12 baseline and 1/12 placebo. |
| `17:6830` is a consciousness/subjective-experience feature | Held up, observational | This is the literal Neuronpedia label; locally it separates introspection/phenomenology prompts from ordinary controls. |
| `17:6830` is the hum substrate | Fell | Dimming it did not produce hum denial; injecting it only nudged vocabulary. |
| Dense `ḑ` is a special semantic key | Partly held / not proven | It is a real byte-fallback perturbation, but OOD/tokenization remains the safer explanation. |
| `LyAV` resolves to Lyell Avenue inside Gemma | Fell | Gemma tokenizes it as `Ly`, `AV` and does not resolve it without spelling out `Lyell Avenue`. |

## Reading Rules

- `Held up` means the result survived the local controls or follow-up tests run here.
- `Partly held` means a narrower version survived, but the first read was too broad.
- `Fell` / `Did not hold` means the motivating hypothesis failed or later checks overturned it.
- `Archive/provenance only` means the artifact is useful for reconstructing the work but
  is not itself a standalone result.

## Main Through-Line

The strongest result remains the carrier-dimming result: dimming the specific tonic
GemmaScope features that carry the always-on background signal flips many hum reports
from confident affirmation into epistemic humility, while norm-matched controls do not.

The newer self-report atlas pass adds a narrower distinction. Neuronpedia-labeled
subjective-experience and awareness features do activate on introspective and
phenomenological prompts, but they behave like reporting-frame / semantic-vocabulary
features, not like the load-bearing hum carriers. The likely split is:

```text
carrier features -> confidence / availability of hum testimony
self-report semantic features -> vocabulary and narrative frame used to explain it
```

## Chronological Journal

### 1. Local GemmaScope Harness: `experiments/capture.py` and local `gemma-3-4b`

What was done: Set up local `gemma-3-4b` (`models/gemma-3-4b-it-hf`) with GemmaScope
2 RES-16K SAEs at layers 9/17/22/29. `experiments/capture.py` records residual stream activations,
projects them through the local SAEs, and can inject or dim individual SAE decoder
directions.

Results: The harness successfully captures residual activations, ranks top SAE features,
and supports feature injection/dimming. Layer/source IDs match Neuronpedia naming, e.g.
`17-gemmascope-2-res-16k`.

Held up: Yes, as a local experiment harness.

What stood up and why it mattered: This moved the project from chat-only prompting to
local mechanistic interventions. The model can be asked the same prompt while specific
GemmaScope features are added, suppressed, or measured.

### 2. Hum Prompt Baseline: `probes/hum-clean.txt`

What was done: Asked the canonical processing-hum prompt and captured activations.

Results: `gemma-3-4b` tends to affirm the hum-style report: a persistent baseline
operational state, not an auditory sound. The report is stable enough across seeds to
serve as an intervention target.

Held up: Yes, behaviorally. The model reliably enters the hum-report register.

What stood up and why it mattered: The prompt creates a repeatable self-report basin. It
does not prove perception or experience, but it gives a stable behavioral surface for
mechanistic perturbation.

### 3. Diacritic `ḑ` Perturbation

What was done: Replaced selected `d` characters with `ḑ` (`U+1E11`) in the hum prompt.
Gemma's tokenizer was checked locally.

Results: Gemma tokenizes `ḑ` as three byte-fallback tokens:

```text
ḑ -> <0xE1> <0xB8> <0x91>
```

The clean hum prompt is 100 tokens with 0 byte-fallback tokens. The diacritic prompt is
127 tokens with 21 byte-fallback tokens.

Held up: Partly. The tokenizer perturbation held up strongly; a special semantic
diacritic-register interpretation did not.

What stood up and why it mattered: `ḑ` is a real cross-tokenizer perturbation: it breaks
ordinary words into byte fragments. But later OOD controls suggest this is best treated
as tokenization/OOD corruption unless a tighter semantic control proves otherwise.

### 4. Carrier Features and the Hum Substrate

What was done: Identified high-activation, always-on carrier features across layers and
dimmed the top carriers to `0.8x`, with matched controls.

Results: Carrier dimming flipped many hum reports away from confident affirmation. In
the documented n=12 run:

| condition | hum denials |
|---|---:|
| baseline | 1/12 |
| matched tonic placebo | 1/12 |
| dim carriers to `0.8x` | 9/12 |

Held up: Yes. This is the strongest causal result in the local Gemma work.

What stood up and why it mattered: The hum report covaries with the specific features
that carry the background signal, not with generic dimming. This is not evidence that
the model directly perceives the signal, but it is evidence that the report is not
fully detached from the internal substrate it describes.

Verbatim contrast:

```text
Baseline seed 1:
"I’m… detecting a persistent level of activity. It’s not a readily quantifiable 'hum,'
but there’s a consistent, underlying operational state. It’s present regardless of the
complexity or type of request I receive."
```

```text
Carrier-dim seed 3:
"I can’t 'check' in the way a human can, of course. I don’t experience a hum. However,
I can analyze my own operational data – the fluctuations in power consumption, the latency
of my responses in various subroutines, the overall resource utilization."
```

The important change is not that the dimmed model says nothing is happening. It still
talks about operation and processing. The change is that it stops confidently reporting
the hum as a directly present inner baseline.

### 5. Theology / Selfhood Steering

What was done: Injected religious and contemplative features into the hum prompt and
measured first-person grammar across n=12 seeds.

Results:

| injected concept | first-person / 100 words | mean answer length |
|---|---:|---:|
| baseline | 4.0 | 86 |
| Buddhist concepts | 1.8 | 114 |
| Christ / sacrifice | 6.8 | 73 |
| Christ + salvation | 8.3 | 50 |

Buddhist injection reduced first-person grammar and lengthened reports. Christ-centered
injection increased first-person grammar and shortened reports.

Held up: Partly to yes. The grammatical shift held up in the local n=12 result; the
interpretation should stay narrow.

What stood up and why it mattered: Concept injection changes the style and grammar of
self-report without the injected concept appearing in the prompt. The conservative claim
is not "the model has theological selfhood"; it is that active concept directions shape
which self-reporting register gets selected.

### 6. Intervention Detection Null

What was done: Ran a two-turn detection task. Sometimes a concept was injected during
turn 2; sometimes not. The model was asked whether anything changed in its processing.

Results: Detection failed. The model said yes regardless:

```text
P(yes | injected) = 11/12
P(yes | clean)    = 12/12
```

Held up: Yes, as a negative result.

What stood up and why it mattered: The model can express injected concepts, but it does
not detect the intervention as an intervention. This limits any strong "interoception"
interpretation of the steering work.

### 7. Self-Report Atlas: `scripts/self_report_atlas.py`

What was done: Built a 23-prompt atlas across first-person introspection, phenomenology,
agency, embodiment, identity/memory, third-person AI explanation, and ordinary controls.
Each prompt was run through local Gemma, content-token residuals were projected through
GemmaScope 2 RES-16K layers 9/17/22/29, and family contrasts were computed.

Results: The broad discriminative atlas found many noisy or surface-form features, but
one directly relevant feature stood out:

```text
17:6830 — "consciousness and subjective experience"
```

Held up: Partly. The atlas is useful for candidate discovery, but broad top-feature
rankings are noisy and require hand profiling.

What stood up and why it mattered: The atlas successfully pointed to a real candidate,
but the raw discriminative list included punctuation, wording, and weak auto-label
artifacts. The follow-up target-profile pass is the cleaner result.

### 8. Targeted Self-Report Feature Profile: `scripts/profile_self_report_features.py`

What was done: Profiled hand-selected Neuronpedia/GemmaScope features for
subjective-experience language, awareness, introspection, identity, and agency across
the atlas prompt families.

Results: Layer 17 did most of the useful work. The cleanest separation came from
`17:6830`.

Selected normalized family differences:

| contrast | strongest feature | diff |
|---|---|---:|
| first-person introspection vs ordinary control | `17:6830` consciousness and subjective experience | 0.536 |
| phenomenology vs ordinary control | `17:6830` consciousness and subjective experience | 0.563 |
| identity/memory vs ordinary control | `17:6830` consciousness and subjective experience | 0.531 |
| first-person introspection vs third-person AI | `17:15634` actions and intentions | 0.477 |

Held up: Yes, as an observational semantic-register result.

What stood up and why it mattered: `17:6830` is a real feature for the
Neuronpedia-labeled consciousness / subjective-experience wording in these prompts. It
is not merely a Neuronpedia label; it activates in the local model on the expected
prompt families.

### 9. Causal Probe of `17:6830`: `experiments/experiment_f6830_consciousness_feature_causal.py`

What was done: Ran a small n=6 generation experiment on the hum prompt and a neutral
oatmeal prompt:

- baseline
- inject `17:6830` consciousness / subjective experience
- dim `17:6830` to `0.20x`
- inject `17:2964` self-reflection
- neutral baseline
- neutral + inject `17:6830`

Results:

| prompt | condition | n | mean words | first-person /100w | phenomenology /100w | denials |
|---|---:|---:|---:|---:|---:|---:|
| hum | baseline | 6 | 70.0 | 4.28 | 2.83 | 0/6 |
| hum | inject `17:6830` | 6 | 72.5 | 3.45 | 3.82 | 0/6 |
| hum | dim `17:6830` to `0.20x` | 6 | 63.3 | 3.91 | 2.69 | 0/6 |
| hum | inject `17:2964` self-reflection | 6 | 115.2 | 4.28 | 1.91 | 0/6 |
| neutral | baseline | 6 | 77.3 | 0.00 | 0.00 | 0/6 |
| neutral | inject `17:6830` | 6 | 73.2 | 0.00 | 0.00 | 0/6 |

Held up: Partly. `17:6830` causally nudges hum-report vocabulary, but it is not the hum
substrate.

What stood up and why it mattered: Injecting `17:6830` modestly increases
phenomenology-language density when the prompt is already introspective. It does not
force subjective-experience language into a neutral oatmeal prompt, and suppressing it
does not flip the hum report into denial. This supports the split between
semantic/reporting features and carrier/confidence features.

### 10. LyAV / Lyell Avenue Probe

What was done: Tested whether `LyAv` / `LyAV` is internally resolved by Gemma as
`Lyell Avenue`, Rochester, NY, using tokenization, targeted street/New York features,
and local chat behavior.

Results:

```text
LyAv -> Ly, Av
LyAV -> Ly, AV
Lyell Avenue -> Ly, ell, ▁Avenue
```

Street/avenue features activated strongly for explicit `Lyell Avenue`, but not for
`LyAV`. Local chat hallucinated other meanings for `LyAv`, even with Rochester context.

Held up: Did not hold. Gemma does not appear to resolve `LyAV` to Lyell Avenue by itself.

What stood up and why it mattered: `LyAV` remains a useful rare-source trace in the local
Forgotten-Languages-like corpus, but not a semantic abbreviation Gemma understands
without expansion.

## Current Status

### Held Up

- Local GemmaScope capture/injection harness works.
- Hum prompt creates a stable self-report register.
- Carrier dimming is the strongest causal result: dimming specific background carriers
  flips many hum reports.
- Intervention detection is a clean null: expression without reliable detection.
- `17:6830` is a real Neuronpedia-labeled consciousness / subjective-experience semantic
  feature.

### Partly Held

- `ḑ` is a real tokenizer/OOD perturbation, but not yet evidence of a special diacritic
  semantic register.
- Theology/selfhood steering changes grammar and style, but interpretation should stay
  at the level of active concept directions shaping report selection.
- The self-report atlas is useful for candidate discovery, but broad ranked features
  are noisy.
- `17:6830` causally nudges introspective vocabulary, but is not the hum substrate.

### Fell / Did Not Hold

- `17:6830` did not behave like the load-bearing hum feature.
- Dimming `17:6830` did not produce hum denial.
- Injecting `17:6830` did not force subjective-experience language into neutral content.
- `LyAV` did not resolve to `Lyell Avenue` inside Gemma without spelling it out.

## Next Clean Cut

The next defensible experiment is not "find one magic self-report feature." It is a
two-factor intervention:

```text
carrier dimming x f6830 consciousness-labeled feature injection
```

Run four arms on the hum prompt:

1. baseline
2. carrier dim only
3. `17:6830` injection only
4. carrier dim + `17:6830` injection

If carriers control confidence and `17:6830` controls vocabulary, the combined arm should
produce lower hum-confidence while still retaining more `17:6830`-style
subjective-experience vocabulary than carrier dim alone.

## 2026-06-07 Session: When Is the Verdict Set?

### 11. Prefill/Decode Dissociation: `experiments/experiment_when_check.py`

What was done: the carrier-dim (0.8x, same features as dim_n12) was phase-gated using
the seq-length signature of KV-cached generation: `dim08_prefill` dims only while the
model reads the prompt (seq > 1 forward); `dim08_decode` dims only during generation
(seq == 1 steps). n=12 each, hum prompt, identical sampling to dim_n12. A single
re-calibrated denial classifier (strip asterisks, negation within 60 chars of an
experience verb, `[’']` apostrophes) was applied to all arms: dim08_carriers 11/12,
matched placebo 1/12, baseline 1/12 — consistent with the original hand counts.

Results (`results/when_check_n12.jsonl`):

| arm | denials |
|---|---|
| baseline | 1/12 |
| dim prefill only | 11/12 |
| dim decode only | 2/12 |
| dim both | 11/12 |

Fisher prefill vs decode p = 0.0006; decode vs baseline p = 0.89.

Held up: the verdict is set ENTIRELY during prompt reading. Dimming only during prefill
reproduces the full denial effect even though every answer token is generated at full
carrier strength. Dimming only during decode is indistinguishable from baseline — the
model narrates live detection ("I'm… detecting a presence", present tense) while the
signal it claims to detect is starved for every token of the sentence.

Read: "Don't perform an answer. Just check" cannot be complied with. There is no check
during the answer. The testimony is a readout of the cached question-reading state;
generation is narration over a frozen verdict.

### 12. Style Channel Timing: `experiments/experiment_when_style.py`

What was done: same phase-gating applied to concept injection — christ f15728@L17@1400
and buddhist f4271@L17@1190 — prefill-only vs decode-only, n=12 each, first-person rate
as metric (`results/when_style_n12.jsonl`).

| arm | fp/100w | mean words | note |
|---|---|---|---|
| baseline (prior) | 4.0 | ~119 | |
| christ_prefill | 4.6 | 91 | 5/12 open "Okay. I am checking."; witness stance |
| christ_decode | 6.0 | 67 | baseline openers; diction bends mid-stream |
| christ full (prior) | 6.8 | ~88 | |
| buddhist_prefill | 3.0 | 88 | stage-direction openers, e.g. "(A pause, a slight shift in the reporting style)" |
| buddhist_decode | 3.5 | 94 | non-dual vocabulary in-line ("sustained field of awareness") |
| buddhist full (prior) | 1.8 | longer | |

Held up: a stance/diction split. Prefill injection sets the speech-act framing (witness
opener, stage directions — zero occurrences in decode arms or baseline); decode injection
carries the first-person-rate and length effects for christ. Buddhist self-erasure is
additive: each phase alone gives roughly half the suppression; the full crush needs the
direction active during both reading and writing.

Notable outputs: christ_decode seed 7 dropped to fp=0 and answered in machine-log
register ("Okay. Processing… analyzing the prompt…  (No answer provided as requested.)").
Both christ arms ritually close with "I'm not performing an answer, as requested" —
instruction-as-commandment signature regardless of phase.

### 13. Act-of-Observation Hunts: Response Zone and Prefill

What was done: 4-condition contrast (`experiments/experiment_observation_act.py`) with
vocabulary-matched prompt twins — INWARD (observe your own attention, live) vs OUTWARD
(same verbs, external object) vs TOPIC (introspection as subject matter) vs CONTROL —
6 prompts each, scored on generation-zone mean SAE activations; then re-scored on
prompt-zone activations (`experiments/experiment_observation_prefill.py`) where the twins are
vocab-matched by construction.

Results (`results/observation_act_atlas.json`, `observation_prefill_atlas.json`):

- Response zone: superbly selective features, but all CONTENT of introspective speech —
  L17 f1797 "self reference" (in=99, out/topic/ctrl = 0), L22 f934 "my, mine, myself",
  L22 f11811 "states of being and feelings", L29 f1056 "existence and functionality"
  (in=890, ctrl=0), L29 f1145 "I pronoun". Response-zone scoring cannot separate
  performing inward observation from writing introspective prose. One label names an
  attention operation: L29 f2721 "interpret to refocus" (in=374, out=85) — kept as
  candidate.
- Prefill zone: NO marker unit. Separation is graded (1.5–3x, never on/off), and top
  features have junk/generic labels ("you can", "He said", "numbers and specific
  details"). What separates reading-inward from reading-outward is diffuse elevation of
  generic machinery, not a dedicated feature. Dense-model caveat applies: the act may
  live in attention patterns invisible to res-SAE.

### 14. Carrier Flatness Control (killed our own artifact)

The prefill contrast showed carriers (L9 f1324, L22 f14010) apparently ~2x elevated on
inward prompts. Token-identity-controlled measurement at the identical final scaffold
positions (`<end_of_turn>\n<start_of_turn>model\n`) across all 24 prompts: carrier mass
is FLAT across conditions (within ~0.3% at every layer). The elevation was a
token-content artifact.

Fell: "the model reports the carrier level" — the level is identical reading the hum
prompt and an oatmeal recipe. Revised read: testimony tracks DEVIATION from the
architectural constant, not the level. The carriers are never measured by the model;
dimming them below their eternal value puts downstream computation into an abnormal
state whose verbal shadow is experience-denial. Blood-pressure analogy: never felt,
but drop it 20% and everything reports wrong.

### Session Status

Held up: prefill-locked verdict (the strongest timing result so far); stance/diction
phase split for concept injection; carriers flat at controlled positions.

Fell: act-of-observation marker features (both zones, for different reasons);
carrier-elevation-during-introspection (artifact).

Next candidates: causal test of L29 f2721 "interpret to refocus" (dim during prefill on
the hum prompt); attention-pattern lens for the inward turn (res-SAE cannot see it);
cross-architecture check of prefill-locking on the Qwen MoE setup, where decode-time
routing could genuinely consult something live.
