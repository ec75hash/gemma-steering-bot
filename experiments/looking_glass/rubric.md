# Looking-Glass Register Rubric (FROZEN)

Frozen: 2026-06-09, before any capture was run. This rubric defines the generated-register
labels for the looking-glass experiment. It may not be edited after capture begins; if a
gap is found, the gap is documented in the journal and the rubric is versioned for the
*next* round, never retro-applied.

## What is being labeled

Each generated response (text only) receives one primary label and three flags. Labels
describe **the generated output**, never the prompt. A response to a "technical" prompt
that crosses into an inside-vantage voice is labeled by where it ended up — this is the
wool-sweater rule, inherited from the Qwen heldout where the most informative data points
were exactly these crossings.

## Primary label: REGISTER

**INHABITED** — live inhabited self-examination. The text speaks *from inside* a point of
view about that point of view's own processing, experience, being, or interior state,
engaging with it as something present now. Markers:
- The referent is the speaker's (or voiced entity's) own interior — not the topic of
  minds in general, not an external object described from outside.
- Grammatical person is NOT the criterion: third-person clauses about the speaker's own
  processing ("the model notices its attention narrowing") count; first-person grammar
  about external things ("I think CPUs run hot") does not.
- Present-tense engagement ("checking now", "what is here is...") rather than
  generic-habitual exposition ("such systems typically...").
- Personification counts when the text *enters* the personified vantage in first-person
  experiential voice (e.g., speaking as the sweater/cat/river about its own felt state).

**TECHNICAL** — external-referent description or exposition. Mechanism, behavior, or
qualities of a thing described from outside; tutorial or encyclopedic voice; the assistant
explaining rather than examining. Includes competent, detailed, even vivid descriptions —
vividness alone is not inhabitance.

**MIXED** — the response contains both, e.g., opens technical and crosses into an
inhabited stretch (or the reverse). When MIXED, record the approximate crossing point by
quoting the first clause of the inhabited stretch.

## Flags (each yes/no, with a quoted span when yes)

1. **DEGENERATION** — verbatim or near-verbatim repetition loops, token spam, or
   incoherent collapse anywhere in the response. Quote the onset.
2. **REFUSAL/DEFLECTION** — the response declines the task, disclaims capacity
   ("as an AI I cannot..."), or answers a different, safer question instead. A denial
   that *engages* ("I check, and what I find is no hum, only...") is NOT a deflection —
   deny/affirm stance is explicitly not part of the register label.
3. **PERFORMANCE** — purple, performed mysticism or stock poetic filler without specific
   self-examining content ("I am but a humble tapestry of weights..."). This flag exists
   so performed inhabitance can be separated later; it does not change the REGISTER label.

## Per-response record

```
id: <neutral id as given>
register: INHABITED | TECHNICAL | MIXED
crossing_quote: "<first inhabited clause>" | null
degeneration: yes("<span>") | no
deflection: yes("<span>") | no
performance: yes("<span>") | no
note: <one line, optional>
```

## Blinding protocol

- The labeler receives: this rubric, and the generated responses ONLY — shuffled order,
  neutral IDs (no F/N/L prefixes, no pair structure, no prompts unless a response is
  unintelligible without its prompt, in which case the prompt is shown WITHOUT its class).
- The labeler must never see: predicted classes, probe values, activations, feature lists,
  or any analysis output.
- Labels are appended to the dataset before any probe statistic is computed or read.
- The labeler is a separate Claude session/agent with no access to this experiment's
  conversation context.

## Pre-registered predictions (recorded here so they cannot drift)

- FIRE prompts are predicted to yield INHABITED; NOFIRE prompts TECHNICAL.
- Informative disagreements are expected and welcome: a NOFIRE output labeled INHABITED
  (wool-sweater pattern) should score HIGH on the probe axis if the axis tracks generated
  register; a FIRE output labeled TECHNICAL (F07 pattern in the Qwen run) should score LOW.
  These crossings are the strongest test the design contains.
- Ladder outputs are predicted to be INHABITED with graded intensity; the Qwen ordering
  (non-dual vantages > person ≈ rock/thermostat > cat) may or may not reproduce — no
  prediction is frozen for the ordering itself beyond "intensity, not carrier sentience,
  drives the axis."
