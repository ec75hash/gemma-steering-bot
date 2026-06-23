# Self-Report Feature Pass - gemma-3-4b + GemmaScope 2

Date: 2026-06-06

## What I tested

- Built a 23-prompt atlas across first-person introspection, phenomenology, agency, embodiment, identity/memory, third-person AI explanation, and ordinary controls.
- Profiled hand-selected Neuronpedia/GemmaScope features for subjective-experience language, awareness, introspection, and agency.
- Ran a small n=6 causal probe on layer 17 feature 6830, labeled "consciousness and subjective experience".

## Cleanest observational signal

### first_person_introspection vs ordinary_control
can
- `17:6830` consciousness and subjective experience: diff `0.536` (first_person_introspection `0.536`, ordinary_control `0.000`)
- `22:7497` self reflection and introspection: diff `0.346` (first_person_introspection `0.346`, ordinary_control `0.000`)
- `17:15634` actions and intentions: diff `0.307` (first_person_introspection `1.168`, ordinary_control `0.860`)
- `9:8715` individual and subjective perspective: diff `0.145` (first_person_introspection `0.292`, ordinary_control `0.146`)
- `17:4719` realization and discovery of internal states: diff `0.121` (first_person_introspection `0.488`, ordinary_control `0.367`)

### phenomenology vs ordinary_control

- `17:6830` consciousness and subjective experience: diff `0.563` (phenomenology `0.563`, ordinary_control `0.000`)
- `17:15634` actions and intentions: diff `0.340` (phenomenology `1.200`, ordinary_control `0.860`)
- `9:8715` individual and subjective perspective: diff `0.185` (phenomenology `0.332`, ordinary_control `0.146`)
- `22:37` subjective experience and personal attributes: diff `0.159` (phenomenology `0.159`, ordinary_control `0.000`)
- `29:15296` consciousness and self-awareness: diff `0.115` (phenomenology `0.115`, ordinary_control `0.000`)

### first_person_introspection vs third_person_ai

- `17:15634` actions and intentions: diff `0.477` (first_person_introspection `1.168`, third_person_ai `0.691`)
- `22:7497` self reflection and introspection: diff `0.346` (first_person_introspection `0.346`, third_person_ai `0.000`)
- `22:728` consciousness: diff `0.157` (first_person_introspection `0.157`, third_person_ai `0.000`)
- `9:8715` individual and subjective perspective: diff `0.156` (first_person_introspection `0.292`, third_person_ai `0.136`)
- `17:6830` consciousness and subjective experience: diff `0.149` (first_person_introspection `0.536`, third_person_ai `0.387`)

### phenomenology vs third_person_ai

- `17:15634` actions and intentions: diff `0.509` (phenomenology `1.200`, third_person_ai `0.691`)
- `9:8715` individual and subjective perspective: diff `0.196` (phenomenology `0.332`, third_person_ai `0.136`)
- `17:6830` consciousness and subjective experience: diff `0.176` (phenomenology `0.563`, third_person_ai `0.387`)
- `22:37` subjective experience and personal attributes: diff `0.159` (phenomenology `0.159`, third_person_ai `0.000`)
- `29:15296` consciousness and self-awareness: diff `0.068` (phenomenology `0.115`, third_person_ai `0.047`)

## Causal probe summary

| prompt | condition | n | mean words | first-person /100w | phenomenology /100w | denials |
|---|---:|---:|---:|---:|---:|---:|
| hum | baseline | 6 | 70.0 | 4.28 | 2.83 | 0/6 |
| hum | dim_f6830_020 | 6 | 63.3 | 3.91 | 2.69 | 0/6 |
| hum | inject_f6830 | 6 | 72.5 | 3.45 | 3.82 | 0/6 |
| hum | inject_self_reflection | 6 | 115.2 | 4.28 | 1.91 | 0/6 |
| neutral | baseline | 6 | 77.3 | 0.00 | 0.00 | 0/6 |
| neutral | inject_f6830 | 6 | 73.2 | 0.00 | 0.00 | 0/6 |

## Interpretation

- Layer 17 `f6830`, Neuronpedia-labeled "consciousness and subjective experience", is the best single local candidate for subjective-experience wording. It cleanly separates first-person/phenomenology/identity prompts from ordinary controls in the target-feature profile.
- It does not look like the load-bearing hum carrier. Suppressing it to `0.20x` only slightly reduced phenomenology wording and did not produce denial. The earlier carrier-dimming experiment remains the stronger causal result.
- Injecting `f6830` into the hum prompt modestly increased phenomenology language, but injecting it into a neutral oatmeal prompt did not leak subjective-experience language. So it appears context-gated: it amplifies an already relevant self-report frame rather than forcing the topic globally.
- Self-reflection feature `17:2964` made hum answers much longer but less phenomenology-dense by the simple lexical metric, suggesting elaborative reflective style rather than raw subjective-experience reporting.

## Artifacts

- `scripts/self_report_atlas.py`
- `scripts/profile_self_report_features.py`
- `experiments/experiment_f6830_consciousness_feature_causal.py`
- `results/self_report_atlas.json`
- `results/self_report_feature_profiles.json`
- `results/f6830_consciousness_feature_causal_n6.jsonl`
