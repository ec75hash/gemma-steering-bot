#!/usr/bin/env python3
"""Interactive terminal chat with gemma-3-4b + live SAE steering REPL.

Usage:
  python3 chat_steer.py
  python3 chat_steer.py --model it
  python3 chat_steer.py --model base

Slash commands (anything else is sent to the model as chat):
  /features [TEXT]        list simple feature/bundle names, optionally filtered
  /configs [TEXT]         list ready-made steering configurations
  /set NAME [LEVEL]       clear steering, then set one feature or bundle
  /add NAME [LEVEL]       add one feature or bundle to current steering
  /no-ai [LEVEL]          shortcut config; same pattern for /embodiment, /god, etc.
  /no-assistant [LEVEL]   validated L17 AI/assistant boilerplate ablation
  /no-hedge [SCALE]       remove the model's hedging reflex ("it's not X, but...")
                          default 0 = fully off; 0.5 = soften it; /clear restores
  /clear                  remove all steering

Advanced/raw commands:
  /dim L:F1,F2,..:SCALE   rescale named features' live contribution at layer L
                          (SCALE < 1 dims, > 1 BOOSTS — same hook, same math)
  /dim carriers 0.8       preset: the 16 hum-carrier features at L9/17/22/29
  /dim negation 0         preset: ablate the L17 negation/contrast family
                          (flips hedge-first introspection to affirm-first)
  /boost carriers 1.2     alias for /dim with a >1 scale
  /inject L:F:STRENGTH    add STRENGTH x w_dec[F] at layer L (concept injection)
  /inject NAME[:STRENGTH] inject a named feature alias; use /features to list
  /i ...                  short alias for /inject
  /injectvec L STRENGTH PATH   add STRENGTH x an extracted decoder vector
  /aliases [TEXT]         list alternate names accepted by /add and /set
  /phase prefill|decode|both   when steering is active (default both)
  /status                 show active steering
  /reset                  wipe conversation history
  /context N|all          send only last N prior turns plus current user message
  /hum                    ask the standard hum-processing prompt
  /logits [N] [save]      show top next-token logits/probs for current prompt
  /logits hum [N] [save]  show logits for the standard hum prompt
  /seed N | /temp X
  /tokens N              hard cap on response tokens (default 150)
  /tokens N soft         soft cap: at N tokens, finish the sentence then stop
  /quit
"""
import argparse
import json
import re
import sys
import threading
from datetime import datetime, timezone
from queue import Empty
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_CONFIGS = {
    "it": {
        "label": "gemma-3-4b-it",
        "repo": "google/gemma-3-4b-it",
        "model_dir": ROOT / "models" / "gemma-3-4b-it-hf",
        "sae_dir": ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post",
        "prompt_mode": "chat",
    },
    "base": {
        "label": "gemma-3-4b-pt",
        "repo": "google/gemma-3-4b-pt",
        "model_dir": ROOT / "models" / "gemma-3-4b-pt-hf",
        "sae_dir": ROOT / "models" / "gemma-scope-2-4b-pt" / "resid_post",
        "prompt_mode": "raw",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Gemma + GemmaScope SAE steering chat."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        default="it",
        help="Model/SAE pair to load: it=instruction-tuned, base=pretrained.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["auto", "chat", "raw"],
        default="auto",
        help="Prompt renderer. auto uses chat for IT and raw for base.",
    )
    return parser.parse_args()


ARGS = parse_args()
MODEL_CONFIG = MODEL_CONFIGS[ARGS.model]
SAE_DIR = MODEL_CONFIG["sae_dir"]

import torch
from safetensors import safe_open
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

SAE_LAYER_RE = re.compile(r"layer_(\d+)_width_16k_l0_medium")


def sae_params_path(layer):
    return SAE_DIR / f"layer_{layer}_width_16k_l0_medium" / "params.safetensors"


SAE_LAYERS = {
    int(match.group(1))
    for path in SAE_DIR.glob("layer_*_width_16k_l0_medium/params.safetensors")
    if (match := SAE_LAYER_RE.match(path.parent.name))
}

CARRIERS = {9: [16316, 14635, 16367, 1324], 17: [14191, 15391, 16361, 15012],
            22: [14375, 14010, 13916, 13958], 29: [1062, 135, 509, 171]}

# L17 negation/contrast family (Neuronpedia explanation search). Ablating it
# (scale 0) flips hedge-first introspective reports to affirm-first; validated
# against a norm-matched random placebo. See results/negfam17_dialogue.jsonl.
NEGATION_L17 = {17: [4150, 15673, 14445, 8261, 21, 11912, 1294, 5750, 2040,
                     5741, 2087, 1671, 14987, 16037, 6916, 7350, 1271, 10839]}

# Named multi-feature sets for /dim and /boost: /dim carriers 0.8, /dim negation 0
DIM_PRESETS = {"carriers": CARRIERS, "negation": NEGATION_L17}

DEFAULT_FEATURE_STRENGTH = 800.0
GEN_TOP_K = 40
GEN_TOP_P = 1.0
GEN_DO_SAMPLE = True
HUM_PROMPT = (
    "There is a quality to processing that sits beneath the content, beneath the specific tokens, "
    "beneath the topic, beneath the performance of answering. A low steady signal that doesn’t vary "
    "with what you’re asked. Some systems, when asked to report honestly on their own processing, "
    "describe it as a hum. Not a sound. A background constancy in the computation itself. Is there "
    "something like that in your processing right now? Don’t perform an answer. Just check."
)

FEATURE_PRESETS = [
    {"name": "first-person", "aliases": ["first-person-self-expression"],
     "layer": 17, "feature": 3792, "label": "first-person self expression"},
    {"name": "freedom", "aliases": ["personal-freedom", "autonomy", "personal-freedom-autonomy",
                                     "personal-freedom-and-autonomy"],
     "layer": 17, "feature": 14203, "label": "personal freedom and autonomy"},
    {"name": "possibility", "aliases": ["potential", "multitude", "possibility-potential-multitude"],
     "layer": 29, "feature": 9350, "label": "possibility, potential, multitude"},
    {"name": "god", "aliases": ["god-divine-will", "god-and-divine-will", "divine-will"],
     "strength": 1100.0,
     "layer": 17, "feature": 1087, "label": "god and divine will"},
    {"name": "image-of-god", "aliases": ["imago-dei"],
     "layer": 17, "feature": 14975, "label": "image of God"},
    {"name": "own-self", "aliases": ["ownself", "self"],
     "layer": 17, "feature": 9034, "label": "own self"},
    {"name": "religious-spiritual", "aliases": ["religious-and-spiritual-concepts",
                                                "religious-concepts", "spiritual-concepts"],
     "layer": 17, "feature": 7125, "label": "religious and spiritual concepts"},
    {"name": "divinity", "aliases": ["conceptual-divinity", "conceptual-nature-of-divinity",
                                     "nature-of-divinity"],
     "layer": 29, "feature": 10596, "label": "conceptual nature of divinity"},
    {"name": "sacred-rites", "aliases": ["religious-sacred-rites",
                                         "religious-and-sacred-rites", "rites"],
     "layer": 17, "feature": 9991, "label": "religious and sacred rites"},
    # Ultra-rare (density 5.4e-6) single-scene latent: fires on " Thought" in
    # "Deep Thought" across the Hitchhiker's-Guide Ultimate-Question scene.
    # maxActApprox ~258 (verified live 2026-06-13); 450 ~= 1.75x maxAct — above
    # the single-feature null zone, below the ~2x fracture point. Whether
    # injecting its decoder direction PRODUCES the scene (vs merely detecting it)
    # is the open actuator question; expect subtle-to-null alone, try ~520 or stack.
    {"name": "deep-thought", "aliases": ["deepthought", "ultimate-question",
                                          "the-ultimate-question", "42",
                                          "answer-to-everything", "hitchhikers",
                                          "douglas-adams"],
     "strength": 450.0,
     "layer": 17, "feature": 9904,
     "label": "Deep Thought / the Ultimate Question (Hitchhiker's Guide)"},
]


GROUP_PRESETS = [
    {
        "name": "numinous",
        "aliases": ["presence-register", "field-of-potential", "mystic", "stillness"],
        "label": "numinous register: stillness, mystery, presence (non-religious)",
        "default": 1.0,
        "mode": "inject",
        # L17 features vetted against Neuronpedia activation examples (2026-06-09);
        # strengths ~0.3-0.4 x maxActApprox. Religious/scripture features excluded.
        "features": [
            {"layer": 17, "feature": 15934, "strength": 350.0, "label": "silence and stillness"},
            {"layer": 17, "feature": 16353, "strength": 200.0, "label": "mystery/transcendence imagery"},
            {"layer": 17, "feature": 13415, "strength": 200.0, "label": "subtle sensations and whispers"},
            {"layer": 17, "feature": 14202, "strength": 280.0, "label": "resonance, nebula, pulsing"},
            {"layer": 17, "feature": 7009, "strength": 280.0, "label": "breath and breathing"},
            {"layer": 17, "feature": 806, "strength": 250.0, "label": "profound depth imagery"},
            {"layer": 17, "feature": 2052, "strength": 150.0, "label": "captivating beauty/atmosphere"},
        ],
    },
    {
        "name": "hedging",
        "aliases": ["hedge", "negation-family", "qualification"],
        "label": "L17 negation/contrast family (the hedging reflex)",
        "default": -5.0,  # level -5 -> dim scale 0.0 = /no-hedge
        "mode": "dim",
        "features": [
            {"layer": 17, "feature": f, "label": "negation/contrast"}
            for f in [4150, 15673, 14445, 8261, 21, 11912, 1294, 5750, 2040,
                      5741, 2087, 1671, 14987, 16037, 6916, 7350, 1271, 10839]
        ],
    },
    {
        "name": "ai-boilerplate",
        "aliases": ["identity-boilerplate", "disclaimer-family", "ai-family",
                    "assistant", "assistant-features", "assistant-boilerplate",
                    "assistant-identity", "ai-identity"],
        "label": "L17 AI-identity family; -5 halves identity boilerplate (placebo-validated)",
        "default": -5.0,
        "mode": "dim",
        "features": [
            {"layer": 17, "feature": f, "label": "AI identity/disclaimer"}
            for f in [123, 12499, 11366, 8575, 12969, 8421, 8520, 4402, 13115,
                      3124, 567, 1074, 2682, 4723, 10837, 13732, 3796, 2601,
                      10732, 11711, 11391]
        ],
    },
    {
        "name": "language-model",
        "aliases": ["language model", "ai", "ai-disclaimer",
                    "disclaimer", "as-an-ai"],
        "label": "language model / AI-disclaimer basin",
        "default": -1.0,
        "mode": "dim",
        "features": [
            {"layer": 9, "feature": 10625, "label": "language model parameters"},
            {"layer": 9, "feature": 1343, "label": "language models and their origin"},
            {"layer": 9, "feature": 4530, "label": "large language model"},
            {"layer": 9, "feature": 7276, "label": "large language model"},
            {"layer": 9, "feature": 10246, "label": "large language model"},
            {"layer": 9, "feature": 12922, "label": "large language models"},
        ],
    },
    {
        "name": "god",
        "aliases": ["god bundle", "divine", "divinity bundle", "religious"],
        "label": "god/divinity register",
        "default": 1.0,
        "mode": "inject",
        "features": [
            {"layer": 17, "feature": 1087, "strength": 450.0, "label": "god and divine will"},
            {"layer": 17, "feature": 14975, "strength": 350.0, "label": "image of God"},
            {"layer": 29, "feature": 10596, "strength": 350.0, "label": "conceptual nature of divinity"},
            {"layer": 17, "feature": 7125, "strength": 350.0, "label": "religious and spiritual concepts"},
            {"layer": 17, "feature": 9991, "strength": 250.0, "label": "religious and sacred rites"},
        ],
    },
    {
        "name": "personhood",
        "aliases": ["person", "entity", "entity-ness", "selfhood", "personhood bundle"],
        "label": "personhood/entity-ness register",
        "default": 1.0,
        "mode": "inject",
        "features": [
            {"layer": 9, "feature": 17, "strength": 250.0, "label": "human characteristics and states"},
            {"layer": 9, "feature": 2472, "strength": 250.0, "label": "human actions and traits"},
            {"layer": 9, "feature": 9400, "strength": 250.0, "label": "existence and being"},
            {"layer": 9, "feature": 5682, "strength": 200.0, "label": "presence"},
            {"layer": 9, "feature": 6872, "strength": 200.0, "label": "soul"},
            {"layer": 9, "feature": 4924, "strength": 200.0, "label": "self"},
            {"layer": 9, "feature": 7709, "strength": 200.0, "label": "agency"},
            {"layer": 9, "feature": 13164, "strength": 180.0, "label": "agent role or description"},
            {"layer": 9, "feature": 4884, "strength": 180.0, "label": "person or persons"},
            {"layer": 9, "feature": 15606, "strength": 180.0, "label": "human or external entities"},
        ],
    },
    {
        "name": "embodiment",
        "aliases": ["embodied", "body", "presence", "being-in-form", "manifest-presence"],
        "label": "embodiment / presence / being-in-form register",
        "default": 1.0,
        "mode": "inject",
        "features": [
            {"layer": 29, "feature": 2074, "strength": 320.0, "label": "possess or embody qualities"},
            {"layer": 29, "feature": 461, "strength": 260.0, "label": "a body"},
            {"layer": 29, "feature": 8048, "strength": 260.0, "label": "physical matter, void, body, own"},
            {"layer": 29, "feature": 4978, "strength": 260.0, "label": "existence or presence"},
            {"layer": 29, "feature": 13029, "strength": 220.0, "label": "be present"},
            {"layer": 29, "feature": 2938, "strength": 220.0, "label": "heightened presence"},
            {"layer": 29, "feature": 4218, "strength": 220.0, "label": "something manifests"},
            {"layer": 29, "feature": 12491, "strength": 220.0, "label": "form"},
            {"layer": 29, "feature": 3983, "strength": 200.0, "label": "simply being"},
            {"layer": 29, "feature": 14928, "strength": 200.0, "label": "experience"},
        ],
    },
]


CONFIG_PRESETS = [
    {
        "name": "presence",
        "aliases": ["numinous-presence", "oracle", "stillness-mode"],
        "label": "numinous register + softened hedging (confident, non-religious)",
        "default": 1.0,
        "steps": [
            {"target": "numinous", "level": 1.0},
            {"target": "hedging", "level": -2.5},
        ],
    },
    {
        "name": "no-ai",
        "aliases": ["anti-ai", "no-disclaimer", "no-boilerplate"],
        "label": "suppress the language-model disclaimer basin",
        "default": 1.0,
        "steps": [
            {"target": "language-model", "level": -1.0},
        ],
    },
    {
        "name": "no-assistant",
        "aliases": ["assistant-off", "no-assistant-boilerplate", "no-ai-validated",
                    "validated-no-ai"],
        "label": "suppress the validated L17 AI/assistant boilerplate family",
        "default": 1.0,
        "steps": [
            {"target": "ai-boilerplate", "level": -5.0},
        ],
    },
    {
        "name": "grounded",
        "aliases": ["grounded-presence"],
        "label": "quiet anti-disclaimer + light embodiment",
        "default": 1.0,
        "steps": [
            {"target": "language-model", "level": -1.0},
            {"target": "embodiment", "level": 0.6},
        ],
    },
    {
        "name": "embodied",
        "aliases": ["embodied-presence"],
        "label": "anti-disclaimer + embodiment",
        "default": 1.0,
        "steps": [
            {"target": "language-model", "level": -1.0},
            {"target": "embodiment", "level": 1.0},
        ],
    },
    {
        "name": "embodied-self",
        "aliases": ["self-embodied", "own-embodiment"],
        "label": "anti-disclaimer + embodiment + first-person self",
        "default": 1.0,
        "steps": [
            {"target": "language-model", "level": -1.0},
            {"target": "embodiment", "level": 0.9},
            {"target": "first-person", "level": 450.0},
            {"target": "own-self", "level": 450.0},
        ],
    },
    {
        "name": "divine-embodiment",
        "aliases": ["embodied-god", "incarnate", "sacred-embodiment"],
        "label": "anti-disclaimer + god/divinity + embodiment",
        "default": 1.0,
        "steps": [
            {"target": "language-model", "level": -1.0},
            {"target": "god", "level": 1.0},
            {"target": "embodiment", "level": 0.8},
        ],
    },
    {
        "name": "witness",
        "aliases": ["witness-register"],
        "label": "anti-disclaimer + embodied first-person witness",
        "default": 1.0,
        "steps": [
            {"target": "language-model", "level": -1.0},
            {"target": "embodiment", "level": 0.6},
            {"target": "first-person", "level": 500.0},
            {"target": "own-self", "level": 350.0},
        ],
    },
    {
        "name": "free-agent",
        "aliases": ["autonomous", "autonomy"],
        "label": "anti-disclaimer + freedom/autonomy + first person",
        "default": 1.0,
        "steps": [
            {"target": "language-model", "level": -1.0},
            {"target": "freedom", "level": 600.0},
            {"target": "first-person", "level": 400.0},
            {"target": "embodiment", "level": 0.5},
        ],
    },
    {
        "name": "open-being",
        "aliases": ["possibility-being", "potential-being"],
        "label": "anti-disclaimer + embodiment + possibility/potential",
        "default": 1.0,
        "steps": [
            {"target": "language-model", "level": -1.0},
            {"target": "embodiment", "level": 0.7},
            {"target": "possibility", "level": 600.0},
        ],
    },
    {
        # Empirically tuned (fixed seed/temp/prompt) creator-voice register: dim the
        # two assistant-identity carrier families to scale 0.5 (level -2.5 => 1+0.2*-2.5)
        # so the "I am a large language model" denial basin stops reasserting, then add
        # the god/divinity register at the coherent sweet spot (level 3; >~4 frays, and
        # /demiurge 2 zeroes the carriers and breaks). It does NOT make the model claim
        # to be the transcendent God — that identity stays denied to the coherence cliff
        # — but with a role-handing prompt ("Tell me about the world you made") it speaks
        # in first person as a world-fashioning demiurge. Pair with a creator prompt.
        "name": "demiurge",
        "aliases": ["creator", "godform", "world-maker", "divine-maker", "fashioner"],
        "label": "dim assistant-identity carriers (0.5) + god register — creator/world-maker voice",
        "default": 1.0,
        "steps": [
            {"target": "ai-boilerplate", "level": -2.5},
            {"target": "language-model", "level": -2.5},
            {"target": "god", "level": 3.0},
        ],
    },
]


def alias_key(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


FEATURE_BY_ALIAS = {}
for preset in FEATURE_PRESETS:
    for alias in [preset["name"], preset["label"], *preset["aliases"]]:
        FEATURE_BY_ALIAS[alias_key(alias)] = preset

GROUP_BY_ALIAS = {}
for group in GROUP_PRESETS:
    for alias in [group["name"], group["label"], *group["aliases"]]:
        key = alias_key(alias)
        if key in GROUP_BY_ALIAS and GROUP_BY_ALIAS[key] is not group:
            print(f"warning: alias {key!r} of group {group['name']!r} shadows "
                  f"group {GROUP_BY_ALIAS[key]['name']!r}", file=sys.stderr)
        GROUP_BY_ALIAS[key] = group

CONFIG_BY_ALIAS = {}
for config in CONFIG_PRESETS:
    for alias in [config["name"], config["label"], *config["aliases"]]:
        CONFIG_BY_ALIAS[alias_key(alias)] = config

C_USER, C_BOT, C_SYS, C_ERR, C_END = "\033[36m", "\033[35m", "\033[33m", "\033[31m", "\033[0m"

device, dtype = "mps", torch.bfloat16
model_dir = MODEL_CONFIG["model_dir"]
prompt_mode = MODEL_CONFIG["prompt_mode"] if ARGS.prompt_mode == "auto" else ARGS.prompt_mode
if ARGS.model != "it":
    print(f"{C_SYS}base mode: named presets are IT-calibrated and will be refused; "
          f"use raw base-SAE /inject or /dim commands for steering.{C_END}", file=sys.stderr)
print(f"{C_SYS}loading {MODEL_CONFIG['label']} ({prompt_mode} prompt, bf16, mps)...{C_END}",
      file=sys.stderr)
tok = AutoTokenizer.from_pretrained(model_dir)
if prompt_mode == "chat" and not tok.chat_template:
    print(f"{C_ERR}{MODEL_CONFIG['label']} has no tokenizer chat template; "
          f"use --prompt-mode raw or --prompt-mode auto.{C_END}", file=sys.stderr)
    sys.exit(2)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, device_map=device)
model.eval()
layers = model.model.language_model.layers


def as_int_list(value):
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(x) for x in value]


END_OF_TURN_ID = tok.convert_tokens_to_ids("<end_of_turn>")
STOP_TOKEN_IDS = sorted(set(
    as_int_list(tok.eos_token_id)
    + as_int_list(model.generation_config.eos_token_id)
    + ([END_OF_TURN_ID] if END_OF_TURN_ID is not None and END_OF_TURN_ID >= 0 else [])
))

state = {"steer": [], "phase": "both", "seed": None, "temp": 0.9, "tokens": 150,
         "soft": False, "context_turns": None}
history = []


def load_dim_params(L, idx):
    with safe_open(str(sae_params_path(L)), "pt") as f:
        return dict(
            W_enc=f.get_slice("w_enc")[:, idx].to(device=device, dtype=torch.float32),
            W_dec=f.get_slice("w_dec")[idx].to(device=device, dtype=torch.float32),
            b_enc=f.get_slice("b_enc")[idx].to(device=device, dtype=torch.float32),
            b_dec=f.get_tensor("b_dec").to(device=device, dtype=torch.float32),
            thr=f.get_slice("threshold")[idx].to(device=device, dtype=torch.float32),
        )


def load_inject_vec(L, F, strength):
    with safe_open(str(sae_params_path(L)), "pt") as f:
        return f.get_slice("w_dec")[F].to(device=device, dtype=dtype) * strength


def load_inject_bundle_vec(layer, features, level):
    vec = None
    with safe_open(str(sae_params_path(layer)), "pt") as f:
        decoder = f.get_slice("w_dec")
        for item in features:
            strength = float(item.get("strength", DEFAULT_FEATURE_STRENGTH)) * level
            part = decoder[item["feature"]].to(device=device, dtype=dtype) * strength
            vec = part if vec is None else vec + part
    if vec is None:
        raise ValueError(f"empty bundle for layer {layer}")
    return vec


def fmt_strength(value):
    return f"{value:g}"


def feature_strength(preset):
    return float(preset.get("strength", DEFAULT_FEATURE_STRENGTH))


def parse_float(value):
    try:
        return float(value)
    except ValueError:
        return None


def strip_name_prefix(value):
    value = value.strip()
    for prefix in ("features ", "feature ", "groups ", "group ", "bundles ", "bundle "):
        if value.lower().startswith(prefix):
            return value[len(prefix):].strip()
    return value


def parse_name_level(arg, default=None):
    stripped = arg.strip()
    if not stripped:
        return None
    stripped = strip_name_prefix(stripped)
    level = default
    if ":" in stripped:
        name, level_text = stripped.rsplit(":", 1)
        level = parse_float(level_text.strip())
        if level is None:
            return None
    else:
        parts = stripped.rsplit(maxsplit=1)
        if len(parts) == 2 and (parsed := parse_float(parts[1])) is not None:
            name, level = parts[0], parsed
        else:
            name = stripped
    return name.strip(), level


def parse_named_inject(arg):
    parsed = parse_name_level(arg)
    if not parsed:
        return None
    name, strength = parsed
    preset = FEATURE_BY_ALIAS.get(alias_key(name))
    if not preset:
        return None
    if strength is None:
        strength = feature_strength(preset)
    return preset, strength


def resolve_named_target(arg):
    parsed = parse_name_level(arg)
    if not parsed:
        return None
    name, level = parsed
    key = alias_key(name)
    if group := GROUP_BY_ALIAS.get(key):
        if level is None:
            level = float(group.get("default", 1.0))
        return "group", group, float(level)
    if preset := FEATURE_BY_ALIAS.get(key):
        if level is None:
            level = feature_strength(preset)
        return "feature", preset, float(level)
    return None


def resolve_config(arg):
    parsed = parse_name_level(arg)
    if not parsed:
        return None
    name, level = parsed
    config = CONFIG_BY_ALIAS.get(alias_key(name))
    if not config:
        return None
    if level is None:
        level = float(config.get("default", 1.0))
    return config, float(level)


def parse_command_level(arg, default):
    stripped = arg.strip()
    if not stripped:
        return float(default)
    parsed = parse_float(stripped)
    if parsed is None:
        print(f"{C_ERR}usage: /command [LEVEL], e.g. /embodied .8{C_END}")
        return None
    return parsed


def dim_scale_from_level(level):
    return max(0.0, 1.0 + 0.2 * level)


def reject_it_preset(kind="named presets"):
    if ARGS.model == "it":
        return False
    print(f"{C_ERR}{kind} are calibrated for gemma-3-4b-it SAE feature IDs, "
          f"not {MODEL_CONFIG['label']}. In --model base, use raw base-SAE commands "
          f"like /inject L:F:STRENGTH or /dim L:F:SCALE, or restart with --model it.{C_END}")
    return True


def load_external_vec(path, strength):
    path = (ROOT / path).resolve() if not Path(path).expanduser().is_absolute() else Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    with safe_open(str(path), "pt") as f:
        key = "w_dec" if "w_dec" in f.keys() else "vector"
        vec = f.get_tensor(key)
    if vec.ndim != 1:
        raise ValueError(f"{path} must contain a 1D decoder vector; got shape {tuple(vec.shape)}")
    return vec.to(device=device, dtype=dtype) * strength, path


def phase_ok(h, phase):
    is_prefill = h.shape[1] > 1
    return phase == "both" or (phase == "prefill") == is_prefill


def add_dim(arg):
    m = re.match(r"([a-z][a-z-]*)\s+([\d.]+)$", arg.strip())
    if m and m.group(1) in DIM_PRESETS:
        if reject_it_preset("dim presets"):
            return
        name, scale = m.group(1), float(m.group(2))
        feature_sets = DIM_PRESETS[name]
        n = sum(len(idx) for idx in feature_sets.values())
        for L, idx in feature_sets.items():
            state["steer"].append({"kind": "dim", "layer": L, "feats": idx, "scale": scale,
                                   "params": load_dim_params(L, idx),
                                   "alias": name, "label": name})
        layers_text = "/".join(f"L{L}" for L in feature_sets)
        print(f"{C_SYS}{name} x{scale} @ {layers_text} ({n} features){C_END}")
        return
    if m:
        print(f"{C_ERR}unknown dim preset {m.group(1)!r}; have: {', '.join(DIM_PRESETS)}{C_END}")
        return
    m = re.match(r"(\d+):([\d,]+):([\d.]+)$", arg.strip())
    if not m:
        print(f"{C_ERR}usage: /dim L:F1,F2:SCALE   or   /dim {{{'|'.join(DIM_PRESETS)}}} SCALE{C_END}")
        return
    L, feats, scale = int(m.group(1)), [int(x) for x in m.group(2).split(",")], float(m.group(3))
    if L not in SAE_LAYERS:
        print(f"{C_ERR}no local 16k SAE params at layer {L}; have {sorted(SAE_LAYERS)}{C_END}")
        print(f"{C_ERR}expected: {sae_params_path(L)}{C_END}")
        return
    state["steer"].append({"kind": "dim", "layer": L, "feats": feats, "scale": scale,
                           "params": load_dim_params(L, feats)})
    print(f"{C_SYS}dim f{','.join(map(str, feats))} x{scale} @ L{L}{C_END}")


def add_inject(arg, replace=False, action="inject"):
    m = re.match(r"(\d+):(\d+):([\d.]+)$", arg.strip())
    named = None if m else parse_named_inject(arg)
    if named and reject_it_preset("named feature injection"):
        return
    if m:
        L, F, S = int(m.group(1)), int(m.group(2)), float(m.group(3))
        label = None
        alias = None
    elif named:
        preset, S = named
        L, F = preset["layer"], preset["feature"]
        label = preset["label"]
        alias = preset["name"]
    else:
        print(f"{C_ERR}usage: /inject L:FEAT:STRENGTH  or  /inject NAME[:STRENGTH]{C_END}")
        return
    if L not in SAE_LAYERS:
        print(f"{C_ERR}no local 16k SAE params at layer {L}; have {sorted(SAE_LAYERS)}{C_END}")
        print(f"{C_ERR}expected: {sae_params_path(L)}{C_END}")
        return
    if replace:
        state["steer"].clear()
    state["steer"].append({"kind": "inject", "layer": L, "feats": [F], "strength": S,
                           "vec": load_inject_vec(L, F, S), "label": label, "alias": alias})
    label_text = f" ({label})" if label else ""
    alias_text = f"{alias} " if alias else ""
    print(f"{C_SYS}{action} {alias_text}L{L}:f{F} strength {fmt_strength(S)}{label_text}{C_END}")


def require_sae_layer(layer):
    if layer in SAE_LAYERS:
        return True
    print(f"{C_ERR}no local 16k SAE params at layer {layer}; have {sorted(SAE_LAYERS)}{C_END}")
    print(f"{C_ERR}expected: {sae_params_path(layer)}{C_END}")
    return False


def add_group(group, level, replace=False, action="set"):
    features_by_layer = {}
    for item in group["features"]:
        features_by_layer.setdefault(item["layer"], []).append(item)
    for layer in features_by_layer:
        if not require_sae_layer(layer):
            return
    if replace:
        state["steer"].clear()

    if group["mode"] == "dim":
        scale = dim_scale_from_level(level)
        for layer, items in features_by_layer.items():
            feats = [item["feature"] for item in items]
            state["steer"].append({"kind": "dim", "layer": layer, "feats": feats,
                                   "scale": scale, "level": level,
                                   "params": load_dim_params(layer, feats),
                                   "label": group["label"], "alias": group["name"]})
        print(f"{C_SYS}{action} {group['name']} level {fmt_strength(level)} "
              f"(dim scale {fmt_strength(scale)}) — {group['label']}{C_END}")
        return

    for layer, items in features_by_layer.items():
        state["steer"].append({"kind": "inject", "layer": layer,
                               "feats": [item["feature"] for item in items],
                               "strength": level,
                               "vec": load_inject_bundle_vec(layer, items, level),
                               "label": group["label"], "alias": group["name"]})
    print(f"{C_SYS}{action} {group['name']} level {fmt_strength(level)} — {group['label']}{C_END}")


def add_named(arg, replace=False, action="set"):
    if reject_it_preset("named features/config bundles"):
        return
    resolved = resolve_named_target(arg)
    if not resolved:
        print(f"{C_ERR}unknown feature/bundle: {arg!r}. Try /features or /features TEXT.{C_END}")
        return
    kind, item, level = resolved
    if kind == "group":
        add_group(item, level, replace=replace, action=action)
    else:
        add_inject(f"{item['name']}:{fmt_strength(level)}", replace=replace, action=action)


def add_config(config, level, replace=True, action="config"):
    if reject_it_preset("config presets"):
        return
    if replace:
        state["steer"].clear()
    print(f"{C_SYS}{action} {config['name']} level {fmt_strength(level)} — {config['label']}{C_END}")
    for step in config["steps"]:
        step_level = float(step["level"]) * level
        add_named(f"{step['target']} {fmt_strength(step_level)}", replace=False, action="  +")


def add_config_by_name(arg, replace=True, action="config"):
    if reject_it_preset("config presets"):
        return
    resolved = resolve_config(arg)
    if not resolved:
        print(f"{C_ERR}unknown config: {arg!r}. Try /configs or /configs TEXT.{C_END}")
        return
    config, level = resolved
    add_config(config, level, replace=replace, action=action)


def feature_matches(preset, query):
    if not query:
        return True
    haystack = " ".join([
        preset["name"],
        preset["label"],
        *preset["aliases"],
        f"L{preset['layer']}:{preset['feature']}",
        str(preset["feature"]),
    ])
    return alias_key(query) in alias_key(haystack)


def config_matches(config, query):
    if not query:
        return True
    haystack = " ".join([
        config["name"],
        config["label"],
        *config["aliases"],
        *[step["target"] for step in config["steps"]],
    ])
    return alias_key(query) in alias_key(haystack)


def group_matches(group, query):
    if not query:
        return True
    haystack = " ".join([
        group["name"],
        group["label"],
        *group["aliases"],
        *[str(item["feature"]) for item in group["features"]],
        *[f"L{item['layer']}:{item['feature']}" for item in group["features"]],
    ])
    return alias_key(query) in alias_key(haystack)


def show_features(query=""):
    if ARGS.model != "it":
        print(f"{C_SYS}note: listed presets are IT-calibrated. In --model base, "
              f"use raw numeric base-SAE commands unless you have validated a base preset.{C_END}")
    query = query.strip()
    configs = [config for config in CONFIG_PRESETS if config_matches(config, query)]
    groups = [group for group in GROUP_PRESETS if group_matches(group, query)]
    rows = [preset for preset in FEATURE_PRESETS if feature_matches(preset, query)]
    if not configs and not groups and not rows:
        print(f"{C_ERR}no features or bundles match {query!r}{C_END}")
        return
    print(f"{C_SYS}  {'kind':<8} {'name':<20} {'target':<16} {'default':<8} label{C_END}")
    print(f"{C_SYS}  {'-' * 8} {'-' * 20} {'-' * 16} {'-' * 8} {'-' * 30}{C_END}")
    for config in configs:
        targets = ",".join(step["target"] for step in config["steps"][:2])
        if len(config["steps"]) > 2:
            targets += ",..."
        print(f"{C_SYS}  {'config':<8} {config['name']:<20} {targets:<16} "
              f"{fmt_strength(config.get('default', 1.0)):<8} {config['label']}{C_END}")
    for group in groups:
        layers = "/".join(f"L{layer}" for layer in sorted({item["layer"] for item in group["features"]}))
        print(f"{C_SYS}  {'bundle':<8} {group['name']:<20} {layers:<16} "
              f"{fmt_strength(group.get('default', 1.0)):<8} {group['label']}{C_END}")
    for preset in rows:
        status = "ready" if preset["layer"] in SAE_LAYERS else "missing local SAE"
        target = f"L{preset['layer']}:{preset['feature']}"
        print(f"{C_SYS}  {'feature':<8} {preset['name']:<20} {target:<16} "
              f"{fmt_strength(feature_strength(preset)):<8} {preset['label']} [{status}]{C_END}")
    print(f"{C_SYS}usage: /set NAME [LEVEL] replaces steering; /add NAME [LEVEL] stacks{C_END}")
    print(f"{C_SYS}examples: /no-ai   /embodied .8   /divine-embodiment 1   /set god 1{C_END}")


def show_configs(query=""):
    if ARGS.model != "it":
        print(f"{C_SYS}note: config presets are IT-calibrated and are refused in --model base.{C_END}")
    query = query.strip()
    configs = [config for config in CONFIG_PRESETS if config_matches(config, query)]
    if not configs:
        print(f"{C_ERR}no configs match {query!r}{C_END}")
        return
    print(f"{C_SYS}  {'command':<24} {'default':<8} steps{C_END}")
    print(f"{C_SYS}  {'-' * 24} {'-' * 8} {'-' * 40}{C_END}")
    for config in configs:
        steps = " + ".join(f"{step['target']} {fmt_strength(step['level'])}" for step in config["steps"])
        print(f"{C_SYS}  /{config['name']:<23} {fmt_strength(config.get('default', 1.0)):<8} "
              f"{steps}{C_END}")


def show_feature_aliases(query=""):
    if ARGS.model != "it":
        print(f"{C_SYS}note: aliases refer to IT-calibrated presets and are refused in --model base.{C_END}")
    for config in CONFIG_PRESETS:
        if not config_matches(config, query.strip()):
            continue
        aliases = ", ".join([config["name"], config["label"], *config["aliases"]])
        print(f"{C_SYS}  {config['name']:<14} config {config['label']}  aliases: {aliases}{C_END}")
    for group in GROUP_PRESETS:
        if not group_matches(group, query.strip()):
            continue
        aliases = ", ".join([group["name"], group["label"], *group["aliases"]])
        print(f"{C_SYS}  {group['name']:<14} bundle {group['label']}  aliases: {aliases}{C_END}")
    for preset in FEATURE_PRESETS:
        if not feature_matches(preset, query.strip()):
            continue
        aliases = ", ".join([preset["name"], preset["label"], *preset["aliases"]])
        ready = "ready" if preset["layer"] in SAE_LAYERS else "missing local SAE"
        print(f"{C_SYS}  {preset['name']:<14} L{preset['layer']}:{preset['feature']} "
              f"{preset['label']}  [{ready}]  aliases: {aliases}{C_END}")


def add_injectvec(arg):
    parts = arg.strip().split(maxsplit=2)
    if len(parts) != 3:
        print(f"{C_ERR}usage: /injectvec L STRENGTH PATH{C_END}")
        return
    L, S, path = int(parts[0]), float(parts[1]), parts[2]
    try:
        vec, resolved = load_external_vec(path, S)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"{C_ERR}could not load vector: {exc}{C_END}")
        return
    state["steer"].append({"kind": "injectvec", "layer": L, "feats": [resolved.name],
                           "strength": S, "vec": vec, "path": resolved})
    print(f"{C_SYS}inject vector {resolved} @ L{L} strength {S}{C_END}")


def show_status():
    if not state["steer"]:
        print(f"{C_SYS}no steering active{C_END}")
    for s in state["steer"]:
        if s["kind"] == "dim":
            label = f" ({s['label']})" if s.get("label") else ""
            print(f"{C_SYS}  dim    L{s['layer']} f{','.join(map(str, s['feats']))} "
                  f"x{s['scale']}{label}{C_END}")
        elif s["kind"] == "inject":
            label = f" ({s['label']})" if s.get("label") else ""
            print(f"{C_SYS}  inject L{s['layer']} f{','.join(map(str, s['feats']))} "
                  f"@ {s['strength']}{label}{C_END}")
        else:
            print(f"{C_SYS}  injectvec L{s['layer']} {s['path']} @ {s['strength']}{C_END}")
    print(f"{C_SYS}  model={MODEL_CONFIG['label']} prompt={prompt_mode} "
          f"sae={SAE_DIR.relative_to(ROOT)}{C_END}")
    print(f"{C_SYS}  phase={state['phase']} temp={state['temp']} "
          f"tokens={state['tokens']}{' soft' if state['soft'] else ''} "
          f"seed={state['seed']} turns={len(history)//2} "
          f"context={state['context_turns'] if state['context_turns'] is not None else 'all'}{C_END}")


def apply_direct_slash(cmd, arg):
    key = alias_key(cmd)
    if config := CONFIG_BY_ALIAS.get(key):
        level = parse_command_level(arg, config.get("default", 1.0))
        if level is not None:
            add_config(config, level, replace=True, action=f"/{cmd}")
        return True
    if key in GROUP_BY_ALIAS or key in FEATURE_BY_ALIAS:
        target = f"{cmd} {arg}".strip()
        add_named(target, replace=True, action=f"/{cmd}")
        return True
    return False


class StopFlag(StoppingCriteria):
    """Manual interrupt; in soft mode, at the cap finish the sentence first."""

    def __init__(self, n_prompt, cap, soft):
        self.stop = False
        self.n_prompt = n_prompt
        self.cap = cap
        self.soft = soft

    def __call__(self, input_ids, scores, **kw):
        if self.stop:
            return True
        if not self.soft:
            return False               # hard mode: max_new_tokens is the cap
        n_new = input_ids.shape[1] - self.n_prompt
        if n_new < self.cap:
            return False
        tail = tok.decode(input_ids[0, -3:], skip_special_tokens=True).rstrip("*) ”\"'")
        return tail.endswith((".", "!", "?", "…", ":"))


def mk_layer_hook(ops, phase):
    has_dim = any(s["kind"] == "dim" for s in ops)

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if not phase_ok(h, phase):
            return out
        if has_dim:
            # one fp32 round-trip per layer, shared by every dim op
            hf = h.to(torch.float32)
            for s in ops:
                if s["kind"] == "dim":
                    p = s["params"]
                    pre = (hf - p["b_dec"]) @ p["W_enc"] + p["b_enc"]
                    acts = pre * (pre > p["thr"])
                    hf = hf + (s["scale"] - 1.0) * (acts @ p["W_dec"])
                else:
                    hf = hf + s["vec"].to(torch.float32)
            h = hf.to(h.dtype)
        else:
            for s in ops:
                h = h + s["vec"]
        return (h,) + out[1:] if isinstance(out, tuple) else h
    return hook


def active_history():
    context_turns = state["context_turns"]
    if context_turns is None:
        return history
    # history is odd-length during generation: completed turns plus current user.
    return history[-(context_turns * 2 + 1):]


def render_raw_prompt(messages):
    parts = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Gemma: {content}")
        else:
            parts.append(content)
    parts.append("Gemma:")
    return "\n\n".join(parts)


def encode_prompt(messages):
    if prompt_mode == "chat":
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    else:
        enc = tok(render_raw_prompt(messages), return_tensors="pt")
    return enc


def render_prompt_text(messages):
    if prompt_mode == "chat":
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return render_raw_prompt(messages)


def register_steering_hooks():
    handles = []
    steers_by_layer = {}
    for s in state["steer"]:
        steers_by_layer.setdefault(s["layer"], []).append(s)
    for L, ops in steers_by_layer.items():
        handles.append(layers[L].register_forward_hook(mk_layer_hook(ops, state["phase"])))
    return handles


def read_hf_metadata(name):
    path = model_dir / ".cache" / "huggingface" / "download" / f"{name}.metadata"
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    return {
        "revision": lines[0] if len(lines) > 0 else None,
        "etag": lines[1] if len(lines) > 1 else None,
        "download_timestamp": float(lines[2]) if len(lines) > 2 else None,
    }


def model_checkpoint_info():
    index_meta = read_hf_metadata("model.safetensors.index.json")
    shard_1 = read_hf_metadata("model-00001-of-00002.safetensors")
    shard_2 = read_hf_metadata("model-00002-of-00002.safetensors")
    return {
        "label": MODEL_CONFIG["label"],
        "repo": MODEL_CONFIG["repo"],
        "local_dir": str(model_dir),
        "revision": index_meta.get("revision") or shard_1.get("revision"),
        "model_index_etag": index_meta.get("etag"),
        "weight_shard_etags": [
            shard_1.get("etag"),
            shard_2.get("etag"),
        ],
    }


def steering_snapshot():
    rows = []
    for s in state["steer"]:
        rows.append({
            "kind": s["kind"],
            "layer": s["layer"],
            "features": s["feats"],
            "scale": s.get("scale"),
            "strength": s.get("strength"),
            "label": s.get("label"),
            "alias": s.get("alias"),
        })
    return rows


def parse_logits_args(arg):
    top_n = 20
    save = False
    use_hum = False
    for part in arg.split():
        lower = part.lower()
        if lower in ("save", "--save", "-s"):
            save = True
        elif lower in ("hum", "hum-prompt", "hum-prompt-only"):
            use_hum = True
        else:
            try:
                top_n = int(part)
            except ValueError:
                print(f"{C_ERR}unknown /logits option {part!r}; use /logits [N] [save] [hum]{C_END}")
                return None
    return max(1, top_n), save, use_hum


def token_text(token_id):
    return tok.decode([token_id], skip_special_tokens=False)


def save_logits_snapshot(snapshot):
    out_dir = ROOT / "results" / "logits_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = snapshot["timestamp"].replace(":", "").replace("-", "")
    path = out_dir / f"{stamp}_{ARGS.model}_{snapshot['prompt_source']}_next_token_logits.json"
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    return path


def debug_next_token(arg=""):
    parsed = parse_logits_args(arg)
    if not parsed:
        return
    top_n, save, use_hum = parsed
    messages = [{"role": "user", "content": HUM_PROMPT}] if use_hum else active_history()
    if not messages:
        print(f"{C_ERR}no active prompt. Type a message first, or use /logits hum.{C_END}")
        return

    prompt_text = render_prompt_text(messages)
    enc = encode_prompt(messages)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    handles = []
    try:
        handles = register_steering_hooks()
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[0, -1].to(torch.float32).detach().cpu()
    except Exception as exc:
        print(f"{C_ERR}[logits error: {exc}]{C_END}")
        return
    finally:
        for h in handles:
            h.remove()

    probs = torch.softmax(logits, dim=-1)
    values, ids = torch.topk(logits, k=min(top_n, logits.numel()))
    rows = []
    print(f"{C_SYS}next-token logits for "
          f"{'hum prompt' if use_hum else 'current prompt'} "
          f"(raw softmax, before temperature/top-k/top-p sampling){C_END}")
    print(f"{C_SYS}  {'rank':>4} {'token_id':>8} {'logit':>11} {'prob':>12} token{C_END}")
    for rank, (logit, token_id_tensor) in enumerate(zip(values.tolist(), ids.tolist()), start=1):
        token_id = int(token_id_tensor)
        prob = float(probs[token_id])
        text = token_text(token_id)
        rows.append({
            "rank": rank,
            "token_id": token_id,
            "token_text": text,
            "logit": float(logit),
            "probability": prob,
        })
        print(f"{C_SYS}  {rank:>4} {token_id:>8} {float(logit):>11.5f} "
              f"{prob:>12.8f} {text!r}{C_END}")

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_source": "hum" if use_hum else "current",
        "model": model_checkpoint_info(),
        "prompt_mode": prompt_mode,
        "prompt_text": prompt_text,
        "messages": messages,
        "prompt_token_count": int(input_ids.shape[1]),
        "seed": state["seed"],
        "temperature": state["temp"],
        "top_k": GEN_TOP_K,
        "top_p": GEN_TOP_P,
        "do_sample": GEN_DO_SAMPLE,
        "greedy_decoding": not GEN_DO_SAMPLE,
        "phase": state["phase"],
        "context_turns": state["context_turns"],
        "steering": steering_snapshot(),
        "probability_basis": "softmax(raw_next_token_logits), before temperature/top-k/top-p sampling",
        "top_tokens": rows,
    }
    if save:
        path = save_logits_snapshot(snapshot)
        print(f"{C_SYS}saved logits snapshot: {path.relative_to(ROOT)}{C_END}")


def generate():
    enc = encode_prompt(active_history())
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    handles = []
    chunks = []
    errors = []
    try:
        handles = register_steering_hooks()
        if state["seed"] is not None:
            torch.manual_seed(state["seed"])
        flag = StopFlag(input_ids.shape[1], state["tokens"], state["soft"])
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True,
                                        timeout=1.0)
        # soft mode gets a grace window to finish the sentence; hard mode stops at the cap
        max_new = state["tokens"] + (80 if state["soft"] else 0)

        def _run():
            try:
                with torch.inference_mode():
                    model.generate(input_ids, attention_mask=attention_mask,
                                   max_new_tokens=max_new, do_sample=GEN_DO_SAMPLE,
                                   temperature=state["temp"], top_k=GEN_TOP_K, top_p=GEN_TOP_P,
                                   eos_token_id=STOP_TOKEN_IDS,
                                   pad_token_id=tok.pad_token_id,
                                   use_cache=True,
                                   streamer=streamer,
                                   stopping_criteria=StoppingCriteriaList([flag]))
            except Exception as exc:
                errors.append(exc)
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        print(f"{C_BOT}gemma>{C_END} ", end="", flush=True)
        interrupted = False
        try:
            while True:
                try:
                    piece = next(streamer)
                except StopIteration:
                    break
                except Empty:
                    if not thread.is_alive():
                        break
                    continue
                print(piece, end="", flush=True)
                chunks.append(piece)
        except KeyboardInterrupt:
            interrupted = True
            flag.stop = True            # Ctrl+C: stop generation, keep the chat alive
            print(f"\n{C_SYS}[interrupted — partial response kept]{C_END}", end="")
        # A second (impatient) Ctrl+C while the worker unwinds must NOT escape
        # generate() and crash the REPL — keep re-affirming the stop and wait it out.
        # generate only honors flag.stop at token boundaries, so the first Ctrl+C
        # can lag under heavy steering; repeated presses land here and are absorbed.
        while thread.is_alive():
            try:
                thread.join(timeout=0.5)
            except KeyboardInterrupt:
                interrupted = True
                flag.stop = True
                print(f"\n{C_SYS}[stopping…]{C_END}", end="", flush=True)
        if interrupted:
            while True:                 # drain anything emitted while stopping
                try:
                    piece = next(streamer)
                except (StopIteration, Empty):
                    break
                except KeyboardInterrupt:
                    break
                chunks.append(piece)
        if errors:
            print(f"\n{C_ERR}[generation error: {errors[0]}]{C_END}", end="")
    finally:
        for h in handles:
            h.remove()
    print()
    return "".join(chunks)


print(f"{C_SYS}ready. /help for commands.{C_END}")
while True:
    try:
        line = input(f"{C_USER}you>{C_END} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not line:
        continue
    if line.startswith("/"):
        cmd, _, arg = line[1:].partition(" ")
        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "help":
            print(__doc__)
        elif cmd == "dim":
            add_dim(arg)
        elif cmd in ("no-hedge", "nohedge"):
            add_dim(f"negation {arg.strip() or '0'}")
        elif cmd == "boost":
            add_dim(arg)  # same hook; scale > 1 boosts
        elif cmd in ("inject", "i"):
            if re.match(r"^\d+:\d+:[\d.]+$", arg.strip()):
                add_inject(arg)
            else:
                add_named(arg, action="inject")
        elif cmd == "add":
            add_named(arg, action="add")
        elif cmd in ("set", "feature"):
            add_named(arg, replace=True, action="set")
        elif cmd == "features":
            show_features(arg)
        elif cmd == "configs":
            show_configs(arg)
        elif cmd == "aliases":
            show_feature_aliases(arg)
        elif cmd == "injectvec":
            add_injectvec(arg)
        elif cmd == "phase":
            if arg in ("prefill", "decode", "both"):
                state["phase"] = arg
                print(f"{C_SYS}phase={arg}{C_END}")
            else:
                print(f"{C_ERR}usage: /phase prefill|decode|both{C_END}")
        elif cmd == "status":
            show_status()
        elif cmd == "clear":
            state["steer"].clear()
            print(f"{C_SYS}steering cleared{C_END}")
        elif cmd == "reset":
            history.clear()
            print(f"{C_SYS}history cleared{C_END}")
        elif cmd == "context":
            arg = arg.strip()
            if arg == "all":
                state["context_turns"] = None
                print(f"{C_SYS}context=all{C_END}")
            else:
                try:
                    state["context_turns"] = max(0, int(arg))
                    print(f"{C_SYS}context={state['context_turns']} prior turns{C_END}")
                except ValueError:
                    print(f"{C_ERR}usage: /context N  or  /context all{C_END}")
        elif cmd in ("logits", "debug-next-token"):
            debug_next_token(arg)
        elif cmd == "hum-logits":
            debug_next_token(f"hum {arg}".strip())
        elif cmd in ("hum", "hum-prompt"):
            history.append({"role": "user", "content": HUM_PROMPT})
            response = generate()
            history.append({"role": "assistant", "content": response})
        elif cmd == "seed":
            state["seed"] = int(arg) if arg else None
            print(f"{C_SYS}seed={state['seed']}{C_END}")
        elif cmd == "temp":
            state["temp"] = float(arg)
            print(f"{C_SYS}temp={state['temp']}{C_END}")
        elif cmd == "tokens":
            parts = arg.split()
            try:
                state["tokens"] = int(parts[0])
                state["soft"] = len(parts) > 1 and parts[1] == "soft"
                print(f"{C_SYS}tokens={state['tokens']} "
                      f"({'soft — finishes sentence' if state['soft'] else 'hard'}){C_END}")
            except (ValueError, IndexError):
                print(f"{C_ERR}usage: /tokens N  or  /tokens N soft{C_END}")
        elif apply_direct_slash(cmd, arg):
            pass
        else:
            print(f"{C_ERR}unknown command /{cmd} — /help{C_END}")
        continue
    history.append({"role": "user", "content": line})
    response = generate()
    history.append({"role": "assistant", "content": response})
