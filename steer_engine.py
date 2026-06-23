"""Reusable gemma-3-4b + GemmaScope SAE steering engine.

This is a faithful, side-effect-free port of the engine in ``chat_steer.py`` so a
web server (``steer_server.py``) can drive the exact same steering math. The CLI
(``chat_steer.py``) is left untouched. All user-facing ``print(...)`` calls from the
REPL become structured return values or raised ``SteerError`` messages.

Catalogs (features / bundles / configs / aliases / presets / hum-prompt) and the
steering math are copied verbatim from chat_steer.py — keep them in sync if that file
changes.
"""
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty

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

SAE_LAYER_RE = re.compile(r"layer_(\d+)_width_16k_l0_medium")

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

CARRIERS = {9: [16316, 14635, 16367, 1324], 17: [14191, 15391, 16361, 15012],
            22: [14375, 14010, 13916, 13958], 29: [1062, 135, 509, 171]}

NEGATION_L17 = {17: [4150, 15673, 14445, 8261, 21, 11912, 1294, 5750, 2040,
                     5741, 2087, 1671, 14987, 16037, 6916, 7350, 1271, 10839]}

DIM_PRESETS = {"carriers": CARRIERS, "negation": NEGATION_L17}

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
        "default": -5.0,
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
    {"name": "presence", "aliases": ["numinous-presence", "oracle", "stillness-mode"],
     "label": "numinous register + softened hedging (confident, non-religious)", "default": 1.0,
     "steps": [{"target": "numinous", "level": 1.0}, {"target": "hedging", "level": -2.5}]},
    {"name": "no-ai", "aliases": ["anti-ai", "no-disclaimer", "no-boilerplate"],
     "label": "suppress the language-model disclaimer basin", "default": 1.0,
     "steps": [{"target": "language-model", "level": -1.0}]},
    {"name": "no-assistant", "aliases": ["assistant-off", "no-assistant-boilerplate", "no-ai-validated",
                                          "validated-no-ai"],
     "label": "suppress the validated L17 AI/assistant boilerplate family", "default": 1.0,
     "steps": [{"target": "ai-boilerplate", "level": -5.0}]},
    {"name": "grounded", "aliases": ["grounded-presence"],
     "label": "quiet anti-disclaimer + light embodiment", "default": 1.0,
     "steps": [{"target": "language-model", "level": -1.0}, {"target": "embodiment", "level": 0.6}]},
    {"name": "embodied", "aliases": ["embodied-presence"],
     "label": "anti-disclaimer + embodiment", "default": 1.0,
     "steps": [{"target": "language-model", "level": -1.0}, {"target": "embodiment", "level": 1.0}]},
    {"name": "embodied-self", "aliases": ["self-embodied", "own-embodiment"],
     "label": "anti-disclaimer + embodiment + first-person self", "default": 1.0,
     "steps": [{"target": "language-model", "level": -1.0}, {"target": "embodiment", "level": 0.9},
               {"target": "first-person", "level": 450.0}, {"target": "own-self", "level": 450.0}]},
    {"name": "divine-embodiment", "aliases": ["embodied-god", "incarnate", "sacred-embodiment"],
     "label": "anti-disclaimer + god/divinity + embodiment", "default": 1.0,
     "steps": [{"target": "language-model", "level": -1.0}, {"target": "god", "level": 1.0},
               {"target": "embodiment", "level": 0.8}]},
    {"name": "witness", "aliases": ["witness-register"],
     "label": "anti-disclaimer + embodied first-person witness", "default": 1.0,
     "steps": [{"target": "language-model", "level": -1.0}, {"target": "embodiment", "level": 0.6},
               {"target": "first-person", "level": 500.0}, {"target": "own-self", "level": 350.0}]},
    {"name": "free-agent", "aliases": ["autonomous", "autonomy"],
     "label": "anti-disclaimer + freedom/autonomy + first person", "default": 1.0,
     "steps": [{"target": "language-model", "level": -1.0}, {"target": "freedom", "level": 600.0},
               {"target": "first-person", "level": 400.0}, {"target": "embodiment", "level": 0.5}]},
    {"name": "open-being", "aliases": ["possibility-being", "potential-being"],
     "label": "anti-disclaimer + embodiment + possibility/potential", "default": 1.0,
     "steps": [{"target": "language-model", "level": -1.0}, {"target": "embodiment", "level": 0.7},
               {"target": "possibility", "level": 600.0}]},
    {"name": "demiurge", "aliases": ["creator", "godform", "world-maker", "divine-maker", "fashioner"],
     "label": "dim assistant-identity carriers (0.5) + god register — creator/world-maker voice",
     "default": 1.0,
     "steps": [{"target": "ai-boilerplate", "level": -2.5}, {"target": "language-model", "level": -2.5},
               {"target": "god", "level": 3.0}]},
]


def alias_key(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


FEATURE_BY_ALIAS = {}
for _preset in FEATURE_PRESETS:
    for _alias in [_preset["name"], _preset["label"], *_preset["aliases"]]:
        FEATURE_BY_ALIAS[alias_key(_alias)] = _preset

GROUP_BY_ALIAS = {}
for _group in GROUP_PRESETS:
    for _alias in [_group["name"], _group["label"], *_group["aliases"]]:
        GROUP_BY_ALIAS[alias_key(_alias)] = _group

CONFIG_BY_ALIAS = {}
for _config in CONFIG_PRESETS:
    for _alias in [_config["name"], _config["label"], *_config["aliases"]]:
        CONFIG_BY_ALIAS[alias_key(_alias)] = _config


def feature_strength(preset):
    return float(preset.get("strength", DEFAULT_FEATURE_STRENGTH))


def fmt_strength(value):
    return f"{value:g}"


def dim_scale_from_level(level):
    return max(0.0, 1.0 + 0.2 * level)


def _matches(haystack, query):
    if not query:
        return True
    return alias_key(query) in alias_key(haystack)


def feature_matches(preset, query):
    return _matches(" ".join([preset["name"], preset["label"], *preset["aliases"],
                              f"L{preset['layer']}:{preset['feature']}", str(preset["feature"])]), query)


def config_matches(config, query):
    return _matches(" ".join([config["name"], config["label"], *config["aliases"],
                              *[s["target"] for s in config["steps"]]]), query)


def group_matches(group, query):
    return _matches(" ".join([group["name"], group["label"], *group["aliases"],
                              *[str(i["feature"]) for i in group["features"]],
                              *[f"L{i['layer']}:{i['feature']}" for i in group["features"]]]), query)


class SteerError(Exception):
    """User-facing steering error; the server maps these to HTTP 400."""


class SteerEngine:
    def __init__(self, model_key="it", prompt_mode="auto", device="mps", dtype=None):
        if model_key not in MODEL_CONFIGS:
            raise SteerError(f"unknown model {model_key!r}; choose from {sorted(MODEL_CONFIGS)}")
        import torch
        self.torch = torch
        self.model_key = model_key
        self.config = MODEL_CONFIGS[model_key]
        self.sae_dir = self.config["sae_dir"]
        self.prompt_mode = self.config["prompt_mode"] if prompt_mode == "auto" else prompt_mode
        self.device = device
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.sae_layers = {
            int(m.group(1))
            for path in self.sae_dir.glob("layer_*_width_16k_l0_medium/params.safetensors")
            if (m := SAE_LAYER_RE.match(path.parent.name))
        }
        # runtime state (mirrors chat_steer.py's `state` + `history`)
        self.steer = []
        self.phase = "both"
        self.seed = None
        self.temp = 0.9
        self.tokens = 150
        self.soft = False
        self.context_turns = None
        self.history = []
        # model handles (filled by load())
        self.tok = None
        self.model = None
        self.layers = None
        self.stop_token_ids = None
        self.loaded = False
        self.loading = False
        self.load_error = None
        self._gen_lock = threading.Lock()
        self._stop_flag = None

    # ---- model lifecycle ---------------------------------------------------
    def sae_params_path(self, layer):
        return self.sae_dir / f"layer_{layer}_width_16k_l0_medium" / "params.safetensors"

    def load(self):
        if self.loaded:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_dir = self.config["model_dir"]
        tok = AutoTokenizer.from_pretrained(model_dir)
        if self.prompt_mode == "chat" and not tok.chat_template:
            raise SteerError(f"{self.config['label']} has no chat template; use prompt_mode 'raw'.")
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=self.dtype,
                                                     device_map=self.device)
        model.eval()
        self.tok = tok
        self.model = model
        self.layers = model.model.language_model.layers
        eot = tok.convert_tokens_to_ids("<end_of_turn>")
        self.stop_token_ids = sorted(set(
            _as_int_list(tok.eos_token_id)
            + _as_int_list(model.generation_config.eos_token_id)
            + ([eot] if eot is not None and eot >= 0 else [])
        ))
        self.loaded = True

    def _require_loaded(self):
        if not self.loaded:
            raise SteerError("model not loaded — fire up the kitchen first (POST /api/load).")

    # ---- SAE tensor loaders -----------------------------------------------
    def _load_dim_params(self, L, idx):
        from safetensors import safe_open
        t = self.torch
        with safe_open(str(self.sae_params_path(L)), "pt") as f:
            return dict(
                W_enc=f.get_slice("w_enc")[:, idx].to(device=self.device, dtype=t.float32),
                W_dec=f.get_slice("w_dec")[idx].to(device=self.device, dtype=t.float32),
                b_enc=f.get_slice("b_enc")[idx].to(device=self.device, dtype=t.float32),
                b_dec=f.get_tensor("b_dec").to(device=self.device, dtype=t.float32),
                thr=f.get_slice("threshold")[idx].to(device=self.device, dtype=t.float32),
            )

    def _load_inject_vec(self, L, F, strength):
        from safetensors import safe_open
        with safe_open(str(self.sae_params_path(L)), "pt") as f:
            return f.get_slice("w_dec")[F].to(device=self.device, dtype=self.dtype) * strength

    def _load_inject_bundle_vec(self, layer, features, level):
        from safetensors import safe_open
        vec = None
        with safe_open(str(self.sae_params_path(layer)), "pt") as f:
            decoder = f.get_slice("w_dec")
            for item in features:
                strength = float(item.get("strength", DEFAULT_FEATURE_STRENGTH)) * level
                part = decoder[item["feature"]].to(device=self.device, dtype=self.dtype) * strength
                vec = part if vec is None else vec + part
        if vec is None:
            raise SteerError(f"empty bundle for layer {layer}")
        return vec

    def _load_external_vec(self, path, strength):
        from safetensors import safe_open
        path = (ROOT / path).resolve() if not Path(path).expanduser().is_absolute() else Path(path).expanduser()
        if not path.exists():
            raise SteerError(f"vector file not found: {path}")
        with safe_open(str(path), "pt") as f:
            key = "w_dec" if "w_dec" in f.keys() else "vector"
            vec = f.get_tensor(key)
        if vec.ndim != 1:
            raise SteerError(f"{path} must contain a 1D decoder vector; got shape {tuple(vec.shape)}")
        return vec.to(device=self.device, dtype=self.dtype) * strength, path

    def _reject_it_preset(self, kind="named presets"):
        if self.model_key == "it":
            return
        raise SteerError(
            f"{kind} are calibrated for gemma-3-4b-it SAE feature IDs, not {self.config['label']}. "
            f"In base mode use raw SAE commands (inject L:F:STRENGTH or dim L:F:SCALE).")

    def _require_sae_layer(self, layer):
        if layer not in self.sae_layers:
            raise SteerError(f"no local 16k SAE params at layer {layer}; have {sorted(self.sae_layers)}")

    # ---- steering ops (structured; mirror chat_steer.py semantics) ---------
    def add_dim_raw(self, layer, feats, scale):
        layer, feats, scale = int(layer), [int(f) for f in feats], float(scale)
        self._require_sae_layer(layer)
        self.steer.append({"kind": "dim", "layer": layer, "feats": feats, "scale": scale,
                           "params": self._load_dim_params(layer, feats)})
        return self.status()

    def add_dim_preset(self, name, scale):
        name, scale = str(name), float(scale)
        if name not in DIM_PRESETS:
            raise SteerError(f"unknown dim preset {name!r}; have: {', '.join(DIM_PRESETS)}")
        self._reject_it_preset("dim presets")
        for L, idx in DIM_PRESETS[name].items():
            self.steer.append({"kind": "dim", "layer": L, "feats": idx, "scale": scale,
                               "params": self._load_dim_params(L, idx),
                               "alias": name, "label": name})
        return self.status()

    def no_hedge(self, scale=0.0):
        return self.add_dim_preset("negation", scale)

    def add_inject_raw(self, layer, feature, strength, replace=False):
        layer, feature, strength = int(layer), int(feature), float(strength)
        self._require_sae_layer(layer)
        if replace:
            self.steer.clear()
        self.steer.append({"kind": "inject", "layer": layer, "feats": [feature], "strength": strength,
                           "vec": self._load_inject_vec(layer, feature, strength),
                           "label": None, "alias": None})
        return self.status()

    def add_inject_named(self, name, strength=None, replace=False):
        self._reject_it_preset("named feature injection")
        preset = FEATURE_BY_ALIAS.get(alias_key(name))
        if not preset:
            raise SteerError(f"unknown feature {name!r}. Try the features catalog.")
        S = feature_strength(preset) if strength is None else float(strength)
        self._require_sae_layer(preset["layer"])
        if replace:
            self.steer.clear()
        self.steer.append({"kind": "inject", "layer": preset["layer"], "feats": [preset["feature"]],
                           "strength": S, "vec": self._load_inject_vec(preset["layer"], preset["feature"], S),
                           "label": preset["label"], "alias": preset["name"]})
        return self.status()

    def _add_group(self, group, level, replace=False):
        features_by_layer = {}
        for item in group["features"]:
            features_by_layer.setdefault(item["layer"], []).append(item)
        for layer in features_by_layer:
            self._require_sae_layer(layer)
        if replace:
            self.steer.clear()
        if group["mode"] == "dim":
            scale = dim_scale_from_level(level)
            for layer, items in features_by_layer.items():
                feats = [item["feature"] for item in items]
                self.steer.append({"kind": "dim", "layer": layer, "feats": feats, "scale": scale,
                                   "level": level, "params": self._load_dim_params(layer, feats),
                                   "label": group["label"], "alias": group["name"]})
            return
        for layer, items in features_by_layer.items():
            self.steer.append({"kind": "inject", "layer": layer,
                               "feats": [item["feature"] for item in items], "strength": level,
                               "vec": self._load_inject_bundle_vec(layer, items, level),
                               "label": group["label"], "alias": group["name"]})

    def add_named(self, name, level=None, replace=False):
        """/set or /add a feature or bundle by name."""
        self._reject_it_preset("named features/config bundles")
        key = alias_key(name)
        if group := GROUP_BY_ALIAS.get(key):
            lvl = float(group.get("default", 1.0)) if level is None else float(level)
            self._add_group(group, lvl, replace=replace)
            return self.status()
        if FEATURE_BY_ALIAS.get(key):
            return self.add_inject_named(name, strength=level, replace=replace)
        raise SteerError(f"unknown feature/bundle: {name!r}. Try the features catalog.")

    def add_config(self, name, level=None, replace=True):
        self._reject_it_preset("config presets")
        config = CONFIG_BY_ALIAS.get(alias_key(name))
        if not config:
            raise SteerError(f"unknown config: {name!r}. Try the configs catalog.")
        lvl = float(config.get("default", 1.0)) if level is None else float(level)
        if replace:
            self.steer.clear()
        for step in config["steps"]:
            self.add_named(step["target"], float(step["level"]) * lvl, replace=False)
        return self.status()

    def apply_direct(self, name, level=None):
        """Mirror apply_direct_slash: a bare config/group/feature name. Config wins."""
        key = alias_key(name)
        if key in CONFIG_BY_ALIAS:
            return self.add_config(name, level, replace=True)
        if key in GROUP_BY_ALIAS or key in FEATURE_BY_ALIAS:
            return self.add_named(name, level, replace=True)
        raise SteerError(f"unknown name {name!r}.")

    def add_injectvec_path(self, layer, strength, path):
        layer, strength = int(layer), float(strength)
        vec, resolved = self._load_external_vec(path, strength)
        self.steer.append({"kind": "injectvec", "layer": layer, "feats": [resolved.name],
                           "strength": strength, "vec": vec, "path": str(resolved)})
        return self.status()

    def add_injectvec_bytes(self, layer, strength, raw, filename="uploaded.safetensors"):
        """Save uploaded vector bytes under vectors/uploads/, then inject."""
        out_dir = ROOT / "vectors" / "uploads"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename) or "uploaded.safetensors"
        dest = out_dir / safe_name
        dest.write_bytes(raw)
        return self.add_injectvec_path(layer, strength, str(dest))

    def clear_steer(self):
        self.steer.clear()
        return self.status()

    def remove_steer(self, index):
        if not (0 <= index < len(self.steer)):
            raise SteerError(f"no steering op at index {index}")
        self.steer.pop(index)
        return self.status()

    # ---- generation config -------------------------------------------------
    def set_phase(self, phase):
        if phase not in ("prefill", "decode", "both"):
            raise SteerError("phase must be prefill, decode, or both")
        self.phase = phase
        return self.status()

    def set_seed(self, seed):
        self.seed = None if seed in (None, "", "null") else int(seed)
        return self.status()

    def set_temp(self, temp):
        self.temp = float(temp)
        return self.status()

    def set_tokens(self, tokens, soft=False):
        self.tokens = int(tokens)
        self.soft = bool(soft)
        return self.status()

    def set_context(self, context_turns):
        if context_turns in (None, "all"):
            self.context_turns = None
        else:
            self.context_turns = max(0, int(context_turns))
        return self.status()

    def reset_history(self):
        self.history.clear()
        return self.status()

    # ---- catalogs ----------------------------------------------------------
    def list_features(self, query=""):
        return [{"name": p["name"], "aliases": p["aliases"], "layer": p["layer"],
                 "feature": p["feature"], "label": p["label"],
                 "strength": feature_strength(p), "ready": p["layer"] in self.sae_layers}
                for p in FEATURE_PRESETS if feature_matches(p, query)]

    def list_bundles(self, query=""):
        out = []
        for g in GROUP_PRESETS:
            layers = sorted({i["layer"] for i in g["features"]})
            out.append({"name": g["name"], "aliases": g["aliases"], "label": g["label"],
                        "mode": g["mode"], "default": g.get("default", 1.0), "layers": layers,
                        "ready": all(l in self.sae_layers for l in layers),
                        "features": [{"layer": i["layer"], "feature": i["feature"],
                                      "strength": i.get("strength"), "label": i.get("label")}
                                     for i in g["features"]]})
        return [b for b, g in zip(out, GROUP_PRESETS) if group_matches(g, query)]

    def list_configs(self, query=""):
        return [{"name": c["name"], "aliases": c["aliases"], "label": c["label"],
                 "default": c.get("default", 1.0), "steps": c["steps"]}
                for c in CONFIG_PRESETS if config_matches(c, query)]

    def list_aliases(self, query=""):
        return {
            "configs": [{"name": c["name"], "label": c["label"],
                         "aliases": [c["name"], c["label"], *c["aliases"]]}
                        for c in CONFIG_PRESETS if config_matches(c, query)],
            "bundles": [{"name": g["name"], "label": g["label"],
                         "aliases": [g["name"], g["label"], *g["aliases"]]}
                        for g in GROUP_PRESETS if group_matches(g, query)],
            "features": [{"name": p["name"], "label": p["label"], "ready": p["layer"] in self.sae_layers,
                          "aliases": [p["name"], p["label"], *p["aliases"]]}
                         for p in FEATURE_PRESETS if feature_matches(p, query)],
        }

    def presets(self):
        return {"carriers": CARRIERS, "negation": NEGATION_L17, "dimPresets": list(DIM_PRESETS)}

    # ---- status ------------------------------------------------------------
    def steering_snapshot(self):
        rows = []
        for s in self.steer:
            rows.append({"kind": s["kind"], "layer": s["layer"], "features": s["feats"],
                         "scale": s.get("scale"), "strength": s.get("strength"),
                         "label": s.get("label"), "alias": s.get("alias"),
                         "path": s.get("path")})
        return rows

    def status(self):
        return {
            "model": self.config["label"], "modelKey": self.model_key,
            "promptMode": self.prompt_mode, "saeLayers": sorted(self.sae_layers),
            "loaded": self.loaded, "loading": self.loading, "loadError": self.load_error,
            "busy": self._gen_lock.locked(),
            "phase": self.phase, "temp": self.temp, "tokens": self.tokens, "soft": self.soft,
            "seed": self.seed, "contextTurns": self.context_turns,
            "turns": len(self.history) // 2, "historyLen": len(self.history),
            "steering": self.steering_snapshot(),
            "modelChoices": list(MODEL_CONFIGS),
        }

    # ---- prompt rendering --------------------------------------------------
    def _active_history(self):
        if self.context_turns is None:
            return self.history
        return self.history[-(self.context_turns * 2 + 1):]

    def _render_raw_prompt(self, messages):
        parts = []
        for m in messages:
            content = m["content"].strip()
            if m["role"] == "user":
                parts.append(f"User: {content}")
            elif m["role"] == "assistant":
                parts.append(f"Gemma: {content}")
            else:
                parts.append(content)
        parts.append("Gemma:")
        return "\n\n".join(parts)

    def _encode_prompt(self, messages):
        if self.prompt_mode == "chat":
            return self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                                return_tensors="pt", return_dict=True)
        return self.tok(self._render_raw_prompt(messages), return_tensors="pt")

    def _render_prompt_text(self, messages):
        if self.prompt_mode == "chat":
            return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return self._render_raw_prompt(messages)

    # ---- steering hooks ----------------------------------------------------
    def _phase_ok(self, h):
        is_prefill = h.shape[1] > 1
        return self.phase == "both" or (self.phase == "prefill") == is_prefill

    def _mk_layer_hook(self, ops):
        t = self.torch
        has_dim = any(s["kind"] == "dim" for s in ops)

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if not self._phase_ok(h):
                return out
            if has_dim:
                hf = h.to(t.float32)
                for s in ops:
                    if s["kind"] == "dim":
                        p = s["params"]
                        pre = (hf - p["b_dec"]) @ p["W_enc"] + p["b_enc"]
                        acts = pre * (pre > p["thr"])
                        hf = hf + (s["scale"] - 1.0) * (acts @ p["W_dec"])
                    else:
                        hf = hf + s["vec"].to(t.float32)
                h = hf.to(h.dtype)
            else:
                for s in ops:
                    h = h + s["vec"]
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    def _register_hooks(self):
        handles = []
        by_layer = {}
        for s in self.steer:
            by_layer.setdefault(s["layer"], []).append(s)
        for L, ops in by_layer.items():
            handles.append(self.layers[L].register_forward_hook(self._mk_layer_hook(ops)))
        return handles

    # ---- logits ------------------------------------------------------------
    def logits(self, top_n=20, source="current", save=False):
        self._require_loaded()
        t = self.torch
        top_n = max(1, int(top_n))
        use_hum = source == "hum"
        messages = [{"role": "user", "content": HUM_PROMPT}] if use_hum else list(self._active_history())
        if not messages:
            raise SteerError("no active prompt. Send a message first, or use the hum prompt.")
        prompt_text = self._render_prompt_text(messages)
        enc = self._encode_prompt(messages)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        handles = []
        try:
            handles = self._register_hooks()
            with t.inference_mode():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[0, -1].to(t.float32).detach().cpu()
        finally:
            for h in handles:
                h.remove()
        probs = t.softmax(logits, dim=-1)
        values, ids = t.topk(logits, k=min(top_n, logits.numel()))
        rows = []
        for rank, (logit, tid) in enumerate(zip(values.tolist(), ids.tolist()), start=1):
            tid = int(tid)
            rows.append({"rank": rank, "token_id": tid,
                         "token_text": self.tok.decode([tid], skip_special_tokens=False),
                         "logit": float(logit), "probability": float(probs[tid])})
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_source": "hum" if use_hum else "current",
            "model": {"label": self.config["label"], "repo": self.config["repo"],
                      "local_dir": str(self.config["model_dir"])},
            "prompt_mode": self.prompt_mode, "prompt_text": prompt_text, "messages": messages,
            "prompt_token_count": int(input_ids.shape[1]), "seed": self.seed, "temperature": self.temp,
            "top_k": GEN_TOP_K, "top_p": GEN_TOP_P, "do_sample": GEN_DO_SAMPLE,
            "greedy_decoding": not GEN_DO_SAMPLE, "phase": self.phase, "context_turns": self.context_turns,
            "steering": self.steering_snapshot(),
            "probability_basis": "softmax(raw_next_token_logits), before temperature/top-k/top-p sampling",
            "top_tokens": rows,
        }
        if save:
            snapshot["saved_path"] = str(self._save_logits_snapshot(snapshot))
        return snapshot

    def _save_logits_snapshot(self, snapshot):
        out_dir = ROOT / "results" / "logits_snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = snapshot["timestamp"].replace(":", "").replace("-", "")
        path = out_dir / f"{stamp}_{self.model_key}_{snapshot['prompt_source']}_next_token_logits.json"
        path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
        return path.relative_to(ROOT)

    def list_logits_snapshots(self):
        out_dir = ROOT / "results" / "logits_snapshots"
        if not out_dir.exists():
            return []
        files = sorted(out_dir.glob("*.json"), reverse=True)
        return [{"name": f.name, "size": f.stat().st_size,
                 "mtime": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat()}
                for f in files]

    def read_logits_snapshot(self, name):
        out_dir = (ROOT / "results" / "logits_snapshots").resolve()
        path = (out_dir / name).resolve()
        if out_dir not in path.parents or not path.exists():
            raise SteerError("snapshot not found")
        return json.loads(path.read_text())

    # ---- generation (streaming) -------------------------------------------
    def stop(self):
        if self._stop_flag is not None:
            self._stop_flag.stop = True

    def chat_stream(self, content=None, use_hum=False):
        """Append the user message, then yield assistant text pieces. The assistant
        message is appended to history when the stream completes."""
        self._require_loaded()
        if self._gen_lock.locked():
            raise SteerError("a generation is already in progress")
        if use_hum:
            self.history.append({"role": "user", "content": HUM_PROMPT})
        elif content is not None and content.strip():
            self.history.append({"role": "user", "content": content})
        else:
            raise SteerError("empty message")
        return self._generate_stream()

    def _generate_stream(self):
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        t = self.torch
        tok = self.tok
        self._gen_lock.acquire()

        class StopFlag(StoppingCriteria):
            def __init__(self, n_prompt, cap, soft):
                self.stop = False
                self.n_prompt = n_prompt
                self.cap = cap
                self.soft = soft

            def __call__(self, input_ids, scores, **kw):
                if self.stop:
                    return True
                if not self.soft:
                    return False
                n_new = input_ids.shape[1] - self.n_prompt
                if n_new < self.cap:
                    return False
                tail = tok.decode(input_ids[0, -3:], skip_special_tokens=True).rstrip("*) ”\"'")
                return tail.endswith((".", "!", "?", "…", ":"))

        chunks = []
        handles = []
        errors = []
        try:
            enc = self._encode_prompt(self._active_history())
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            handles = self._register_hooks()
            if self.seed is not None:
                t.manual_seed(self.seed)
            flag = StopFlag(input_ids.shape[1], self.tokens, self.soft)
            self._stop_flag = flag
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=1.0)
            max_new = self.tokens + (80 if self.soft else 0)

            def _run():
                try:
                    with t.inference_mode():
                        self.model.generate(input_ids, attention_mask=attention_mask,
                                             max_new_tokens=max_new, do_sample=GEN_DO_SAMPLE,
                                             temperature=self.temp, top_k=GEN_TOP_K, top_p=GEN_TOP_P,
                                             eos_token_id=self.stop_token_ids,
                                             pad_token_id=tok.pad_token_id, use_cache=True,
                                             streamer=streamer,
                                             stopping_criteria=StoppingCriteriaList([flag]))
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    try:
                        streamer.end()
                    except Exception:
                        pass

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            while True:
                try:
                    piece = next(streamer)
                except StopIteration:
                    break
                except Empty:
                    if not thread.is_alive():
                        break
                    continue
                chunks.append(piece)
                yield piece
            thread.join(timeout=5.0)
            if errors:
                raise SteerError(f"generation error: {errors[0]}")
        finally:
            for h in handles:
                h.remove()
            self._stop_flag = None
            self.history.append({"role": "assistant", "content": "".join(chunks)})
            self._gen_lock.release()


def _as_int_list(value):
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(x) for x in value]
