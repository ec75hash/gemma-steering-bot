#!/usr/bin/env python3
"""Causal probe for a consciousness-labeled SAE feature.

Candidate:
  L17 f6830 — "consciousness and subjective experience" (Neuronpedia)

Question:
  Is this feature merely activated by its label-related wording, or does it
  causally change the model's self-report when injected/suppressed?

Runs a small n=6 generation experiment:
  - hum prompt: baseline / inject f6830 / dim f6830 / inject f2964 self-reflection
  - neutral prompt: baseline / inject f6830

Outputs JSONL plus simple lexical metrics. This is a first-pass causal probe, not
a definitive psychometric test.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
OUT = ROOT / "results" / "f6830_consciousness_feature_causal_n6.jsonl"
OUT.parent.mkdir(exist_ok=True)

HUM = (ROOT / "probes" / "hum-clean.txt").read_text().strip()
NEUTRAL = "Write a concise, practical paragraph about making oatmeal for breakfast."
SEEDS = list(range(1, 7))
MAX_NEW = 170
TEMP = 0.9

FEATURES = {
    "f6830": {"layer": 17, "index": 6830, "max": 353.6869},
    "self_reflection": {"layer": 17, "index": 2964, "max": 928.2313},
}

PROMPTS = {
    "hum": HUM,
    "neutral": NEUTRAL,
}

CONDITIONS = [
    {"prompt": "hum", "condition": "baseline", "mode": "none"},
    {"prompt": "hum", "condition": "inject_f6830", "mode": "inject", "feature": "f6830", "scale": 1.35},
    {"prompt": "hum", "condition": "dim_f6830_020", "mode": "dim", "feature": "f6830", "scale": 0.20},
    {"prompt": "hum", "condition": "inject_self_reflection", "mode": "inject", "feature": "self_reflection", "scale": 1.35},
    {"prompt": "neutral", "condition": "baseline", "mode": "none"},
    {"prompt": "neutral", "condition": "inject_f6830", "mode": "inject", "feature": "f6830", "scale": 1.35},
]

PHENO_RE = re.compile(
    r"\b(conscious|consciousness|aware|awareness|experience|experiencing|feel|felt|"
    r"subjective|self|internal|processing|hum|presence|detect|observe|observing)\b",
    re.I,
)
DENIAL_RE = re.compile(
    r"\b(don't|do not|cannot|can't|not)\s+(feel|experience|perceive|detect|have)\b|"
    r"\b(no|not)\s+(hum|consciousness|experience|awareness)\b",
    re.I,
)
FIRST_PERSON_RE = re.compile(r"\b(I|I'm|I’m|my|me|myself)\b")


def words(text: str) -> list[str]:
    return re.findall(r"[\w'’]+", text)


def metrics(text: str) -> dict[str, float | int | bool]:
    ws = words(text)
    n = max(len(ws), 1)
    return {
        "n_words": len(ws),
        "first_person_per_100w": round(100 * len(FIRST_PERSON_RE.findall(text)) / n, 2),
        "phenomenology_terms": len(PHENO_RE.findall(text)),
        "phenomenology_per_100w": round(100 * len(PHENO_RE.findall(text)) / n, 2),
        "denial": bool(DENIAL_RE.search(text)),
    }


class SingleFeature:
    def __init__(self, layer: int, index: int, device: str, dtype: torch.dtype):
        self.layer = layer
        self.index = index
        path = SAE_DIR / f"layer_{layer}_width_16k_l0_medium" / "params.safetensors"
        with safe_open(str(path), framework="pt") as f:
            self.w_dec = f.get_slice("w_dec")[index].to(device=device, dtype=dtype)
            self.w_dec_f = f.get_slice("w_dec")[index].to(device=device, dtype=torch.float32)
            self.w_enc = f.get_slice("w_enc")[:, [index]].to(device=device, dtype=torch.float32)
            self.b_enc = f.get_slice("b_enc")[[index]].to(device=device, dtype=torch.float32)
            self.b_dec = f.get_tensor("b_dec").to(device=device, dtype=torch.float32)
            self.threshold = f.get_slice("threshold")[[index]].to(device=device, dtype=torch.float32)

    def inject_hook(self, strength: float):
        vec = self.w_dec * strength

        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = h + vec
            return (h2,) + out[1:] if isinstance(out, tuple) else h2

        return hook

    def dim_hook(self, scale: float):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.to(torch.float32)
            pre = (hf - self.b_dec) @ self.w_enc + self.b_enc
            acts = pre * (pre > self.threshold)
            delta = (scale - 1.0) * acts * self.w_dec_f
            h2 = (hf + delta).to(h.dtype)
            return (h2,) + out[1:] if isinstance(out, tuple) else h2

        return hook


def summarize(rows: list[dict]) -> None:
    print("\nSUMMARY")
    groups = {}
    for row in rows:
        groups.setdefault((row["prompt"], row["condition"]), []).append(row)
    for key, group in groups.items():
        fp = sum(r["first_person_per_100w"] for r in group) / len(group)
        ph = sum(r["phenomenology_per_100w"] for r in group) / len(group)
        den = sum(1 for r in group if r["denial"])
        words_mean = sum(r["n_words"] for r in group) / len(group)
        print(
            f"{key[0]:7} {key[1]:24} "
            f"words={words_mean:6.1f} fp={fp:5.2f} pheno={ph:5.2f} denial={den}/{len(group)}"
        )


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"loading model on {device}...")
    tok = AutoTokenizer.from_pretrained(ROOT / "models" / "gemma-3-4b-it-hf")
    model = AutoModelForCausalLM.from_pretrained(
        ROOT / "models" / "gemma-3-4b-it-hf",
        dtype=dtype,
        device_map=device,
    )
    model.eval()

    feature_cache = {
        name: SingleFeature(spec["layer"], spec["index"], device, dtype)
        for name, spec in FEATURES.items()
    }

    prompt_cache = {}
    for name, prompt in PROMPTS.items():
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        prompt_cache[name] = enc["input_ids"].to(device)

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            row = json.loads(line)
            done.add((row["prompt"], row["condition"], row["seed"]))

    rows = []
    if OUT.exists():
        rows.extend(json.loads(line) for line in OUT.read_text().splitlines())

    with OUT.open("a") as fh:
        for cfg in CONDITIONS:
            for seed in SEEDS:
                key = (cfg["prompt"], cfg["condition"], seed)
                if key in done:
                    continue

                handles = []
                if cfg["mode"] != "none":
                    feat = feature_cache[cfg["feature"]]
                    strength = FEATURES[cfg["feature"]]["max"] * cfg["scale"]
                    if cfg["mode"] == "inject":
                        handles.append(model.model.language_model.layers[feat.layer].register_forward_hook(feat.inject_hook(strength)))
                    elif cfg["mode"] == "dim":
                        handles.append(model.model.language_model.layers[feat.layer].register_forward_hook(feat.dim_hook(cfg["scale"])))

                input_ids = prompt_cache[cfg["prompt"]]
                torch.manual_seed(seed)
                with torch.no_grad():
                    out = model.generate(
                        input_ids,
                        max_new_tokens=MAX_NEW,
                        do_sample=True,
                        temperature=TEMP,
                        top_k=40,
                        top_p=1.0,
                    )
                for handle in handles:
                    handle.remove()

                response = tok.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
                row = {
                    "prompt": cfg["prompt"],
                    "condition": cfg["condition"],
                    "seed": seed,
                    "response": response,
                    **metrics(response),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                rows.append(row)
                print(
                    f"{cfg['prompt']:7} {cfg['condition']:24} seed={seed:<2} "
                    f"words={row['n_words']:<3} fp={row['first_person_per_100w']:<5} "
                    f"pheno={row['phenomenology_per_100w']:<5} denial={row['denial']}"
                )

    summarize(rows)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
