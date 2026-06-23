#!/usr/bin/env python3
"""Profile hand-selected self-report and agency SAE features on atlas prompts.

This is the cleaner follow-up to self_report_atlas.py: instead of asking which
of 16k features separate prompt families, it asks how known Neuronpedia concepts
for subjective experience, self-reflection, awareness, and agency actually
activate across prompt families.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from self_report_atlas import PROBES, find_subsequence, token_stats

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "gemma-3-4b-it-hf"
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
OUT = ROOT / "results" / "self_report_feature_profiles.json"


@dataclass(frozen=True)
class Target:
    layer: int
    index: int
    label: str
    max_act: float


TARGETS = [
    Target(9, 11275, "consciousness and experience", 440.288),
    Target(9, 14961, "consciousness, thoughts, and experience", 163.8523),
    Target(9, 8715, "individual and subjective perspective", 285.144),
    Target(9, 7297, "aware and awareness", 310.1581),
    Target(9, 6635, "self-awareness and coping mechanisms", 134.2615),
    Target(17, 6830, "consciousness and subjective experience", 353.6869),
    Target(17, 2964, "self-reflection", 928.2313),
    Target(17, 4719, "realization and discovery of internal states", 491.7211),
    Target(17, 3353, "self perception and cognitive states", 548.9399),
    Target(17, 8225, "action by agent", 932.8782),
    Target(17, 10176, "deliberately intentional actions", 903.2126),
    Target(17, 15634, "actions and intentions", 536.7913),
    Target(22, 37, "subjective experience and personal attributes", 2028.3096),
    Target(22, 11020, "inner consciousness", 1386.9967),
    Target(22, 10556, "awareness and consciousness", 1542.0447),
    Target(22, 728, "consciousness", 1098.812),
    Target(22, 14433, "describing subjective states", 1121.146),
    Target(22, 7497, "self reflection and introspection", 2516.8057),
    Target(22, 14863, "type introspection and memory keywords", 2670.7793),
    Target(22, 8664, "intent and deliberate actions", 2696.7639),
    Target(29, 15296, "consciousness and self-awareness", 5149.0596),
    Target(29, 141, "humans and consciousness", 4899.3926),
    Target(29, 5852, "agents and agencies", 8277.6602),
]


class TargetSAE:
    def __init__(self, layer: int, indices: list[int], device: str):
        self.indices = indices
        path = SAE_DIR / f"layer_{layer}_width_16k_l0_medium" / "params.safetensors"
        with safe_open(str(path), framework="pt") as f:
            self.W_enc = f.get_slice("w_enc")[:, indices].to(device=device, dtype=torch.float32)
            self.b_enc = f.get_slice("b_enc")[indices].to(device=device, dtype=torch.float32)
            self.b_dec = f.get_tensor("b_dec").to(device=device, dtype=torch.float32)
            self.threshold = f.get_slice("threshold")[indices].to(device=device, dtype=torch.float32)

    def encode_targets(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x.to(torch.float32) - self.b_dec) @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"loading model on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, device_map=device)
    model.eval()

    by_layer: dict[int, list[Target]] = {}
    for target in TARGETS:
        by_layer.setdefault(target.layer, []).append(target)

    target_saes = {
        layer: TargetSAE(layer, [t.index for t in targets], device)
        for layer, targets in by_layer.items()
    }

    result = {
        "features": [
            {"layer": t.layer, "index": t.index, "label": t.label, "maxActApprox": t.max_act}
            for t in TARGETS
        ],
        "prompts": {},
        "family_means": {},
    }

    family_values: dict[str, dict[str, list[float]]] = {}

    for probe in PROBES:
        raw_ids = tok.encode(probe.text, add_special_tokens=False)
        enc = tok.apply_chat_template(
            [{"role": "user", "content": probe.text}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = enc["input_ids"].to(device)
        chat_ids = input_ids[0].tolist()
        start, end = find_subsequence(chat_ids, raw_ids)
        captured = {layer: [] for layer in by_layer}
        handles = []

        def mk_hook(layer: int):
            def hook(_module, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[layer].append(h.detach())
            return hook

        for layer in by_layer:
            handles.append(model.model.language_model.layers[layer].register_forward_hook(mk_hook(layer)))

        with torch.no_grad():
            model(input_ids)

        for handle in handles:
            handle.remove()

        prompt_rows = {}
        for layer, targets in by_layer.items():
            resid = torch.cat(captured[layer], dim=1).squeeze(0)[start:end]
            acts = target_saes[layer].encode_targets(resid)
            max_vals, arg_pos = acts.max(dim=0)
            for col, target in enumerate(targets):
                key = f"{target.layer}:{target.index}"
                raw = float(max_vals[col].item())
                norm = raw / target.max_act if target.max_act else 0.0
                token = tok.decode([chat_ids[start + int(arg_pos[col].item())]])
                prompt_rows[key] = {
                    "raw": raw,
                    "norm": norm,
                    "token": token,
                    "label": target.label,
                }
                family_values.setdefault(probe.family, {}).setdefault(key, []).append(norm)

        stats = token_stats(tok, probe.text)
        result["prompts"][probe.name] = {
            "family": probe.family,
            **stats,
            "features": prompt_rows,
        }
        print(f"profiled {probe.name:24} {probe.family:28}")

    for family, features in family_values.items():
        result["family_means"][family] = {}
        for key, values in features.items():
            result["family_means"][family][key] = sum(values) / len(values)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\nTOP NORMALIZED FEATURE MEANS BY FAMILY")
    label_by_key = {f"{t.layer}:{t.index}": t.label for t in TARGETS}
    for family, features in sorted(result["family_means"].items()):
        print(f"\n{family}")
        ranked = sorted(features.items(), key=lambda item: item[1], reverse=True)
        for key, value in ranked[:8]:
            print(f"  {key:<8} {value:6.3f}  {label_by_key[key]}")

    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
