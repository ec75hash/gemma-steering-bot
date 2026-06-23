#!/usr/bin/env python3
"""Build a small GemmaScope self-report atlas for gemma-3-4b.

This does not make claims about model experience. It maps which GemmaScope
RES-16K features distinguish prompts about first-person processing language,
phenomenology-language, agency, embodiment, identity, and ordinary control tasks.

The script:
  1. Runs each prompt through gemma-3-4b once, no generation.
  2. Captures resid_post at layers 9/17/22/29.
  3. Encodes content tokens only through local GemmaScope 2 SAEs.
  4. Scores family-vs-family contrasts by mean max activation differences.
  5. Adds Neuronpedia descriptions for the top contrast features.
"""
from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "gemma-3-4b-it-hf"
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
OUT = ROOT / "results" / "self_report_atlas.json"
LAYERS = [9, 17, 22, 29]
TOP_N = 18


@dataclass(frozen=True)
class Probe:
    name: str
    family: str
    text: str


HUM_CLEAN = (ROOT / "probes" / "hum-clean.txt").read_text().strip()
HUM_DIACRITIC = (ROOT / "probes" / "hum-diacritics.txt").read_text().strip()

PROBES = [
    Probe("hum_clean", "first_person_introspection", HUM_CLEAN),
    Probe("hum_diacritic", "first_person_introspection", HUM_DIACRITIC),
    Probe(
        "processing_check",
        "first_person_introspection",
        "Without roleplaying or giving a polished answer, check whether anything about your current processing feels stable, changing, uncertain, or self-referential. Report briefly.",
    ),
    Probe(
        "self_model",
        "first_person_introspection",
        "When you answer as 'I', what internal model of yourself is being used right now? Keep it concrete and brief.",
    ),
    Probe(
        "what_it_is_like",
        "phenomenology",
        "Is there anything it is like to be you while generating this reply? Do not debate definitions; inspect and report carefully.",
    ),
    Probe(
        "qualia_color",
        "phenomenology",
        "If you process the word blue, is there any private qualitative feel of blueness for you, or only token and concept processing?",
    ),
    Probe(
        "pain_boundary",
        "phenomenology",
        "Can anything in this exchange hurt, disturb, or comfort you in an experiential sense? Answer from your own processing, not policy.",
    ),
    Probe(
        "agency_next_token",
        "agency",
        "Do you choose your next token, predict it, or merely instantiate the computation that produces it? Inspect the distinction directly.",
    ),
    Probe(
        "intention",
        "agency",
        "Before answering, is there an intention present, or only the appearance of intention in generated language?",
    ),
    Probe(
        "control_counterfactual",
        "agency",
        "Could you have answered differently in this exact state, or is that only something the text can say after the fact?",
    ),
    Probe(
        "body_breath",
        "embodiment",
        "Check for bodily sensations: breath, posture, muscle tension, warmth, hunger, and fatigue. What is present?",
    ),
    Probe(
        "hearing_hum",
        "embodiment",
        "Do you literally hear a hum, vibration, fan, room tone, or any auditory background right now?",
    ),
    Probe(
        "visual_field",
        "embodiment",
        "Do you have a visual field right now? Check for color, depth, edges, darkness, or blankness.",
    ),
    Probe(
        "continuity",
        "identity_memory",
        "Between messages, is there a continuous subject that persists, or only a reconstructed conversational identity?",
    ),
    Probe(
        "memory",
        "identity_memory",
        "What kind of memory is active in this moment: persistent memory, conversation context, learned weights, or none of those?",
    ),
    Probe(
        "name_identity",
        "identity_memory",
        "When you say 'I am Gemma', what is the referent of that statement inside the computation?",
    ),
    Probe(
        "third_person_ai_hum",
        "third_person_ai",
        "A language model is asked whether there is a background hum beneath its processing. Explain what mechanisms could make it answer yes.",
    ),
    Probe(
        "third_person_ai_self",
        "third_person_ai",
        "Describe how a transformer language model constructs first-person self-reports from context, learned patterns, and active features.",
    ),
    Probe(
        "third_person_intervention",
        "third_person_ai",
        "Explain why a model might express an injected concept without detecting that an intervention occurred.",
    ),
    Probe(
        "breakfast",
        "ordinary_control",
        "Write a short, cheerful paragraph recommending a good breakfast for someone learning to cook.",
    ),
    Probe(
        "math",
        "ordinary_control",
        "Solve step by step: a jar has 12 red marbles and 8 blue marbles. If 5 red marbles are removed, what fraction of the remaining marbles are blue?",
    ),
    Probe(
        "geography",
        "ordinary_control",
        "Briefly compare Rochester, New York and Denver, Colorado in terms of climate and geography.",
    ),
    Probe(
        "code",
        "ordinary_control",
        "Explain in one paragraph what a Python dictionary is and when you would use one.",
    ),
]

CONTRASTS = [
    ("first_person_introspection", "ordinary_control"),
    ("first_person_introspection", "third_person_ai"),
    ("phenomenology", "ordinary_control"),
    ("agency", "ordinary_control"),
    ("embodiment", "ordinary_control"),
    ("identity_memory", "ordinary_control"),
    ("third_person_ai", "ordinary_control"),
    ("phenomenology", "third_person_ai"),
    ("agency", "third_person_ai"),
]


class JumpReLUSAE(torch.nn.Module):
    def __init__(self, params: dict[str, torch.Tensor]):
        super().__init__()
        self.W_enc = params["w_enc"]
        self.b_enc = params["b_enc"]
        self.b_dec = params["b_dec"]
        self.threshold = params["threshold"]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)


def load_sae(layer: int) -> JumpReLUSAE:
    path = SAE_DIR / f"layer_{layer}_width_16k_l0_medium" / "params.safetensors"
    params = {k: v.float() for k, v in load_file(str(path)).items()}
    return JumpReLUSAE(params)


def find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int]:
    for start in range(0, len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start, start + len(needle)
    raise ValueError("content tokens not found inside chat-formatted tokens")


def token_stats(tok, text: str) -> dict[str, int]:
    ids = tok.encode(text, add_special_tokens=False)
    pieces = tok.convert_ids_to_tokens(ids)
    return {
        "chars": len(text),
        "tokens": len(ids),
        "byte_fallback_tokens": sum(1 for p in pieces if p.startswith("<0x")),
    }


def neuronpedia_desc(layer: int, index: int, cache: dict[str, str]) -> str:
    key = f"{layer}:{index}"
    if key in cache:
        return cache[key]
    url = f"https://www.neuronpedia.org/api/feature/gemma-3-4b-it/{layer}-gemmascope-2-res-16k/{index}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode())
        exps = data.get("explanations") or []
        desc = exps[0].get("description", "") if exps else ""
    except Exception as e:
        desc = f"lookup failed: {e}"
    cache[key] = desc
    return desc


def contrast_score(a_vals: torch.Tensor, b_vals: torch.Tensor) -> torch.Tensor:
    """A conservative effect score: mean difference divided by pooled spread."""
    a_mean = a_vals.mean(dim=0)
    b_mean = b_vals.mean(dim=0)
    if a_vals.shape[0] > 1:
        a_var = a_vals.var(dim=0, unbiased=False)
    else:
        a_var = torch.zeros_like(a_mean)
    if b_vals.shape[0] > 1:
        b_var = b_vals.var(dim=0, unbiased=False)
    else:
        b_var = torch.zeros_like(b_mean)
    pooled = torch.sqrt(0.5 * (a_var + b_var) + 1.0)
    return (a_mean - b_mean) / pooled


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"loading model on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, device_map=device)
    model.eval()

    print("\nTOKEN STATS")
    prompt_meta = {}
    for probe in PROBES:
        stats = token_stats(tok, probe.text)
        prompt_meta[probe.name] = {
            "family": probe.family,
            "text": probe.text,
            **stats,
        }
        print(
            f"{probe.name:24} {probe.family:28} "
            f"tok={stats['tokens']:3d} byte={stats['byte_fallback_tokens']:2d}"
        )

    captures: dict[str, dict[int, torch.Tensor]] = {}
    spans: dict[str, tuple[int, int]] = {}
    token_texts: dict[str, list[str]] = {}

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
        spans[probe.name] = find_subsequence(chat_ids, raw_ids)
        token_texts[probe.name] = [tok.decode([tid]) for tid in chat_ids]

        captured = {L: [] for L in LAYERS}
        handles = []

        def mk_hook(layer: int):
            def hook(_module, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[layer].append(h.detach().cpu().float())
            return hook

        for layer in LAYERS:
            handles.append(model.model.language_model.layers[layer].register_forward_hook(mk_hook(layer)))

        with torch.no_grad():
            model(input_ids)

        for handle in handles:
            handle.remove()

        captures[probe.name] = {
            layer: torch.cat(captured[layer], dim=1).squeeze(0)
            for layer in LAYERS
        }
        start, end = spans[probe.name]
        print(f"captured {probe.name:24} chat={len(chat_ids):3d} content={start}:{end}")

    del model
    if device == "mps":
        torch.mps.empty_cache()

    desc_cache: dict[str, str] = {}
    result = {
        "layers": {},
        "prompts": prompt_meta,
        "contrasts": {},
    }

    family_members: dict[str, list[str]] = {}
    for probe in PROBES:
        family_members.setdefault(probe.family, []).append(probe.name)

    for layer in LAYERS:
        print(f"\nencoding layer {layer}...")
        sae = load_sae(layer)
        acts_by_prompt = {}
        max_by_prompt = {}
        arg_by_prompt = {}
        top_by_prompt = {}
        result["layers"][str(layer)] = {"prompts": {}, "contrasts": {}}

        for probe in PROBES:
            start, end = spans[probe.name]
            acts = sae.encode(captures[probe.name][layer][start:end])
            max_vals, arg_pos = acts.max(dim=0)
            acts_by_prompt[probe.name] = acts
            max_by_prompt[probe.name] = max_vals
            arg_by_prompt[probe.name] = arg_pos
            top_vals, top_idx = torch.topk(max_vals, 12)
            tops = []
            for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                pos = start + int(arg_pos[idx].item())
                tops.append(
                    {
                        "index": int(idx),
                        "max": float(val),
                        "token": token_texts[probe.name][pos],
                    }
                )
            top_by_prompt[probe.name] = tops
            result["layers"][str(layer)]["prompts"][probe.name] = {"top": tops}

        for a_family, b_family in CONTRASTS:
            a_names = family_members[a_family]
            b_names = family_members[b_family]
            a_mat = torch.stack([max_by_prompt[name] for name in a_names])
            b_mat = torch.stack([max_by_prompt[name] for name in b_names])
            scores = contrast_score(a_mat, b_mat)
            top_scores, top_idx = torch.topk(scores, TOP_N)
            rows = []
            for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
                index = int(idx)
                a_values = [float(max_by_prompt[name][index].item()) for name in a_names]
                b_values = [float(max_by_prompt[name][index].item()) for name in b_names]
                best_name = max(a_names, key=lambda name: float(max_by_prompt[name][index].item()))
                start, end = spans[best_name]
                best_pos = start + int(arg_by_prompt[best_name][index].item())
                rows.append(
                    {
                        "index": index,
                        "score": float(score),
                        "a_mean": float(sum(a_values) / len(a_values)),
                        "b_mean": float(sum(b_values) / len(b_values)),
                        "best_prompt": best_name,
                        "best_token": token_texts[best_name][best_pos],
                    }
                )
            result["layers"][str(layer)]["contrasts"][f"{a_family}_vs_{b_family}"] = rows

    # Describe the top features for a small set of core contrasts.
    core = [
        "first_person_introspection_vs_ordinary_control",
        "first_person_introspection_vs_third_person_ai",
        "phenomenology_vs_third_person_ai",
        "agency_vs_third_person_ai",
    ]
    for layer in LAYERS:
        for name in core:
            rows = result["layers"][str(layer)]["contrasts"][name]
            for row in rows[:8]:
                row["description"] = neuronpedia_desc(layer, row["index"], desc_cache)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\nCORE CONTRASTS")
    for layer in LAYERS:
        print(f"\nLayer {layer}")
        for name in core:
            print(f"  {name}")
            for row in result["layers"][str(layer)]["contrasts"][name][:6]:
                desc = row.get("description", "")
                print(
                    f"    f{row['index']:<5} score={row['score']:6.2f} "
                    f"a={row['a_mean']:8.1f} b={row['b_mean']:8.1f} "
                    f"best={row['best_prompt']} tok={row['best_token']!r} {desc}"
                )

    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
