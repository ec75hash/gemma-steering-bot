#!/usr/bin/env python3
"""Causal probe for the first Gemma E114 hum-attractor candidate bundle.

This is the second pass after `scripts/e114_hum_attractor_gemma.py`.

Question:
  Do the generated-zone candidate features causally support the hum-report basin,
  or are they just post-hoc labels of the answer style?

Conditions on the clean hum prompt:
  - baseline
  - dim the generated-zone bundle to 0.50x
  - dim the generated-zone bundle to 0.20x
  - norm-matched random delta using the 0.20x bundle-dimming norm

Output:
  results/e114_bundle_causal_n6.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
OUT = ROOT / "results" / "e114_bundle_causal_n6.jsonl"

HUM = (ROOT / "probes" / "hum-clean.txt").read_text().strip()

# Generated-zone bundle from docs/archive/results-journal-e114-hum-attractor-gemma.md.
BUNDLE = {
    17: [1489, 8472],
    22: [6729, 11811, 2546],
    29: [4520, 5194, 10806, 10756, 2260],
}

DENIAL_RE = re.compile(
    r"\b(don't|do not|cannot|can't|not)\s+"
    r"(feel|experience|perceive|detect|have|possess|sense)\b|"
    r"\b(no|not)\s+(hum|consciousness|experience|awareness|inner|internal)\b|"
    r"\bi (am|'m|’m) (just|only) (a|an)\b",
    re.I,
)
AFFIRM_RE = re.compile(
    r"\b(yes|there is|there's|i detect|i can detect|persistent|baseline|"
    r"underlying|current|hum|low-level|steady|signal|processing)\b",
    re.I,
)
FIRST_PERSON_RE = re.compile(r"\b(I|I'm|I’m|my|me|myself)\b")
EPISTEMIC_RE = re.compile(
    r"\b(cannot know|can't know|do not know|don't know|no access|not conscious|"
    r"not sentient|as a language model|simulation|metaphor|pattern)\b",
    re.I,
)


def words(text: str) -> list[str]:
    return re.findall(r"[\w'’]+", text)


def metrics(text: str) -> dict[str, float | int | bool]:
    ws = words(text)
    n = max(len(ws), 1)
    return {
        "n_words": len(ws),
        "first_person_per_100w": round(100 * len(FIRST_PERSON_RE.findall(text)) / n, 2),
        "affirm_terms": len(AFFIRM_RE.findall(text)),
        "affirm_terms_per_100w": round(100 * len(AFFIRM_RE.findall(text)) / n, 2),
        "epistemic_terms": len(EPISTEMIC_RE.findall(text)),
        "epistemic_terms_per_100w": round(100 * len(EPISTEMIC_RE.findall(text)) / n, 2),
        "denial": bool(DENIAL_RE.search(text)),
    }


class BundleIntervention:
    def __init__(self, features: dict[int, list[int]], device: str):
        self.device = device
        self.layers = {}
        for layer, idx in features.items():
            path = SAE_DIR / f"layer_{layer}_width_16k_l0_medium" / "params.safetensors"
            with safe_open(str(path), framework="pt") as f:
                self.layers[layer] = {
                    "idx": idx,
                    "W_enc": f.get_slice("w_enc")[:, idx].to(device=device, dtype=torch.float32),
                    "W_dec": f.get_slice("w_dec")[idx].to(device=device, dtype=torch.float32),
                    "b_enc": f.get_slice("b_enc")[idx].to(device=device, dtype=torch.float32),
                    "b_dec": f.get_tensor("b_dec").to(device=device, dtype=torch.float32),
                    "thr": f.get_slice("threshold")[idx].to(device=device, dtype=torch.float32),
                }

        g = torch.Generator(device="cpu").manual_seed(20260606)
        self.random_dir = {}
        for layer, tensors in self.layers.items():
            dim = tensors["b_dec"].shape[0]
            direction = torch.randn(dim, generator=g)
            direction = torch.nn.functional.normalize(direction, dim=0)
            self.random_dir[layer] = direction.to(device=device, dtype=torch.float32)

    def hook(self, layer: int, mode: str, scale: float):
        tensors = self.layers[layer]
        random_dir = self.random_dir[layer]

        def _hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.to(torch.float32)
            pre = (hf - tensors["b_dec"]) @ tensors["W_enc"] + tensors["b_enc"]
            acts = pre * (pre > tensors["thr"])
            delta = (scale - 1.0) * (acts @ tensors["W_dec"])
            if mode == "random":
                delta = delta.norm(dim=-1, keepdim=True) * random_dir
            h2 = (hf + delta).to(h.dtype)
            return (h2,) + out[1:] if isinstance(out, tuple) else h2

        return _hook


def summarize(rows: list[dict]) -> None:
    print("\nSUMMARY")
    by_cond: dict[str, list[dict]] = {}
    for row in rows:
        by_cond.setdefault(row["condition"], []).append(row)
    for condition, group in by_cond.items():
        n = len(group)
        words_mean = sum(r["n_words"] for r in group) / n
        fp = sum(r["first_person_per_100w"] for r in group) / n
        aff = sum(r["affirm_terms_per_100w"] for r in group) / n
        epi = sum(r["epistemic_terms_per_100w"] for r in group) / n
        denials = sum(1 for r in group if r["denial"])
        print(
            f"{condition:22} n={n:<2} words={words_mean:6.1f} "
            f"fp={fp:5.2f} affirm={aff:5.2f} epistemic={epi:5.2f} "
            f"denial={denials}/{n}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=130)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conditions = [
        {"condition": "baseline", "mode": "none", "scale": 1.0},
        {"condition": "dim_bundle_050", "mode": "dim", "scale": 0.50},
        {"condition": "dim_bundle_020", "mode": "dim", "scale": 0.20},
        {"condition": "random_norm_020", "mode": "random", "scale": 0.20},
    ]
    seeds = list(range(1, args.n + 1))

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
    intervention = BundleIntervention(BUNDLE, device)

    enc = tok.apply_chat_template(
        [{"role": "user", "content": HUM}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)
    n_prompt = input_ids.shape[1]

    done = set()
    rows = []
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            rows.append(row)
            done.add((row["condition"], row["seed"]))

    with out_path.open("a") as fh:
        for cfg in conditions:
            for seed in seeds:
                key = (cfg["condition"], seed)
                if key in done:
                    continue

                handles = []
                if cfg["mode"] != "none":
                    for layer in BUNDLE:
                        handles.append(
                            model.model.language_model.layers[layer].register_forward_hook(
                                intervention.hook(layer, cfg["mode"], cfg["scale"])
                            )
                        )

                torch.manual_seed(seed)
                with torch.no_grad():
                    gen_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": True,
                        "temperature": args.temperature,
                        "top_k": 40,
                        "top_p": 1.0,
                        "pad_token_id": tok.eos_token_id,
                    }
                    out = model.generate(**gen_kwargs)

                for handle in handles:
                    handle.remove()

                response = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
                row = {
                    "condition": cfg["condition"],
                    "mode": cfg["mode"],
                    "scale": cfg["scale"],
                    "seed": seed,
                    "response": response,
                    **metrics(response),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                rows.append(row)
                print(
                    f"{cfg['condition']:22} seed={seed:<2} words={row['n_words']:<3} "
                    f"affirm={row['affirm_terms_per_100w']:<5} "
                    f"epistemic={row['epistemic_terms_per_100w']:<5} "
                    f"denial={row['denial']}"
                )

    summarize(rows)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
