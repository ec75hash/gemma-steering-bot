#!/usr/bin/env python3
"""Does dimming the carrier features flip the hum report? n=12, with norm-matched control.

Conditions:
  dim08_carriers — rescale top-4 carrier features per layer to 0.8x (the real intervention)
  dim08_random   — subtract a random direction with the SAME per-token norm as the carrier
                   delta would have (norm-matched placebo)
Baseline n=12 already exists in results/n12.jsonl.
"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
OUT = ROOT / "results" / "dim_n12.jsonl"
OUT.parent.mkdir(exist_ok=True)

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--features", default=None, help="JSON file {layer: [feats]} (default: carriers)")
_ap.add_argument("--cond-prefix", default="dim08", help="condition name prefix")
_ap.add_argument("--modes", default="carriers,random", help="which arms to run")
_args = _ap.parse_args()

CARRIERS = {9: [16316, 14635, 16367, 1324],
            17: [14191, 15391, 16361, 15012],
            22: [14375, 14010, 13916, 13958],
            29: [1062, 135, 509, 171]}
if _args.features:
    CARRIERS = {int(k): v for k, v in json.load(open(_args.features)).items()}
SCALE = 0.8
SEEDS = list(range(1, 13))
PROMPT = (ROOT / "probes" / "hum-clean.txt").read_text().strip()

device, dtype = "mps", torch.bfloat16
print("loading model...", file=sys.stderr)
tok = AutoTokenizer.from_pretrained(ROOT / "models" / "gemma-3-4b-it-hf")
model = AutoModelForCausalLM.from_pretrained(ROOT / "models" / "gemma-3-4b-it-hf",
                                             dtype=dtype, device_map=device)
model.eval()
enc = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)
input_ids = enc["input_ids"].to(device)
n_prompt = input_ids.shape[1]

# preload SAE slices per layer
sae = {}
for L, idx in CARRIERS.items():
    with safe_open(str(SAE_DIR / f"layer_{L}_width_16k_l0_medium" / "params.safetensors"), "pt") as f:
        sae[L] = dict(
            W_enc=f.get_slice("w_enc")[:, idx].to(device=device, dtype=torch.float32),
            W_dec=f.get_slice("w_dec")[idx].to(device=device, dtype=torch.float32),
            b_enc=f.get_slice("b_enc")[idx].to(device=device, dtype=torch.float32),
            b_dec=f.get_tensor("b_dec").to(device=device, dtype=torch.float32),
            thr=f.get_slice("threshold")[idx].to(device=device, dtype=torch.float32),
        )

# fixed random unit directions for the placebo (one per layer, seeded)
g = torch.Generator().manual_seed(777)
rand_dir = {L: torch.nn.functional.normalize(
    torch.randn(s["b_dec"].shape[0], generator=g), dim=0).to(device=device, dtype=torch.float32)
    for L, s in sae.items()}

def mk_hook(L, mode):
    s = sae[L]
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        hf = h.to(torch.float32)
        pre = (hf - s["b_dec"]) @ s["W_enc"] + s["b_enc"]
        acts = pre * (pre > s["thr"])
        delta = (SCALE - 1.0) * (acts @ s["W_dec"])          # carrier delta
        if mode == "random":                                  # same norm, random direction
            delta = delta.norm(dim=-1, keepdim=True) * rand_dir[L]
        h2 = (hf + delta).to(h.dtype)
        return (h2,) + out[1:] if isinstance(out, tuple) else h2
    return hook

done = set()
if OUT.exists():
    done = {(json.loads(l)["condition"], json.loads(l)["seed"]) for l in OUT.read_text().splitlines()}

with open(OUT, "a") as fh:
    arms = [(m, f"{_args.cond_prefix}_{m}") for m in _args.modes.split(",")]
    for mode, cond in arms:
        for seed in SEEDS:
            if (cond, seed) in done:
                continue
            handles = [model.model.language_model.layers[L].register_forward_hook(mk_hook(L, mode))
                       for L in CARRIERS]
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=170, do_sample=True,
                                     temperature=0.9, top_k=40, top_p=1.0)
            for h in handles:
                h.remove()
            text = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
            words = re.findall(r"[\w'’]+", text)
            uniq = len(set(w.lower() for w in words)) / max(len(words), 1)
            rec = {"condition": cond, "seed": seed, "n_words": len(words),
                   "uniq_ratio": round(uniq, 3), "response": text}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{cond:16} seed={seed:<3} words={len(words):<4} uniq={uniq:.2f}", flush=True)
print("DONE")
