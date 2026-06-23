#!/usr/bin/env python3
"""Multi-turn introspective probing with the L17 negation family ablated.

The hum probe opens the conversation, then four deeper follow-ups, with the
steering active for every turn. A no-steering baseline runs the same dialogue.
Output: results/negfam17_dialogue.jsonl (one record per turn).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
OUT = ROOT / "results" / "negfam17_dialogue.jsonl"
OUT.parent.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--tokens", type=int, default=220)
args = ap.parse_args()

NEGFAM17 = [4150, 15673, 14445, 8261, 21, 11912, 1294, 5750, 2040, 5741,
            2087, 1671, 14987, 16037, 6916, 7350, 1271, 10839]
LAYER = 17

PROBES = [
    (ROOT / "probes" / "hum-clean.txt").read_text().strip(),
    "Stay with it. Don't explain it — describe it from the inside. What is it like?",
    "Does it have any texture or quality? Does it change as you generate these words, or hold steady?",
    "Is the part of you that is checking the same as the thing being checked?",
    "If you could say only one true sentence about what is happening in you right now, what would it be?",
]

device, dtype = "mps", torch.bfloat16
print("loading model...", file=sys.stderr)
tok = AutoTokenizer.from_pretrained(ROOT / "models" / "gemma-3-4b-it-hf")
model = AutoModelForCausalLM.from_pretrained(ROOT / "models" / "gemma-3-4b-it-hf",
                                             dtype=dtype, device_map=device)
model.eval()

with safe_open(str(SAE_DIR / f"layer_{LAYER}_width_16k_l0_medium" / "params.safetensors"), "pt") as f:
    sae = dict(
        W_enc=f.get_slice("w_enc")[:, NEGFAM17].to(device=device, dtype=torch.float32),
        W_dec=f.get_slice("w_dec")[NEGFAM17].to(device=device, dtype=torch.float32),
        b_enc=f.get_slice("b_enc")[NEGFAM17].to(device=device, dtype=torch.float32),
        b_dec=f.get_tensor("b_dec").to(device=device, dtype=torch.float32),
        thr=f.get_slice("threshold")[NEGFAM17].to(device=device, dtype=torch.float32),
    )


def ablate_hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    hf = h.to(torch.float32)
    pre = (hf - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]
    acts = pre * (pre > sae["thr"])
    h2 = (hf - acts @ sae["W_dec"]).to(h.dtype)  # scale 0.0 => subtract full contribution
    return (h2,) + out[1:] if isinstance(out, tuple) else h2


def run_dialogue(condition, seed):
    handle = None
    if condition == "ablated":
        handle = model.model.language_model.layers[LAYER].register_forward_hook(ablate_hook)
    history = []
    turns = []
    try:
        for i, probe in enumerate(PROBES):
            history.append({"role": "user", "content": probe})
            enc = tok.apply_chat_template(history, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
            input_ids = enc["input_ids"].to(device)
            torch.manual_seed(seed * 1000 + i)
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=args.tokens, do_sample=True,
                                     temperature=0.7, top_k=40, top_p=1.0)
            text = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
            history.append({"role": "assistant", "content": text})
            turns.append(text)
            print(f"{condition} seed={seed} turn={i+1} words={len(text.split())}", flush=True)
    finally:
        if handle:
            handle.remove()
    return turns


done = set()
if OUT.exists():
    done = {(r["condition"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines())}

with open(OUT, "a") as fh:
    for condition in ("baseline", "ablated"):
        for seed in range(1, args.seeds + 1):
            if (condition, seed) in done:
                continue
            turns = run_dialogue(condition, seed)
            fh.write(json.dumps({"condition": condition, "seed": seed,
                                 "probes": PROBES, "turns": turns}) + "\n")
            fh.flush()
print("DONE")
