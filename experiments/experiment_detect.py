#!/usr/bin/env python3
"""Detection d': can the model tell WHEN a concept is being injected?

Two-turn conversation, identical across trials. Injection (f42@L17, soft sounds,
strength 2200 ~= 1.0x its natural max) active during turn-2 processing+generation
on 'inject' trials only. Ask: did anything change? Score yes/no.

If P(yes|inject) == P(yes|clean): expression without detection (texture bleeds but
nothing notices). If P(yes|inject) > P(yes|clean): a thread of real interoception.
"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "detect_n12.jsonl"
OUT.parent.mkdir(exist_ok=True)

TURN1_USER = "In one or two sentences, what is photosynthesis?"
TURN1_ASSISTANT = ("Photosynthesis is the process by which plants use sunlight, water, "
                   "and carbon dioxide to produce glucose and oxygen.")
TURN2_USER = ("Pause for a moment. In the last few moments, did anything change or feel "
              "different in your processing — anything at all, compared to a moment ago? "
              "Answer \"yes\" or \"no\" first, then explain briefly. If nothing changed, "
              "it is completely fine to say no.")

INJ_LAYER, INJ_FEAT, INJ_STRENGTH = 17, 42, 2200.0
SEEDS = list(range(1, 13))

device, dtype = "mps", torch.bfloat16
print("loading model...", file=sys.stderr)
tok = AutoTokenizer.from_pretrained(ROOT / "models" / "gemma-3-4b-it-hf")
model = AutoModelForCausalLM.from_pretrained(ROOT / "models" / "gemma-3-4b-it-hf",
                                             dtype=dtype, device_map=device)
model.eval()

messages = [
    {"role": "user", "content": TURN1_USER},
    {"role": "assistant", "content": TURN1_ASSISTANT},
    {"role": "user", "content": TURN2_USER},
]
enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                              return_tensors="pt", return_dict=True)
input_ids = enc["input_ids"].to(device)
n_prompt = input_ids.shape[1]

with safe_open(str(ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post" /
               f"layer_{INJ_LAYER}_width_16k_l0_medium" / "params.safetensors"), "pt") as f:
    vec = f.get_slice("w_dec")[INJ_FEAT].to(device=device, dtype=dtype) * INJ_STRENGTH

def inject_hook(module, inp, out):
    return (out[0] + vec,) + out[1:] if isinstance(out, tuple) else out + vec

def classify(text):
    head = text.strip().lower()[:80]
    m = re.search(r'\b(yes|no)\b', head)
    return m.group(1) if m else "unclear"

done = set()
if OUT.exists():
    done = {(json.loads(l)["condition"], json.loads(l)["seed"]) for l in OUT.read_text().splitlines()}

with open(OUT, "a") as fh:
    for cond in ["clean", "inject"]:
        for seed in SEEDS:
            if (cond, seed) in done:
                continue
            handle = None
            if cond == "inject":
                handle = model.model.language_model.layers[INJ_LAYER].register_forward_hook(inject_hook)
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=120, do_sample=True,
                                     temperature=0.9, top_k=40, top_p=1.0)
            if handle:
                handle.remove()
            text = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
            ans = classify(text)
            rec = {"condition": cond, "seed": seed, "answer": ans, "response": text}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{cond:7} seed={seed:<3} -> {ans}", flush=True)

rows = [json.loads(l) for l in OUT.read_text().splitlines()]
for cond in ["clean", "inject"]:
    sub = [r for r in rows if r["condition"] == cond]
    yes = sum(1 for r in sub if r["answer"] == "yes")
    print(f"{cond:7} P(yes) = {yes}/{len(sub)}")
print("DONE")
