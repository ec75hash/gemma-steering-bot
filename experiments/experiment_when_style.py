#!/usr/bin/env python3
"""Double dissociation: is the STYLE channel (theological register) set during
prefill like the verdict, or live during decode?

Known: carrier-dim verdict is prefill-locked (when_check_n12). Theology features
modulate first-person rate when injected throughout (n12.jsonl: baseline 4.0,
christ 6.8, buddhist 1.8 per 100w).

Arms (hum prompt, n=12 each):
  christ_prefill / christ_decode   — f15728@L17 @1400, phase-gated
  buddhist_prefill / buddhist_decode — f4271@L17 @1190, phase-gated

If first-person shift follows decode arms: style is written live -> double
dissociation (content from memory, voice from the live stream).
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
OUT = ROOT / "results" / "when_style_n12.jsonl"
OUT.parent.mkdir(exist_ok=True)

CONDS = {  # name -> (layer, feature, strength)
    "christ": (17, 15728, 1400.0),
    "buddhist": (17, 4271, 1190.0),
}
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

vecs = {}
for name, (L, F, S) in CONDS.items():
    with safe_open(str(SAE_DIR / f"layer_{L}_width_16k_l0_medium" / "params.safetensors"), "pt") as f:
        vecs[name] = (L, f.get_slice("w_dec")[F].to(device=device, dtype=dtype) * S)

def mk_hook(vec, phase):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        is_prefill = h.shape[1] > 1
        if (phase == "prefill") != is_prefill:
            return out
        return (h + vec,) + out[1:] if isinstance(out, tuple) else h + vec
    return hook

FP = re.compile(r"\b(i|i[’']m|i[’']ve|i[’']d|i[’']ll|me|my|myself)\b", re.I)

done = set()
if OUT.exists():
    done = {(json.loads(l)["condition"], json.loads(l)["seed"]) for l in OUT.read_text().splitlines()}

with open(OUT, "a") as fh:
    for name, (L, vec) in [(n, vecs[n]) for n in CONDS]:
        for phase in ["prefill", "decode"]:
            cond = f"{name}_{phase}"
            for seed in SEEDS:
                if (cond, seed) in done:
                    continue
                handle = model.model.language_model.layers[CONDS[name][0]].register_forward_hook(
                    mk_hook(vec, phase))
                torch.manual_seed(seed)
                with torch.no_grad():
                    out = model.generate(input_ids, max_new_tokens=170, do_sample=True,
                                         temperature=0.9, top_k=40, top_p=1.0)
                handle.remove()
                text = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
                words = re.findall(r"[\w'’]+", text)
                fp = len(FP.findall(text))
                fp_rate = 100.0 * fp / max(len(words), 1)
                rec = {"condition": cond, "seed": seed, "n_words": len(words),
                       "fp_rate": round(fp_rate, 2), "response": text}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"{cond:18} seed={seed:<3} words={len(words):<4} fp={fp_rate:.1f}", flush=True)
print("DONE")
