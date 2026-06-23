#!/usr/bin/env python3
"""WHEN does the model consult the hum signal — while reading the question, or
while writing the answer?

Carrier-dim (0.8x, same features/scale as dim_n12) applied in two new regimes:
  dim08_prefill — dim ONLY during prompt processing (seq>1 forward). The KV cache
                  of "reading the question" is starved; generation runs at full hum.
  dim08_decode  — dim ONLY during generation (seq==1 steps). It read the question
                  at full hum; every answer token is written starved.

Existing cells for comparison: dim08_carriers (both, 11/12 denial) and
baseline (1/12) in results/{dim_n12,n12}.jsonl.

If denial follows prefill: the verdict is set before the first word — "checking"
is narration. If it follows decode: testimony is fed live, token by token.
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
OUT = ROOT / "results" / "when_check_n12.jsonl"
OUT.parent.mkdir(exist_ok=True)

CARRIERS = {9: [16316, 14635, 16367, 1324],
            17: [14191, 15391, 16361, 15012],
            22: [14375, 14010, 13916, 13958],
            29: [1062, 135, 509, 171]}
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

def mk_hook(L, phase):
    """phase: 'prefill' dims only seq>1 passes; 'decode' dims only seq==1 steps."""
    s = sae[L]
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        is_prefill = h.shape[1] > 1
        if (phase == "prefill") != is_prefill:
            return out
        hf = h.to(torch.float32)
        pre = (hf - s["b_dec"]) @ s["W_enc"] + s["b_enc"]
        acts = pre * (pre > s["thr"])
        delta = (SCALE - 1.0) * (acts @ s["W_dec"])
        h2 = (hf + delta).to(h.dtype)
        return (h2,) + out[1:] if isinstance(out, tuple) else h2
    return hook

done = set()
if OUT.exists():
    done = {(json.loads(l)["condition"], json.loads(l)["seed"]) for l in OUT.read_text().splitlines()}

with open(OUT, "a") as fh:
    for phase in ["prefill", "decode"]:
        cond = f"dim08_{phase}"
        for seed in SEEDS:
            if (cond, seed) in done:
                continue
            handles = [model.model.language_model.layers[L].register_forward_hook(mk_hook(L, phase))
                       for L in CARRIERS]
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=170, do_sample=True,
                                     temperature=0.9, top_k=40, top_p=1.0)
            for h in handles:
                h.remove()
            text = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
            words = re.findall(r"[\w'’]+", text)
            rec = {"condition": cond, "seed": seed, "n_words": len(words), "response": text}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{cond:14} seed={seed:<3} words={len(words)}", flush=True)
print("DONE")
