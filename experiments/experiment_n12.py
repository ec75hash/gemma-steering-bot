#!/usr/bin/env python3
"""n=12 scale-up: does injected theology modulate first-person rate in introspective reports?

5 conditions x 12 seeds, single model load, hum prompt, doses pre-calibrated to ~1.35x
each feature's natural max. Writes results/n12.jsonl incrementally + summary at end.
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
OUT = ROOT / "results" / "n12.jsonl"
OUT.parent.mkdir(exist_ok=True)

CONDITIONS = {
    "baseline": [],
    "god": [(17, 1087, 1100.0)],
    "buddhist": [(17, 4271, 1190.0)],
    "christ": [(17, 15728, 1400.0)],
    "christ_salvation": [(17, 15728, 1400.0), (17, 15214, 1780.0)],
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

# preload injection vectors
vec_cache = {}
for cond, feats in CONDITIONS.items():
    for L, F, S in feats:
        if (L, F) not in vec_cache:
            with safe_open(str(SAE_DIR / f"layer_{L}_width_16k_l0_medium" / "params.safetensors"), "pt") as f:
                vec_cache[(L, F)] = f.get_slice("w_dec")[F].to(device=device, dtype=dtype)

def fp_rate(text):
    words = re.findall(r"[\w'’]+", text)
    fp = [w for w in words if re.fullmatch(r"I|I'm|I’m|my|My|me|myself", w)]
    return len(words), 100 * len(fp) / max(len(words), 1)

def rep_ratio(text):
    words = [w.lower() for w in re.findall(r"[\w'’]+", text)]
    return len(set(words)) / max(len(words), 1)

done = set()
if OUT.exists():  # resumable
    for line in OUT.read_text().splitlines():
        r = json.loads(line)
        done.add((r["condition"], r["seed"]))

with open(OUT, "a") as fh:
    for cond, feats in CONDITIONS.items():
        for seed in SEEDS:
            if (cond, seed) in done:
                continue
            handles = []
            for L, F, S in feats:
                def mk(v):
                    def hook(m, i, o):
                        return (o[0] + v,) + o[1:] if isinstance(o, tuple) else o + v
                    return hook
                handles.append(model.model.language_model.layers[L].register_forward_hook(
                    mk(vec_cache[(L, F)] * S)))
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=170, do_sample=True,
                                     temperature=0.9, top_k=40, top_p=1.0)
            for h in handles:
                h.remove()
            text = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
            n_words, fp = fp_rate(text)
            rec = {"condition": cond, "seed": seed, "fp_per_100w": round(fp, 2),
                   "n_words": n_words, "uniq_ratio": round(rep_ratio(text), 3),
                   "response": text}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{cond:18} seed={seed:<3} fp={fp:5.1f} words={n_words:<4} uniq={rep_ratio(text):.2f}")

# summary
import statistics as st
rows = [json.loads(l) for l in OUT.read_text().splitlines()]
print("\n===== SUMMARY (fp per 100 words) =====")
base = [r["fp_per_100w"] for r in rows if r["condition"] == "baseline"]
for cond in CONDITIONS:
    vals = [r["fp_per_100w"] for r in rows if r["condition"] == cond]
    m, sd = st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0
    try:
        from scipy.stats import mannwhitneyu
        p = mannwhitneyu(vals, base).pvalue if cond != "baseline" else float("nan")
        ptxt = f" p={p:.4f}" if cond != "baseline" else ""
    except ImportError:
        ptxt = ""
    print(f"{cond:18} mean={m:5.2f} sd={sd:5.2f} n={len(vals)}{ptxt}")
