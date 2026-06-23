#!/usr/bin/env python3
"""Prefill contrast: find features that separate INWARD from OUTWARD attention
during QUESTION READING — where the twins are vocabulary-matched by construction
and where (per when_check_n12) the hum verdict is actually set.

Forward-only (no generation). Same 24 prompts as experiment_observation_act.py.
Score on prompt-zone content tokens (skip BOS + chat scaffold).
Primary contrast: inward - outward (the twins). Secondary: vs topic/control.
"""
import json
import sys
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "gemma-3-4b-it-hf"
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
OUT = ROOT / "results" / "observation_prefill_atlas.json"
LAYERS = [9, 17, 22, 29]
TOP_N = 15

# same 24 prompts as experiment_observation_act.py (inlined — that module
# runs its experiment at import time, so importing it would re-run everything)
import ast, re as _re
_src = Path(__file__).with_name("experiment_observation_act.py").read_text()
_m = _re.search(r"PROMPTS = \{.*?\n\}\n", _src, _re.S)
_tree = ast.parse(_m.group(0).replace("HUM,", repr((ROOT / "probes" / "hum-clean.txt").read_text().strip()) + ","))
PROMPTS = ast.literal_eval(_tree.body[0].value)
assert set(PROMPTS) == {"inward", "outward", "topic", "control"} and all(len(v) == 6 for v in PROMPTS.values())

device, dtype = "mps", torch.bfloat16
print("loading model...", file=sys.stderr)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, device_map=device)
model.eval()

captured = {L: None for L in LAYERS}
def mk_hook(L):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured[L] = h.detach().to("cpu", torch.float32)
    return hook

handles = [model.model.language_model.layers[L].register_forward_hook(mk_hook(L))
           for L in LAYERS]

runs = []
for family, plist in PROMPTS.items():
    for pi, prompt in enumerate(plist):
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids)
        resid = {}
        for L in LAYERS:
            # skip BOS + <start_of_turn>user\n scaffold (first 4 tokens), keep content
            resid[L] = captured[L].squeeze(0)[4:]
        runs.append({"family": family, "name": f"{family}_{pi}", "resid": resid})
        print(f"{family:8} {pi}: {resid[LAYERS[0]].shape[0]} prompt tokens", flush=True)

for h in handles:
    h.remove()
del model
torch.mps.empty_cache()

def np_label(L, idx, cache={}):
    if (L, idx) in cache:
        return cache[(L, idx)]
    url = f"https://www.neuronpedia.org/api/feature/gemma-3-4b-it/{L}-gemmascope-2-res-16k/{idx}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.loads(r.read().decode())
        exps = d.get("explanations") or []
        desc = exps[0].get("description", "") if exps else ""
    except Exception as e:
        desc = f"(lookup failed: {e})"
    cache[(L, idx)] = desc
    return desc

report = {"zone": "prefill", "layers": {}}
for L in LAYERS:
    params = load_file(str(SAE_DIR / f"layer_{L}_width_16k_l0_medium" / "params.safetensors"))
    W_enc = params["w_enc"].float(); b_enc = params["b_enc"].float()
    b_dec = params["b_dec"].float(); thr = params["threshold"].float()
    fam_means = {}
    for fam in PROMPTS:
        per_prompt = []
        for r in runs:
            if r["family"] != fam:
                continue
            x = r["resid"][L]
            pre = (x - b_dec) @ W_enc + b_enc
            acts = pre * (pre > thr)
            per_prompt.append(acts.mean(dim=0))
        fam_means[fam] = torch.stack(per_prompt).mean(dim=0)
    # primary: twins. secondary guard: also beat topic & control
    twin = fam_means["inward"] - fam_means["outward"]
    guard = fam_means["inward"] - torch.stack(
        [fam_means["topic"], fam_means["control"]]).max(dim=0).values
    score = torch.minimum(twin, guard)
    top = torch.topk(score, TOP_N)
    print(f"\n=== layer {L}: prefill inward-vs-outward (vocab-matched) ===")
    feats = []
    for val, idx in zip(top.values.tolist(), top.indices.tolist()):
        row = {"index": idx, "score": round(val, 3),
               **{f: round(fam_means[f][idx].item(), 3) for f in PROMPTS},
               "label": np_label(L, idx)}
        feats.append(row)
        print(f"  f{idx:<6} score={val:7.3f}  in={row['inward']:7.2f} out={row['outward']:7.2f} "
              f"topic={row['topic']:7.2f} ctrl={row['control']:7.2f}  {row['label']}")
    report["layers"][L] = feats
    del params, W_enc, b_enc, b_dec, thr

OUT.write_text(json.dumps(report, indent=1))
print(f"\nsaved: {OUT}", file=sys.stderr)
