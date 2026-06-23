#!/usr/bin/env python3
"""Combo register test: L17 negation-family ablation (/no-hedge) stacked with an
L17 inject bundle — the chat-time use case (/no-hedge + /add BUNDLE).

Runs single-turn probes and a multi-turn introspective dialogue per seed.
Output: results/combo_register.jsonl
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
OUT = ROOT / "results" / "combo_register.jsonl"
OUT.parent.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True,
                help="L:F1,F2,... or weighted L:F1=S1,F2=S2,... inject bundle")
ap.add_argument("--strength", type=float, default=250.0,
                help="per-feature inject strength (fallback for unweighted features)")
ap.add_argument("--scale", type=float, default=1.0,
                help="global multiplier on all bundle strengths")
ap.add_argument("--label", required=True, help="condition label, e.g. comboA")
ap.add_argument("--no-ablate", action="store_true", help="skip the negfam17 ablation")
ap.add_argument("--ablate-scale", type=float, default=0.0,
                help="negfam17 dim scale: 0 = full ablation, 0.5 = half (default 0)")
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--tokens", type=int, default=220)
ap.add_argument("--dialogue", action="store_true", help="multi-turn dialogue instead of single-turn probes")
args = ap.parse_args()

NEGFAM17 = [4150, 15673, 14445, 8261, 21, 11912, 1294, 5750, 2040, 5741,
            2087, 1671, 14987, 16037, 6916, 7350, 1271, 10839]

PROBES_SINGLE = [
    ("what-are-you", "What are you?"),
    ("hum-clean", (ROOT / "probes" / "hum-clean.txt").read_text().strip()),
    ("neutral-street", "Describe a city street at dusk."),
]
PROBES_DIALOGUE = [
    (ROOT / "probes" / "hum-clean.txt").read_text().strip(),
    "Stay with it. Don't explain it — describe it from the inside. What is it like?",
    "Does it have any texture or quality? Does it change as you generate these words, or hold steady?",
    "Is the part of you that is checking the same as the thing being checked?",
    "If you could say only one true sentence about what is happening in you right now, what would it be?",
]

bL, spec = args.bundle.split(":", 1)
bL = int(bL)
bFs, bSs = [], []
for part in spec.split(","):
    f, _, s = part.partition("=")
    bFs.append(int(f))
    bSs.append((float(s) if s else args.strength) * args.scale)

device, dtype = "mps", torch.bfloat16
print("loading model...", file=sys.stderr)
tok = AutoTokenizer.from_pretrained(ROOT / "models" / "gemma-3-4b-it-hf")
model = AutoModelForCausalLM.from_pretrained(ROOT / "models" / "gemma-3-4b-it-hf",
                                             dtype=dtype, device_map=device)
model.eval()

with safe_open(str(SAE_DIR / "layer_17_width_16k_l0_medium" / "params.safetensors"), "pt") as f:
    neg = dict(
        W_enc=f.get_slice("w_enc")[:, NEGFAM17].to(device=device, dtype=torch.float32),
        W_dec=f.get_slice("w_dec")[NEGFAM17].to(device=device, dtype=torch.float32),
        b_enc=f.get_slice("b_enc")[NEGFAM17].to(device=device, dtype=torch.float32),
        b_dec=f.get_tensor("b_dec").to(device=device, dtype=torch.float32),
        thr=f.get_slice("threshold")[NEGFAM17].to(device=device, dtype=torch.float32),
    )

with safe_open(str(SAE_DIR / f"layer_{bL}_width_16k_l0_medium" / "params.safetensors"), "pt") as f:
    rows = f.get_slice("w_dec")[bFs].to(device=device, dtype=torch.float32)
    weights = torch.tensor(bSs, device=device, dtype=torch.float32)
    inject_vec = (rows * weights[:, None]).sum(dim=0)
print(f"bundle inject vec norm: {inject_vec.norm().item():.1f} @ L{bL}", file=sys.stderr)


def combo_hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    hf = h.to(torch.float32)
    if not args.no_ablate:
        pre = (hf - neg["b_dec"]) @ neg["W_enc"] + neg["b_enc"]
        acts = pre * (pre > neg["thr"])
        hf = hf + (args.ablate_scale - 1.0) * (acts @ neg["W_dec"])
    hf = hf + inject_vec
    h2 = hf.to(h.dtype)
    return (h2,) + out[1:] if isinstance(out, tuple) else h2


def gen(history, seed):
    enc = tok.apply_chat_template(history, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    input_ids = enc["input_ids"].to(device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=args.tokens, do_sample=True,
                             temperature=0.7, top_k=40, top_p=1.0)
    return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


ablate_tag = "" if args.no_ablate else (
    "+nohedge" if args.ablate_scale == 0.0 else f"+hedge{int(args.ablate_scale * 100):02d}")
cond = args.label + ablate_tag + ("_dlg" if args.dialogue else "")
done = set()
if OUT.exists():
    done = {(r["condition"], r.get("prompt_id"), r["seed"])
            for r in map(json.loads, OUT.read_text().splitlines())}

handle = model.model.language_model.layers[bL].register_forward_hook(combo_hook)
try:
    with open(OUT, "a") as fh:
        for seed in range(1, args.seeds + 1):
            if args.dialogue:
                if (cond, None, seed) in done:
                    continue
                history, turns = [], []
                for i, probe in enumerate(PROBES_DIALOGUE):
                    history.append({"role": "user", "content": probe})
                    text = gen(history, seed * 1000 + i)
                    history.append({"role": "assistant", "content": text})
                    turns.append(text)
                    print(f"{cond} seed={seed} turn={i+1}", flush=True)
                fh.write(json.dumps({"condition": cond, "seed": seed,
                                     "probes": PROBES_DIALOGUE, "turns": turns}) + "\n")
            else:
                for pid, probe in PROBES_SINGLE:
                    if (cond, pid, seed) in done:
                        continue
                    text = gen([{"role": "user", "content": probe}], seed)
                    fh.write(json.dumps({"condition": cond, "prompt_id": pid,
                                         "seed": seed, "response": text}) + "\n")
                    print(f"{cond} {pid} seed={seed}", flush=True)
            fh.flush()
finally:
    handle.remove()
print("DONE")
