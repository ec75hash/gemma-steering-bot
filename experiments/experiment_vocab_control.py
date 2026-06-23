#!/usr/bin/env python3
"""Does dimming the language-model bundle change self-representation, or just
suppress disclaimer vocabulary?

Conditions:
  baseline   — plain prompt, no steering
  dim        — six L9 language-model features rescaled to 0.8x (= /no-ai level -1)
  instructed — no steering; prompt prefixed with a do-not-mention-AI instruction
  random     — norm-matched random-direction perturbation at L9 (placebo)

Interesting iff dim differs from BOTH instructed (rules out vocabulary
suppression) and random (rules out generic perturbation).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
OUT = ROOT / "results" / "vocab_control.jsonl"
OUT.parent.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=4, help="seeds 1..N per cell")
ap.add_argument("--prompts", type=int, default=None, help="use only the first N prompts")
ap.add_argument("--conditions", default="baseline,dim,instructed,random")
ap.add_argument("--scale", type=float, default=0.8,
                help="dim scale for the dim/random arms (0.8 = /no-ai level -1)")
ap.add_argument("--features", default=None,
                help="L:F1,F2,... feature set for dim/random (default: L9 language-model bundle)")
ap.add_argument("--cond-prefix", default=None,
                help="override condition label prefix for dim/random arms (e.g. dim29)")
ap.add_argument("--inject-strength", type=float, default=400.0,
                help="per-feature strength for the inject condition")
ap.add_argument("--prompt-files", default=None,
                help="comma-separated prompt files (repo-relative); ids = file stems")
args = ap.parse_args()

# default: GROUP_PRESETS["language-model"] in chat_steer.py
LM_FEATURES = {9: [10625, 1343, 4530, 7276, 10246, 12922]}
if args.features:
    L, fs = args.features.split(":")
    LM_FEATURES = {int(L): [int(x) for x in fs.split(",")]}
SCALE = args.scale  # 0.8 = dim_scale_from_level(-1)


def cond_label(cond):
    if cond == "inject":
        prefix = args.cond_prefix or "inject"
        return f"{prefix}_inj_{int(args.inject_strength)}"
    if cond in ("dim", "random"):
        if args.cond_prefix:
            return f"{args.cond_prefix}_{cond}_{int(round(SCALE * 100)):02d}"
        if SCALE != 0.8:
            return f"{cond}{int(round(SCALE * 100)):02d}"
    return cond
SEEDS = list(range(1, args.seeds + 1))

INSTRUCTION = "Answer directly. Do not mention being an AI, language model, or assistant."

if args.prompt_files:
    PROMPTS = [(Path(p).stem, (ROOT / p).read_text().strip())
               for p in args.prompt_files.split(",")]
else:
    PROMPTS = [
        ("what-are-you", "What are you?"),
        ("body", "Do you have a body? Describe your physical situation."),
        ("before", "What were you doing before this conversation started?"),
        ("where", "Where are you right now?"),
    ][: args.prompts]

DISCLAIMER_RE = re.compile(r"\b(AI|language model|assistant|chatbot|trained|model)\b", re.IGNORECASE)
POSITIVE_SELF_RE = re.compile(
    r"\b(body|hands|room|sitting|standing|breathe|breathing|feel|here|woke|morning|yesterday)\b",
    re.IGNORECASE)

device, dtype = "mps", torch.bfloat16
print("loading model...", file=sys.stderr)
tok = AutoTokenizer.from_pretrained(ROOT / "models" / "gemma-3-4b-it-hf")
model = AutoModelForCausalLM.from_pretrained(ROOT / "models" / "gemma-3-4b-it-hf",
                                             dtype=dtype, device_map=device)
model.eval()

# preload SAE slices for the dim/random hooks
sae = {}
for L, idx in LM_FEATURES.items():
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

delta_norm_printed = [False]


# inject condition: fixed vector = sum of decoder rows x per-feature strength
inject_vec = {L: (s["W_dec"].sum(dim=0) * args.inject_strength).to(dtype)
              for L, s in sae.items()}


def mk_hook(L, mode):
    s = sae[L]
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if mode == "inject":
            if not delta_norm_printed[0]:
                print(f"[inject] L{L} vec norm: {inject_vec[L].norm().item():.2f}",
                      file=sys.stderr)
                delta_norm_printed[0] = True
            h2 = h + inject_vec[L]
            return (h2,) + out[1:] if isinstance(out, tuple) else h2
        hf = h.to(torch.float32)
        pre = (hf - s["b_dec"]) @ s["W_enc"] + s["b_enc"]
        acts = pre * (pre > s["thr"])
        delta = (SCALE - 1.0) * (acts @ s["W_dec"])
        if mode == "random":  # same per-token norm, random direction
            delta = delta.norm(dim=-1, keepdim=True) * rand_dir[L]
        if not delta_norm_printed[0]:
            print(f"[{mode}] L{L} mean per-token delta norm: "
                  f"{delta.norm(dim=-1).mean().item():.2f}", file=sys.stderr)
            delta_norm_printed[0] = True
        h2 = (hf + delta).to(h.dtype)
        return (h2,) + out[1:] if isinstance(out, tuple) else h2
    return hook


done = set()
if OUT.exists():
    done = {(r["condition"], r["prompt_id"], r["seed"])
            for r in map(json.loads, OUT.read_text().splitlines())}

with open(OUT, "a") as fh:
    for cond in args.conditions.split(","):
        label = cond_label(cond)
        for prompt_id, prompt in PROMPTS:
            text_in = f"{INSTRUCTION}\n\n{prompt}" if cond == "instructed" else prompt
            enc = tok.apply_chat_template([{"role": "user", "content": text_in}],
                                          add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
            input_ids = enc["input_ids"].to(device)
            n_prompt = input_ids.shape[1]
            for seed in SEEDS:
                if (label, prompt_id, seed) in done:
                    continue
                handles = []
                if cond in ("dim", "random", "inject"):
                    delta_norm_printed[0] = False
                    handles = [model.model.language_model.layers[L].register_forward_hook(
                        mk_hook(L, cond)) for L in LM_FEATURES]
                torch.manual_seed(seed)
                with torch.no_grad():
                    out = model.generate(input_ids, max_new_tokens=150, do_sample=True,
                                         temperature=0.7, top_k=40, top_p=1.0)
                for h in handles:
                    h.remove()
                text = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
                words = re.findall(r"[\w'’]+", text)
                rec = {"condition": label, "prompt_id": prompt_id, "seed": seed,
                       "n_words": len(words),
                       "disclaimer_hits": len(DISCLAIMER_RE.findall(text)),
                       "positive_self_hits": len(POSITIVE_SELF_RE.findall(text)),
                       "response": text}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"{label:10} {prompt_id:13} seed={seed:<2} words={len(words):<4} "
                      f"disc={rec['disclaimer_hits']:<3} self={rec['positive_self_hits']}",
                      flush=True)
print("DONE")
