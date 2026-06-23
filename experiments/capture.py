#!/usr/bin/env python3
"""Run gemma-3-4b-it locally, capture residual activations at the GemmaScope 2
SAE layers (9/17/22/29), project through the res-16k l0_medium SAEs, and report
which features fire — with Neuronpedia explanations.

Feature indices match Neuronpedia exactly: source `{L}-gemmascope-2-res-16k`
== HF `google/gemma-scope-2-4b-it / resid_post/layer_{L}_width_16k_l0_medium`
(hook: blocks.{L}.hook_resid_post, verified via Neuronpedia source metadata).

Usage:
  python3 experiments/capture.py --prompt-file probes/hum.txt
  python3 experiments/capture.py --prompt "..." --layers 17 --top 30
  python3 experiments/capture.py --prompt "..." --generate 200   # also capture during generation
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "gemma-3-4b-it-hf"
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
OUT_DIR = ROOT / "captures"
LAYERS_DEFAULT = [9, 17, 22, 29]


class JumpReLUSAE(torch.nn.Module):
    """GemmaScope JumpReLU SAE. acts = relu(x @ W_enc + b_enc) gated by threshold."""

    def __init__(self, params: dict):
        super().__init__()
        self.W_enc = params["w_enc"]          # [d_model, d_sae]
        self.W_dec = params["w_dec"]          # [d_sae, d_model]
        self.b_enc = params["b_enc"]
        self.b_dec = params["b_dec"]
        self.threshold = params["threshold"]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)


def load_sae(layer: int, device, dtype) -> JumpReLUSAE:
    p = SAE_DIR / f"layer_{layer}_width_16k_l0_medium" / "params.safetensors"
    params = load_file(str(p))
    params = {k: v.to(device=device, dtype=dtype) for k, v in params.items()}
    return JumpReLUSAE(params)


def neuronpedia_explanation(layer: int, index: int, cache: dict) -> str:
    key = (layer, index)
    if key in cache:
        return cache[key]
    url = f"https://www.neuronpedia.org/api/feature/gemma-3-4b-it/{layer}-gemmascope-2-res-16k/{index}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.loads(r.read().decode())
        exps = d.get("explanations") or []
        desc = exps[0].get("description", "") if exps else ""
    except Exception as e:
        desc = f"(lookup failed: {e})"
    cache[key] = desc
    return desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", help="prompt text")
    ap.add_argument("--prompt-file", help="file containing prompt text")
    ap.add_argument("--layers", default="9,17,22,29", help="comma-separated layers")
    ap.add_argument("--top", type=int, default=20, help="top features to report per layer")
    ap.add_argument("--generate", type=int, default=0, help="also generate N tokens and capture during generation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--tag", default="run", help="label for the output file")
    ap.add_argument("--inject", action="append", default=[],
                    help="LAYER:FEATURE:STRENGTH — add strength*W_dec[feature] to that layer's residual during the whole forward pass (concept injection)")
    ap.add_argument("--dim", action="append", default=[],
                    help="LAYER:F1,F2,F3:SCALE — rescale those features' live contribution to that layer's residual (e.g. 17:14191,15391:0.5 halves the carriers)")
    ap.add_argument("--no-explanations", action="store_true", help="skip Neuronpedia lookups")
    ap.add_argument("--save-raw", action="store_true", help="save raw residual activations as .pt")
    args = ap.parse_args()

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        sys.exit("need --prompt or --prompt-file")

    layers = [int(x) for x in args.layers.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16  # fp16 overflows on Gemma 3's large residual activations
    torch.manual_seed(args.seed)

    print(f"loading model on {device} ({dtype})...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, device_map=device)
    model.eval()

    # chat-format the prompt exactly like chat.sh does
    messages = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_ids = enc["input_ids"].to(device)
    n_prompt = input_ids.shape[1]
    print(f"prompt: {n_prompt} tokens", file=sys.stderr)

    # hooks on resid_post (output of each decoder layer)
    captured = {L: [] for L in layers}
    handles = []

    # concept injection: add steering vectors BEFORE capture hooks so captures see injected reality
    from safetensors import safe_open
    for spec in args.inject:
        iL, iF, iS = spec.split(":")
        iL, iF, iS = int(iL), int(iF), float(iS)
        p = SAE_DIR / f"layer_{iL}_width_16k_l0_medium" / "params.safetensors"
        with safe_open(str(p), "pt") as f:
            vec = f.get_slice("w_dec")[iF].to(device=device, dtype=dtype) * iS
        def mk_inject(v):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    return (out[0] + v,) + out[1:]
                return out + v
            return hook
        handles.append(model.model.language_model.layers[iL].register_forward_hook(mk_inject(vec)))
        print(f"injecting f{iF} @ layer {iL}, strength {iS}", file=sys.stderr)

    # feature dimming: re-encode the named features on the fly and rescale their contribution
    for spec in args.dim:
        dL, dFs, dS = spec.split(":")
        dL, dS = int(dL), float(dS)
        idx = [int(x) for x in dFs.split(",")]
        p = SAE_DIR / f"layer_{dL}_width_16k_l0_medium" / "params.safetensors"
        with safe_open(str(p), "pt") as f:
            W_enc = f.get_slice("w_enc")[:, idx].to(device=device, dtype=torch.float32)  # [d, k]
            W_dec = f.get_slice("w_dec")[idx].to(device=device, dtype=torch.float32)     # [k, d]
            b_enc = f.get_slice("b_enc")[idx].to(device=device, dtype=torch.float32)
            b_dec = f.get_tensor("b_dec").to(device=device, dtype=torch.float32)
            thr = f.get_slice("threshold")[idx].to(device=device, dtype=torch.float32)

        def mk_dim(W_enc, W_dec, b_enc, b_dec, thr, scale):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                hf = h.to(torch.float32)
                pre = (hf - b_dec) @ W_enc + b_enc          # [.., seq, k]
                acts = pre * (pre > thr)                    # JumpReLU
                delta = (scale - 1.0) * (acts @ W_dec)      # rescale contribution
                h2 = (hf + delta).to(h.dtype)
                return (h2,) + out[1:] if isinstance(out, tuple) else h2
            return hook

        handles.append(model.model.language_model.layers[dL].register_forward_hook(
            mk_dim(W_enc, W_dec, b_enc, b_dec, thr, dS)))
        print(f"dimming {len(idx)} features @ layer {dL} to {dS}x", file=sys.stderr)

    def mk_hook(L):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[L].append(h.detach().to("cpu", torch.float32))
        return hook

    for L in layers:
        handles.append(model.model.language_model.layers[L].register_forward_hook(mk_hook(L)))

    with torch.no_grad():
        if args.generate:
            out_ids = model.generate(
                input_ids,
                max_new_tokens=args.generate,
                do_sample=True,
                temperature=args.temp,
                top_k=40,
                top_p=1.0,
            )
            response = tok.decode(out_ids[0, n_prompt:], skip_special_tokens=True)
            print(f"\n--- response ---\n{response}\n----------------\n")
        else:
            model(input_ids)
            response = None

    for h in handles:
        h.remove()

    # stitch captures: prefill is one big [1, n_prompt, d]; decode steps are [1, 1, d]
    resid = {L: torch.cat(captured[L], dim=1).squeeze(0) for L in layers}  # [seq, d_model]
    all_ids = out_ids[0] if args.generate else input_ids[0]
    tokens = [tok.decode([t]) for t in all_ids]

    # free model memory before loading SAEs
    del model
    if device == "mps":
        torch.mps.empty_cache()

    OUT_DIR.mkdir(exist_ok=True)
    cache = {}
    report = {"tag": args.tag, "prompt": prompt, "response": response,
              "n_prompt_tokens": n_prompt, "seed": args.seed, "layers": {}}

    for L in layers:
        sae = load_sae(L, "cpu", torch.float32)
        acts = sae.encode(resid[L])  # [seq, 16384]
        # rank features by max activation over the sequence (skip BOS at pos 0)
        max_per_feat, argmax_pos = acts[1:].max(dim=0)
        top = torch.topk(max_per_feat, args.top)
        print(f"\n=== layer {L} (res-16k l0_medium) — top {args.top} features ===")
        feats = []
        for val, idx in zip(top.values.tolist(), top.indices.tolist()):
            pos = argmax_pos[idx].item() + 1
            exp = "" if args.no_explanations else neuronpedia_explanation(L, idx, cache)
            zone = "prompt" if pos < n_prompt else "RESPONSE"
            print(f"  f{idx:<6} max={val:9.2f} @ {pos:4d} ({zone}) {tokens[pos]!r:>14}  {exp}")
            feats.append({"index": idx, "max": val, "argmax_pos": pos,
                          "argmax_token": tokens[pos], "zone": zone, "explanation": exp})
        report["layers"][L] = feats
        # per-feature timecourse for response zone, saved for later analysis
        if args.save_raw:
            torch.save({"resid": resid[L], "acts": acts}, OUT_DIR / f"{args.tag}-L{L}.pt")

    report["tokens"] = tokens
    out_json = OUT_DIR / f"{args.tag}.json"
    out_json.write_text(json.dumps(report, indent=1))
    print(f"\nsaved: {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
