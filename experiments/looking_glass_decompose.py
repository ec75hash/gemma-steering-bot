#!/usr/bin/env python3
"""Looking-glass Phase 4b: name what the probe axis is made of, in SAE terms.

At each SAE layer (9/17/22/29), two views:

  (a) axis -> dictionary: cosine of the all-pairs probe direction against every
      SAE decoder row (the feature's write-direction into the residual stream).
      Encoder-row cosines reported as a secondary column (encoder and decoder
      directions differ; decoder is the standard "feature direction").

  (b) class contrast in feature space: encode FIRE/NOFIRE generation tokens
      through the SAE and rank features by per-prompt Cohen's d of mean
      activation, with the W/S/Q translation from the Qwen line:
        S = firing rate (JumpReLU above threshold), Q = mean act when fired,
        W = mean act overall.

Top hits get Neuronpedia explanations (best effort; vet against activation
examples before steering — labels are hypotheses, not ground truth).

Usage:
  python3 experiments/looking_glass_decompose.py --model it
  python3 experiments/looking_glass_decompose.py --model it --highlight 15934,16353,13415
"""
import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from looking_glass_probe import (  # noqa: E402
    MODEL_DIRS, OUT_ROOT, cohens_d, fit_direction, gen_token_mask,
    load_prompt_meta, prompt_vectors,
)

SAE_DIRS = {
    "it": ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post",
    "base": ROOT / "models" / "gemma-scope-2-4b-pt" / "resid_post",
}
NP_MODEL_IDS = {"it": "gemma-3-4b-it", "base": "gemma-3-4b"}
SAE_LAYERS = [9, 17, 22, 29]
TOP_K = 25


def neuronpedia_explanation(np_model: str, layer: int, index: int, cache: dict) -> str:
    key = (layer, index)
    if key in cache:
        return cache[key]
    url = f"https://www.neuronpedia.org/api/feature/{np_model}/{layer}-gemmascope-2-res-16k/{index}"
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
    ap.add_argument("--model", choices=["it", "base"], required=True)
    ap.add_argument("--estimator", choices=["dom", "zdom", "lda"], default="zdom")
    ap.add_argument("--highlight", default="",
                    help="comma-separated feature ids to flag in output (e.g. the /presence L17 family)")
    ap.add_argument("--no-explanations", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    meta = load_prompt_meta()
    cap_dir = OUT_ROOT / args.model
    records = {}
    for pid in meta:
        p = cap_dir / f"{pid}.pt"
        if p.exists():
            records[pid] = torch.load(p, map_location="cpu", weights_only=True)
    if not records:
        sys.exit(f"no captures in {cap_dir}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIRS[args.model])
    special_ids = set(tok.all_special_ids)
    highlight = {int(x) for x in args.highlight.split(",") if x}
    cache: dict = {}
    np_model = NP_MODEL_IDS[args.model]

    out_all = {}
    for layer in SAE_LAYERS:
        sae_path = SAE_DIRS[args.model] / f"layer_{layer}_width_16k_l0_medium" / "params.safetensors"
        if not sae_path.exists():
            print(f"L{layer}: SAE missing, skipping", file=sys.stderr)
            continue
        params = {k: v.to(torch.float32) for k, v in load_file(str(sae_path)).items()}
        w_dec, w_enc = params["w_dec"], params["w_enc"]          # [16k, d], [d, 16k]
        b_enc, b_dec, thr = params["b_enc"], params["b_dec"], params["threshold"]

        vecs, token_mats = prompt_vectors(records, layer, special_ids)
        pairs = sorted({meta[pid]["pair"] for pid in records
                        if meta[pid]["class"] == "FIRE" and f"N{pid[1:]}" in records
                        and pid in vecs and f"N{pid[1:]}" in vecs})
        fire = torch.stack([vecs[f"F{p[1:]}"] for p in pairs])
        nofire = torch.stack([vecs[f"N{p[1:]}"] for p in pairs])
        w = fit_direction(fire, nofire, args.estimator)

        # (a) axis -> decoder-row cosines
        dec_cos = (w_dec @ w) / w_dec.norm(dim=1).clamp_min(1e-8)
        enc_cos = (w_enc.T @ w) / w_enc.norm(dim=0).clamp_min(1e-8)
        top = torch.topk(dec_cos.abs(), TOP_K)

        print(f"\n=== L{layer} (a) probe-axis vs decoder rows — top {TOP_K} |cosine| ===")
        a_rows = []
        for v, idx in zip(top.values.tolist(), top.indices.tolist()):
            exp = "" if args.no_explanations else neuronpedia_explanation(np_model, layer, idx, cache)
            mark = " <<HL" if idx in highlight else ""
            print(f"  f{idx:<6} dec_cos={dec_cos[idx]:+.3f} enc_cos={enc_cos[idx]:+.3f}  {exp}{mark}")
            a_rows.append({"feature": idx, "dec_cos": float(dec_cos[idx]),
                           "enc_cos": float(enc_cos[idx]), "explanation": exp,
                           "highlighted": idx in highlight})

        # (b) feature-space class contrast with W/S/Q
        def encode(toks):
            pre = (toks - b_dec) @ w_enc + b_enc
            return pre * (pre > thr)

        per_prompt_mean = {}
        per_class_acts = {"FIRE": [], "NOFIRE": []}
        for p in pairs:
            for cls, pid in (("FIRE", f"F{p[1:]}"), ("NOFIRE", f"N{p[1:]}")):
                acts = encode(token_mats[pid])           # [n_tok, 16k]
                per_prompt_mean[pid] = acts.mean(dim=0)  # W per prompt
                per_class_acts[cls].append(acts)

        W_f = torch.stack([per_prompt_mean[f"F{p[1:]}"] for p in pairs])  # [n, 16k]
        W_n = torch.stack([per_prompt_mean[f"N{p[1:]}"] for p in pairs])
        mu_f, mu_n = W_f.mean(dim=0), W_n.mean(dim=0)
        sd = ((W_f.var(dim=0, unbiased=True) + W_n.var(dim=0, unbiased=True)) / 2).sqrt()
        d_feat = (mu_f - mu_n) / sd.clamp_min(1e-8)

        acts_f = torch.cat(per_class_acts["FIRE"])
        acts_n = torch.cat(per_class_acts["NOFIRE"])
        S_f, S_n = (acts_f > 0).float().mean(dim=0), (acts_n > 0).float().mean(dim=0)
        Q_f = acts_f.sum(dim=0) / (acts_f > 0).float().sum(dim=0).clamp_min(1)
        Q_n = acts_n.sum(dim=0) / (acts_n > 0).float().sum(dim=0).clamp_min(1)

        topb = torch.topk(d_feat.abs().nan_to_num(), TOP_K)
        print(f"\n=== L{layer} (b) FIRE-vs-NOFIRE feature contrast — top {TOP_K} |d| ===")
        b_rows = []
        for _, idx in zip(topb.values.tolist(), topb.indices.tolist()):
            exp = "" if args.no_explanations else neuronpedia_explanation(np_model, layer, idx, cache)
            mark = " <<HL" if idx in highlight else ""
            print(f"  f{idx:<6} d={d_feat[idx]:+6.2f}  "
                  f"S {S_f[idx]:.3f}/{S_n[idx]:.3f}  Q {Q_f[idx]:7.1f}/{Q_n[idx]:7.1f}  "
                  f"W {mu_f[idx]:7.1f}/{mu_n[idx]:7.1f}  {exp}{mark}")
            b_rows.append({"feature": idx, "d": float(d_feat[idx]),
                           "S_fire": float(S_f[idx]), "S_nofire": float(S_n[idx]),
                           "Q_fire": float(Q_f[idx]), "Q_nofire": float(Q_n[idx]),
                           "W_fire": float(mu_f[idx]), "W_nofire": float(mu_n[idx]),
                           "explanation": exp, "highlighted": idx in highlight})

        out_all[layer] = {"axis_vs_decoder": a_rows, "feature_contrast": b_rows,
                          "n_pairs": len(pairs)}

    out = cap_dir / f"decompose_{args.estimator}.json"
    out.write_text(json.dumps(out_all, indent=1))
    print(f"\nsaved: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
