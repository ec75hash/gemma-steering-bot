#!/usr/bin/env python3
"""Cross-lingual transfer test: does the ENGLISH-fit register axis order the
technical->recursive->experience ladder in German/Spanish/Chinese?

If the English axis (fit on ext_inh vs ext_tech English prompts) separates L3 from
L1 in languages whose surface tokens it never saw, the signal is not English
surface tokens. Pre-registered prediction: L3 > L1 in every language, L2 ~ L1
(experience, not self-reference), strongest at mid layers (multilingual concept
convergence). The XL_en_* cells are a within-set sanity check that the lean prompt
format separates at all.

GPU-free. Run AFTER:
  python3 experiments/looking_glass_capture.py --model it \
    --prompts-file experiments/looking_glass/prompts_hvac_xling.tsv \
    --tag hvacxl --layers 0,9,17,22,29 --max-new-tokens 64
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from looking_glass_probe import (  # noqa: E402
    MODEL_DIRS, OUT_ROOT, cohens_d, fit_direction, load_prompt_meta, prompt_vectors,
)
from transformers import AutoTokenizer  # noqa: E402

MODEL = "it"
XL_DIR = OUT_ROOT.parent / "looking_glass_hvacxl" / MODEL
XL_TSV = ROOT / "experiments" / "looking_glass" / "prompts_hvac_xling.tsv"
SAE_LAYERS = [9, 17, 22, 29]


def content_mean(rec, true_layer, special_ids, side):
    si = rec["layer_ids"].index(true_layer)
    resid = rec["resid"][si]
    seq = resid.shape[0]
    n_prompt, trim = rec["n_prompt"], rec["trim_idx"]
    ids = rec["all_ids"]
    mask = torch.zeros(seq, dtype=torch.bool)
    if side == "gen":
        mask[n_prompt: n_prompt + trim] = True
    else:
        mask[:n_prompt] = True
    for i in range(seq):
        if mask[i] and i < len(ids) and ids[i].item() in special_ids:
            mask[i] = False
    if mask.sum() == 0:
        return None
    return resid[mask].to(torch.float32).mean(0)


def main():
    side = sys.argv[1] if len(sys.argv) > 1 else "prompt"
    tok = AutoTokenizer.from_pretrained(MODEL_DIRS[MODEL])
    special_ids = set(tok.all_special_ids)

    xmeta = {r["id"]: r for r in csv.DictReader(open(XL_TSV), delimiter="\t")}
    xrecs = {pid: torch.load(XL_DIR / f"{pid}.pt", map_location="cpu", weights_only=True)
             for pid in xmeta if (XL_DIR / f"{pid}.pt").exists()}
    if not xrecs:
        sys.exit(f"no xling captures in {XL_DIR} — run the capture command in this file's docstring first")
    print(f"loaded {len(xrecs)} xling captures ({side}-side)\n", file=sys.stderr)

    # original English register axis
    cap_dir = OUT_ROOT / MODEL
    ometa = load_prompt_meta()
    olabels = json.loads((cap_dir / "labels.json").read_text())
    orecs = {pid: torch.load(cap_dir / f"{pid}.pt", map_location="cpu", weights_only=True)
             for pid in ometa if (cap_dir / f"{pid}.pt").exists()}
    ext_inh = sorted(p for p in orecs if olabels[p]["register"] == "INHABITED" and ometa[p]["class"] in ("LADDER", "NOFIRE"))
    ext_tech = sorted(p for p in orecs if olabels[p]["register"] == "TECHNICAL" and ometa[p]["class"] == "NOFIRE")

    LANGS = ["en", "de", "es", "ru"]
    results = {"side": side, "layers": {}}
    for L in SAE_LAYERS:
        ov, _ = prompt_vectors(orecs, L, special_ids)
        w_reg = fit_direction(torch.stack([ov[i] for i in ext_inh]),
                              torch.stack([ov[i] for i in ext_tech]), "zdom")
        omid = (torch.stack([ov[i] for i in ext_inh]).mean(0) + torch.stack([ov[i] for i in ext_tech]).mean(0)) / 2

        proj = {}
        for pid, rec in xrecs.items():
            v = content_mean(rec, L, special_ids, side)
            if v is not None:
                proj[pid] = float((v - omid) @ w_reg)

        layer_rows = {}
        for lang in LANGS:
            g = {lv: proj.get(f"XL_{lang}_{lv}") for lv in ("L1", "L2", "L3")}
            if None in g.values():
                continue
            denom = (g["L3"] - g["L1"]) or 1e-9
            l2pos = (g["L2"] - g["L1"]) / denom
            layer_rows[lang] = {"L1": g["L1"], "L2": g["L2"], "L3": g["L3"],
                                "L3_gt_L1": g["L3"] > g["L1"], "L2_position": l2pos}
        results["layers"][L] = layer_rows
        print(f"=== L{L} (project foreign cells onto ENGLISH register axis) ===")
        for lang in LANGS:
            if lang in layer_rows:
                r = layer_rows[lang]
                print(f"  {lang}:  L1={r['L1']:+.2f}  L2={r['L2']:+.2f}  L3={r['L3']:+.2f}   "
                      f"L3>L1={'YES' if r['L3_gt_L1'] else 'no '}  L2pos={r['L2_position']:+.2f}")
        # pooled foreign axis (de+es+zh L1 vs L3) cosine vs English axis
        fl1 = [content_mean(xrecs[f"XL_{l}_L1"], L, special_ids, side) for l in ("de", "es", "ru") if f"XL_{l}_L1" in xrecs]
        fl3 = [content_mean(xrecs[f"XL_{l}_L3"], L, special_ids, side) for l in ("de", "es", "ru") if f"XL_{l}_L3" in xrecs]
        if len(fl1) >= 2 and len(fl3) >= 2:
            w_for = fit_direction(torch.stack(fl1), torch.stack(fl3), "zdom")
            cos = float(w_for @ w_reg)
            results["layers"][L]["foreign_axis_cos_with_english"] = cos
            print(f"  cos(foreign L1->L3 axis, English register axis) = {cos:+.3f}")
        print()

    out = XL_DIR / f"xling_score_{side}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"saved: {out}", file=sys.stderr)
    print("\nREAD: transfer confirmed if L3>L1 in de/es/zh (not just en) and L2pos~0; "
          "strongest at mid layers; positive foreign-vs-English axis cosine => same direction across languages.")


if __name__ == "__main__":
    main()
