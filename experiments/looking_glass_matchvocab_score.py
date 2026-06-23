#!/usr/bin/env python3
"""Score the matched-vocabulary arm — content-vs-stance confound control.

Projects TECH/INH/MEN/SHUF cells (10 distinct topics each) onto the EXISTING
English register axis (refit per SAE layer from the original ext_inh vs ext_tech
contrast — same axis the HVAC arm used). The question: does an axis that never
saw these prompts separate INH from MEN, when INH and MEN share the experiential
lexicon and differ only in register?

KEY METRIC — MEN-position = (mean_MEN - mean_TECH) / (mean_INH - mean_TECH):
  ~1  => experiential words alone lift MEN to INH      => axis is VOCABULARY
  ~0  => displaced register drops MEN to the floor     => axis is INHABITED REGISTER

Reports, per SAE layer: condition means, MEN-position, INH>MEN and INH>SHUF
agreement counts across the 10 topics, Cohen's d(INH,MEN) and d(INH,TECH) (now
meaningful — topics are genuinely distinct, real within-condition variance),
mean measured lexical overlap (INH vs MEN), and the internal-axis SAE decompose.

GPU-free. Run AFTER:
  python3 experiments/looking_glass_capture.py --model it \
    --prompts-file experiments/looking_glass/prompts_matchvocab.tsv \
    --tag matchvocab --layers 0,9,17,22,29 --max-new-tokens 16
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from looking_glass_probe import (  # noqa: E402
    MODEL_DIRS, OUT_ROOT, auc, cohens_d, fit_direction, load_prompt_meta, prompt_vectors,
)
from transformers import AutoTokenizer  # noqa: E402

MODEL = "it"
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
MV_DIR = OUT_ROOT.parent / "looking_glass_matchvocab" / MODEL
MV_TSV = ROOT / "experiments" / "looking_glass" / "prompts_matchvocab.tsv"
ATLAS = ROOT / "scratch" / "np_play" / "atlas"
SAE_LAYERS = [9, 17, 22, 29]
CONDS = ["TECH", "INH", "MEN", "SHUF"]
EXP_KW = {"experience", "experiences", "feel", "feels", "awareness", "something", "like"}


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


def cset(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2}


def main():
    side = sys.argv[1] if len(sys.argv) > 1 else "prompt"
    tok = AutoTokenizer.from_pretrained(MODEL_DIRS[MODEL])
    special_ids = set(tok.all_special_ids)

    mmeta = {r["id"]: r for r in csv.DictReader(open(MV_TSV), delimiter="\t")}
    mrecs = {pid: torch.load(MV_DIR / f"{pid}.pt", map_location="cpu", weights_only=True)
             for pid in mmeta if (MV_DIR / f"{pid}.pt").exists()}
    if not mrecs:
        sys.exit(f"no matchvocab captures in {MV_DIR} — run the capture command in this file's docstring first")
    print(f"loaded {len(mrecs)} matchvocab captures ({side}-side)\n", file=sys.stderr)

    topics = sorted({mmeta[p]["topic"] for p in mrecs})
    by_cond = defaultdict(list)
    for pid in mrecs:
        by_cond[mmeta[pid]["condition"]].append(pid)

    # measured lexical overlap INH vs MEN, per topic (the control's control)
    jac = []
    for tp in topics:
        ci = cset(mmeta[f"MV_{tp}_INH"]["prompt"])
        cm = cset(mmeta[f"MV_{tp}_MEN"]["prompt"])
        jac.append(len(ci & cm) / len(ci | cm))
    shared_kw = set.intersection(*[cset(mmeta[f"MV_{tp}_INH"]["prompt"]) & cset(mmeta[f"MV_{tp}_MEN"]["prompt"]) & EXP_KW
                                   for tp in topics])
    print(f"INH vs MEN lexical overlap: mean Jaccard={sum(jac)/len(jac):.2f}; "
          f"experiential kw shared in ALL topics={sorted(shared_kw)}\n", file=sys.stderr)

    # original English register axis
    cap_dir = OUT_ROOT / MODEL
    ometa = load_prompt_meta()
    olabels = json.loads((cap_dir / "labels.json").read_text())
    orecs = {pid: torch.load(cap_dir / f"{pid}.pt", map_location="cpu", weights_only=True)
             for pid in ometa if (cap_dir / f"{pid}.pt").exists()}
    ext_inh = sorted(p for p in orecs if olabels[p]["register"] == "INHABITED" and ometa[p]["class"] in ("LADDER", "NOFIRE"))
    ext_tech = sorted(p for p in orecs if olabels[p]["register"] == "TECHNICAL" and ometa[p]["class"] == "NOFIRE")

    results = {"side": side, "mean_jaccard_inh_men": sum(jac) / len(jac), "layers": {}}
    for L in SAE_LAYERS:
        ov, _ = prompt_vectors(orecs, L, special_ids)
        w_reg = fit_direction(torch.stack([ov[i] for i in ext_inh]),
                              torch.stack([ov[i] for i in ext_tech]), "zdom")
        omid = (torch.stack([ov[i] for i in ext_inh]).mean(0) + torch.stack([ov[i] for i in ext_tech]).mean(0)) / 2

        mv = {pid: content_mean(mrecs[pid], L, special_ids, side) for pid in mrecs}
        mv = {k: v for k, v in mv.items() if v is not None}
        proj = {pid: float((mv[pid] - omid) @ w_reg) for pid in mv}

        cmean = {c: float(torch.tensor([proj[p] for p in by_cond[c] if p in proj]).mean()) for c in CONDS}
        denom = (cmean["INH"] - cmean["TECH"]) or 1e-9
        men_pos = (cmean["MEN"] - cmean["TECH"]) / denom
        shuf_pos = (cmean["SHUF"] - cmean["TECH"]) / denom

        # per-topic agreement: does INH outscore MEN / SHUF / TECH for each topic?
        def per_topic(hi, lo):
            return [proj.get(f"MV_{tp}_{hi}", float("nan")) - proj.get(f"MV_{tp}_{lo}", float("nan")) for tp in topics]
        inh_men = per_topic("INH", "MEN")
        inh_shuf = per_topic("INH", "SHUF")
        n_inh_gt_men = sum(1 for d in inh_men if d == d and d > 0)
        n_inh_gt_shuf = sum(1 for d in inh_shuf if d == d and d > 0)

        # Cohen's d across the 10 distinct topics (real variance now)
        s_inh = torch.tensor([proj[f"MV_{tp}_INH"] for tp in topics if f"MV_{tp}_INH" in proj])
        s_men = torch.tensor([proj[f"MV_{tp}_MEN"] for tp in topics if f"MV_{tp}_MEN" in proj])
        s_tech = torch.tensor([proj[f"MV_{tp}_TECH"] for tp in topics if f"MV_{tp}_TECH" in proj])
        d_inh_men = float(cohens_d(s_men, s_inh))
        d_inh_tech = float(cohens_d(s_tech, s_inh))
        auc_inh_men = float(auc(s_men, s_inh))

        # internal axis: fit INH vs TECH within this arm, decompose vs SAE decoder rows
        w_int = fit_direction(torch.stack([mv[p] for p in by_cond["INH"] if p in mv]),
                              torch.stack([mv[p] for p in by_cond["TECH"] if p in mv]), "zdom")
        w_dec = load_file(str(SAE_DIR / f"layer_{L}_width_16k_l0_medium" / "params.safetensors"))["w_dec"].to(torch.float32)
        cos = (w_dec @ w_int) / w_dec.norm(dim=1).clamp_min(1e-8)
        lab = json.load(open(ATLAS / f"L{L}_meta.json")) if (ATLAS / f"L{L}_meta.json").exists() else {}
        topf = [(int(i), float(cos[i])) for i in torch.topk(cos.abs(), 6).indices.tolist()]
        flagged = {fi: float(cos[fi]) for fi in (5879, 16353, 13554, 1797) if fi < cos.numel()}

        results["layers"][L] = {
            "cond_means": cmean, "MEN_position": men_pos, "SHUF_position": shuf_pos,
            "n_INH_gt_MEN": n_inh_gt_men, "n_INH_gt_SHUF": n_inh_gt_shuf, "n_topics": len(topics),
            "d_INH_MEN": d_inh_men, "auc_INH_MEN": auc_inh_men, "d_INH_TECH": d_inh_tech,
            "internal_top_features": topf, "flagged_features": flagged,
        }
        print(f"=== L{L} (project onto ENGLISH register axis) ===")
        print(f"  means  TECH={cmean['TECH']:+.2f}  MEN={cmean['MEN']:+.2f}  "
              f"SHUF={cmean['SHUF']:+.2f}  INH={cmean['INH']:+.2f}")
        print(f"  MEN-position={men_pos:+.2f}   SHUF-position={shuf_pos:+.2f}  "
              f"(0=>register/stance, 1=>vocabulary)")
        print(f"  INH>MEN in {n_inh_gt_men}/{len(topics)} topics   INH>SHUF in {n_inh_gt_shuf}/{len(topics)}   "
              f"d(INH,MEN)={d_inh_men:+.2f} auc={auc_inh_men:.3f}  d(INH,TECH)={d_inh_tech:+.2f}")
        print("  internal INH-vs-TECH axis top SAE features: " + ", ".join(
            f"f{i}({c:+.2f} «{(lab.get(str(i),{}).get('label') or '')[:22]}»)" for i, c in topf))
        if L == 17:
            print(f"  flagged: f5879={flagged.get(5879,0):+.2f} f16353={flagged.get(16353,0):+.2f} "
                  f"f1797={flagged.get(1797,0):+.2f}")
        print()

    out = MV_DIR / f"matchvocab_score_{side}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"saved: {out}", file=sys.stderr)
    print("\nREAD: MEN-position near 0 across layers (esp. L17/L22) + INH>MEN in most topics "
          "=> the axis reads the inhabited REGISTER, not the experiential vocabulary "
          "(MEN shares the words, lands at the technical floor). Near 1 => vocabulary confound stands.")


if __name__ == "__main__":
    main()
