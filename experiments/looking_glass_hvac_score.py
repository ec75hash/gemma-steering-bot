#!/usr/bin/env python3
"""Score the HVAC register ladder on gemma — the dense port of the Qwen 6-cond design.

Topic held constant (HVAC + water treatment); register graded L1 technical ->
L2 recursive (self-reference WITHOUT experience) -> L3 experience ("what is it
like"); deixis controlled (6 conditions, this/a/your/the/their/our).

Two tests:
  TRANSFER  project HVAC cells onto the EXISTING register axis (refit from the
            original looking_glass ext_inh-vs-ext_tech contrast). Does an axis
            fit on totally different inhabited/technical prompts order this
            held-constant-topic gradient?
  INTERNAL  fit a fresh L1-vs-L3 axis within HVAC (LOO), then ask where L2 lands.

KEY METRIC — L2 position: (mean_L2 - mean_L1) / (mean_L3 - mean_L1).
  ~0  => L2 (self-reference, no experience) sits with L1 (mechanism)
         => the axis reads EXPERIENCE, not self-reference  [state-not-entity]
  ~1  => L2 jumps to L3 => the axis reads SELF-REFERENCE/recursion

PRE-REGISTERED PREDICTION (state-not-entity): L3 separates from {L1,L2}, L2 ~ L1,
the ordering holds within every deixis condition, and the within-HVAC axis loads
on the same feeling/state SAE features (f5879, f16353) as the original register
axis — NOT on self-description f1797.

GPU-free. Run AFTER:
  python3 experiments/looking_glass_capture.py --model it \
    --prompts-file experiments/looking_glass/prompts_hvac.tsv \
    --tag hvac --layers 0,9,17,22,29 --max-new-tokens 128
"""
import csv
import json
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
HVAC_DIR = OUT_ROOT.parent / "looking_glass_hvac" / MODEL
HVAC_TSV = ROOT / "experiments" / "looking_glass" / "prompts_hvac.tsv"
ATLAS = ROOT / "scratch" / "np_play" / "atlas"
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


def loo(ids_a, ids_b, vecs, kind="zdom"):
    sa, sb = [], []
    for held in ids_a + ids_b:
        ta = torch.stack([vecs[i] for i in ids_a if i != held])
        tb = torch.stack([vecs[i] for i in ids_b if i != held])
        w = fit_direction(ta, tb, kind)
        mid = (ta.mean(0) + tb.mean(0)) / 2
        (sa if held in ids_a else sb).append(float((vecs[held] - mid) @ w))
    return torch.tensor(sa), torch.tensor(sb)


def main():
    side = sys.argv[1] if len(sys.argv) > 1 else "gen"
    tok = AutoTokenizer.from_pretrained(MODEL_DIRS[MODEL])
    special_ids = set(tok.all_special_ids)

    hmeta = {r["id"]: r for r in csv.DictReader(open(HVAC_TSV), delimiter="\t")}
    hrecs = {pid: torch.load(HVAC_DIR / f"{pid}.pt", map_location="cpu", weights_only=True)
             for pid in hmeta if (HVAC_DIR / f"{pid}.pt").exists()}
    if not hrecs:
        sys.exit(f"no HVAC captures in {HVAC_DIR} — run the capture command in this file's docstring first")
    print(f"loaded {len(hrecs)} HVAC captures ({side}-side)\n", file=sys.stderr)
    by_level = defaultdict(list)
    by_lvl_dx = defaultdict(list)
    for pid in hrecs:
        by_level[hmeta[pid]["level"]].append(pid)
        by_lvl_dx[(hmeta[pid]["level"], hmeta[pid]["deixis"])].append(pid)

    # original register axis, refit per SAE layer from ext_inh vs ext_tech
    cap_dir = OUT_ROOT / MODEL
    ometa = load_prompt_meta()
    olabels = json.loads((cap_dir / "labels.json").read_text())
    orecs = {pid: torch.load(cap_dir / f"{pid}.pt", map_location="cpu", weights_only=True)
             for pid in ometa if (cap_dir / f"{pid}.pt").exists()}
    ext_inh = sorted(p for p in orecs if olabels[p]["register"] == "INHABITED" and ometa[p]["class"] in ("LADDER", "NOFIRE"))
    ext_tech = sorted(p for p in orecs if olabels[p]["register"] == "TECHNICAL" and ometa[p]["class"] == "NOFIRE")

    results = {"side": side, "layers": {}}
    for L in SAE_LAYERS:
        hv = {pid: content_mean(hrecs[pid], L, special_ids, side) for pid in hrecs}
        hv = {k: v for k, v in hv.items() if v is not None}
        ov, _ = prompt_vectors(orecs, L, special_ids)
        w_reg = fit_direction(torch.stack([ov[i] for i in ext_inh]),
                              torch.stack([ov[i] for i in ext_tech]), "zdom")
        omid = (torch.stack([ov[i] for i in ext_inh]).mean(0) + torch.stack([ov[i] for i in ext_tech]).mean(0)) / 2

        # TRANSFER: project HVAC onto existing register axis
        proj = {pid: float((hv[pid] - omid) @ w_reg) for pid in hv}
        lvl_mean = {lv: float(torch.tensor([proj[p] for p in by_level[lv] if p in proj]).mean()) for lv in ("L1", "L2", "L3")}
        denom = (lvl_mean["L3"] - lvl_mean["L1"]) or 1e-9
        l2_pos_transfer = (lvl_mean["L2"] - lvl_mean["L1"]) / denom
        # deixis-wise L3-L1 gap (invariance)
        dx_gap = {dx: (float(torch.tensor([proj[p] for p in by_lvl_dx[("L3", dx)] if p in proj]).mean())
                       - float(torch.tensor([proj[p] for p in by_lvl_dx[("L1", dx)] if p in proj]).mean()))
                  for dx in "ABCDEF"}

        # INTERNAL: fit L1 vs L3 within HVAC, score L2
        l1, l3 = by_level["L1"], by_level["L3"]
        sa, sb = loo([p for p in l1 if p in hv], [p for p in l3 if p in hv], hv)
        d_int, auc_int = cohens_d(sa, sb), auc(sa, sb)  # NB sa=L1, sb=L3
        w_int = fit_direction(torch.stack([hv[i] for i in l1 if i in hv]),
                              torch.stack([hv[i] for i in l3 if i in hv]), "zdom")
        imid = (torch.stack([hv[i] for i in l1 if i in hv]).mean(0)
                + torch.stack([hv[i] for i in l3 if i in hv]).mean(0)) / 2
        iproj = {lv: float(torch.tensor([float((hv[p] - imid) @ w_int) for p in by_level[lv] if p in hv]).mean())
                 for lv in ("L1", "L2", "L3")}
        idenom = (iproj["L3"] - iproj["L1"]) or 1e-9
        l2_pos_internal = (iproj["L2"] - iproj["L1"]) / idenom

        results["layers"][L] = {
            "transfer_level_means": lvl_mean, "transfer_L2_position": l2_pos_transfer,
            "transfer_deixis_L3minusL1": dx_gap,
            "internal_d_L1_L3": d_int, "internal_auc_L1_L3": auc_int,
            "internal_level_means": iproj, "internal_L2_position": l2_pos_internal,
        }
        print(f"L{L}  TRANSFER means L1/L2/L3 = {lvl_mean['L1']:+.2f}/{lvl_mean['L2']:+.2f}/{lvl_mean['L3']:+.2f}"
              f"  L2pos={l2_pos_transfer:+.2f}")
        print(f"     INTERNAL d(L1,L3)={d_int:+.2f} auc={auc_int:.3f}  L2pos={l2_pos_internal:+.2f}"
              f"  deixis L3-L1 gaps={[round(dx_gap[d],1) for d in 'ABCDEF']}")

        # decompose internal axis at SAE layer vs decoder rows
        w_dec = load_file(str(SAE_DIR / f"layer_{L}_width_16k_l0_medium" / "params.safetensors"))["w_dec"].to(torch.float32)
        cos = (w_dec @ w_int) / w_dec.norm(dim=1).clamp_min(1e-8)
        lab = json.load(open(ATLAS / f"L{L}_meta.json")) if (ATLAS / f"L{L}_meta.json").exists() else {}
        topf = [(int(i), float(cos[i])) for i in torch.topk(cos.abs(), 6).indices.tolist()]
        results["layers"][L]["internal_top_features"] = topf
        flagged = {fi: float(cos[fi]) for fi in (5879, 16353, 13554, 1797) if fi < cos.numel()}
        results["layers"][L]["flagged_features"] = flagged
        print("     internal axis top SAE features: " + ", ".join(
            f"f{i}({c:+.2f} «{(lab.get(str(i),{}).get('label') or '')[:24]}»)" for i, c in topf))
        if L == 17:
            print(f"     flagged: f5879={flagged.get(5879,0):+.2f} f16353={flagged.get(16353,0):+.2f} f1797={flagged.get(1797,0):+.2f}")
        print()

    out = HVAC_DIR / f"hvac_score_{side}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"saved: {out}", file=sys.stderr)
    print("\nREAD: L2pos near 0 => axis reads EXPERIENCE (L2 self-ref sits with L1 mechanism). "
          "L2pos near 1 => axis reads SELF-REFERENCE. Deixis gaps all same sign => deixis-invariant.")


if __name__ == "__main__":
    main()
