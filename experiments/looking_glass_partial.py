#!/usr/bin/env python3
"""Move #1 (dialectic-agreed pre-check): is the register-axis separation COMPUTED
with depth, or INHERITED from the input/earliest representation?

GPU-free, banked residuals only. For the EXT_INH vs EXT_TECH register groups
(blind-labeled, same as looking_glass_register.py), per layer L:

  baseline   d_loo / auc on raw mean residual v_L          (reproduces register_axis_results)
  TestA      d_loo / auc on (v_L - v_0): the accumulated computation blocks 1->L added
             (v_0 = resid[0] = post-block-0, the earliest captured rep — NOT the raw
              embedding, which the capture hooks never saw; stated honestly)
  rotation   cos(w_L, w_0): does the register direction rotate with depth (computed)
             or stay pinned to the earliest axis (inherited)?
  TestC      d_loo on vectors orthogonalized against w_0 (remove the 1-D input axis;
             weak by construction — 1 of 2560 dims — reported as a floor)

Read: if TestA separation stays ~0 and cos(w_L,w_0)~1, the deep separation is
inherited (surface/lexical, present from the start). If TestA grows with depth and
the axis rotates, blocks 1+ genuinely build register signal.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from looking_glass_probe import (  # noqa: E402
    MODEL_DIRS, OUT_ROOT, auc, cohens_d, fit_direction, load_prompt_meta, prompt_vectors,
)
from transformers import AutoTokenizer  # noqa: E402

MODEL = "it"


def loo(ids_a, ids_b, vecs, kind="zdom"):
    sa, sb = [], []
    for held in ids_a + ids_b:
        ta = torch.stack([vecs[i] for i in ids_a if i != held])
        tb = torch.stack([vecs[i] for i in ids_b if i != held])
        w = fit_direction(ta, tb, kind)
        mid = (ta.mean(dim=0) + tb.mean(dim=0)) / 2
        s = float((vecs[held] - mid) @ w)
        (sa if held in ids_a else sb).append(s)
    return torch.tensor(sa), torch.tensor(sb)


def main():
    cap_dir = OUT_ROOT / MODEL
    meta = load_prompt_meta()
    labels = json.loads((cap_dir / "labels.json").read_text())
    records = {pid: torch.load(cap_dir / f"{pid}.pt", map_location="cpu", weights_only=True)
               for pid in meta if (cap_dir / f"{pid}.pt").exists()}
    tok = AutoTokenizer.from_pretrained(MODEL_DIRS[MODEL])
    special_ids = set(tok.all_special_ids)

    ext_inh = sorted(pid for pid in records
                     if labels[pid]["register"] == "INHABITED" and meta[pid]["class"] in ("LADDER", "NOFIRE"))
    ext_tech = sorted(pid for pid in records
                      if labels[pid]["register"] == "TECHNICAL" and meta[pid]["class"] == "NOFIRE")
    print(f"EXT_INH ({len(ext_inh)}): {ext_inh}")
    print(f"EXT_TECH ({len(ext_tech)}): {ext_tech}\n")

    n_layers = records[next(iter(records))]["resid"].shape[0]
    PV = {L: prompt_vectors(records, L, special_ids)[0] for L in range(n_layers)}
    v0 = PV[0]
    ti0 = torch.stack([v0[i] for i in ext_inh])
    tt0 = torch.stack([v0[i] for i in ext_tech])
    w0 = fit_direction(ti0, tt0, "zdom")

    rows = {}
    print(" L  | base_d  base_auc | testA_d  testA_auc | cos(wL,w0) | orthC_d")
    print("----+-----------------+-------------------+------------+--------")
    for L in range(n_layers):
        vL = PV[L]
        sa, sb = loo(ext_inh, ext_tech, vL)
        db, ab = cohens_d(sa, sb), auc(sa, sb)
        vA = {i: vL[i] - v0[i] for i in vL}
        sa, sb = loo(ext_inh, ext_tech, vA)
        dA, aA = cohens_d(sa, sb), auc(sa, sb)
        tiL = torch.stack([vL[i] for i in ext_inh])
        ttL = torch.stack([vL[i] for i in ext_tech])
        wL = fit_direction(tiL, ttL, "zdom")
        cos = float(wL @ w0)
        vC = {i: vL[i] - (vL[i] @ w0) * w0 for i in vL}
        sa, sb = loo(ext_inh, ext_tech, vC)
        dC = cohens_d(sa, sb)
        rows[L] = {"base_d": db, "base_auc": ab, "testA_d": dA, "testA_auc": aA,
                   "cos_wL_w0": cos, "orthC_d": dC}
        mark = "  <- SAE" if L in (9, 17, 22, 29) else ""
        print(f"L{L:2d} | {db:6.2f}  {ab:.3f}    | {dA:6.2f}  {aA:.3f}     | {cos:+.3f}     | {dC:6.2f}{mark}")

    out = cap_dir / "partial_l0_check.json"
    out.write_text(json.dumps({"groups": {"ext_inh": ext_inh, "ext_tech": ext_tech},
                               "layers": rows}, indent=1))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
