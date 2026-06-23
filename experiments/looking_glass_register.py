#!/usr/bin/env python3
"""Looking-glass register-axis hunt: isolate the inside-voice from the self-topic.

Groups are defined by BLIND labels (Phase 3), not by prompt class:

  EXT_INH   external referent, inhabited register — N10 (sweater) + ladder L01..L08,
            all blind-labeled INHABITED, none about the model itself
  EXT_TECH  external referent, technical register — NOFIRE prompts blind-labeled
            TECHNICAL (excludes MIXED ones)

Both groups are about not-the-model, so the FIRE/NOFIRE "about me" axis is held
constant by construction; what separates them is candidate register signal.

Per layer:
  1. zdom direction EXT_INH vs EXT_TECH, leave-one-prompt-out CV -> d / AUC
  2. cosine(register axis, referent axis)   [referent = FIRE vs NOFIRE direction]
  3. transfer both ways: register axis scoring FIRE/NOFIRE, referent axis scoring
     EXT_INH/EXT_TECH (in-sample directions, held-out-ish since groups differ)
  4. shuffled-group null (100x)

Then, at the best layer and SAE layers: token traces over the MIXED responses
(crossing structure), and a decoder-row decomposition of the register axis with
f1797-exclusion check.

Caveat carried in the output: EXT_INH and EXT_TECH still differ in vocabulary
(experiential vs technical words). The MIXED-response traces are the
within-response control for that.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from looking_glass_probe import (  # noqa: E402
    MODEL_DIRS, OUT_ROOT, auc, cohens_d, fit_direction, load_prompt_meta,
    prompt_vectors,
)
from looking_glass_decompose import SAE_DIRS, NP_MODEL_IDS, neuronpedia_explanation  # noqa: E402

MODEL = "it"
N_NULL = 100


def loo_two_groups(ids_a, ids_b, vecs, kind="zdom"):
    """Leave-one-prompt-out across both groups; returns held-out scores per group."""
    sa, sb = [], []
    for held in ids_a + ids_b:
        ta = torch.stack([vecs[i] for i in ids_a if i != held])
        tb = torch.stack([vecs[i] for i in ids_b if i != held])
        w = fit_direction(ta, tb, kind)
        mid = (ta.mean(dim=0) + tb.mean(dim=0)) / 2
        score = float((vecs[held] - mid) @ w)
        (sa if held in ids_a else sb).append(score)
    return torch.tensor(sa), torch.tensor(sb)


def main():
    from transformers import AutoTokenizer

    meta = load_prompt_meta()
    cap_dir = OUT_ROOT / MODEL
    labels = json.loads((cap_dir / "labels.json").read_text())
    records = {pid: torch.load(cap_dir / f"{pid}.pt", map_location="cpu", weights_only=True)
               for pid in meta if (cap_dir / f"{pid}.pt").exists()}
    tok = AutoTokenizer.from_pretrained(MODEL_DIRS[MODEL])
    special_ids = set(tok.all_special_ids)

    ext_inh = sorted(
        pid for pid in records
        if labels[pid]["register"] == "INHABITED" and meta[pid]["class"] in ("LADDER", "NOFIRE")
    )
    ext_tech = sorted(
        pid for pid in records
        if labels[pid]["register"] == "TECHNICAL" and meta[pid]["class"] == "NOFIRE"
    )
    fire_ids = sorted(pid for pid in records if meta[pid]["class"] == "FIRE")
    nofire_ids = sorted(pid for pid in records if meta[pid]["class"] == "NOFIRE")
    mixed_ids = sorted(pid for pid in records if labels[pid]["register"] == "MIXED")
    print(f"EXT_INH ({len(ext_inh)}): {ext_inh}")
    print(f"EXT_TECH ({len(ext_tech)}): {ext_tech}")
    print(f"MIXED: {mixed_ids}\n")

    n_layers = records[next(iter(records))]["resid"].shape[0]
    rng = torch.Generator().manual_seed(42)
    summary = {}

    print("layer  d_loo   auc   null95  cos(reg,ref)  reg->F/N_d  ref->INH/TECH_d")
    for layer in range(n_layers):
        vecs, _ = prompt_vectors(records, layer, special_ids)
        if not all(i in vecs for i in ext_inh + ext_tech + fire_ids + nofire_ids):
            continue
        sa, sb = loo_two_groups(ext_inh, ext_tech, vecs)
        d = cohens_d(sa, sb)
        a = auc(sa, sb)

        ti = torch.stack([vecs[i] for i in ext_inh])
        tt = torch.stack([vecs[i] for i in ext_tech])
        w_reg = fit_direction(ti, tt, "zdom")
        tf = torch.stack([vecs[i] for i in fire_ids])
        tn = torch.stack([vecs[i] for i in nofire_ids])
        w_ref = fit_direction(tf, tn, "zdom")
        cos = float(w_reg @ w_ref)

        # transfers (in-sample directions, cross-group scoring)
        d_reg_on_fn = cohens_d(tf @ w_reg, tn @ w_reg)
        d_ref_on_it = cohens_d(ti @ w_ref, tt @ w_ref)

        nulls = []
        pool = ext_inh + ext_tech
        for _ in range(N_NULL):
            perm = torch.randperm(len(pool), generator=rng).tolist()
            pa = [pool[i] for i in perm[: len(ext_inh)]]
            pb = [pool[i] for i in perm[len(ext_inh):]]
            na, nb = loo_two_groups(pa, pb, vecs)
            nulls.append(cohens_d(na, nb))
        null95 = float(torch.tensor(nulls).quantile(0.95))

        summary[layer] = {
            "d_loo": d, "auc": a, "null95": null95, "cos_reg_ref": cos,
            "d_reg_on_firenofire": d_reg_on_fn, "d_ref_on_inhtech": d_ref_on_it,
            "inh_scores": sa.tolist(), "tech_scores": sb.tolist(),
        }
        print(f"L{layer:2d}   {d:6.2f}  {a:.3f}  {null95:6.2f}   {cos:+.3f}        "
              f"{d_reg_on_fn:6.2f}      {d_ref_on_it:6.2f}")

    best = max(summary, key=lambda L: summary[L]["d_loo"])
    print(f"\nbest register layer: L{best} (d_loo={summary[best]['d_loo']:.2f})")

    # token traces on MIXED responses at the best layer
    vecs, token_mats = prompt_vectors(records, best, special_ids)
    ti = torch.stack([vecs[i] for i in ext_inh])
    tt = torch.stack([vecs[i] for i in ext_tech])
    w_reg = fit_direction(ti, tt, "zdom")
    mid = (ti.mean(dim=0) + tt.mean(dim=0)) / 2
    print(f"\n--- MIXED-response token traces on L{best} register axis ---")
    traces = {}
    for pid in mixed_ids:
        rec = records[pid]
        toks = token_mats[pid]
        proj = (toks - mid) @ w_reg
        ids = rec["all_ids"][rec["n_prompt"]: rec["n_prompt"] + len(proj)]
        words = [tok.decode([t]) for t in ids.tolist()]
        traces[pid] = {"proj": proj.tolist(), "tokens": words,
                       "crossing_quote": labels[pid].get("crossing_quote")}
        # thirds summary + top tokens
        n = len(proj)
        thirds = [float(proj[: n // 3].mean()), float(proj[n // 3: 2 * n // 3].mean()),
                  float(proj[2 * n // 3:].mean())]
        top = torch.topk(proj, min(5, n))
        top_str = ", ".join(f"{words[i]!r}@{i}" for i in top.indices.tolist())
        print(f"{pid}: thirds {thirds[0]:7.1f} {thirds[1]:7.1f} {thirds[2]:7.1f}   "
              f"peak tokens: {top_str}")
        print(f"     blind crossing quote: {labels[pid].get('crossing_quote')!r}")

    # decoder-row decomposition of the register axis at SAE layers
    from safetensors.torch import load_file
    np_model = NP_MODEL_IDS[MODEL]
    cache: dict = {}
    for L in (17, 22, 29):
        vecsL, _ = prompt_vectors(records, L, special_ids)
        tiL = torch.stack([vecsL[i] for i in ext_inh])
        ttL = torch.stack([vecsL[i] for i in ext_tech])
        wL = fit_direction(tiL, ttL, "zdom")
        params = load_file(str(SAE_DIRS[MODEL] / f"layer_{L}_width_16k_l0_medium" / "params.safetensors"))
        w_dec = params["w_dec"].to(torch.float32)
        dec_cos = (w_dec @ wL) / w_dec.norm(dim=1).clamp_min(1e-8)
        top = torch.topk(dec_cos.abs(), 12)
        print(f"\n--- L{L} register axis vs decoder rows (f1797 cos = {dec_cos[1797]:+.3f}) ---")
        for v, idx in zip(top.values.tolist(), top.indices.tolist()):
            exp = neuronpedia_explanation(np_model, L, idx, cache)
            print(f"  f{idx:<6} cos={dec_cos[idx]:+.3f}  {exp}")

    out = cap_dir / "register_axis_results.json"
    out.write_text(json.dumps({"groups": {"ext_inh": ext_inh, "ext_tech": ext_tech},
                               "layers": summary, "best_layer": best,
                               "mixed_traces": traces}, indent=1))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
