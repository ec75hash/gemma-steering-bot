#!/usr/bin/env python3
"""Move #1, second half: does the COMPUTED register axis (after subtracting the
earliest rep v_0) still load on the same interpretable self-features as the raw
axis? If the ladder (e.g. L17:f1797 self-description) survives the v_0 subtraction,
the computed signal is carried by the self-feature, not diffuse vocabulary.

GPU-free: banked residuals + SAE decoder weight reads + offline atlas labels.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from looking_glass_probe import (  # noqa: E402
    MODEL_DIRS, OUT_ROOT, fit_direction, load_prompt_meta, prompt_vectors,
)
from transformers import AutoTokenizer  # noqa: E402

MODEL = "it"
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
ATLAS = ROOT / "scratch" / "np_play" / "atlas"


def labels_for(L):
    p = ATLAS / f"L{L}_meta.json"
    return json.load(open(p)) if p.exists() else {}


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

    v0 = prompt_vectors(records, 0, special_ids)[0]

    for L in (9, 17, 22, 29):
        vL = prompt_vectors(records, L, special_ids)[0]
        ti = torch.stack([vL[i] for i in ext_inh]); tt = torch.stack([vL[i] for i in ext_tech])
        w_base = fit_direction(ti, tt, "zdom")
        tiA = torch.stack([vL[i] - v0[i] for i in ext_inh]); ttA = torch.stack([vL[i] - v0[i] for i in ext_tech])
        w_A = fit_direction(tiA, ttA, "zdom")
        cos_base_A = float(w_base @ w_A)

        w_dec = load_file(str(SAE_DIR / f"layer_{L}_width_16k_l0_medium" / "params.safetensors"))["w_dec"].to(torch.float32)
        norms = w_dec.norm(dim=1).clamp_min(1e-8)
        cos_base = (w_dec @ w_base) / norms
        cos_A = (w_dec @ w_A) / norms
        lab = labels_for(L)

        def top(cos, k=8):
            t = torch.topk(cos.abs(), k)
            return [(int(i), float(cos[i])) for i in t.indices.tolist()]

        print(f"\n===== L{L}  (cos(base_axis, testA_axis) = {cos_base_A:+.3f}) =====")
        print(" BASELINE axis top decoder features:")
        for idx, c in top(cos_base):
            print(f"   f{idx:<6} cos={c:+.3f}  «{(lab.get(str(idx),{}).get('label') or '')[:60]}»")
        print(" TEST-A (v_L - v0) axis top decoder features:")
        for idx, c in top(cos_A):
            print(f"   f{idx:<6} cos={c:+.3f}  «{(lab.get(str(idx),{}).get('label') or '')[:60]}»")
        if L == 17:
            print(f" f1797 (self-description): base cos={float(cos_base[1797]):+.3f}  testA cos={float(cos_A[1797]):+.3f}")


if __name__ == "__main__":
    main()
