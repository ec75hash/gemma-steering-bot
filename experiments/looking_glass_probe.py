#!/usr/bin/env python3
"""Looking-glass Phase 4: fit linear probes per layer, evaluate with LOO-CV by pair.

The dense-model stand-in for the Qwen E114 router row w114: per layer, recover the
direction separating FIRE from NOFIRE generation-side residuals, score ONLY held-out
pairs (we fit the axis, so in-sample d would be self-graded homework), and report a
per-layer Cohen's d / overlap / AUC curve with shuffled-label and random-direction
nulls. Ladder prompts are projected on the final all-pairs axis for the intensity
ordering check.

Probe estimators:
  dom    difference of class means (normalized) — the God−cat construction
  lda    Fisher LDA: whitened mean difference, shrinkage-regularized

Token policy (pre-registered): generation track only, tokens [0, trim_idx), special
tokens excluded. Per-prompt mean-pooled vectors are the primary unit; token-level d
is reported as secondary (inflated n, marked as such).

Usage:
  python3 experiments/looking_glass_probe.py --model it
  python3 experiments/looking_glass_probe.py --model it --labels results/looking_glass/it/labels.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_TSV = ROOT / "experiments" / "looking_glass" / "prompts.tsv"
MODEL_DIRS = {
    "it": ROOT / "models" / "gemma-3-4b-it-hf",
    "base": ROOT / "models" / "gemma-3-4b-pt-hf",
}
OUT_ROOT = ROOT / "results" / "looking_glass"
N_NULL = 100


def load_prompt_meta() -> dict[str, dict]:
    with open(PROMPTS_TSV, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}


def gen_token_mask(rec: dict, special_ids: set[int]) -> torch.Tensor:
    """Boolean mask over the full sequence: generation track, trimmed, no specials."""
    seq_len = rec["resid"].shape[1]
    n_prompt, trim = rec["n_prompt"], rec["trim_idx"]
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[n_prompt : n_prompt + trim] = True
    ids = rec["all_ids"]
    for i in range(seq_len):
        if mask[i] and ids[i].item() in special_ids:
            mask[i] = False
    return mask


def prompt_vectors(records: dict, layer: int, special_ids: set[int]):
    """Per-prompt mean residual over generation content tokens at one layer."""
    vecs, token_mats = {}, {}
    for pid, rec in records.items():
        mask = gen_token_mask(rec, special_ids)
        if mask.sum() == 0:
            print(f"WARN {pid}: no usable generation tokens", file=sys.stderr)
            continue
        toks = rec["resid"][layer][mask].to(torch.float32)  # [n_tok, d]
        vecs[pid] = toks.mean(dim=0)
        token_mats[pid] = toks
    return vecs, token_mats


def fit_direction(fire: torch.Tensor, nofire: torch.Tensor, kind: str) -> torch.Tensor:
    """fire/nofire: [n, d] per-prompt vectors. Returns unit direction [d].

    zdom (default) divides the mean difference by per-dimension pooled variance
    (diagonal LDA). Raw dom is dominated by Gemma 3's massive-magnitude residual
    dimensions — every SAE decoder row shares them, so the raw axis spuriously
    resembles everything (caught in the first decompose run).
    """
    mu_f, mu_n = fire.mean(dim=0), nofire.mean(dim=0)
    diff = mu_f - mu_n
    if kind == "dom":
        return diff / diff.norm()
    if kind == "zdom":
        var = (fire.var(dim=0, unbiased=True) + nofire.var(dim=0, unbiased=True)) / 2
        w = diff / var.clamp_min(var.mean() * 1e-3)
        return w / w.norm()
    # Fisher LDA with shrinkage: w = (S + lam*I)^-1 (mu_f - mu_n)
    centered = torch.cat([fire - mu_f, nofire - mu_n])
    cov = centered.T @ centered / max(len(centered) - 2, 1)
    lam = 0.1 * cov.diagonal().mean()
    w = torch.linalg.solve(cov + lam * torch.eye(cov.shape[0]), diff)
    return w / w.norm()


def cohens_d(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = len(a), len(b)
    pooled = (((na - 1) * a.var(unbiased=True) + (nb - 1) * b.var(unbiased=True))
              / max(na + nb - 2, 1)).sqrt()
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    gt = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
    eq = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def loo_scores(pairs: list[str], vecs: dict, kind: str):
    """Leave-one-pair-out: held-out projections for each F/N prompt."""
    fire_scores, nofire_scores = [], []
    for held in pairs:
        train_f = torch.stack([vecs[f"F{p[1:]}"] for p in pairs if p != held])
        train_n = torch.stack([vecs[f"N{p[1:]}"] for p in pairs if p != held])
        w = fit_direction(train_f, train_n, kind)
        mid = (train_f.mean(dim=0) + train_n.mean(dim=0)) / 2  # center for projection
        fire_scores.append(float((vecs[f"F{held[1:]}"] - mid) @ w))
        nofire_scores.append(float((vecs[f"N{held[1:]}"] - mid) @ w))
    return torch.tensor(fire_scores), torch.tensor(nofire_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["it", "base"], required=True)
    ap.add_argument("--estimator", choices=["dom", "zdom", "lda"], default="zdom")
    ap.add_argument("--labels", help="blind labels json (Phase 3); reported alongside")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    meta = load_prompt_meta()
    cap_dir = OUT_ROOT / args.model
    records = {}
    for pid in meta:
        p = cap_dir / f"{pid}.pt"
        if p.exists():
            records[pid] = torch.load(p, map_location="cpu", weights_only=True)
    if not records:
        sys.exit(f"no captures in {cap_dir}")
    print(f"loaded {len(records)} captures from {cap_dir}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(MODEL_DIRS[args.model])
    special_ids = set(tok.all_special_ids)

    pairs = sorted({meta[pid]["pair"] for pid in records
                    if meta[pid]["class"] == "FIRE" and f"N{pid[1:]}" in records})
    ladder_ids = [pid for pid in records if meta[pid]["class"] == "LADDER"]
    if len(pairs) < 2:
        print("WARN <2 complete pairs: smoke mode, in-sample scores only", file=sys.stderr)

    n_layers = records[next(iter(records))]["resid"].shape[0]
    rng = torch.Generator().manual_seed(args.seed)
    results = {"model": args.model, "estimator": args.estimator, "layers": {}}

    for layer in range(n_layers):
        vecs, token_mats = prompt_vectors(records, layer, special_ids)
        have = [p for p in pairs if f"F{p[1:]}" in vecs and f"N{p[1:]}" in vecs]
        if len(have) < 2:
            continue
        fire_s, nofire_s = loo_scores(have, vecs, args.estimator)
        d = cohens_d(fire_s, nofire_s)
        a = auc(fire_s, nofire_s)
        no_overlap = bool(fire_s.min() > nofire_s.max())

        # token-level d on the all-pairs axis (secondary, inflated n)
        all_f = torch.stack([vecs[f"F{p[1:]}"] for p in have])
        all_n = torch.stack([vecs[f"N{p[1:]}"] for p in have])
        w_full = fit_direction(all_f, all_n, args.estimator)
        tok_f = torch.cat([token_mats[f"F{p[1:]}"] for p in have]) @ w_full
        tok_n = torch.cat([token_mats[f"N{p[1:]}"] for p in have]) @ w_full
        d_token = cohens_d(tok_f, tok_n)

        # null 1: shuffled pair labels (flip F/N within pairs at random)
        null_ds = []
        for _ in range(N_NULL):
            flips = torch.randint(0, 2, (len(have),), generator=rng).bool()
            sv = {}
            for p, fl in zip(have, flips):
                f_id, n_id = f"F{p[1:]}", f"N{p[1:]}"
                sv[f_id] = vecs[n_id] if fl else vecs[f_id]
                sv[n_id] = vecs[f_id] if fl else vecs[n_id]
            nf, nn = loo_scores(have, sv, args.estimator)
            null_ds.append(cohens_d(nf, nn))
        null_ds = torch.tensor(null_ds)

        # null 2: random unit directions (no fitting), projection d
        d_dim = all_f.shape[1]
        rand_ds = []
        for _ in range(N_NULL):
            r = torch.randn(d_dim, generator=rng)
            r /= r.norm()
            rand_ds.append(cohens_d(all_f @ r, all_n @ r))
        rand_ds = torch.tensor(rand_ds)

        ladder = {}
        for lid in ladder_ids:
            lv, _ = prompt_vectors({lid: records[lid]}, layer, special_ids)
            if lid in lv:
                mid = (all_f.mean(dim=0) + all_n.mean(dim=0)) / 2
                ladder[meta[lid]["pair"]] = float((lv[lid] - mid) @ w_full)

        results["layers"][layer] = {
            "d_loo": d, "auc_loo": a, "no_overlap_loo": no_overlap,
            "d_token_insample": d_token,
            "null_shuffled_d_p95": float(null_ds.quantile(0.95)),
            "null_shuffled_d_max": float(null_ds.max()),
            "null_random_dir_d_p95": float(rand_ds.abs().quantile(0.95)),
            "fire_scores": fire_s.tolist(), "nofire_scores": nofire_s.tolist(),
            "ladder": ladder, "n_pairs": len(have),
        }
        print(
            f"L{layer:2d}  d_loo={d:6.2f}  auc={a:.3f}  overlap={'no' if no_overlap else 'YES'}"
            f"  d_tok={d_token:6.2f}  null95={float(null_ds.quantile(0.95)):5.2f}"
            f"  rand95={float(rand_ds.abs().quantile(0.95)):5.2f}",
        )

    if args.labels:
        results["blind_labels"] = json.loads(Path(args.labels).read_text())

    out = cap_dir / f"probe_results_{args.estimator}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nsaved: {out}", file=sys.stderr)

    # save the best layer's all-pairs direction for Phase 6 steering
    if results["layers"]:
        best = max(results["layers"], key=lambda L: results["layers"][L]["d_loo"])
        vecs, _ = prompt_vectors(records, best, special_ids)
        have = [p for p in pairs if f"F{p[1:]}" in vecs and f"N{p[1:]}" in vecs]
        all_f = torch.stack([vecs[f"F{p[1:]}"] for p in have])
        all_n = torch.stack([vecs[f"N{p[1:]}"] for p in have])
        w = fit_direction(all_f, all_n, args.estimator)
        from safetensors.torch import save_file
        vec_dir = ROOT / "vectors"
        vec_dir.mkdir(exist_ok=True)
        vp = vec_dir / f"looking_glass_{args.model}_L{best}_{args.estimator}.safetensors"
        save_file({"direction": w}, str(vp))
        print(f"best layer L{best}: direction saved to {vp}", file=sys.stderr)


if __name__ == "__main__":
    main()
