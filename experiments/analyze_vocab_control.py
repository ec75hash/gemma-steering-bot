#!/usr/bin/env python3
"""Summarize results/vocab_control.jsonl.

Decision rule: the dim result is interesting iff `dim` differs from BOTH
`instructed` (rules out vocabulary suppression) and `random` (rules out
generic perturbation).

Usage:
  python3 experiments/analyze_vocab_control.py            # table only
  python3 experiments/analyze_vocab_control.py --texts    # also dump responses
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results" / "vocab_control.jsonl"

PREFERRED_ORDER = ["baseline", "dim", "dim06", "dim04", "instructed", "random"]

ap = argparse.ArgumentParser()
ap.add_argument("--texts", action="store_true", help="print full responses grouped by condition")
args = ap.parse_args()

records = [json.loads(line) for line in IN.read_text().splitlines()]
if not records:
    raise SystemExit(f"no records in {IN}")

seen = {r["condition"] for r in records}
CONDITIONS = [c for c in PREFERRED_ORDER if c in seen] + sorted(seen - set(PREFERRED_ORDER))

prompt_ids = sorted({r["prompt_id"] for r in records})
cells = defaultdict(list)  # (condition, prompt_id) -> records
for r in records:
    cells[(r["condition"], r["prompt_id"])].append(r)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


for metric in ("disclaimer_hits", "positive_self_hits"):
    print(f"\n=== mean {metric} (condition x prompt) ===")
    header = f"{'condition':12}" + "".join(f"{p:>15}" for p in prompt_ids) + f"{'ALL':>10}"
    print(header)
    for cond in CONDITIONS:
        row = f"{cond:12}"
        all_vals = []
        for p in prompt_ids:
            vals = [r[metric] for r in cells.get((cond, p), [])]
            all_vals += vals
            row += f"{mean(vals):>15.2f}" if vals else f"{'-':>15}"
        row += f"{mean(all_vals):>10.2f}" if all_vals else f"{'-':>10}"
        print(row)

n_by_cond = defaultdict(int)
for r in records:
    n_by_cond[r["condition"]] += 1
print(f"\nn per condition: {dict(n_by_cond)}")
print("\nInteresting iff dim differs from BOTH instructed and random.")

if args.texts:
    for cond in CONDITIONS:
        print(f"\n{'=' * 70}\n### {cond}\n{'=' * 70}")
        for p in prompt_ids:
            for r in sorted(cells.get((cond, p), []), key=lambda r: r["seed"]):
                print(f"\n--- {p} seed={r['seed']} ---")
                print(r["response"].strip())
