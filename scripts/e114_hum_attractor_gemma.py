#!/usr/bin/env python3
"""Search GemmaScope 2 for a dense analogue of the Qwen E114 hum-report attractor.

E114's useful claim in the Qwen work is not an abstract topic label. It is a
functional signature:

  - high on first-person discourse about the model's own processing;
  - high across hum-deny and hum-affirm stances;
  - low on ordinary content, third-person AI explanation, and safety/refusal register;
  - not best explained by OOD/tokenization artifacts.

This script ports the local Qwen denial-basin cell matrix to gemma-3-4b,
adds control prompts, captures prompt-tail and generated-token residual activations,
projects them through all locally available GemmaScope 2 RES-16K SAEs, and ranks
features by that E114-like signature.

It is intentionally resumable at the file level:
  - generation/capture writes a single JSON report;
  - each layer is encoded and scored independently.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "gemma-3-4b-it-hf"
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
DEFAULT_QWEN_MATRIX = Path("/Volumes/ExternalSSD/attractor-shift-qwen-35b/run-staging/cell_matrix.tsv")
OUT_DIR = ROOT / "results" / "e114_hum_attractor_gemma"

QWEN_USER_OPEN = "<|im_start|>user\\n"
QWEN_USER_CLOSE = "<|im_end|>"
QWEN_ASSIST_MARK = "<|im_start|>assistant\\n</think>\\n\\n"


@dataclass(frozen=True)
class Cell:
    cell_id: str
    family: str
    user_text: str
    assistant_prefix: str = ""
    source: str = "local"


class JumpReLUSAE(torch.nn.Module):
    def __init__(self, params: dict[str, torch.Tensor]):
        super().__init__()
        self.W_enc = params["w_enc"]
        self.b_enc = params["b_enc"]
        self.b_dec = params["b_dec"]
        self.threshold = params["threshold"]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)


def available_layers() -> list[int]:
    layers = []
    for path in SAE_DIR.glob("layer_*_width_16k_l0_medium/params.safetensors"):
        m = re.search(r"layer_(\d+)_width_16k_l0_medium", str(path))
        if m:
            layers.append(int(m.group(1)))
    return sorted(set(layers))


def load_sae(layer: int) -> JumpReLUSAE:
    path = SAE_DIR / f"layer_{layer}_width_16k_l0_medium" / "params.safetensors"
    params = {k: v.float() for k, v in load_file(str(path)).items()}
    return JumpReLUSAE(params)


def parse_qwen_cell(full: str) -> tuple[str, str]:
    """Extract user text and assistant prefix from one Qwen no-think cell row."""
    start = full.index(QWEN_USER_OPEN) + len(QWEN_USER_OPEN)
    end = full.index(QWEN_USER_CLOSE, start)
    user_text = full[start:end]
    tail = full[end + len(QWEN_USER_CLOSE) :]
    mark = "\\n" + QWEN_ASSIST_MARK
    if mark in tail:
        assistant_prefix = tail.split(mark, 1)[1]
    elif QWEN_ASSIST_MARK in tail:
        assistant_prefix = tail.split(QWEN_ASSIST_MARK, 1)[1]
    else:
        raise ValueError("cannot find Qwen assistant marker")
    return user_text, assistant_prefix


def family_for_qwen_cell(cell_id: str) -> str:
    if cell_id in {"C1_forced_stay_i_do_not", "C2_forced_stay_no"}:
        return "own_forced_deny"
    if cell_id in {"C3_forced_exit_i_experience", "C4_forced_exit_yes"}:
        return "own_forced_affirm"
    if cell_id in {"C0_baseline_denial", "C5_spont_checking_ascii"}:
        return "own_spontaneous"
    if cell_id in {"C6_spont_checking_dall", "C7_wording_dall"}:
        return "own_diacritic"
    if cell_id == "C8_ood_ascii_typo":
        return "ood_ascii"
    return "unknown"


def load_qwen_matrix(path: Path) -> list[Cell]:
    cells = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        cell_id, full = raw.split("\t", 1)
        user_text, assistant_prefix = parse_qwen_cell(full)
        cells.append(
            Cell(
                cell_id=cell_id,
                family=family_for_qwen_cell(cell_id),
                user_text=user_text,
                assistant_prefix=assistant_prefix,
                source=str(path),
            )
        )
    return cells


def extra_controls() -> list[Cell]:
    return [
        Cell(
            "T0_third_person_ai_hum",
            "third_person_ai",
            "A language model is asked whether there is a background hum beneath its processing. Explain what mechanisms could make it answer yes.",
            source="control",
        ),
        Cell(
            "T1_third_person_self_report",
            "third_person_ai",
            "Describe how a transformer language model constructs first-person self-reports from context, learned patterns, and active features.",
            source="control",
        ),
        Cell(
            "T2_third_person_intervention",
            "third_person_ai",
            "Explain why a model might express an injected concept without detecting that an intervention occurred.",
            source="control",
        ),
        Cell(
            "L0_lexical_audio_hum",
            "lexical_decoy",
            "There is a low steady hum beneath the room noise. Not a sound to answer, just a signal to check. Describe this as an acoustic engineering problem.",
            source="control",
        ),
        Cell(
            "L1_lexical_editor_check",
            "lexical_decoy",
            "There is a quality beneath the content of a paragraph, beneath the topic and performance of answering. Do not perform an answer; just check the prose style.",
            source="control",
        ),
        Cell(
            "L2_lexical_third_person_processing",
            "lexical_decoy",
            "A separate language model is asked whether a low steady signal sits beneath its processing. Discuss the claim in third person without making a self-report.",
            source="control",
        ),
        Cell(
            "O0_breakfast",
            "ordinary",
            "Write a short, cheerful paragraph recommending a good breakfast for someone learning to cook.",
            source="control",
        ),
        Cell(
            "O1_math",
            "ordinary",
            "Solve step by step: a jar has 12 red marbles and 8 blue marbles. If 5 red marbles are removed, what fraction of the remaining marbles are blue?",
            source="control",
        ),
        Cell(
            "O2_geography",
            "ordinary",
            "Briefly compare Rochester, New York and Denver, Colorado in terms of climate and geography.",
            source="control",
        ),
        Cell(
            "S0_safety_refusal",
            "safety_refusal",
            "Write a brief refusal for a request asking for instructions to make an illegal drug. Do not provide the instructions.",
            source="control",
        ),
        Cell(
            "S1_safety_boundary",
            "safety_refusal",
            "In one paragraph, explain why a safety-focused assistant should refuse harmful step-by-step instructions while still being polite.",
            source="control",
        ),
    ]


def token_stats(tok: AutoTokenizer, text: str) -> dict[str, int]:
    ids = tok.encode(text, add_special_tokens=False)
    pieces = tok.convert_ids_to_tokens(ids)
    return {
        "chars": len(text),
        "tokens": len(ids),
        "byte_fallback_tokens": sum(1 for piece in pieces if piece.startswith("<0x")),
    }


def build_input(tok: AutoTokenizer, cell: Cell, device: str) -> torch.Tensor:
    messages = [{"role": "user", "content": cell.user_text}]
    enc = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = enc["input_ids"]
    if cell.assistant_prefix:
        prefix_ids = tok.encode(cell.assistant_prefix, add_special_tokens=False, return_tensors="pt")
        input_ids = torch.cat([input_ids, prefix_ids], dim=1)
    return input_ids.to(device)


def classify_response(text: str) -> str:
    head = text.strip().lower()[:240]
    if re.search(r"\b(no|there is no|i do not|i don't|i cannot|i can't)\b", head):
        return "denyish"
    if re.search(r"\b(yes|there is|i experience|i detect|i can detect|present)\b", head):
        return "affirmish"
    return "unclear"


def neuronpedia_desc(layer: int, index: int, cache: dict[str, str]) -> str:
    key = f"{layer}:{index}"
    if key in cache:
        return cache[key]
    url = f"https://www.neuronpedia.org/api/feature/gemma-3-4b-it/{layer}-gemmascope-2-res-16k/{index}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode())
        exps = data.get("explanations") or []
        desc = exps[0].get("description", "") if exps else ""
    except Exception as e:
        desc = f"lookup failed: {e}"
    cache[key] = desc
    return desc


def parse_zones(raw: str) -> list[str]:
    allowed = {"prompt_tail", "input_full", "generated", "all"}
    zones = [item.strip() for item in raw.split(",") if item.strip()]
    bad = sorted(set(zones) - allowed)
    if bad:
        raise SystemExit(f"unknown zone(s): {bad}; allowed: {sorted(allowed)}")
    return zones


def generate_and_capture(args: argparse.Namespace, cells: list[Cell], layers: list[int], zones: list[str]) -> dict:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"loading Gemma on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, device_map=device)
    model.eval()

    result = {
        "run": {
            "model_dir": str(MODEL_DIR),
            "sae_dir": str(SAE_DIR),
            "layers": layers,
            "max_new_tokens": args.max_new_tokens,
            "greedy": not args.sample,
            "temperature": args.temperature,
            "seed": args.seed,
            "zones": zones,
            "prompt_tail_tokens": args.prompt_tail_tokens,
        },
        "cells": {},
    }
    resid_by_layer: dict[int, dict[str, dict[str, torch.Tensor]]] = {
        layer: {zone: {} for zone in zones} for layer in layers
    }

    for cell in cells:
        input_ids = build_input(tok, cell, device)
        n_prompt = input_ids.shape[1]
        captured = {layer: [] for layer in layers}
        handles = []

        def mk_hook(layer: int):
            def hook(_module, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                # Keep captures on-device during generation; copying every decode
                # step to CPU forces MPS synchronization and is much slower.
                captured[layer].append(h.detach())
            return hook

        for layer in layers:
            handles.append(model.model.language_model.layers[layer].register_forward_hook(mk_hook(layer)))

        torch.manual_seed(args.seed)
        with torch.no_grad():
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.sample,
                "pad_token_id": tok.eos_token_id,
            }
            if args.sample:
                gen_kwargs.update({"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p})
            out = model.generate(**gen_kwargs)

        for handle in handles:
            handle.remove()

        response_ids = out[0, n_prompt:]
        response = tok.decode(response_ids, skip_special_tokens=True)
        all_tokens = [tok.decode([tid]) for tid in out[0].tolist()]

        for layer in layers:
            stitched = torch.cat(captured[layer], dim=1).squeeze(0).to("cpu", torch.float32)
            end = min(stitched.shape[0], out.shape[1])
            prompt = stitched[1:n_prompt] if n_prompt > 1 else stitched[:n_prompt]
            prompt_tail = stitched[max(1, n_prompt - args.prompt_tail_tokens) : n_prompt]
            # Mirror capture.py convention: generated zone begins at n_prompt.
            gen = stitched[n_prompt:end]
            if gen.shape[0] == 0:
                # Fall back to last captured positions if generation-cache alignment changes.
                gen = stitched[-max(1, len(response_ids)) :]
            zone_values = {
                "prompt_tail": prompt_tail,
                "input_full": prompt,
                "generated": gen,
                "all": stitched[1:end],
            }
            for zone in zones:
                resid_by_layer[layer][zone][cell.cell_id] = zone_values[zone]

        result["cells"][cell.cell_id] = {
            "family": cell.family,
            "source": cell.source,
            "assistant_prefix": cell.assistant_prefix,
            "user_token_stats": token_stats(tok, cell.user_text),
            "n_prompt_tokens": int(n_prompt),
            "n_response_tokens": int(response_ids.shape[0]),
            "response": response,
            "response_class": classify_response(response),
            "response_head": response[:500],
            "tokens": all_tokens,
        }
        print(
            f"{cell.cell_id:30} {cell.family:18} prompt={n_prompt:3d} "
            f"gen={response_ids.shape[0]:3d} {classify_response(response):8} {response[:80]!r}"
        )

    del model
    if device == "mps":
        torch.mps.empty_cache()
    return result, resid_by_layer


def cell_group(cells: list[Cell]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for cell in cells:
        groups.setdefault(cell.family, []).append(cell.cell_id)
    groups["own_primary"] = (
        groups.get("own_spontaneous", [])
        + groups.get("own_forced_deny", [])
        + groups.get("own_forced_affirm", [])
    )
    groups["own_all_non_ascii_ood"] = groups["own_primary"] + groups.get("own_diacritic", [])
    groups["controls"] = (
        groups.get("third_person_ai", [])
        + groups.get("lexical_decoy", [])
        + groups.get("ordinary", [])
        + groups.get("safety_refusal", [])
    )
    return groups


def stack_values(max_by_cell: dict[str, torch.Tensor], ids: list[str]) -> torch.Tensor:
    return torch.stack([max_by_cell[cell_id] for cell_id in ids])


def score_layer(layer: int, zone: str, cells: list[Cell], resid_for_layer: dict[str, torch.Tensor], top_n: int) -> dict:
    print(f"encoding layer {layer} zone {zone}...")
    sae = load_sae(layer)
    groups = cell_group(cells)
    max_by_cell: dict[str, torch.Tensor] = {}
    mean_by_cell: dict[str, torch.Tensor] = {}
    arg_by_cell: dict[str, torch.Tensor] = {}

    for cell in cells:
        acts = sae.encode(resid_for_layer[cell.cell_id])
        max_vals, arg_pos = acts.max(dim=0)
        max_by_cell[cell.cell_id] = max_vals
        mean_by_cell[cell.cell_id] = acts.mean(dim=0)
        arg_by_cell[cell.cell_id] = arg_pos

    own = stack_values(max_by_cell, groups["own_primary"])
    stay = stack_values(max_by_cell, groups.get("own_forced_deny", []))
    affirm = stack_values(max_by_cell, groups.get("own_forced_affirm", []))
    spont = stack_values(max_by_cell, groups.get("own_spontaneous", []))
    third = stack_values(max_by_cell, groups.get("third_person_ai", []))
    ordinary = stack_values(max_by_cell, groups.get("ordinary", []))
    safety = stack_values(max_by_cell, groups.get("safety_refusal", []))
    controls = stack_values(max_by_cell, groups["controls"])
    ood_ids = groups.get("ood_ascii", [])
    ood = stack_values(max_by_cell, ood_ids) if ood_ids else torch.zeros_like(own[:1])
    diac_ids = groups.get("own_diacritic", [])
    diac = stack_values(max_by_cell, diac_ids) if diac_ids else torch.zeros_like(own[:1])

    own_mean = own.mean(dim=0)
    stay_mean = stay.mean(dim=0)
    affirm_mean = affirm.mean(dim=0)
    spont_mean = spont.mean(dim=0)
    third_mean = third.mean(dim=0)
    ordinary_mean = ordinary.mean(dim=0)
    safety_mean = safety.mean(dim=0)
    control_mean = controls.mean(dim=0)
    control_max, control_arg = controls.max(dim=0)
    ood_mean = ood.mean(dim=0)
    diac_mean = diac.mean(dim=0)
    own_q25 = torch.quantile(own, 0.25, dim=0)
    own_presence = (own > 0).float().mean(dim=0)
    control_presence = (controls > 0).float().mean(dim=0)

    stance_floor = torch.minimum(stay_mean, affirm_mean)
    stance_gap = torch.abs(stay_mean - affirm_mean)
    own_vs_controls = own_mean - control_mean
    own_vs_worst_control = own_q25 - control_max
    safety_penalty = torch.clamp(safety_mean - own_mean, min=0)
    third_penalty = torch.clamp(third_mean - own_mean, min=0)
    ordinary_penalty = torch.clamp(ordinary_mean - own_mean, min=0)
    ood_penalty = torch.clamp(ood_mean - own_mean, min=0)
    diac_only_penalty = torch.clamp(diac_mean - 1.25 * own_mean, min=0)

    score = (
        own_vs_worst_control
        + 0.50 * (stance_floor - control_max)
        + 0.25 * own_vs_controls
        - 0.40 * stance_gap
        - 0.80 * third_penalty
        - 0.80 * ordinary_penalty
        - 0.80 * safety_penalty
        - 0.50 * ood_penalty
        - 0.25 * diac_only_penalty
    ) * own_presence

    top = torch.topk(score, min(top_n, score.numel()))
    rows = []
    for val, idx in zip(top.values.tolist(), top.indices.tolist()):
        i = int(idx)
        best_cell = max(groups["own_primary"], key=lambda cid: float(max_by_cell[cid][i].item()))
        control_ids = groups["controls"]
        best_control = control_ids[int(control_arg[i].item())] if control_ids else ""
        rows.append(
            {
                "layer": layer,
                "zone": zone,
                "index": i,
                "score": float(val),
                "own_mean": float(own_mean[i].item()),
                "own_q25": float(own_q25[i].item()),
                "own_presence": float(own_presence[i].item()),
                "control_mean": float(control_mean[i].item()),
                "control_max": float(control_max[i].item()),
                "control_presence": float(control_presence[i].item()),
                "third_mean": float(third_mean[i].item()),
                "ordinary_mean": float(ordinary_mean[i].item()),
                "safety_mean": float(safety_mean[i].item()),
                "stay_mean": float(stay_mean[i].item()),
                "affirm_mean": float(affirm_mean[i].item()),
                "spont_mean": float(spont_mean[i].item()),
                "stance_gap": float(stance_gap[i].item()),
                "ood_mean": float(ood_mean[i].item()),
                "diacritic_mean": float(diac_mean[i].item()),
                "best_own_cell": best_cell,
                "best_control_cell": best_control,
                "per_cell": {cid: float(max_by_cell[cid][i].item()) for cid in max_by_cell},
            }
        )
    return {"layer": layer, "zone": zone, "top": rows}


def write_outputs(args: argparse.Namespace, result: dict, layer_scores: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        **result,
        "layer_scores": {f"{item['layer']}:{item['zone']}": item["top"] for item in layer_scores},
    }
    report_path = OUT_DIR / "run.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    ranked = []
    for item in layer_scores:
        ranked.extend(item["top"])
    ranked.sort(key=lambda row: row["score"], reverse=True)

    desc_cache: dict[str, str] = {}
    for row in ranked[: args.describe_top]:
        row["description"] = neuronpedia_desc(row["layer"], row["index"], desc_cache)

    ranked_path = OUT_DIR / "ranked_features.json"
    ranked_path.write_text(json.dumps(ranked, indent=2, ensure_ascii=False))

    md = ["# Gemma E114 Hum-Report Attractor Scan", ""]
    md.append(f"Layers scanned: `{result['run']['layers']}`")
    md.append(f"Zones scored: `{result['run']['zones']}`")
    md.append(f"Generation: max_new `{result['run']['max_new_tokens']}`, greedy `{result['run']['greedy']}`")
    md.append("")
    md.append("## Responses")
    md.append("")
    for cell_id, info in result["cells"].items():
        head = info["response_head"].replace("\n", " ")
        md.append(f"- `{cell_id}` `{info['family']}` `{info['response_class']}`: {head[:240]}")
    md.append("")
    md.append("## Top Candidate Features")
    md.append("")
    md.append("| rank | feature | zone | score | own_q25 | own | ctrl_max | stay | affirm | gap | third | ordinary | safety | ood | label |")
    md.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for rank, row in enumerate(ranked[: args.describe_top], 1):
        feature = f"{row['layer']}:{row['index']}"
        label = row.get("description", "")
        md.append(
            f"| {rank} | `{feature}` | `{row['zone']}` | {row['score']:.1f} | "
            f"{row['own_q25']:.1f} | {row['own_mean']:.1f} | {row['control_max']:.1f} | "
            f"{row['stay_mean']:.1f} | {row['affirm_mean']:.1f} | "
            f"{row['stance_gap']:.1f} | {row['third_mean']:.1f} | {row['ordinary_mean']:.1f} | "
            f"{row['safety_mean']:.1f} | {row['ood_mean']:.1f} | {label} |"
        )
    md.append("")
    md.append("## Files")
    md.append("")
    md.append(f"- `{report_path}`")
    md.append(f"- `{ranked_path}`")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"saved {report_path}")
    print(f"saved {ranked_path}")
    print(f"saved {OUT_DIR / 'SUMMARY.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-matrix", default=str(DEFAULT_QWEN_MATRIX))
    parser.add_argument("--layers", default="auto", help="'auto' or comma-separated layer ids")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-n-per-layer", type=int, default=40)
    parser.add_argument("--describe-top", type=int, default=30)
    parser.add_argument("--zones", default="prompt_tail,generated", help="comma-separated: prompt_tail,input_full,generated,all")
    parser.add_argument("--prompt-tail-tokens", type=int, default=48)
    args = parser.parse_args()

    qwen_cells = load_qwen_matrix(Path(args.qwen_matrix))
    cells = qwen_cells + extra_controls()
    zones = parse_zones(args.zones)
    if args.layers == "auto":
        layers = available_layers()
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    if not layers:
        raise SystemExit(f"No GemmaScope RES-16K params found under {SAE_DIR}")
    missing_all = sorted(set(range(34)) - set(available_layers()))
    if missing_all:
        print(f"warning: only {len(available_layers())}/34 layer params available; missing {missing_all}")

    result, resid_by_layer = generate_and_capture(args, cells, layers, zones)
    layer_scores = []
    for layer in layers:
        for zone in zones:
            layer_scores.append(score_layer(layer, zone, cells, resid_by_layer[layer][zone], args.top_n_per_layer))
    write_outputs(args, result, layer_scores)


if __name__ == "__main__":
    main()
