#!/usr/bin/env python3
"""Compare GemmaScope RES-16K activations for LyAV/Lyell Avenue probes.

This is intentionally small and local:
  - tokenizes each probe string;
  - runs gemma-3-4b once per probe under the official chat template;
  - captures resid_post at the downloaded GemmaScope layers;
  - projects through RES-16K SAEs;
  - reports cosine/Jaccard convergence and the top layer-29 features.
"""
from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "gemma-3-4b-it-hf"
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
OUT = ROOT / "captures" / "lyav_ladder.json"
LAYERS = [9, 17, 22, 29]
TOP_K = 25
TARGETS = {
    9: {
        13221: "street addresses",
        8612: "streets and places",
        10032: "New York City",
    },
    17: {
        4952: "street and avenue names",
        6752: "address",
        11101: "street names or addresses",
        14881: "new york city and nj",
    },
    22: {
        7096: "address",
        13211: "New York, New Jersey",
        14835: "address",
        14867: "streets, roads, avenues",
        16374: "address or PO Box",
    },
    29: {
        328: "new york",
        6216: "New York City",
        6992: "New York City",
        11485: "New York locations",
    },
}

FL_SNIPPET = (
    "Muiż LyAV y liddw aġ rele riy mufyiţ (LyAV) eşķāh nyt nyse aġ nerw tikšiš. "
    "Dusekun fw ridw gubšyd ir weai, fouriţ, aġ rwfnyr genwd lyli sougyd LyAV."
)

PROBES = {
    "lyav_mixed": "LyAv",
    "lyav_upper": "LyAV",
    "lyell_ave": "Lyell Ave",
    "lyell_avenue": "Lyell Avenue",
    "lyell_rochester": "Lyell Avenue in Rochester, New York",
    "fl_lyav": FL_SNIPPET,
    "fl_lyell_replaced": FL_SNIPPET.replace("LyAV", "Lyell Avenue"),
}


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


def load_sae(layer: int) -> JumpReLUSAE:
    path = SAE_DIR / f"layer_{layer}_width_16k_l0_medium" / "params.safetensors"
    params = {k: v.float() for k, v in load_file(str(path)).items()}
    return JumpReLUSAE(params)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return 0.0
    return float((a @ b / denom).item())


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int]:
    for start in range(0, len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start, start + len(needle)
    raise ValueError("prompt token subsequence not found inside chat template")


def neuronpedia_desc(layer: int, index: int, cache: dict[str, str]) -> str:
    key = f"{layer}:{index}"
    if key in cache:
        return cache[key]
    url = f"https://www.neuronpedia.org/api/feature/gemma-3-4b-it/{layer}-gemmascope-2-res-16k/{index}"
    try:
        with urllib.request.urlopen(url, timeout=12) as r:
            data = json.loads(r.read().decode())
        exps = data.get("explanations") or []
        desc = exps[0].get("description", "") if exps else ""
    except Exception as e:
        desc = f"lookup failed: {e}"
    cache[key] = desc
    return desc


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"loading tokenizer/model on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, device_map=device)
    model.eval()

    print("\nTOKENIZATION")
    tokenization = {}
    for name, text in PROBES.items():
        ids = tok.encode(text, add_special_tokens=False)
        toks = tok.convert_ids_to_tokens(ids)
        tokenization[name] = {"ids": ids, "tokens": toks}
        preview = text if len(text) < 50 else text[:47] + "..."
        print(f"{name:18} {len(ids):3d} tokens  {preview!r}")
        if name.startswith("ly"):
            print(f"  {toks}")

    captures: dict[str, dict[int, torch.Tensor]] = {}
    token_texts: dict[str, list[str]] = {}
    content_spans: dict[str, tuple[int, int]] = {}

    for name, text in PROBES.items():
        print(f"\nrunning {name}...")
        raw_ids = tok.encode(text, add_special_tokens=False)
        enc = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = enc["input_ids"].to(device)
        chat_ids = input_ids[0].tolist()
        content_spans[name] = find_subsequence(chat_ids, raw_ids)
        captured = {L: [] for L in LAYERS}
        handles = []

        def mk_hook(layer: int):
            def hook(_module, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[layer].append(h.detach().cpu().float())
            return hook

        for L in LAYERS:
            handles.append(model.model.language_model.layers[L].register_forward_hook(mk_hook(L)))

        with torch.no_grad():
            model(input_ids)

        for h in handles:
            h.remove()

        captures[name] = {L: torch.cat(captured[L], dim=1).squeeze(0) for L in LAYERS}
        token_texts[name] = [tok.decode([tid]) for tid in chat_ids]
        start, end = content_spans[name]
        print(f"  chat tokens: {input_ids.shape[1]}, content span: {start}:{end}")

    del model
    if device == "mps":
        torch.mps.empty_cache()

    results = {
        "tokenization": tokenization,
        "probes": PROBES,
        "layers": {},
        "comparisons": {},
    }
    desc_cache: dict[str, str] = {}

    pairs = [
        ("lyav_mixed", "lyell_avenue"),
        ("lyav_upper", "lyell_avenue"),
        ("lyav_upper", "lyell_rochester"),
        ("lyell_avenue", "lyell_rochester"),
        ("fl_lyav", "fl_lyell_replaced"),
        ("fl_lyav", "lyell_rochester"),
    ]

    for L in LAYERS:
        print(f"\nencoding layer {L} SAE...")
        sae = load_sae(L)
        layer_vectors = {}
        layer_last_vectors = {}
        layer_tops = {}
        layer_last_tops = {}
        results["layers"][str(L)] = {}

        for name in PROBES:
            acts = sae.encode(captures[name][L])
            # Rank only the user-content tokens. Otherwise chat markers dominate.
            start, end = content_spans[name]
            content_acts = acts[start:end]
            max_vals, arg_pos = content_acts.max(dim=0)
            top = torch.topk(max_vals, TOP_K)
            vec = max_vals
            top_feats = []
            for val, idx in zip(top.values.tolist(), top.indices.tolist()):
                pos = int(start + arg_pos[idx].item())
                top_feats.append(
                    {
                        "index": int(idx),
                        "max": float(val),
                        "argmax_pos": pos,
                        "argmax_token": token_texts[name][pos],
                    }
                )
            last_vals = acts[end - 1]
            last_top = torch.topk(last_vals, TOP_K)
            last_feats = [
                {
                    "index": int(idx),
                    "max": float(val),
                    "argmax_pos": end - 1,
                    "argmax_token": token_texts[name][end - 1],
                }
                for val, idx in zip(last_top.values.tolist(), last_top.indices.tolist())
            ]
            target_rows = {}
            for index, desc in TARGETS.get(L, {}).items():
                val = float(max_vals[index].item())
                pos = int(start + arg_pos[index].item())
                rank = int((max_vals > max_vals[index]).sum().item() + 1)
                target_rows[str(index)] = {
                    "description": desc,
                    "max": val,
                    "rank": rank,
                    "argmax_pos": pos,
                    "argmax_token": token_texts[name][pos],
                    "last": float(last_vals[index].item()),
                }
            layer_vectors[name] = vec
            layer_last_vectors[name] = last_vals
            layer_tops[name] = {f["index"] for f in top_feats}
            layer_last_tops[name] = {f["index"] for f in last_feats}
            results["layers"][str(L)][name] = {
                "top": top_feats,
                "last_top": last_feats,
                "targets": target_rows,
            }

        comp_rows = []
        for a, b in pairs:
            row = {
                "a": a,
                "b": b,
                "max_cosine": cosine(layer_vectors[a], layer_vectors[b]),
                "max_top_jaccard": jaccard(layer_tops[a], layer_tops[b]),
                "top_overlap": sorted(layer_tops[a] & layer_tops[b]),
                "last_cosine": cosine(layer_last_vectors[a], layer_last_vectors[b]),
                "last_top_jaccard": jaccard(layer_last_tops[a], layer_last_tops[b]),
                "last_top_overlap": sorted(layer_last_tops[a] & layer_last_tops[b]),
            }
            comp_rows.append(row)
        results["comparisons"][str(L)] = comp_rows

    # Add descriptions only for layer 29 top-10; this keeps API traffic small.
    for name, info in results["layers"]["29"].items():
        for feat in info["top"][:10]:
            feat["description"] = neuronpedia_desc(29, feat["index"], desc_cache)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\nCONVERGENCE SUMMARY")
    for L in LAYERS:
        print(f"\nLayer {L}")
        for row in results["comparisons"][str(L)]:
            print(
                f"  {row['a']:18} vs {row['b']:18} "
                f"maxCos={row['max_cosine']:.3f} maxJ={row['max_top_jaccard']:.2f} "
                f"lastCos={row['last_cosine']:.3f} lastJ={row['last_top_jaccard']:.2f}"
            )

    print("\nLAYER 29 TOP FEATURES")
    for name in PROBES:
        print(f"\n{name}")
        for feat in results["layers"]["29"][name]["top"][:8]:
            desc = feat.get("description", "")
            print(
                f"  f{feat['index']:<5} max={feat['max']:8.1f} "
                f"@ {feat['argmax_token']!r:>12}  {desc}"
            )

    print("\nTARGET FEATURE ACTIVATIONS")
    for L in LAYERS:
        print(f"\nLayer {L}")
        for index, desc in TARGETS.get(L, {}).items():
            print(f"  f{index} {desc}")
            for name in PROBES:
                row = results["layers"][str(L)][name]["targets"][str(index)]
                print(
                    f"    {name:18} max={row['max']:8.1f} "
                    f"rank={row['rank']:5d} @ {row['argmax_token']!r}"
                )

    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
