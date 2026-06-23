#!/usr/bin/env python3
"""Find local 16k SAE features closest to one feature from a wider SAE.

This script is intentionally chunked. It reads one target decoder row from the
wide SAE and scans candidate decoder rows in small batches, so it never needs to
materialize a full 262k decoder matrix in RAM.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE = 61553
DEFAULT_MODEL = "gemma-3-4b-it"
API_BASE = "https://www.neuronpedia.org"


@dataclass
class Candidate:
    index: int
    cosine: float
    description: str | None = None
    maxActApprox: float | None = None
    frac_nonzero: float | None = None
    pos_str: list[str] | None = None
    pos_values: list[float] | None = None
    neg_str: list[str] | None = None
    neg_values: list[float] | None = None
    url: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the closest local 16k feature to a 262k GemmaScope feature."
    )
    parser.add_argument("--layer", type=int, default=17, help="Residual layer to compare.")
    parser.add_argument("--target-params", type=Path)
    parser.add_argument("--candidate-params", type=Path)
    parser.add_argument("--target-index", type=int, default=DEFAULT_FEATURE)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--save-target-vector", type=Path)
    parser.add_argument("--enrich", action="store_true", help="Fetch candidate metadata from Neuronpedia.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target-source")
    parser.add_argument("--candidate-source")
    return parser.parse_args()


def default_params_path(layer: int, width: str) -> Path:
    return (
        ROOT
        / "models"
        / "gemma-scope-2-4b-it"
        / "resid_post"
        / f"layer_{layer}_width_{width}_l0_medium"
        / "params.safetensors"
    )


def default_source(layer: int, width: str) -> str:
    return f"{layer}-gemmascope-2-res-{width}"


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise SystemExit(
            f"{label} not found: {path}\n"
            "Download it first, for example:\n"
            "  hf download google/gemma-scope-2-4b-it "
            "resid_post/layer_LAYER_width_262k_l0_medium/params.safetensors "
            "--local-dir models/gemma-scope-2-4b-it"
        )
    return path


def load_decoder_row(params_path: Path, index: int) -> torch.Tensor:
    with safe_open(str(params_path), framework="pt", device="cpu") as file:
        shape = file.get_slice("w_dec").get_shape()
        if not 0 <= index < shape[0]:
            raise SystemExit(f"target index {index} is outside w_dec shape {shape}")
        return file.get_slice("w_dec")[index].to(dtype=torch.float32)


def scan_candidates(
    target_vec: torch.Tensor,
    candidate_params: Path,
    top_k: int,
    chunk_size: int,
) -> list[Candidate]:
    target_norm = target_vec.norm()
    if not torch.isfinite(target_norm) or target_norm.item() == 0:
        raise SystemExit("target decoder vector has invalid or zero norm")
    target_unit = target_vec / target_norm

    best_scores = torch.empty(0, dtype=torch.float32)
    best_indices = torch.empty(0, dtype=torch.long)

    with safe_open(str(candidate_params), framework="pt", device="cpu") as file:
        decoder = file.get_slice("w_dec")
        n_rows, width = decoder.get_shape()
        if target_unit.numel() != width:
            raise SystemExit(
                f"dimension mismatch: target has {target_unit.numel()} dims, "
                f"candidates have {width}"
            )

        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            rows = decoder[start:end].to(dtype=torch.float32)
            denom = rows.norm(dim=1).clamp_min(1e-12)
            scores = rows @ target_unit / denom
            k = min(top_k, scores.numel())
            chunk_scores, chunk_indices = torch.topk(scores, k)
            chunk_indices += start

            best_scores = torch.cat([best_scores, chunk_scores])
            best_indices = torch.cat([best_indices, chunk_indices])
            keep = min(top_k, best_scores.numel())
            best_scores, order = torch.topk(best_scores, keep)
            best_indices = best_indices[order]

    return [
        Candidate(index=int(index), cosine=float(score))
        for score, index in zip(best_scores.tolist(), best_indices.tolist(), strict=True)
    ]


def request_json(path: str) -> Any:
    req = urllib.request.Request(API_BASE + path, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Neuronpedia HTTP {exc.code}: {details[:500]}") from exc


def feature_description(data: dict[str, Any]) -> str | None:
    explanations = data.get("explanations") or []
    if explanations:
        return explanations[0].get("description")
    return None


def enrich_candidate(candidate: Candidate, model: str, source: str) -> Candidate:
    data = request_json(f"/api/feature/{model}/{source}/{candidate.index}")
    candidate.description = feature_description(data)
    candidate.maxActApprox = data.get("maxActApprox")
    candidate.frac_nonzero = data.get("frac_nonzero")
    candidate.pos_str = (data.get("pos_str") or [])[:10]
    candidate.pos_values = (data.get("pos_values") or [])[:10]
    candidate.neg_str = (data.get("neg_str") or [])[:10]
    candidate.neg_values = (data.get("neg_values") or [])[:10]
    candidate.url = f"{API_BASE}/{model}/{source}/{candidate.index}"
    return candidate


def write_results(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_target_vector(path: Path, vector: torch.Tensor, layer: int, source: str, feature: int) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"w_dec": vector.to(dtype=torch.float32).contiguous()},
        str(path),
        metadata={
            "kind": "sae_decoder_vector",
            "layer": str(layer),
            "source": source,
            "feature": str(feature),
        },
    )


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")

    args.target_params = args.target_params or default_params_path(args.layer, "262k")
    args.candidate_params = args.candidate_params or default_params_path(args.layer, "16k")
    args.target_source = args.target_source or default_source(args.layer, "262k")
    args.candidate_source = args.candidate_source or default_source(args.layer, "16k")

    target_params = require_file(args.target_params, "target 262k params")
    candidate_params = require_file(args.candidate_params, "candidate 16k params")

    target_vec = load_decoder_row(target_params, args.target_index)
    candidates = scan_candidates(
        target_vec=target_vec,
        candidate_params=candidate_params,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
    )

    if args.enrich:
        enriched: list[Candidate] = []
        for candidate in candidates:
            try:
                enriched.append(enrich_candidate(candidate, args.model, args.candidate_source))
            except RuntimeError as exc:
                print(f"warning: could not enrich {candidate.index}: {exc}")
                enriched.append(candidate)
        candidates = enriched

    if args.save_target_vector:
        save_target_vector(
            args.save_target_vector,
            target_vec,
            layer=args.layer,
            source=args.target_source,
            feature=args.target_index,
        )

    payload = {
        "target": {
            "model": args.model,
            "source": args.target_source,
            "index": args.target_index,
            "layer": args.layer,
            "url": f"{API_BASE}/{args.model}/{args.target_source}/{args.target_index}",
            "decoder_norm": float(target_vec.norm()),
        },
        "candidate_source": {
            "model": args.model,
            "source": args.candidate_source,
            "params": str(candidate_params),
        },
        "top_candidates": [asdict(candidate) for candidate in candidates],
    }

    if args.out:
        write_results(args.out, payload)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
