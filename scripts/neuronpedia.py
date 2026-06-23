#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "https://www.neuronpedia.org"
DEFAULT_MODEL = "gemma-3-4b-it"
DEFAULT_RES16K_LAYERS = [
    "9-gemmascope-2-res-16k",
    "17-gemmascope-2-res-16k",
    "22-gemmascope-2-res-16k",
    "29-gemmascope-2-res-16k",
]


def load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def api_key() -> str | None:
    load_env()
    return os.environ.get("NEURONPEDIA_API_KEY")


def request_json(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
    }
    key = api_key()
    if key:
        headers["x-api-key"] = key
    req = urllib.request.Request(
        API_BASE + path,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Neuronpedia HTTP {exc.code}: {details[:1000]}") from exc


def compact_feature(row: dict[str, Any], include_url: bool = True) -> dict[str, Any]:
    neuron = row.get("neuron") or row
    model_id = row.get("modelId") or neuron.get("modelId")
    layer = row.get("layer") or neuron.get("layer")
    index = str(row.get("index") or neuron.get("index"))
    explanations = neuron.get("explanations") or row.get("explanations") or []
    description = row.get("description")
    if not description and explanations:
        description = explanations[0].get("description")
    compact = {
        "modelId": model_id,
        "layer": layer,
        "index": index,
        "description": description or "",
        "typeName": row.get("typeName") or (explanations[0].get("typeName") if explanations else ""),
        "explanationModelName": row.get("explanationModelName")
        or (explanations[0].get("explanationModelName") if explanations else ""),
        "cosine_similarity": row.get("cosine_similarity"),
        "maxActApprox": neuron.get("maxActApprox"),
        "frac_nonzero": neuron.get("frac_nonzero"),
        "hookName": neuron.get("hookName"),
        "pos_str": neuron.get("pos_str", [])[:10],
        "pos_values": neuron.get("pos_values", [])[:10],
        "neg_str": neuron.get("neg_str", [])[:10],
        "neg_values": neuron.get("neg_values", [])[:10],
    }
    if include_url and model_id and layer and index:
        compact["url"] = f"{API_BASE}/{model_id}/{layer}/{index}"
    return compact


def search(args: argparse.Namespace) -> None:
    if args.layers:
        endpoint = "/api/explanation/search"
        body = {
            "modelId": args.model,
            "layers": args.layers,
            "query": args.query,
            "offset": args.offset,
        }
    else:
        endpoint = "/api/explanation/search-model"
        body = {
            "modelId": args.model,
            "query": args.query,
            "offset": args.offset,
        }
    data = request_json("POST", endpoint, body)
    results = data.get("results", [])[: args.limit]
    compact = [compact_feature(row) for row in results]
    write_output(compact, args.out)
    if not args.quiet:
        for item in compact:
            print(
                f"{item['layer']}:{item['index']} "
                f"{item.get('description', '')!r} "
                f"maxAct={item.get('maxActApprox')} "
                f"url={item.get('url')}"
            )


def feature(args: argparse.Namespace) -> None:
    data = request_json("GET", f"/api/feature/{args.model}/{args.layer}/{args.index}")
    compact = compact_feature(data)
    if args.full:
        write_output(data, args.out)
        if not args.quiet:
            print(json.dumps(data, indent=2)[: args.preview_chars])
        return
    write_output(compact, args.out)
    if not args.quiet:
        print(json.dumps(compact, indent=2))


def activate(args: argparse.Namespace) -> None:
    body = {
        "feature": {
            "modelId": args.model,
            "layer": args.layer,
            "index": str(args.index),
        },
        "customText": args.text,
    }
    data = request_json("POST", "/api/activation/new", body)
    write_output(data, args.out)
    if args.quiet:
        return
    tokens = data.get("tokens") or data.get("activation", {}).get("tokens") or []
    values = data.get("values") or data.get("activation", {}).get("values") or []
    if not tokens or not values:
        print(json.dumps(data, indent=2)[: args.preview_chars])
        return
    pairs = sorted(
        [(float(value), index, tokens[index]) for index, value in enumerate(values)],
        reverse=True,
    )[: args.top]
    for value, index, token in pairs:
        print(f"{index:4d} {value:10.4f} {token!r}")


def write_output(data: Any, out: str | None) -> None:
    if not out:
        return
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        rows = data if isinstance(data, list) else [data]
        fieldnames = sorted({key for row in rows if isinstance(row, dict) for key in row})
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        path.write_text(json.dumps(data, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small Neuronpedia API client for this Gemma steering project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search Neuronpedia explanations for candidate features.")
    search_parser.add_argument("query", help="Search query, e.g. golf")
    search_parser.add_argument("--model", default=DEFAULT_MODEL)
    search_parser.add_argument("--layers", nargs="*", help="Restrict to specific source/layer IDs.")
    search_parser.add_argument("--res16k", action="store_true", help="Restrict to GemmaScope 2 RES-16K layers.")
    search_parser.add_argument("--limit", type=int, default=20)
    search_parser.add_argument("--offset", type=int, default=0)
    search_parser.add_argument("--out")
    search_parser.add_argument("--quiet", action="store_true")
    search_parser.set_defaults(func=search)

    feature_parser = subparsers.add_parser("feature", help="Fetch one feature dashboard.")
    feature_parser.add_argument("layer")
    feature_parser.add_argument("index")
    feature_parser.add_argument("--model", default=DEFAULT_MODEL)
    feature_parser.add_argument("--full", action="store_true")
    feature_parser.add_argument("--out")
    feature_parser.add_argument("--quiet", action="store_true")
    feature_parser.add_argument("--preview-chars", type=int, default=4000)
    feature_parser.set_defaults(func=feature)

    activate_parser = subparsers.add_parser("activate", help="Test one feature on custom text.")
    activate_parser.add_argument("layer")
    activate_parser.add_argument("index")
    activate_parser.add_argument("text")
    activate_parser.add_argument("--model", default=DEFAULT_MODEL)
    activate_parser.add_argument("--top", type=int, default=10)
    activate_parser.add_argument("--out")
    activate_parser.add_argument("--quiet", action="store_true")
    activate_parser.add_argument("--preview-chars", type=int, default=4000)
    activate_parser.set_defaults(func=activate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "res16k", False):
        args.layers = DEFAULT_RES16K_LAYERS
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
