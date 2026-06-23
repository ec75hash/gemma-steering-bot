#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "gemma-3-4b-it-hf"
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
BASE_MODEL_DIR = ROOT / "models" / "gemma-3-4b-pt-hf"
BASE_SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-pt" / "resid_post"
SAE_LAYERS = [9, 17, 22, 29]


def ok(message: str) -> None:
    print(f"OK   {message}")


def warn(message: str) -> None:
    print(f"WARN {message}")


def fail(message: str) -> None:
    print(f"FAIL {message}")
    raise SystemExit(1)


def require_file(path: Path) -> None:
    if not path.exists():
        fail(f"missing {path.relative_to(ROOT)}")
    ok(f"found {path.relative_to(ROOT)}")


def check_import(name: str) -> object:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        fail(f"cannot import {name}: {exc}")
    version = getattr(module, "__version__", "unknown")
    ok(f"import {name} {version}")
    return module


def check_python() -> None:
    ok(f"python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        fail("Python 3.10+ is required")


def check_packages() -> None:
    torch = check_import("torch")
    check_import("transformers")
    check_import("safetensors")
    check_import("huggingface_hub")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        ok("torch MPS backend available")
    else:
        warn("torch MPS backend not available; CPU will be very slow and CUDA is untested here")


def check_model_files() -> None:
    require_file(MODEL_DIR / "config.json")
    require_file(MODEL_DIR / "generation_config.json")
    require_file(MODEL_DIR / "tokenizer.json")
    require_file(MODEL_DIR / "tokenizer.model")
    require_file(MODEL_DIR / "model.safetensors.index.json")

    index_path = MODEL_DIR / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
    except Exception as exc:
        fail(f"cannot read {index_path.relative_to(ROOT)}: {exc}")

    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        fail("model.safetensors.index.json has no weight_map shards")
    for shard_name in shard_names:
        require_file(MODEL_DIR / shard_name)


def check_sae_files() -> None:
    for layer in SAE_LAYERS:
        layer_dir = SAE_DIR / f"layer_{layer}_width_16k_l0_medium"
        require_file(layer_dir / "params.safetensors")
        require_file(layer_dir / "config.json")


def check_base_assets() -> None:
    """Base (pre-instruction-tuned) model + native base SAEs.

    Optional for the chat demo; required for the looking-glass experiment
    (experiments/looking_glass_*). Missing assets warn rather than fail.
    """
    base_index = BASE_MODEL_DIR / "model.safetensors.index.json"
    if not base_index.exists():
        warn(
            "base model not found (models/gemma-3-4b-pt-hf); "
            "needed only for experiments/looking_glass_* — run scripts/download_weights.sh"
        )
    else:
        try:
            index = json.loads(base_index.read_text())
        except Exception as exc:
            fail(f"cannot read {base_index.relative_to(ROOT)}: {exc}")
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        if not shard_names:
            fail("base model.safetensors.index.json has no weight_map shards")
        for shard_name in shard_names:
            require_file(BASE_MODEL_DIR / shard_name)
        ok("base model (gemma-3-4b-pt-hf) shards verified")

    missing_layers = [
        layer
        for layer in SAE_LAYERS
        if not (BASE_SAE_DIR / f"layer_{layer}_width_16k_l0_medium" / "params.safetensors").exists()
    ]
    if missing_layers:
        warn(
            f"base GemmaScope SAEs missing for layers {missing_layers} "
            "(models/gemma-scope-2-4b-pt); needed only for experiments/looking_glass_*"
        )
    else:
        ok("base GemmaScope 2 RES-16K SAEs present for layers 9/17/22/29")


def check_neuronpedia() -> None:
    url = "https://www.neuronpedia.org/api/feature/gemma-3-4b-it/17-gemmascope-2-res-16k/4271"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.load(response)
    except Exception as exc:
        warn(f"Neuronpedia feature fetch failed: {exc}")
        return
    if str(data.get("index")) == "4271" or data.get("index") == 4271:
        ok("Neuronpedia public feature fetch works")
    else:
        warn("Neuronpedia responded, but the payload did not look like feature 4271")


def main() -> int:
    print("Checking gemma-3-4b-it-sae-demo setup\n")
    check_python()
    check_packages()
    check_model_files()
    check_sae_files()
    check_base_assets()
    check_neuronpedia()
    print("\nSetup looks ready. Start chat with:")
    print("  python3 chat_steer.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
