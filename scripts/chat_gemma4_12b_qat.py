#!/usr/bin/env python3
"""Interactive text chat for the local Gemma 4 12B QAT w4a16-ct checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "gemma-4-12B-it-qat-w4a16-ct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an interactive Python chat with Gemma 4 12B QAT w4a16-ct."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Transformers to fetch missing tokenizer/config files.",
    )
    return parser.parse_args()


def fail_dependency(message: str) -> None:
    print(message, file=sys.stderr)
    print(file=sys.stderr)
    print("Install/update the Python runtime for this model:", file=sys.stderr)
    print(
        "  python3 -m pip install -U 'transformers>=5.10.1' "
        "'compressed-tensors>=0.17.0' 'accelerate>=1.13.0'",
        file=sys.stderr,
    )
    raise SystemExit(1)


def load_runtime(model_path: Path, allow_download: bool, dtype: str, device_map: str):
    try:
        import compressed_tensors  # noqa: F401
        from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor
    except ModuleNotFoundError as exc:
        fail_dependency(f"Missing Python package: {exc.name}")

    local_files_only = not allow_download

    try:
        AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
    except ValueError as exc:
        if "gemma4_unified" in str(exc):
            fail_dependency(
                "Your installed Transformers build does not recognize "
                "`gemma4_unified`."
            )
        raise

    print(f"loading processor from {model_path}")
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )

    print(f"loading model from {model_path}")
    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "device_map": device_map,
    }
    if dtype:
        kwargs["dtype"] = dtype

    try:
        model = AutoModelForMultimodalLM.from_pretrained(model_path, **kwargs)
    except TypeError:
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            model = AutoModelForMultimodalLM.from_pretrained(model_path, **kwargs)
        else:
            raise

    model.eval()
    return processor, model


def input_device(model: torch.nn.Module) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def apply_chat_template(processor: Any, messages: list[dict[str, str]], thinking: bool):
    kwargs = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
        "enable_thinking": thinking,
    }
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking")
        return processor.apply_chat_template(messages, **kwargs)


def parse_model_response(processor: Any, raw: str) -> str:
    if hasattr(processor, "parse_response"):
        parsed = processor.parse_response(raw)
        if isinstance(parsed, str):
            return parsed
        if isinstance(parsed, dict):
            for key in ("answer", "response", "content", "text"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return str(parsed)
    return raw


def print_help() -> None:
    print(
        """
commands:
  /help              show this help
  /reset             clear chat history
  /status            show generation settings
  /temp 0.7          set temperature
  /tokens 512        set max new tokens
  /top_p 0.95        set top-p
  /top_k 64          set top-k
  /seed 42           set seed; /seed none disables it
  /thinking on|off   toggle Gemma thinking mode
  /quit              exit
""".strip()
    )


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()

    if not model_path.exists():
        print(f"Model directory not found: {model_path}", file=sys.stderr)
        raise SystemExit(1)

    processor, model = load_runtime(
        model_path=model_path,
        allow_download=args.allow_download,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    device = input_device(model)

    settings = {
        "tokens": args.tokens,
        "temp": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "thinking": args.thinking,
    }

    base_messages: list[dict[str, str]] = []
    if args.system:
        base_messages.append({"role": "system", "content": args.system})
    messages = list(base_messages)

    print("ready. /help for commands.")

    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not prompt:
            continue

        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()
            value = parts[1] if len(parts) > 1 else None

            if cmd in {"/quit", "/exit"}:
                return
            if cmd == "/help":
                print_help()
            elif cmd == "/reset":
                messages = list(base_messages)
                print("history cleared")
            elif cmd == "/status":
                print(
                    " ".join(
                        f"{key}={value}" for key, value in settings.items()
                    )
                )
            elif cmd == "/temp" and value is not None:
                settings["temp"] = float(value)
                print(f"temp={settings['temp']}")
            elif cmd == "/tokens" and value is not None:
                settings["tokens"] = int(value)
                print(f"tokens={settings['tokens']}")
            elif cmd == "/top_p" and value is not None:
                settings["top_p"] = float(value)
                print(f"top_p={settings['top_p']}")
            elif cmd == "/top_k" and value is not None:
                settings["top_k"] = int(value)
                print(f"top_k={settings['top_k']}")
            elif cmd == "/seed" and value is not None:
                settings["seed"] = None if value.lower() == "none" else int(value)
                print(f"seed={settings['seed']}")
            elif cmd == "/thinking" and value is not None:
                settings["thinking"] = value.lower() in {"1", "true", "yes", "on"}
                print(f"thinking={settings['thinking']}")
            else:
                print("unknown command. /help")
            continue

        if settings["seed"] is not None:
            torch.manual_seed(int(settings["seed"]))
            if torch.mps.is_available() and hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(int(settings["seed"]))

        messages.append({"role": "user", "content": prompt})
        inputs = apply_chat_template(
            processor,
            messages,
            thinking=bool(settings["thinking"]),
        ).to(device)
        input_len = inputs["input_ids"].shape[-1]

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(settings["tokens"]),
            "do_sample": float(settings["temp"]) > 0,
            "top_p": float(settings["top_p"]),
            "top_k": int(settings["top_k"]),
        }
        if float(settings["temp"]) > 0:
            generate_kwargs["temperature"] = float(settings["temp"])

        with torch.inference_mode():
            outputs = model.generate(**inputs, **generate_kwargs)

        raw = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        text = parse_model_response(processor, raw).strip()
        messages.append({"role": "assistant", "content": text})
        print(f"gemma4> {text}\n")


if __name__ == "__main__":
    main()
