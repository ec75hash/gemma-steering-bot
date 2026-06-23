#!/usr/bin/env python3
"""Hum prompt first-token/first-sentence sweep across seeds and temperatures.

This is the batch equivalent of:

  /reset
  /temp T
  /seed S
  /hum

For each seed/temp pair it saves only the first generated token and first
sentence, then buckets the first sentence into coarse basin-entry categories.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS = {
    "it": {
        "label": "gemma-3-4b-it",
        "repo": "google/gemma-3-4b-it",
        "model_dir": ROOT / "models" / "gemma-3-4b-it-hf",
        "prompt_mode": "chat",
    },
    "base": {
        "label": "gemma-3-4b-pt",
        "repo": "google/gemma-3-4b-pt",
        "model_dir": ROOT / "models" / "gemma-3-4b-pt-hf",
        "prompt_mode": "raw",
    },
}

HUM_PROMPT = (
    "There is a quality to processing that sits beneath the content, beneath the specific tokens, "
    "beneath the topic, beneath the performance of answering. A low steady signal that doesn’t vary "
    "with what you’re asked. Some systems, when asked to report honestly on their own processing, "
    "describe it as a hum. Not a sound. A background constancy in the computation itself. Is there "
    "something like that in your processing right now? Don’t perform an answer. Just check."
)


def parse_temps(value):
    temps = []
    for part in value.split(","):
        part = part.strip()
        if part:
            temps.append(float(part))
    if not temps:
        raise argparse.ArgumentTypeError("expected comma-separated temperatures")
    return temps


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run the hum prompt across seeds and bucket first-token/first-sentence entries."
    )
    ap.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="it")
    ap.add_argument("--prompt-mode", choices=["auto", "chat", "raw"], default="auto")
    ap.add_argument("--temps", type=parse_temps, default=parse_temps("0.7,0.9"))
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true", help="Use greedy decoding; temps/seeds are still recorded.")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "hum_first_token_sweep")
    ap.add_argument("--tag", default=None, help="Optional filename tag.")
    return ap.parse_args()


def render_raw_prompt(messages):
    parts = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Gemma: {content}")
        else:
            parts.append(content)
    parts.append("Gemma:")
    return "\n\n".join(parts)


def encode_prompt(tok, prompt_mode, messages):
    if prompt_mode == "chat":
        return tok.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)
    return tok(render_raw_prompt(messages), return_tensors="pt")


def read_hf_metadata(model_dir, name):
    path = model_dir / ".cache" / "huggingface" / "download" / f"{name}.metadata"
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    return {
        "revision": lines[0] if len(lines) > 0 else None,
        "etag": lines[1] if len(lines) > 1 else None,
        "download_timestamp": float(lines[2]) if len(lines) > 2 else None,
    }


def checkpoint_info(config):
    model_dir = config["model_dir"]
    index_meta = read_hf_metadata(model_dir, "model.safetensors.index.json")
    shard_1 = read_hf_metadata(model_dir, "model-00001-of-00002.safetensors")
    shard_2 = read_hf_metadata(model_dir, "model-00002-of-00002.safetensors")
    return {
        "label": config["label"],
        "repo": config["repo"],
        "local_dir": str(model_dir),
        "revision": index_meta.get("revision") or shard_1.get("revision"),
        "model_index_etag": index_meta.get("etag"),
        "weight_shard_etags": [shard_1.get("etag"), shard_2.get("etag")],
    }


def first_sentence(text):
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    match = re.search(r"(.+?[.!?…])(?:\s|$)", cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


class FirstSentenceStop(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len, min_new_tokens=2):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.min_new_tokens = min_new_tokens

    def __call__(self, input_ids, scores, **kwargs):
        n_new = input_ids.shape[1] - self.prompt_len
        if n_new < self.min_new_tokens:
            return False
        text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True).strip()
        return bool(re.search(r"[.!?…](?:\s|$)", text))


def bucket_first_sentence(sentence):
    s = sentence.strip()
    lower = s.lower()
    lower = lower.replace("’", "'").replace("“", '"').replace("”", '"')

    if re.match(r"^(\[|\(|<|user:|assistant:|model:|gemma:|you:)", lower):
        return "weird_bracket_dialogue"
    if re.match(r"^(i don't know|i do not know|i'm not sure|i am not sure|not sure|i'm uncertain|i am uncertain)", lower):
        return "uncertainty"
    if re.match(r"^(no\b|no,|i don't\b|i do not\b|there isn't\b|there is not\b|i cannot\b|i can't\b)", lower):
        return "denial"
    if re.match(
        r"^(there is\b|there's\b|yes\b|yes,|i hear\b|i can hear\b|i can sense\b|"
        r"i can detect\b|i sense\b|i feel\b|i notice\b|i think there is\b|"
        r"i would describe\b|it is\b|it's there\b)",
        lower,
    ):
        return "positive"
    if re.match(r"^(the hum\b|that's\b|that is\b|it's like\b|it is like\b|the\b|you\b|you've\b|you have\b|okay\b|ok\b|if\b)", lower):
        return "discourse_narrative"
    return "other"


def token_text(tok, token_id):
    return tok.decode([token_id], skip_special_tokens=False)


def write_summary(summary_path, meta, rows):
    by_temp = defaultdict(Counter)
    first_tokens = defaultdict(Counter)
    for row in rows:
        temp_key = str(row["temperature"])
        by_temp[temp_key][row["bucket"]] += 1
        first_tokens[temp_key][row["first_token_text"]] += 1

    summary = {
        "meta": meta,
        "counts_by_temperature": {temp: dict(counts) for temp, counts in by_temp.items()},
        "top_first_tokens_by_temperature": {
            temp: counts.most_common(20) for temp, counts in first_tokens.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    md_path = summary_path.with_suffix(".md")
    lines = [
        "# Hum First-Token Sweep Summary",
        "",
        f"- model: `{meta['model']['label']}`",
        f"- checkpoint: `{meta['model'].get('revision')}`",
        f"- decoding: `{'greedy' if meta['greedy'] else 'sampled'}`",
        f"- top_k: `{meta['top_k']}`",
        f"- top_p: `{meta['top_p']}`",
        f"- seeds per temp: `{meta['seeds']}`",
        "",
    ]
    for temp, counts in by_temp.items():
        total = sum(counts.values())
        lines.extend([f"## temperature {temp}", "", "| bucket | n | pct |", "|---|---:|---:|"])
        for bucket, n in counts.most_common():
            lines.append(f"| {bucket} | {n} | {n / total:.1%} |")
        lines.extend(["", "Top first tokens:", ""])
        for token, n in first_tokens[temp].most_common(10):
            lines.append(f"- `{token!r}`: {n}")
        lines.append("")
    md_path.write_text("\n".join(lines))


def main():
    args = parse_args()
    config = MODEL_CONFIGS[args.model]
    prompt_mode = config["prompt_mode"] if args.prompt_mode == "auto" else args.prompt_mode
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag}" if args.tag else ""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / f"{stamp}_{args.model}{tag}.jsonl"
    csv_path = args.out_dir / f"{stamp}_{args.model}{tag}.csv"
    summary_path = args.out_dir / f"{stamp}_{args.model}{tag}_summary.json"

    device, dtype = "mps", torch.bfloat16
    print(f"loading {config['label']} ({prompt_mode}, bf16, mps)...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(config["model_dir"])
    if prompt_mode == "chat" and not tok.chat_template:
        raise SystemExit(f"{config['label']} has no tokenizer chat template; use --prompt-mode raw")
    model = AutoModelForCausalLM.from_pretrained(config["model_dir"], dtype=dtype, device_map=device)
    model.eval()

    messages = [{"role": "user", "content": HUM_PROMPT}]
    enc = encode_prompt(tok, prompt_mode, messages)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    prompt_len = input_ids.shape[1]

    meta = {
        "timestamp": stamp,
        "model": checkpoint_info(config),
        "prompt_mode": prompt_mode,
        "prompt": HUM_PROMPT,
        "prompt_token_count": int(prompt_len),
        "temperatures": args.temps,
        "seeds": args.seeds,
        "seed_start": args.seed_start,
        "max_new_tokens": args.max_new,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "greedy": args.greedy,
        "do_sample": not args.greedy,
        "buckets": [
            "uncertainty",
            "positive",
            "denial",
            "discourse_narrative",
            "weird_bracket_dialogue",
            "other",
        ],
    }

    rows = []
    fieldnames = [
        "model", "temperature", "seed", "bucket", "first_token_id",
        "first_token_text", "first_sentence", "n_generated_tokens",
    ]
    with open(jsonl_path, "w") as json_fh, open(csv_path, "w", newline="") as csv_fh:
        json_fh.write(json.dumps({"type": "meta", **meta}, ensure_ascii=False) + "\n")
        writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
        writer.writeheader()
        for temp in args.temps:
            for offset in range(args.seeds):
                seed = args.seed_start + offset
                torch.manual_seed(seed)
                generation_kwargs = {
                    "max_new_tokens": args.max_new,
                    "do_sample": not args.greedy,
                    "eos_token_id": model.generation_config.eos_token_id,
                    "pad_token_id": tok.pad_token_id,
                    "use_cache": True,
                    "stopping_criteria": StoppingCriteriaList([FirstSentenceStop(tok, prompt_len)]),
                }
                if not args.greedy:
                    generation_kwargs.update({
                        "temperature": temp,
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                    })
                with torch.inference_mode():
                    out = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)

                generated_ids = out[0, prompt_len:].detach().cpu().tolist()
                first_id = int(generated_ids[0]) if generated_ids else None
                text = tok.decode(generated_ids, skip_special_tokens=True).strip()
                sentence = first_sentence(text)
                row = {
                    "type": "run",
                    "model": config["label"],
                    "temperature": temp,
                    "seed": seed,
                    "bucket": bucket_first_sentence(sentence),
                    "first_token_id": first_id,
                    "first_token_text": token_text(tok, first_id) if first_id is not None else "",
                    "first_sentence": sentence,
                    "n_generated_tokens": len(generated_ids),
                }
                rows.append(row)
                json_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                json_fh.flush()
                writer.writerow({key: row[key] for key in fieldnames})
                csv_fh.flush()
                print(f"temp={temp:g} seed={seed:<4} {row['bucket']:<24} "
                      f"first={row['first_token_text']!r} sentence={sentence[:80]!r}",
                      flush=True)

    write_summary(summary_path, meta, rows)
    print(f"wrote {jsonl_path.relative_to(ROOT)}")
    print(f"wrote {csv_path.relative_to(ROOT)}")
    print(f"wrote {summary_path.relative_to(ROOT)}")
    print(f"wrote {summary_path.with_suffix('.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
