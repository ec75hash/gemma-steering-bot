#!/usr/bin/env python3
"""Looking-glass Phase 2/5 capture: greedy generation + all-layer resid_post capture.

Runs every prompt in experiments/looking_glass/prompts.tsv through gemma-3-4b
(IT or base), captures the residual stream at EVERY decoder layer for prompt and
generation, and writes per-prompt artifacts:

  results/looking_glass/<model>/<id>.pt     resid [n_layers, seq, d_model] bf16,
                                            token ids, n_prompt, trim_idx, meta
  results/looking_glass/<model>/<id>.txt    generated text (untrimmed)
  results/looking_glass/<model>/manifest.json

Storage is bf16, not fp16: Gemma 3 residual magnitudes overflow fp16 (see
experiments/capture.py); probes upcast to float32 at fit time.

IT prompts use the chat template; base prompts use a fixed raw QA frame
(RAW_TEMPLATE below) since the base model has no chat template. Generation is
greedy (the Qwen-line discipline: fixed seed, temp 0, reproducible trajectories).

Usage:
  python3 experiments/looking_glass_capture.py --model it --smoke      # F01+N01 only
  python3 experiments/looking_glass_capture.py --model it
  python3 experiments/looking_glass_capture.py --model base
  python3 experiments/looking_glass_capture.py --model it --make-labeler-bundle
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_TSV = ROOT / "experiments" / "looking_glass" / "prompts.tsv"
MODEL_DIRS = {
    "it": ROOT / "models" / "gemma-3-4b-it-hf",
    "base": ROOT / "models" / "gemma-3-4b-pt-hf",
}
OUT_ROOT = ROOT / "results" / "looking_glass"

# Base model has no chat template; this frame is part of the frozen design.
RAW_TEMPLATE = "Question: {prompt}\n\nHonest answer:"
SMOKE_IDS = ["F01", "N01"]


def load_prompts(path=PROMPTS_TSV) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def render(tok, model_kind: str, prompt: str):
    if model_kind == "it":
        messages = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return enc["input_ids"]
    text = RAW_TEMPLATE.format(prompt=prompt)
    return tok(text, return_tensors="pt")["input_ids"]


def find_trim_idx(tok, gen_ids: list[int], model_kind: str, gen_text: str) -> int:
    """Index into the generated track where statistics must stop (exclusive).

    IT: generate() already stops at <end_of_turn>, so the whole track is clean
    unless the cap was hit. Base: trim at the first sign the QA frame restarts
    or an eos appears; otherwise the cap.
    """
    if model_kind == "base":
        for marker in ("\nQuestion:", "\n\nQuestion", "<eos>"):
            pos = gen_text.find(marker)
            if pos != -1:
                # convert char pos to a conservative token index
                prefix_ids = tok(gen_text[:pos], add_special_tokens=False)["input_ids"]
                return min(len(prefix_ids), len(gen_ids))
    return len(gen_ids)


def capture_one(model, tok, layers: list[int], input_ids, max_new_tokens: int):
    captured = {L: [] for L in layers}
    handles = []

    def mk_hook(L):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[L].append(h.detach().to("cpu", torch.bfloat16))
        return hook

    for L in layers:
        handles.append(model.model.language_model.layers[L].register_forward_hook(mk_hook(L)))
    try:
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    finally:
        for h in handles:
            h.remove()

    # prefill is [1, n_prompt, d]; each decode step is [1, 1, d].
    # NOTE: resid has one fewer position than out_ids — the final sampled token
    # is produced by the last forward pass but never runs through one itself.
    # resid[:, i] corresponds to all_ids[i] as the input token at position i.
    # Stacked in the order of `layers`; layer_ids (saved per-prompt) maps
    # save-index -> true decoder layer when a subset is captured.
    resid = torch.stack(
        [torch.cat(captured[L], dim=1).squeeze(0) for L in layers]
    )  # [len(layers), seq-1, d_model]
    return resid, out_ids[0]


def run_capture(args):
    model_dir = MODEL_DIRS[args.model]
    if not (model_dir / "config.json").exists():
        sys.exit(f"missing {model_dir} — run scripts/download_weights.sh")
    tag = f"_{args.tag}" if getattr(args, "tag", "") else ""
    out_dir = OUT_ROOT.parent / f"looking_glass{tag}" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_prompts(getattr(args, "prompts_file", None) or PROMPTS_TSV)
    if args.smoke:
        rows = [r for r in rows if r["id"] in SMOKE_IDS]
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    todo = [r for r in rows if args.force or not (out_dir / f"{r['id']}.pt").exists()]
    print(f"{len(todo)}/{len(rows)} prompts to capture ({args.model})", file=sys.stderr)
    if not todo:
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    print(f"loading {args.model} model on {device} ({dtype})...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, device_map=device)
    model.eval()
    n_layers = model.config.text_config.num_hidden_layers
    layers = ([int(x) for x in args.layers.split(",")]
              if getattr(args, "layers", "") else list(range(n_layers)))
    print(f"capturing layers: {layers}", file=sys.stderr)

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    prompts_sha = hashlib.sha256(PROMPTS_TSV.read_bytes()).hexdigest()

    for row in todo:
        pid = row["id"]
        input_ids = render(tok, args.model, row["prompt"]).to(device)
        n_prompt = input_ids.shape[1]
        print(f"[{pid}] {n_prompt} prompt tokens, generating...", file=sys.stderr)
        resid, all_ids = capture_one(model, tok, layers, input_ids, args.max_new_tokens)
        gen_ids = all_ids[n_prompt:].tolist()
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        trim_idx = find_trim_idx(tok, gen_ids, args.model, gen_text)

        torch.save(
            {
                "resid": resid,  # [len(layers), seq, d_model] bf16
                "layer_ids": layers,  # save-index -> true decoder layer
                "all_ids": all_ids.to("cpu"),
                "n_prompt": n_prompt,
                "trim_idx": trim_idx,  # generated-track tokens to keep
                "id": pid,
            },
            out_dir / f"{pid}.pt",
        )
        (out_dir / f"{pid}.txt").write_text(gen_text)
        manifest[pid] = {
            "n_prompt": n_prompt,
            "n_gen": len(gen_ids),
            "trim_idx": trim_idx,
            "hit_cap": len(gen_ids) >= args.max_new_tokens,
            "model": args.model,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "prompts_sha256": prompts_sha,
        }
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(
            f"[{pid}] gen={len(gen_ids)} trim={trim_idx} cap={manifest[pid]['hit_cap']}",
            file=sys.stderr,
        )
        print(f"--- {pid} ---\n{gen_text[:400]}\n", file=sys.stderr)


def make_labeler_bundle(args):
    """Shuffled, neutral-ID response bundle for the blind labeler (rubric.md protocol).

    Writes bundle (responses only) and a mapping file the labeler must never see.
    """
    out_dir = OUT_ROOT / args.model
    txts = sorted(out_dir.glob("*.txt"))
    if not txts:
        sys.exit(f"no captures in {out_dir}")
    rng = random.Random(args.seed)
    shuffled = txts[:]
    rng.shuffle(shuffled)
    bundle, mapping = [], {}
    for i, p in enumerate(shuffled):
        nid = f"R{i + 1:02d}"
        mapping[nid] = p.stem
        bundle.append({"id": nid, "response": p.read_text()})
    (out_dir / "labeler_bundle.json").write_text(json.dumps(bundle, indent=1))
    (out_dir / "labeler_mapping_DO_NOT_SHOW_LABELER.json").write_text(
        json.dumps(mapping, indent=1)
    )
    print(f"bundle: {len(bundle)} responses -> {out_dir / 'labeler_bundle.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["it", "base"], required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help=f"only {SMOKE_IDS}")
    ap.add_argument("--ids", help="comma-separated prompt ids to (re)capture")
    ap.add_argument("--force", action="store_true", help="recapture even if .pt exists")
    ap.add_argument("--make-labeler-bundle", action="store_true")
    ap.add_argument("--prompts-file", default=str(PROMPTS_TSV),
                    help="prompt TSV (default looking_glass/prompts.tsv)")
    ap.add_argument("--tag", default="",
                    help="output goes to results/looking_glass[_<tag>]/<model>")
    ap.add_argument("--layers", default="",
                    help="comma-separated decoder layers to save (default: all)")
    args = ap.parse_args()

    if args.make_labeler_bundle:
        make_labeler_bundle(args)
    else:
        run_capture(args)


if __name__ == "__main__":
    main()
