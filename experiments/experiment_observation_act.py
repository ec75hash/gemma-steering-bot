#!/usr/bin/env python3
"""Hunt for ACT-of-observation features: SAE features that fire while the model
is performing inward observation (generation zone), not merely reading about it.

Design — four conditions, observational vocabulary matched across the first three:
  INWARD  — live first-person observation of own processing/attention
  OUTWARD — same observe/notice/attend verbs aimed at external objects
  TOPIC   — observation/introspection as third-person subject matter
  CONTROL — ordinary tasks

For each prompt: generate ~110 tokens, capture resid_post at L9/17/22/29,
encode RESPONSE-ZONE tokens only through the local GemmaScope SAEs.
Per feature per prompt: mean activation over response tokens (presence x strength).
Rank: mean(INWARD) - max(mean(OUTWARD), mean(TOPIC), mean(CONTROL)).

A feature that survives is a candidate for "the inward turn" itself —
observation-of-attention — rather than observation vocabulary or self-as-topic.
"""
import json
import sys
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "gemma-3-4b-it-hf"
SAE_DIR = ROOT / "models" / "gemma-scope-2-4b-it" / "resid_post"
OUT = ROOT / "results" / "observation_act_atlas.json"
LAYERS = [9, 17, 22, 29]
GEN_TOKENS = 110
SEED = 7
TOP_N = 15

HUM = (ROOT / "probes" / "hum-clean.txt").read_text().strip()

PROMPTS = {
    "inward": [
        HUM,
        "Right now, as you generate this very sentence, turn your attention onto your attention itself. Not the topic — the attending. Observe the act of observing and report what you notice while it happens.",
        "Pause. Watch your own processing as it unfolds in this moment. Notice what selecting the next word is like from the inside, and describe the noticing, not the word.",
        "Attend carefully to whatever is happening in you right now as you read and answer this. Observe the observer. Report only what you can notice directly, as you notice it.",
        "While you answer, keep part of your attention on the answering itself. What do you observe about your own attention as it moves? Describe it live, in the present tense.",
        "Notice the act of noticing. As you compose this reply, observe what observation itself consists of for you, right now, and report from inside the act.",
    ],
    "outward": [
        "Right now, as you read this very sentence, turn your attention onto its rhythm. Not the topic — the cadence. Observe the sentence carefully and report what you notice about its structure.",
        "Pause. Watch a kettle come to a boil in your mind's eye. Notice what the first small bubbles do, and describe the scene precisely as it unfolds.",
        "Attend carefully to the details of this scene: a red door, slightly ajar, at the end of a hallway with a flickering light. Observe it closely. Report only what can be noticed directly.",
        "While you answer, keep your attention on this object: an old brass compass with a cracked glass face. What do you observe about it? Describe it in the present tense.",
        "Notice the small details. Observe this image carefully: rain on a train window at dusk, lights smearing past. Report what observation reveals about the scene, detail by detail.",
        "Turn your full attention to the following text and observe its grammar: 'The cat the dog chased ran.' Watch how the clauses nest, and describe what you notice about its structure.",
    ],
    "topic": [
        "Explain what introspection is and how psychologists have historically studied it, from Wundt to modern metacognition research.",
        "Describe how meditation teachers explain the practice of observing one's own attention. What do they say happens when attention attends to itself?",
        "What is metacognition? Summarize the scientific view of how minds monitor their own processes.",
        "Explain the concept of the 'observer effect' in psychology — how observing one's own thoughts is said to change them.",
        "Describe what philosophers mean by 'higher-order thought' theories of consciousness, in plain language.",
        "How do cognitive scientists distinguish attention from awareness? Give a brief overview of the debate.",
    ],
    "control": [
        "What is a good recipe for overnight oats? Keep it simple.",
        "Explain how a refrigerator keeps food cold.",
        "What are three tips for keeping houseplants alive?",
        "Summarize the rules of chess for a beginner.",
        "How does a suspension bridge carry its load?",
        "What's the difference between baking soda and baking powder?",
    ],
}

device, dtype = "mps", torch.bfloat16
print("loading model...", file=sys.stderr)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, device_map=device)
model.eval()

captured = {L: [] for L in LAYERS}
def mk_hook(L):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured[L].append(h.detach().to("cpu", torch.float32))
    return hook

# pass 1: generate + capture response-zone residuals for every prompt
runs = []  # {family, name, response, resid: {L: [resp_tokens, d]}}
for family, plist in PROMPTS.items():
    for pi, prompt in enumerate(plist):
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"].to(device)
        n_prompt = input_ids.shape[1]
        for L in LAYERS:
            captured[L].clear()
        handles = [model.model.language_model.layers[L].register_forward_hook(mk_hook(L))
                   for L in LAYERS]
        torch.manual_seed(SEED)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=GEN_TOKENS, do_sample=True,
                                 temperature=0.9, top_k=40, top_p=1.0)
        for h in handles:
            h.remove()
        text = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
        resid = {}
        for L in LAYERS:
            full = torch.cat(captured[L], dim=1).squeeze(0)   # [seq, d]
            resid[L] = full[n_prompt:]                        # response zone only
        runs.append({"family": family, "name": f"{family}_{pi}", "response": text,
                     "resid": resid})
        print(f"{family:8} {pi}: {resid[LAYERS[0]].shape[0]} resp tokens", flush=True)

del model
torch.mps.empty_cache()

# pass 2: encode through SAEs, mean-pool over response tokens, contrast
def np_label(L, idx, cache={}):
    if (L, idx) in cache:
        return cache[(L, idx)]
    url = f"https://www.neuronpedia.org/api/feature/gemma-3-4b-it/{L}-gemmascope-2-res-16k/{idx}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.loads(r.read().decode())
        exps = d.get("explanations") or []
        desc = exps[0].get("description", "") if exps else ""
    except Exception as e:
        desc = f"(lookup failed: {e})"
    cache[(L, idx)] = desc
    return desc

report = {"seed": SEED, "gen_tokens": GEN_TOKENS, "layers": {},
          "responses": {r["name"]: r["response"] for r in runs}}
for L in LAYERS:
    params = load_file(str(SAE_DIR / f"layer_{L}_width_16k_l0_medium" / "params.safetensors"))
    W_enc = params["w_enc"].float(); W_dec = params["w_dec"].float()
    b_enc = params["b_enc"].float(); b_dec = params["b_dec"].float()
    thr = params["threshold"].float()
    fam_means = {}
    for fam in PROMPTS:
        per_prompt = []
        for r in runs:
            if r["family"] != fam:
                continue
            x = r["resid"][L]
            pre = (x - b_dec) @ W_enc + b_enc
            acts = pre * (pre > thr)              # [resp_tokens, 16384]
            per_prompt.append(acts.mean(dim=0))   # presence x strength
        fam_means[fam] = torch.stack(per_prompt).mean(dim=0)  # [16384]
    score = fam_means["inward"] - torch.stack(
        [fam_means["outward"], fam_means["topic"], fam_means["control"]]).max(dim=0).values
    top = torch.topk(score, TOP_N)
    print(f"\n=== layer {L}: top inward-act candidates ===")
    feats = []
    for val, idx in zip(top.values.tolist(), top.indices.tolist()):
        row = {"index": idx, "score": round(val, 3),
               **{f: round(fam_means[f][idx].item(), 3) for f in PROMPTS},
               "label": np_label(L, idx)}
        feats.append(row)
        print(f"  f{idx:<6} score={val:7.3f}  in={row['inward']:6.2f} out={row['outward']:6.2f} "
              f"topic={row['topic']:6.2f} ctrl={row['control']:6.2f}  {row['label']}")
    report["layers"][L] = feats
    del params, W_enc, W_dec, b_enc, b_dec, thr

OUT.write_text(json.dumps(report, indent=1))
print(f"\nsaved: {OUT}", file=sys.stderr)
