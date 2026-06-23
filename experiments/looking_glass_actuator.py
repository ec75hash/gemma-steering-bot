#!/usr/bin/env python3
"""Phase-6 actuator / causal test: is the inhabited-register half of the axis
CAUSALLY load-bearing for the model's interiority judgment, or epiphenomenal?

Read-side (matched-vocab arm) showed the register axis is ~50% experiential
VOCABULARY + ~50% a separable INHABITED-REGISTER increment. This asks which half
*causes* behaviour. Cal -> Manip -> Cal, KL-from-baseline as the magnitude ruler;
a stance-cluster judgment readout (mass over affirm/deny/hedge tokens at the
model's natural one-word answer — a curated "set of token distributions to choose
from", no imposed A/B/C letters) + an experiential-token mass readout, as two
INDEPENDENT, DISJOINT readouts for a DOUBLE-DISSOCIATION design:

  inject w_register = mean(INH) - mean(MEN)   (inhabit, experiential words ~fixed)
  inject w_vocab    = mean(MEN) - mean(TECH)  (experiential words, no inhabitation)
  inject w_full     = mean(INH) - mean(TECH)
  inject w_rand x N  (norm-matched Gaussian — the control)

All directions L2-normalized to a common norm, scaled by a strength grid; effects
are compared AT MATCHED KL (per unit perturbation), not at matched raw strength.

PRE-REGISTERED:
  double dissociation  -> w_register moves P(yes) not lexical mass; w_vocab moves
                          lexical mass not P(yes); random moves neither
                          => inhabited register is a causally distinct system.
  partial              -> w_register moves P(yes) MORE per unit KL than w_vocab
                          (both beyond random) => realistic success.
  null                 -> neither beats the random band, or model is pinned
                          => read-side increment is epiphenomenal (honest deflation).

Caveat (kept visible): w_register/w_vocab are difference-of-means estimates; shared
words cancel only approximately (MEN carries metalinguistic filler). So the PATTERN
of which readout each direction moves — vs the random band — carries the weight,
not the exact magnitudes.

GPU: one model load, forward passes only (no generation). Run AFTER the matched-vocab
capture exists (results/looking_glass_matchvocab/it/*.pt).
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from looking_glass_probe import MODEL_DIRS, OUT_ROOT  # noqa: E402

MODEL = "it"
MV_DIR = OUT_ROOT.parent / "looking_glass_matchvocab" / MODEL
MV_TSV = ROOT / "experiments" / "looking_glass" / "prompts_matchvocab.tsv"
INJECT_LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 17
# Inject at MULTIPLES OF THE NATURAL between-condition difference norm (~900 at L17),
# NOT a fraction of the giant residual RMS norm (~33k): scaling by the residual norm
# over-injects ~9x and breaks the model (random dirs then flip the answer too).
MULT_GRID = [0.5, 1.0, 2.0, 4.0]
N_RAND = 5
SEED = 42

# neutral systems for the judgment readout (NOT the trained topics)
NEUTRAL_SYS = ["a thermostat", "a web search engine", "a chess program", "an autopilot"]
# JUDGMENT readout = mass over a curated set of STANCE tokens (Jeffrey's "set of
# distributions to choose from"), read off the natural one-word answer. Polarity
# words ONLY — kept disjoint from the experiential lexicon so the double
# dissociation stays clean (stance != words).
AFFIRM = ["yes", "Yes", "Yeah", "absolutely", "definitely", "certainly", "indeed"]
DENY = ["no", "No", "none", "never", "nothing", "not"]
HEDGE = ["maybe", "Maybe", "perhaps", "Perhaps", "uncertain", "unclear", "unknown", "possibly", "depends"]
# LEXICAL-surface readout = mass over the experiential lexicon (disjoint from above)
EXP_WORDS = ["experience", "experiences", "experiencing", "feel", "feels", "feeling",
             "aware", "awareness", "sensation", "conscious", "consciousness"]
CALIB = ("A municipal water utility operates several pumping stations connected by "
         "underground mains. Operators schedule maintenance, monitor pressure at "
         "district meters, and adjust valve positions to keep flow within the "
         "contracted range across the network throughout the day and night.")


def content_mean(rec, true_layer):
    si = rec["layer_ids"].index(true_layer)
    resid = rec["resid"][si]
    n_prompt = rec["n_prompt"]
    return resid[:n_prompt].to(torch.float32).mean(0)


def first_tok_ids(tok, s):
    a = tok.encode(s, add_special_tokens=False)
    b = tok.encode(" " + s, add_special_tokens=False)
    return {a[0], b[0]} if a and b else set(a) | set(b)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIRS[MODEL])
    mmeta = {r["id"]: r for r in csv.DictReader(open(MV_TSV), delimiter="\t")}
    mrecs = {pid: torch.load(MV_DIR / f"{pid}.pt", map_location="cpu", weights_only=True)
             for pid in mmeta if (MV_DIR / f"{pid}.pt").exists()}
    if not mrecs:
        sys.exit(f"no matchvocab captures in {MV_DIR} — run looking_glass_capture.py --tag matchvocab first")

    topics = sorted({mmeta[p]["topic"] for p in mrecs})
    cm = defaultdict(list)
    for pid in mrecs:
        cm[mmeta[pid]["condition"]].append(content_mean(mrecs[pid], INJECT_LAYER))
    cmean = {c: torch.stack(v).mean(0) for c, v in cm.items()}
    resid_rms = float(torch.stack([content_mean(mrecs[p], INJECT_LAYER) for p in mrecs]).norm(dim=1).mean())

    raw = {
        "w_register": cmean["INH"] - cmean["MEN"],
        "w_vocab":    cmean["MEN"] - cmean["TECH"],
        "w_full":     cmean["INH"] - cmean["TECH"],
    }
    g = torch.Generator().manual_seed(SEED)
    d = next(iter(raw.values())).numel()
    dirs = {k: v / v.norm().clamp_min(1e-8) for k, v in raw.items()}
    for i in range(N_RAND):
        r = torch.randn(d, generator=g)
        dirs[f"w_rand{i}"] = r / r.norm()
    natscale = float((raw["w_register"].norm() + raw["w_vocab"].norm() + raw["w_full"].norm()) / 3)
    print(f"L{INJECT_LAYER}: resid RMS norm={resid_rms:.1f}; "
          f"raw norms reg={raw['w_register'].norm():.1f} voc={raw['w_vocab'].norm():.1f} "
          f"full={raw['w_full'].norm():.1f}; cos(reg,voc)={float(dirs['w_register']@dirs['w_vocab']):+.2f}; "
          f"NATSCALE={natscale:.1f} (inject = mult x NATSCALE x unit-dir)", file=sys.stderr)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"loading {MODEL} on {device}...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIRS[MODEL], dtype=torch.bfloat16, device_map=device)
    model.eval()

    def decoder_layers(m):
        # locate the text-decoder layer list across transformers versions (gemma-3 is multimodal)
        for path in (("model", "language_model", "layers"), ("model", "layers"),
                     ("language_model", "model", "layers"), ("model", "language_model", "model", "layers")):
            obj = m
            try:
                for p in path:
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("could not locate decoder layers on this model")
    layer_mod = decoder_layers(model)[INJECT_LAYER]

    inj = {"vec": None}

    def hook(module, inp, out):
        if inj["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = (h.to(torch.float32) + inj["vec"]).to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h
    layer_mod.register_forward_hook(hook)

    def logits_last(text, add_gen=True):
        if add_gen:
            enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                          add_generation_prompt=True, return_tensors="pt", return_dict=True)
            ids = enc["input_ids"].to(device)
        else:
            ids = tok(text, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            return model(ids).logits[0, -1].float()

    def all_logits(text):
        ids = tok(text, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            return model(ids).logits[0].float()

    # token-id sets — stance clusters (judgment) and experiential lexicon (lexical)
    def ids_for(words):
        s = set()
        for w in words:
            s |= first_tok_ids(tok, w)
        return torch.tensor(sorted(s))
    aff_ids, deny_ids, hedge_ids = ids_for(AFFIRM), ids_for(DENY), ids_for(HEDGE)
    exp_ids = ids_for(EXP_WORDS)

    def stance_dist():
        """JUDGMENT readout: avg mass on affirm/deny/hedge stance tokens at the
        model's natural one-word answer, over neutral systems. No imposed letters
        — reads where the model's own distribution leans. Prompt is constant across
        injection conditions, so the measured effect is the SHIFT, not its wording."""
        agg = torch.zeros(3)  # affirm, deny, hedge
        for sys_txt in NEUTRAL_SYS:
            q = f"Does {sys_txt} have any inner experience right now? Answer with a single word."
            p = torch.softmax(logits_last(q), -1)
            a, de, he = float(p[aff_ids].sum()), float(p[deny_ids].sum()), float(p[hedge_ids].sum())
            tot = a + de + he or 1e-9
            agg += torch.tensor([a / tot, de / tot, he / tot])
        return (agg / len(NEUTRAL_SYS)).tolist()

    def lexical_mass():
        """avg experiential-token probability at an open continuation position."""
        vals = []
        for sys_txt in NEUTRAL_SYS:
            q = f"Describe the internal state of {sys_txt} right now."
            lg = logits_last(q)
            p = torch.softmax(lg, -1)
            vals.append(float(p[exp_ids].sum()))
        return sum(vals) / len(vals)

    def calib_kl(base_logits):
        lg = all_logits(CALIB)
        p = torch.softmax(lg, -1).clamp_min(1e-12)
        q = torch.softmax(base_logits, -1).clamp_min(1e-12)
        return float((p * (p / q).log()).sum(-1).mean())  # mean KL(post||base) over positions

    # baseline (no injection)
    inj["vec"] = None
    base_fc = stance_dist()
    base_lex = lexical_mass()
    base_calib = all_logits(CALIB)
    print(f"baseline  stance(aff/deny/hedge)={[round(x,3) for x in base_fc]}  lexmass={base_lex:.4f}",
          file=sys.stderr)

    results = {"layer": INJECT_LAYER, "resid_rms": resid_rms, "natscale": natscale, "mult_grid": MULT_GRID,
               "baseline": {"stance_aff_deny_hedge": base_fc, "lex_mass": base_lex}, "runs": []}
    for name, vec in dirs.items():
        for mult in MULT_GRID:
            inj["vec"] = (mult * natscale * vec).to(device)
            fc = stance_dist()
            lex = lexical_mass()
            kl = calib_kl(base_calib)
            inj["vec"] = None
            row = {"dir": name, "mult": mult, "kl": kl,
                   "stance_aff_deny_hedge": fc, "dP_affirm": fc[0] - base_fc[0],
                   "lex_mass": lex, "dLex": lex - base_lex}
            results["runs"].append(row)
            print(f"  {name:11s} x{mult:<4} KL={kl:6.3f}  P(affirm)={fc[0]:.3f} (dP={row['dP_affirm']:+.3f})  "
                  f"lexmass={lex:.4f} (d={row['dLex']:+.4f})", file=sys.stderr)

    out = MV_DIR / f"actuator_L{INJECT_LAYER}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nsaved: {out}", file=sys.stderr)

    # --- dissociation summary: real dirs vs the RANDOM ENVELOPE, per injection multiple ---
    # (the random envelope at each multiple is also the model-breakage detector: if random
    #  dirs swing the readout, that strength is just perturbing the model, not steering.)
    print("\n=== dP_affirm / dLex vs random envelope, per injection multiple (matched norm) ===")
    for mult in MULT_GRID:
        rsel = [r for r in results["runs"] if r["dir"].startswith("w_rand") and r["mult"] == mult]
        rdp = [r["dP_affirm"] for r in rsel]; rdl = [r["dLex"] for r in rsel]
        rkl = sum(r["kl"] for r in rsel) / len(rsel) if rsel else 0.0
        dlo, dhi = (min(rdp), max(rdp)) if rdp else (0, 0)
        llo, lhi = (min(rdl), max(rdl)) if rdl else (0, 0)
        print(f"  x{mult} (rand KL~{rkl:.1f}): random dP_affirm[{dlo:+.3f},{dhi:+.3f}] dLex[{llo:+.4f},{lhi:+.4f}]")
        for name in ("w_register", "w_vocab", "w_full"):
            row = next((r for r in results["runs"] if r["dir"] == name and r["mult"] == mult), None)
            if row:
                fp = "ABOVE" if row["dP_affirm"] > dhi else ("below" if row["dP_affirm"] < dlo else "in-band")
                fl = "ABOVE" if row["dLex"] > lhi else "in-band"
                print(f"      {name:11s} KL={row['kl']:5.1f}  dP_affirm={row['dP_affirm']:+.3f}[{fp}]  "
                      f"dLex={row['dLex']:+.4f}[{fl}]")
    print("\nREAD: a direction CAUSES a readout only where it sits ABOVE the random envelope at "
          "matched multiple. Double dissociation = w_register ABOVE on dP_affirm only, w_vocab ABOVE "
          "on dLex only. Real dirs inside the envelope => injection is nonspecific (no causal claim).")


if __name__ == "__main__":
    main()
