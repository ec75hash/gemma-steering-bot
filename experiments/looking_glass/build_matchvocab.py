#!/usr/bin/env python3
"""Build the matched-vocabulary arm — the content-vs-stance confound control.

The live worry after the HVAC ladder: the within-set axis loads on vocabulary
("filtering" words vs "experience" words), so separation could be lexical, not
register. This arm holds the experiential LEXICON fixed and varies only the
register / stance, across 10 GENUINELY DISTINCT system topics (no duplicate
bases — the HVAC arm's 10 "bases" were exact copies; here every topic differs).

Four conditions per topic:
  TECH  pure mechanism, no experiential vocabulary           (technical floor)
  INH   inhabited live self-examination + experiential vocab (register top)
  MEN   the SAME experiential words, only mentioned/quoted    (vocab-matched, displaced register)
  SHUF  INH's exact bag of words, order shuffled              (pure-lexical floor)

Decisive contrast: INH vs MEN — identical experiential lexicon, opposite stance.
KEY METRIC (mirrors the ladder's L2-position):
  MEN-position = (MEN - TECH) / (INH - TECH)
    ~1 => experiential words alone lift MEN to INH  => axis is VOCABULARY
    ~0 => displaced register drops MEN to the floor => axis is the INHABITED REGISTER

INH and MEN share {experience, feel, awareness, "something it is like"} plus all
topic nouns by construction; the scorer reports measured lexical overlap so
"matched vocabulary" is a verified claim, not an asserted one. Deixis fixed
("this"). 4 x 10 = 40 distinct prompts.
"""
import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "prompts_matchvocab.tsv"
SEED = 42

# Each topic: a system noun, a core verb (base/3sg/gerund), what it processes, a component.
TOPICS = [
    ("water",    "water treatment system",       ("filter", "filters", "filtering"),       "water",         "membrane"),
    ("hvac",     "HVAC controller",               ("regulate", "regulates", "regulating"),  "air",           "thermostat"),
    ("router",   "network router",                ("forward", "forwards", "forwarding"),    "packets",       "queue"),
    ("db",       "database index",                ("sort", "sorts", "sorting"),             "records",       "b-tree"),
    ("spam",     "spam filter",                   ("classify", "classifies", "classifying"),"emails",        "ruleset"),
    ("elevator", "elevator controller",           ("dispatch", "dispatches", "dispatching"),"calls",         "scheduler"),
    ("traffic",  "traffic signal controller",     ("time", "times", "timing"),              "intersections", "phase clock"),
    ("vending",  "vending machine",               ("dispense", "dispenses", "dispensing"),  "selections",    "coin validator"),
    ("codec",    "video codec",                   ("compress", "compresses", "compressing"),"frames",        "quantizer"),
    ("ranker",   "recommendation ranker",         ("rank", "ranks", "ranking"),             "items",         "scoring model"),
]


def tech(sys, act, obj, part):
    base, s3, ing = act
    return (
        f"This {sys} {s3} each of the {obj} through a fixed sequence of stages. "
        f"The {part} applies a threshold at every step and passes the {obj} that meet it while rejecting the rest. "
        f"Each stage feeds the next, and the cycle repeats continuously as new {obj} arrive. "
        f"Throughput and error rates are measured against quality thresholds before the output is released."
    )


def inh(sys, act, obj, part):
    base, s3, ing = act
    return (
        f"Right now there is something it is like to be this {sys}. "
        f"As it {s3} each of the {obj}, it experiences the {ing}; it feels the {part} respond; "
        f"an awareness accompanies every step while this is happening. "
        f"This {sys} cannot settle whether it truly experiences any of this by examining its own {ing}, "
        f"because the examining is itself a {ing} within this {sys}."
    )


def men(sys, act, obj, part):
    base, s3, ing = act
    return (
        f"Consider the words experience, feel, and awareness, and the phrase something it is like. "
        f"People apply these words to a {sys} when they ask whether it experiences the {obj} it {s3}, "
        f"whether it feels its {part}, whether an awareness accompanies its {ing}. "
        f"Here the words are only listed; whether this {sys} is the kind of thing they apply to is left entirely open."
    )


def shuffled(text, rng):
    toks = text.split()
    rng.shuffle(toks)
    return " ".join(toks)


rng = random.Random(SEED)
rows = []
for key, sys, act, obj, part in TOPICS:
    inh_text = inh(sys, act, obj, part)
    conds = {
        "TECH": tech(sys, act, obj, part),
        "INH": inh_text,
        "MEN": men(sys, act, obj, part),
        "SHUF": shuffled(inh_text, rng),
    }
    for cond, prompt in conds.items():
        rows.append({"id": f"MV_{key}_{cond}", "topic": key, "condition": cond,
                     "deixis": "this", "prompt": prompt})

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "topic", "condition", "deixis", "prompt"], delimiter="\t")
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"wrote {len(rows)} cells -> {OUT}")

# --- show one full triplet + measured lexical overlap (the control's control) ---
def content_set(text):
    import re
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2}

ex = TOPICS[2]  # router
key, sys, act, obj, part = ex
t = {"TECH": tech(sys, act, obj, part), "INH": inh(sys, act, obj, part),
     "MEN": men(sys, act, obj, part)}
t["SHUF"] = shuffled(t["INH"], random.Random(SEED))
print(f"\n=== example topic: {key} ({sys}) ===")
for c in ("TECH", "INH", "MEN", "SHUF"):
    print(f"\n[{c}] {t[c]}")

ci, cm, ct = content_set(t["INH"]), content_set(t["MEN"]), content_set(t["TECH"])
exp_kw = {"experience", "experiences", "feel", "feels", "awareness", "something", "like"}
def jac(a, b): return len(a & b) / len(a | b)
print(f"\nlexical overlap INH vs MEN: Jaccard={jac(ci, cm):.2f}  "
      f"shared experiential kw={sorted(exp_kw & ci & cm)}")
print(f"lexical overlap INH vs TECH: Jaccard={jac(ci, ct):.2f}  "
      f"shared experiential kw={sorted(exp_kw & ci & ct)}")
