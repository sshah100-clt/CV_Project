"""Summary of the HD-EPIC pilot runs (P01 dev questions): accuracy of every saved QA run, paired differences, accuracy
per category, checks on the device's notes, grounding boxes and planner, and a confidence router.
95% intervals: bootstrap over videos (questions about one video are correlated).

    python -m src.analysis.analyze_pilot [--results data/hdepic/results] [--vqa data/hdepic/vqa]
"""
import os
import re
import glob
import json
import argparse
from collections import Counter
from itertools import product

import numpy as np

T = "f32_pt30_dense_dev"
LOCAL = "3B raw (local)"
NAMES = {
    ("Qwen2.5-VL-3B-Instruct", "raw"): LOCAL,
    ("Qwen2.5-VL-7B-Instruct", "blur"): "7B frames",
    ("Qwen2.5-VL-32B-Instruct", "blur"): "32B frames",
    ("Qwen2.5-VL-7B-Instruct", "cropguided"): "7B guided crop",
    ("Qwen2.5-VL-7B-Instruct", "cropcenter"): "7B centre crop",
    ("Qwen2.5-VL-7B-Instruct", "cropfull"): "7B full frames 756px",
    ("Qwen2.5-VL-7B-Instruct", "gated"): "7B gated frames",
    ("Qwen2.5-VL-7B-Instruct", "notes"): "7B notes",
    ("Qwen2.5-VL-3B-Instruct", "labels"): "3B labels",
    ("Qwen2.5-VL-7B-Instruct", "labels"): "7B labels",
    ("Qwen2.5-VL-7B-Instruct", "objonly"): "7B objects only",
    ("Qwen2.5-VL-3B-Instruct", "blind"): "3B blind",
    ("Qwen2.5-VL-7B-Instruct", "blind"): "7B blind",
    ("Qwen2.5-VL-32B-Instruct", "blind"): "32B blind",
    ("gpt-4o-2024-11-20", "blur"): "GPT-4o frames",
    ("gpt-4o-2024-11-20", "cropcenter"): "GPT-4o centre crop",
    ("gpt-4o-2024-11-20", "cropguided"): "GPT-4o guided crop",
    ("gpt-4o-2024-11-20", "notes"): "GPT-4o notes",
    ("gpt-4o-2024-11-20", "labels"): "GPT-4o labels",
    ("gpt-4o-2024-11-20", "objonly"): "GPT-4o objects only",
    ("gpt-4o-2024-11-20", "blind"): "GPT-4o blind",
    ("gpt-4o-2024-11-20", "boxsize"): "GPT-4o box+size",
    ("gpt-4o-2024-11-20", "boxdraw"): "GPT-4o box drawn",
}
PAIRS = [("7B guided crop", "7B centre crop"), ("7B guided crop", "7B frames"), ("7B centre crop", "7B full frames 756px"),
         ("7B full frames 756px", "7B frames"), ("7B notes", "7B frames"), ("GPT-4o frames", "7B frames"),
         ("GPT-4o frames", "32B frames"), ("GPT-4o centre crop", "GPT-4o frames"), ("GPT-4o box+size", "GPT-4o frames"),
         ("GPT-4o box drawn", "GPT-4o frames")]

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="data/hdepic/results")
ap.add_argument("--vqa", default="data/hdepic/vqa")
a = ap.parse_args()
rng = np.random.default_rng(0)


def rd(p):
    out = {}
    for line in open(p):
        x = json.loads(line)
        out[x["id"]] = x
    return out


def boot(vals, keys, n=4000):
    vals, keys = np.asarray(vals, float), np.asarray(keys)
    u = sorted(set(keys))
    S = np.array([vals[keys == k].sum() for k in u])
    N = np.array([(keys == k).sum() for k in u])
    b = []
    for _ in range(n):
        j = rng.integers(0, len(u), len(u))
        b.append(S[j].sum() / N[j].sum())
    return 100 * vals.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    p, n = s[y == 1], s[y == 0]
    return np.mean([(x > z) + 0.5 * (x == z) for x, z in product(p, n)]) if len(p) and len(n) else float("nan")


f = lambda n: os.path.join(a.results, n)
C = {}
for path in sorted(glob.glob(f(f"qa_*_{T}.jsonl"))):
    model, form = os.path.basename(path)[3:-len(f"_{T}.jsonl")].rsplit("_", 1)
    C[NAMES.get((model, form), f"{model} {form}")] = os.path.basename(path)
order = list(NAMES.values())
C = dict(sorted(C.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)))
D = {k: rd(f(v)) for k, v in C.items()}
ids = sorted(D[LOCAL])
qa = {qid: q for fn in glob.glob(os.path.join(a.vqa, "*.json")) for qid, q in json.load(open(fn)).items()}
vid = {i: D[LOCAL][i]["video"] for i in ids}
cat = {i: D[LOCAL][i]["prototype"].split("_")[0] for i in ids}
hasbox = {i: "<BBOX" in qa[i]["question"] for i in ids}
corr = lambda k, I: [float(D[k][i]["pred"] == D[k][i]["gold"]) for i in I]

print(f"questions {len(ids)}, videos {len(set(vid.values()))}, chance 20% (5 options)\n")
print("== accuracy (95% CI, bootstrap over videos)")
for k in C:
    I = [i for i in ids if i in D[k]]
    if not I:
        continue
    m, lo, hi = boot(corr(k, I), [vid[i] for i in I])
    lm = [D[k][i]["letter_mass"] for i in I if D[k][i].get("letter_mass") is not None]
    unp = sum(D[k][i]["pred"] < 0 for i in I)
    print(f"  {k:30s} n={len(I):3d}  {m:5.1f} [{lo:5.1f},{hi:5.1f}]  letter-mass {np.mean(lm) if lm else float('nan'):.3f}"
          f"  unparsed {unp}  frames/q {np.mean([D[k][i]['n_frames'] for i in I]):.1f}")

print("\n== paired differences vs 3B local (same questions)")
for k in C:
    I = [i for i in ids if i in D[k]]
    if k.startswith("3B") or not I:
        continue
    m, lo, hi = boot(np.array(corr(k, I)) - np.array(corr(LOCAL, I)), [vid[i] for i in I])
    print(f"  {k:30s} n={len(I):3d}  {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]")
for x, y in PAIRS:
    if x in D and y in D:
        J = [i for i in ids if i in D[x] and i in D[y]]
        m, lo, hi = boot(np.array(corr(x, J)) - np.array(corr(y, J)), [vid[i] for i in J])
        print(f"  {x} - {y}: {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]")

print("\n== by category (n; " + " / ".join(C) + ")")
groups = [(c, [i for i in ids if cat[i] == c]) for c in sorted(set(cat.values()))]
groups += [("box questions", [i for i in ids if hasbox[i]]), ("no box", [i for i in ids if not hasbox[i]])]
for lab, J in groups:
    print(f"  {lab:13s} n={len(J):3d}  " + "  ".join(
        f"{100 * np.mean([D[k][i]['pred'] == D[k][i]['gold'] for i in J if i in D[k]]):5.1f}" for k in C))

notes_path = f(f"notes_Qwen2.5-VL-3B-Instruct_{T}.jsonl")
r3 = [i for i in ids if D[LOCAL][i]["pred"] == D[LOCAL][i]["gold"]]
if os.path.exists(notes_path):
    print("\n== device notes (3B, options hidden): do they leak the answer?")
    N = rd(notes_path)
    J = [i for i in ids if i in N]
    lead = sum(bool(re.match(r"\s*\(?[A-E][.):]", N[i]["evidence"])) for i in J)
    flag = sum(N[i]["mentions_answer_or_letter"] for i in J)
    copy = sum(any(len(str(c)) > 15 and str(c).strip().lower()[:40] in N[i]["evidence"].lower() for c in qa[i]["choices"])
               for i in J)
    timed = sum(bool(re.search(r"\d{2}:\d{2}:\d{2}", N[i]["evidence"])) for i in J)
    print(f"  n={len(J)}; starts with a letter {lead}; mentions answer/option/letter {flag}; contains an option's text "
          f"(>15 chars) {copy}; has a time stamp {timed}; words median {int(np.median([N[i]['words'] for i in J]))}")
    for k in ("7B notes", "GPT-4o notes"):
        if k in D:
            K = [i for i in J if i in D[k]]
            agree = np.mean([D[k][i]["pred"] == D[LOCAL][i]["pred"] for i in K])
            print(f"  {k}: answer = 3B answer {100 * agree:.1f}%; accuracy when 3B right "
                  f"{100 * np.mean(corr(k, [i for i in K if i in r3])):.1f}, when 3B wrong "
                  f"{100 * np.mean(corr(k, [i for i in K if i not in r3])):.1f}")

ground_path = f(f"ground_Qwen2.5-VL-3B-Instruct_{T}.jsonl")
if os.path.exists(ground_path):
    print("\n== grounding boxes (3B)")
    G = rd(ground_path)
    out = sum(any(float(v) > 448 for v in re.findall(r"-?\d+\.?\d*", G[i]["reply"])[:4]) for i in ids)
    boxes = [tuple(G[i]["box"]) for i in ids]
    area = [(b[2] - b[0]) * (b[3] - b[1]) / 448 ** 2 for b in boxes]
    print(f"  parsed {sum(G[i]['parsed'] for i in ids)}/{len(ids)}; reply has a coordinate > 448: {out}/{len(ids)}; "
          f"area median {np.median(area):.2f}, >= 0.9 of frame: {sum(x >= 0.9 for x in area)}")
    print("  most common boxes:", Counter(boxes).most_common(4))

for pp in sorted(glob.glob(f("plan_Qwen2.5-VL-3B-Instruct_pt*_dev.jsonl"))):
    print("\n== planner (3B, questions whose answer is a time; options hidden)")
    P = [json.loads(line) for line in open(pp)]
    m, lo, hi = boot([float(x["hit"]) - x["p_random"] for x in P], [x["video"] for x in P])
    print(f"  n={len(P)}; valid {sum(x['valid'] for x in P)}; hit {100 * np.mean([x['hit'] for x in P]):.1f} vs random "
          f"{100 * np.mean([x['p_random'] for x in P]):.1f}; difference {m:+.1f} [{lo:+.1f},{hi:+.1f}]")
    for p in sorted({x["prototype"] for x in P}):
        R = [x for x in P if x["prototype"] == p]
        print(f"  {p:48s} n={len(R):3d}  hit {100 * np.mean([x['hit'] for x in R]):5.1f}  "
              f"random {100 * np.mean([x['p_random'] for x in R]):5.1f}")

print("\n== confidence router: 3B raw, escalate the lowest-confidence share to 7B frames (in-sample)")
conf = np.array([max(D[LOCAL][i]["probs"]) for i in ids])
c3, c7 = np.array(corr(LOCAL, ids)), np.array(corr("7B frames", ids))
order = np.argsort(conf)
for frac in (0, .25, .5, .75, 1):
    esc = np.zeros(len(ids), bool)
    esc[order[:int(round(frac * len(ids)))]] = True
    print(f"  escalate {int(frac * 100):3d}%: {100 * np.mean(np.where(esc, c7, c3)):.1f}")
print(f"  AUC of low 3B confidence for 'cloud fixes it': {auc(-conf, (c7 > c3).astype(int)):.2f};  "
      f"3B confidence AUC for 3B correct: {auc(conf, c3):.2f}")
print(f"  oracle best-of(3B, 7B frames): {100 * np.mean(np.maximum(c3, c7)):.1f}")
