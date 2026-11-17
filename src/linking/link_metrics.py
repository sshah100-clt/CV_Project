"""Household-linking metrics beyond K = 4 for the trained linking model.

5-fold cross-validation over subjects (households), as in link_cv.py. For each query
upload of a test household, every test household with enough videos gets a history of
m uploads (m different videos, one random frame each; for the true household the query
video and its near-duplicates are excluded). Score = max cosine between the query and
the history. From the same scores we report:

  gallery   rank-1 over ALL eligible test households of the fold (chance = 1 / gallery size)
  K=4, K=16 top-1 among the true household and K-1 random others
  pairwise  same-household verification: ROC-AUC, TPR at 1% and 0.1% FPR

95% intervals: bootstrap over households (1,000 resamples).

    python -m src.linking.link_metrics blur silhouette personly bgblur linkbreak
    python -m src.linking.link_metrics blur --cross_room
    python -m src.linking.link_metrics blur linkbreak --siglip   # linking head on SigLIP features
    python -m src.linking.link_metrics blur silhouette personly --person   # ablation on frames that contain a person
    python -m src.linking.link_metrics blur personly --face                 # only frames where a face was detected (and blurred)
"""
import os
import sys
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.linking import link_eval
from src.linking.train_linker import train_head

OUT = link_eval.OUT
MS = (1, 2, 4, 8)


def tpr_at(pos, neg, fpr):
    thr = np.quantile(neg, 1 - fpr)
    return float((pos > thr).mean())


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot(by, stat, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    S = list(by)
    vals = []
    for _ in range(n):
        pick = [by[S[i]] for i in rng.choice(len(S), len(S))]
        vals.append(stat(pick))
    return np.percentile(vals, [2.5, 97.5])


def run_variant(v, frames, subs, cross_room=False, folds=5, n_q=600, seed=0, bb="dinov2"):
    A = np.load(f"{OUT}/emb_{bb}_{v}.npy").astype(np.float32)
    res = {m: {"gal": [], "k4": [], "k16": [], "pos": [], "neg": [], "gsize": []} for m in MS}
    for k in range(folds):
        test_s = set(subs[k::folds])
        tr = [i for i, f in enumerate(frames) if f["subject"] not in test_s and not f["private"]]
        sid = {s: i for i, s in enumerate(sorted({frames[i]["subject"] for i in tr}))}
        head = train_head(A[tr], np.array([sid[frames[i]["subject"]] for i in tr]))
        with torch.no_grad():
            Z = F.normalize(head(torch.tensor(A)), dim=-1).numpy()
        idx = link_eval.build_index([dict(f, private=f["private"] or f["subject"] not in test_s) for f in frames])
        rng = random.Random(1000 * k + seed)
        for m in MS:
            gallery = [s for s, vids in idx.items() if len(vids) >= m]
            targets = [s for s, vids in idx.items() if len(vids) >= m + 1]
            done = 0
            while done < n_q:
                t = rng.choice(targets)
                qv = rng.choice(list(idx[t]))
                q, q_scene = rng.choice(idx[t][qv])
                pool = [x for x in idx[t] if x != qv and x not in link_eval.NEAR_DUP.get(qv, ())
                        and (not cross_room or idx[t][x][0][1] != q_scene)]
                if len(pool) < m:
                    continue
                scores = {}
                for s in gallery:
                    vs = rng.sample(pool, m) if s == t else rng.sample(list(idx[s]), m)
                    h = [rng.choice(idx[s][x])[0] for x in vs]
                    scores[s] = float((Z[h] @ Z[q]).max())
                others = [s for s in gallery if s != t]
                r = res[m]
                r["gal"].append((t, int(max(scores, key=scores.get) == t)))
                r["gsize"].append(len(gallery))
                for K, key in ((4, "k4"), (16, "k16")):
                    if len(others) >= K - 1:
                        neg = rng.sample(others, K - 1)
                        r[key].append((t, int(scores[t] > max(scores[s] for s in neg))))
                r["pos"].append((t, scores[t]))
                r["neg"].extend((t, scores[s]) for s in others)
                done += 1
        print(f"  fold {k} done", flush=True)
    out = {}
    for m in MS:
        r = res[m]
        row = {}
        for key in ("gal", "k4", "k16"):
            by = {}
            for s, ok in r[key]:
                by.setdefault(s, []).append(ok)
            acc = np.mean([ok for _, ok in r[key]])
            lo, hi = boot(by, lambda p: np.mean(np.concatenate(p)))
            row[key] = (acc, lo, hi)
        pos = np.array([x for _, x in r["pos"]])
        neg = np.array([x for _, x in r["neg"]])
        byp = {}
        for s, x in r["pos"]:
            byp.setdefault(s, [[], []])[0].append(x)
        for s, x in r["neg"]:
            byp.setdefault(s, [[], []])[1].append(x)
        lo, hi = boot(byp, lambda p: auc(np.concatenate([a for a, _ in p]), np.concatenate([b for _, b in p])), n=300)
        row["auc"] = (auc(pos, neg), lo, hi)
        row["tpr1"] = tpr_at(pos, neg, 0.01)
        row["tpr01"] = tpr_at(pos, neg, 0.001)
        row["n_pos"], row["n_neg"] = len(pos), len(neg)
        row["gsize"] = float(np.mean(r["gsize"]))
        out[m] = row
        f = lambda t: f"{100 * t[0]:.1f} [{100 * t[1]:.1f}, {100 * t[2]:.1f}]"
        print(f"{bb}/{v}{' cross_room' if cross_room else ''} m={m}: gallery rank-1 {f(row['gal'])} (gallery {row['gsize']:.0f}, "
              f"chance {100 / row['gsize']:.1f}) | K=4 {f(row['k4'])} | K=16 {f(row['k16'])} | AUC {row['auc'][0]:.3f} "
              f"[{row['auc'][1]:.3f}, {row['auc'][2]:.3f}] | TPR@1%FPR {100 * row['tpr1']:.1f} | TPR@0.1%FPR "
              f"{100 * row['tpr01']:.1f} | pos {row['n_pos']} neg {row['n_neg']}", flush=True)
    return out


def main(args):
    cross = "--cross_room" in args
    bb = "siglip" if "--siglip" in args else "dinov2"
    person = "--person" in args  # ablation: keep only frames where a person covers > 1% of the image
    face = "--face" in args      # check: keep only frames with >= 1 detected (hence blurred) face
    variants = [a for a in args if not a.startswith("--")]
    frames = json.load(open(OUT + "/frames.json"))
    subs = sorted({f["subject"] for f in frames})
    random.Random(0).shuffle(subs)
    if person:
        has = json.load(open(OUT + "/person_present.json"))
        frames = [dict(f, private=f["private"] or not has[f"{f['video']}_{f['k']}"]) for f in frames]
    if face:
        frames = [dict(f, private=f["private"] or f["faces"] < 1) for f in frames]
    allres = {}
    for v in variants:
        print(f"== {v}", flush=True)
        allres[v] = run_variant(v, frames, subs, cross_room=cross, bb=bb)
    tag = bb + "_" + "_".join(variants) + ("_cross" if cross else "") + ("_person" if person else "") + ("_face" if face else "")
    os.makedirs("results/linking", exist_ok=True)
    json.dump({v: {str(m): r for m, r in d.items()} for v, d in allres.items()},
              open(f"results/linking/link_metrics_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
