"""Edge leak score under 5-fold cross-validation over households.

The ATTACKER (trained head, seed 0, train folds) decides whether a query upload is linked
to the right household among K = 4. The EDGE never sees the linking model or the other
candidate households; it knows only its own household's disclosure log and a fixed bank
of 32 reference households from the training folds. Edge scores:

  mean_raw   max cos(q, log) - mean_r max cos(q, ref_r)      (raw embedder)
  max_raw    max cos(q, log) - max_r  max cos(q, ref_r)      (margin over the closest other home)
  max_head   as max_raw, on the edge's own contrastive head (trained on the train folds, seed 1)
  ens_head   mean of max_head over two edge heads, on DINOv2 and on SigLIP features
  oracle     the linking model's own margin over the true distractors (upper bound, not available on device)

Reported: AUC for predicting attack success (bootstrap over households), and the share of
episodes in which an upload was SENT AND LINKED when the edge holds back the
highest-scoring 50% / 75% of uploads, versus holding back at random.

    python -m src.linking.leak_eval dinov2 blur [siglip]   # edge backbone, upload variant, linking-model backbone (default dinov2)
"""
import os
import sys
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.linking import link_eval
from src.linking.train_linker import train_head

OUT = link_eval.OUT


def embed_head(X, tr, y, seed):
    h = train_head(X[tr], y, seed=seed)
    with torch.no_grad():
        return F.normalize(h(torch.tensor(X)), dim=-1).numpy()


def main(edge_bb, variant, att_bb="dinov2", folds=5, n=1000, n_ref=32, ms=(2, 4, 8)):
    frames = json.load(open(OUT + "/frames.json"))
    subs = sorted({f["subject"] for f in frames})
    random.Random(0).shuffle(subs)
    A = np.load(f"{OUT}/emb_{att_bb}_{variant}.npy").astype(np.float32)
    E0 = np.load(f"{OUT}/emb_{edge_bb}_{variant}.npy").astype(np.float32)
    ED = np.load(f"{OUT}/emb_dinov2_{variant}.npy").astype(np.float32)   # edge ensemble members
    ES = np.load(f"{OUT}/emb_siglip_{variant}.npy").astype(np.float32)
    rows = {m: [] for m in ms}
    for k in range(folds):
        test_s = set(subs[k::folds])
        tr = [i for i, f in enumerate(frames) if f["subject"] not in test_s and not f["private"]]
        sid = {s: i for i, s in enumerate(sorted({frames[i]["subject"] for i in tr}))}
        y = np.array([sid[frames[i]["subject"]] for i in tr])
        Z = embed_head(A, tr, y, seed=0)       # attacker
        EH = embed_head(E0, tr, y, seed=1)     # edge's own linker
        EHD, EHS = embed_head(ED, tr, y, seed=1), embed_head(ES, tr, y, seed=1)
        idx = link_eval.build_index([dict(f, private=f["private"] or f["subject"] not in test_s) for f in frames])
        ref_idx = link_eval.build_index([dict(f, private=f["private"] or f["subject"] in test_s) for f in frames])
        rng = random.Random(k)
        for m in ms:
            refs = [[rng.choice(v)[0] for v in rng.sample(list(ref_idx[s].values()), m)]
                    for s in rng.sample([s for s in ref_idx if len(ref_idx[s]) >= m], n_ref)]
            for q, hists, t in link_eval.episodes(idx, m, 4, n, False, rng):
                att = [float((Z[h] @ Z[q]).max()) for h in hists]
                hit = int(np.argmax(att) == t)
                log = hists[t]
                r_raw = [float((E0[r] @ E0[q]).max()) for r in refs]
                r_head = [float((EH[r] @ EH[q]).max()) for r in refs]
                l_raw = float((E0[log] @ E0[q]).max())
                l_head = float((EH[log] @ EH[q]).max())
                ens = np.mean([float((H[log] @ H[q]).max()) - max(float((H[r] @ H[q]).max()) for r in refs) for H in (EHD, EHS)])
                rows[m].append(dict(s=frames[q]["subject"], hit=hit,
                                    mean_raw=l_raw - np.mean(r_raw), max_raw=l_raw - max(r_raw),
                                    max_head=l_head - max(r_head), ens_head=ens,
                                    oracle=att[t] - max(a for i, a in enumerate(att) if i != t)))
        print(f"  fold {k} done", flush=True)
    out = {}
    brng = np.random.default_rng(0)
    for m in ms:
        R = rows[m]
        hit = np.array([r["hit"] for r in R])
        S = sorted({r["s"] for r in R})
        by = {s: [i for i, r in enumerate(R) if r["s"] == s] for s in S}
        line, res = [f"edge {edge_bb} vs attacker {att_bb}/{variant} m={m}: attack success {hit.mean():.3f} (n={len(R)}, {len(S)} households)"], {}
        for key in ("mean_raw", "max_raw", "max_head", "ens_head", "oracle"):
            sc = np.array([r[key] for r in R])
            a = roc_auc_score(hit, sc)
            bs = []
            for _ in range(300):
                ii = np.concatenate([by[S[j]] for j in brng.choice(len(S), len(S))])
                if 0 < hit[ii].mean() < 1:
                    bs.append(roc_auc_score(hit[ii], sc[ii]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            order = np.argsort(sc)  # lowest score = safest, sent first
            sent = {rate: float(hit[order[: int(round(rate * len(hit)))]].sum() / len(hit)) for rate in (0.75, 0.5, 0.25)}
            res[key] = dict(auc=a, lo=lo, hi=hi, **{f"send{int(r * 100)}": v for r, v in sent.items()})
            line.append(f"{key}: AUC {a:.3f} [{lo:.3f}, {hi:.3f}], linked@send50 {sent[0.5]:.3f}")
        res["random"] = {f"send{int(r * 100)}": float(hit.mean() * r) for r in (0.75, 0.5, 0.25)}
        line.append(f"random linked@send50 {hit.mean() * 0.5:.3f}")
        print(" | ".join(line), flush=True)
        out[m] = res
    os.makedirs("results/linking", exist_ok=True)
    json.dump(out, open(f"results/linking/leak_eval_{edge_bb}_vs_{att_bb}_{variant}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
