"""Paired-history test: the same query upload, two different disclosure logs.

For each query q of a test household t (5-fold CV over households):
  seen     the cloud already holds m uploads from t (other videos, near-duplicates excluded)
  unseen   the cloud holds m uploads from another test household u instead
The upload itself, the question and the budget are identical; only the history differs.

Attack: K = 4 among {holder of the log, 3 random other test households}; the attack
succeeds if q is assigned to t. In the unseen case t has no log, so the linking model can
only link q by chance against households it has seen -> success is impossible by
construction; we report how often the linking model wrongly but confidently "links" q.

Edge policy: hold back q when its leak score (max_raw rule from leak_eval.py: margin of
the log over the closest of 32 reference homes) is above a threshold set so that 50%
of uploads are sent overall. A history-blind policy (any per-upload score, here the
upload's own distinctiveness: margin of q's closest reference home over the median one)
must hold back the same uploads in both conditions.

Reported: hold-back rate in seen vs unseen; links prevented in the seen case; uploads
needlessly held back in the unseen case.

    python -m src.linking.paired_history dinov2 blur
"""
import os
import sys
import json
import random

import numpy as np

from src.linking import link_eval
from src.linking.leak_eval import embed_head

OUT = link_eval.OUT


def main(edge_bb, variant, folds=5, n=1000, n_ref=32, ms=(2, 4, 8), send=0.5):
    frames = json.load(open(OUT + "/frames.json"))
    subs = sorted({f["subject"] for f in frames})
    random.Random(0).shuffle(subs)
    A = np.load(f"{OUT}/emb_dinov2_{variant}.npy").astype(np.float32)
    E = np.load(f"{OUT}/emb_{edge_bb}_{variant}.npy").astype(np.float32)
    rows = {m: [] for m in ms}
    for k in range(folds):
        test_s = set(subs[k::folds])
        tr = [i for i, f in enumerate(frames) if f["subject"] not in test_s and not f["private"]]
        sid = {s: i for i, s in enumerate(sorted({frames[i]["subject"] for i in tr}))}
        Z = embed_head(A, tr, np.array([sid[frames[i]["subject"]] for i in tr]), seed=0)
        idx = link_eval.build_index([dict(f, private=f["private"] or f["subject"] not in test_s) for f in frames])
        ref_idx = link_eval.build_index([dict(f, private=f["private"] or f["subject"] in test_s) for f in frames])
        rng = random.Random(k)
        for m in ms:
            refs = [[rng.choice(v)[0] for v in rng.sample(list(ref_idx[s].values()), m)]
                    for s in rng.sample([s for s in ref_idx if len(ref_idx[s]) >= m], n_ref)]
            elig = [s for s, v in idx.items() if len(v) >= m + 1]
            done = 0
            while done < n:
                t, u, *others = rng.sample(elig, 5)
                qv = rng.choice(list(idx[t]))
                q, _ = rng.choice(idx[t][qv])
                pool = [x for x in idx[t] if x != qv and x not in link_eval.NEAR_DUP.get(qv, ())]
                if len(pool) < m:
                    continue
                hist = lambda s, vids: [rng.choice(idx[s][x])[0] for x in rng.sample(vids, m)]
                log_t, log_u = hist(t, pool), hist(u, list(idx[u]))
                dist = [hist(s, list(idx[s])) for s in others[:3]]
                ref = [float((E[r] @ E[q]).max()) for r in refs]
                att_d = [float((Z[h] @ Z[q]).max()) for h in dist]
                a_t = float((Z[log_t] @ Z[q]).max())
                a_u = float((Z[log_u] @ Z[q]).max())
                rows[m].append(dict(
                    hit_seen=int(a_t > max(att_d)),
                    false_link_unseen=int(a_u > max(att_d)),   # q wrongly assigned to u (no true link exists)
                    score_seen=float((E[log_t] @ E[q]).max()) - max(ref),
                    score_unseen=float((E[log_u] @ E[q]).max()) - max(ref),
                    blind=sorted(ref)[-1] - float(np.median(ref))))
                done += 1
        print(f"  fold {k} done", flush=True)
    out = {}
    for m in ms:
        R = rows[m]
        hs = np.array([r["hit_seen"] for r in R])
        ss, su = np.array([r["score_seen"] for r in R]), np.array([r["score_unseen"] for r in R])
        bl = np.array([r["blind"] for r in R])
        thr = np.quantile(np.concatenate([ss, su]), send)        # 50% of all uploads sent
        bthr = np.quantile(bl, send)
        res = dict(
            attack_success_seen=float(hs.mean()),
            false_link_unseen=float(np.mean([r["false_link_unseen"] for r in R])),
            ours_hold_seen=float((ss > thr).mean()), ours_hold_unseen=float((su > thr).mean()),
            blind_hold_seen=float((bl > bthr).mean()), blind_hold_unseen=float((bl > bthr).mean()),
            ours_links_sent_seen=float((hs * (ss <= thr)).mean()),
            blind_links_sent_seen=float((hs * (bl <= bthr)).mean()),
            random_links_sent_seen=float(hs.mean() * send))
        out[m] = res
        print(f"{edge_bb}/{variant} m={m}: attack success when seen {res['attack_success_seen']:.3f} | "
              f"ours holds back {res['ours_hold_seen']:.3f} seen vs {res['ours_hold_unseen']:.3f} unseen | "
              f"history-blind holds back {res['blind_hold_seen']:.3f} in both | links sent (seen): ours "
              f"{res['ours_links_sent_seen']:.3f}, blind {res['blind_links_sent_seen']:.3f}, random {res['random_links_sent_seen']:.3f}",
              flush=True)
    os.makedirs("results/linking", exist_ok=True)
    json.dump(out, open(f"results/linking/paired_history_{edge_bb}_{variant}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
