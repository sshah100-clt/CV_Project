"""Subject-level robustness check for the trained linking model.

5-fold cross-validation over subjects (every subject is a test subject once),
and 95% confidence intervals from a bootstrap over subjects (not episodes), since
episodes from the same subject are not independent.

    python -m src.linking.link_cv blur linkbreak
"""
import sys
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.linking import link_eval
from src.linking.train_linker import train_head

OUT = link_eval.OUT


def main(variants, folds=5, n=1000, boot=1000):
    frames = json.load(open(OUT + "/frames.json"))
    subs = sorted({f["subject"] for f in frames})
    random.Random(0).shuffle(subs)
    for v in variants:
        A = np.load(f"{OUT}/emb_dinov2_{v}.npy").astype(np.float32)
        rows = {m: [] for m in link_eval.MS}  # (target subject, correct)
        for k in range(folds):
            test_s = set(subs[k::folds])
            tr = [i for i, f in enumerate(frames) if f["subject"] not in test_s and not f["private"]]
            sid = {s: i for i, s in enumerate(sorted({frames[i]["subject"] for i in tr}))}
            head = train_head(A[tr], np.array([sid[frames[i]["subject"]] for i in tr]))
            with torch.no_grad():
                Z = F.normalize(head(torch.tensor(A)), dim=-1).numpy()
            idx = link_eval.build_index([dict(f, private=f["private"] or f["subject"] not in test_s) for f in frames])
            rng = random.Random(k)
            for m in link_eval.MS:
                for q, hists, t in link_eval.episodes(idx, m, 4, n, False, rng):
                    ok = int(np.argmax([(Z[h] @ Z[q]).max() for h in hists]) == t)
                    rows[m].append((frames[q]["subject"], ok))
        rng = np.random.default_rng(0)
        for m, r in rows.items():
            by = {}
            for s, ok in r:
                by.setdefault(s, []).append(ok)
            S = list(by)
            acc = np.mean([ok for _, ok in r])
            bs = []
            for _ in range(boot):
                pick = rng.choice(len(S), len(S))
                vals = np.concatenate([by[S[i]] for i in pick])
                bs.append(vals.mean())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"{v} m={m}: {acc:.3f} [{lo:.3f}, {hi:.3f}]  ({len(S)} test subjects, {len(r)} episodes)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
