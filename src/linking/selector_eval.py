"""Can the edge predict leakage, and does withholding by that score beat random?

Per episode: the target's m earlier uploads are the disclosure log; the query is
the next candidate upload. The ATTACKER (trained head, held-out subjects) decides
success. The EDGE never sees the linking model: its leak score is

    dE(q) = max cos(q, log) - mean_r max cos(q, ref_r)

using its own embedder and a fixed bank of public reference households (train
subjects). At equal sending rate, a policy that withholds the highest-dE uploads
is compared with random withholding.

    python -m src.linking.selector_eval <edge_backbone> <variant> [learned]
"""
import sys
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.linking import link_eval
from src.linking.train_linker import split_subjects, train_head

OUT = link_eval.OUT
RATES = (1.0, 0.75, 0.5, 0.25)


def main(edge_bb, variant, n=2000, n_ref=32, learned=False):
    frames = json.load(open(OUT + "/frames.json"))
    train_s, test_s = split_subjects(frames)
    tr = [i for i, f in enumerate(frames) if f["subject"] in train_s and not f["private"]]
    sid = {s: i for i, s in enumerate(sorted(train_s))}

    A = np.load(f"{OUT}/emb_dinov2_{variant}.npy").astype(np.float32)
    head = train_head(A[tr], np.array([sid[frames[i]["subject"]] for i in tr]))
    with torch.no_grad():
        Z = F.normalize(head(torch.tensor(A)), dim=-1).numpy()
    E = np.load(f"{OUT}/emb_{edge_bb}_{variant}.npy").astype(np.float32)
    if learned:  # the edge's own linker, trained on train subjects with a different backbone
        eh = train_head(E[tr], np.array([sid[frames[i]["subject"]] for i in tr]), seed=1)
        with torch.no_grad():
            E = F.normalize(eh(torch.tensor(E)), dim=-1).numpy()

    test_frames = [dict(f, private=f["private"] or f["subject"] not in test_s) for f in frames]
    idx = link_eval.build_index(test_frames)
    ref_idx = link_eval.build_index([dict(f, private=f["private"] or f["subject"] not in train_s) for f in frames])
    rng = random.Random(0)
    for m in (2, 4, 8):
        refs = [[rng.choice(v)[0] for v in rng.sample(list(ref_idx[s].values()), m)]
                for s in rng.sample([s for s in ref_idx if len(ref_idx[s]) >= m], n_ref)]
        eps = link_eval.episodes(idx, m, 4, n, False, rng)
        hit, score = [], []
        for q, hists, t in eps:
            hit.append(int(np.argmax([(Z[h] @ Z[q]).max() for h in hists]) == t))
            log = hists[t]
            score.append((E[log] @ E[q]).max() - np.mean([(E[r] @ E[q]).max() for r in refs]))
        hit, score = np.array(hit), np.array(score)
        auc = roc_auc_score(hit, score)
        line = [f"m={m}: attack success {hit.mean():.3f}; edge dE AUC {auc:.3f};"]
        order = np.argsort(score)  # lowest dE first = safest
        for r in RATES:
            k = int(round(r * len(hit)))
            line.append(f"send {int(r * 100)}%: random {hit.mean() * r:.3f} vs dE {hit[order[:k]].sum() / len(hit):.3f}")
        print(" | ".join(line), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], learned=len(sys.argv) > 3 and sys.argv[3] == "learned")
