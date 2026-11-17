"""Trained linking model: a projection head on frozen DINOv2 features, learned with a
supervised contrastive loss on subject labels from TRAIN subjects only, then
tested with the same K-way episodes on held-out subjects. It is trained on the
same upload variant it is tested on (it adapts to the cleaning operator).

    python -m src.linking.train_linker blur bgblur linkbreak ...
"""
import sys
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.linking import link_eval

OUT = link_eval.OUT


def split_subjects(frames, seed=0, frac_test=0.4):
    subs = sorted({f["subject"] for f in frames})
    random.Random(seed).shuffle(subs)
    return set(subs[int(len(subs) * frac_test):]), set(subs[: int(len(subs) * frac_test)])


def train_head(X, y, epochs=30, dim=256, tau=0.1, seed=0):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(X.shape[1], 1024), nn.GELU(), nn.Dropout(0.2), nn.Linear(1024, dim))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    X, y = torch.tensor(X), torch.tensor(y)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            b = perm[i:i + 512]
            z = F.normalize(head(X[b]), dim=-1)
            sim = z @ z.T / tau
            same = (y[b][:, None] == y[b][None, :]).float()
            eye = torch.eye(len(b))
            same = same - eye
            logp = sim - torch.logsumexp(sim - 1e9 * eye, dim=1, keepdim=True)
            loss = -((logp * same).sum(1) / same.sum(1).clamp(min=1))[same.sum(1) > 0].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.eval()


def main(variants, backbone="dinov2"):
    frames = json.load(open(OUT + "/frames.json"))
    train_s, test_s = split_subjects(frames)
    sid = {s: i for i, s in enumerate(sorted(train_s))}
    tr = [i for i, f in enumerate(frames) if f["subject"] in train_s and not f["private"]]
    test_frames = [dict(f, private=f["private"] or f["subject"] not in test_s) for f in frames]
    for v in variants:
        E = np.load(f"{OUT}/emb_{backbone}_{v}.npy").astype(np.float32)
        head = train_head(E[tr], np.array([sid[frames[i]["subject"]] for i in tr]))
        with torch.no_grad():
            Z = F.normalize(head(torch.tensor(E)), dim=-1).numpy()
        print(f"== {v}: untrained NN on test subjects", flush=True)
        link_eval.run(E, test_frames)
        print(f"== {v}: trained linking model on test subjects", flush=True)
        link_eval.run(Z, test_frames)


if __name__ == "__main__":
    main(sys.argv[1:])
