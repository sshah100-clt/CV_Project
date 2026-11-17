"""Local utility proxy: zero-shot SigLIP action recognition on each upload variant.

For every non-private frame we know which Charades actions are happening at that
timestamp. SigLIP scores the 157 action names against the (cleaned) frame; the
macro mAP tells how much task-relevant content an operator keeps. It stands in
for VLM question answering (cheap check on a CPU).

    python -m src.qa.action_proxy blur bgblur linkbreak ...   (needs emb_siglip_<variant>.npy)
"""
import sys
import csv
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoModel, AutoTokenizer

OUT = "data/charades_link"
CH = "data/charades"
POSITIONS = (0.25, 0.5, 0.75)


def labels(frames):
    meta = {}
    for split in ("train", "test"):
        for r in csv.DictReader(open(f"{CH}/Charades_v1_{split}.csv")):
            meta[r["id"]] = r
    classes = [l.strip().split(" ", 1) for l in open(f"{CH}/Charades_v1_classes.txt") if l.strip()]
    cid = {c: i for i, (c, _) in enumerate(classes)}
    Y = np.zeros((len(frames), len(classes)), np.uint8)
    for i, f in enumerate(frames):
        r = meta[f["video"]]
        t = POSITIONS[f["k"]] * float(r["length"])
        for a in filter(None, r["actions"].split(";")):
            c, s, e = a.split()
            if float(s) <= t <= float(e):
                Y[i, cid[c]] = 1
    return Y, [n for _, n in classes]


def text_emb(names):
    m = "google/siglip-base-patch16-224"
    tok, model = AutoTokenizer.from_pretrained(m), AutoModel.from_pretrained(m).eval()
    x = tok([f"a photo of a person {n.lower()}" for n in names], padding="max_length", return_tensors="pt")
    with torch.no_grad():
        e = model.get_text_features(**x)
    return torch.nn.functional.normalize(e, dim=-1).numpy()


def main(variants):
    frames = json.load(open(OUT + "/frames.json"))
    Y, names = labels(frames)
    keep = np.array([not f["private"] for f in frames]) & (Y.sum(1) > 0)
    T = text_emb(names)
    cols = Y[keep].sum(0) >= 20
    for v in variants:
        E = np.load(f"{OUT}/emb_siglip_{v}.npy").astype(np.float32)[keep]
        S = E @ T.T
        mAP = np.mean([average_precision_score(Y[keep][:, j], S[:, j]) for j in np.where(cols)[0]])
        print(f"{v}: zero-shot action mAP {mAP:.4f}  ({keep.sum()} frames, {cols.sum()} classes)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
