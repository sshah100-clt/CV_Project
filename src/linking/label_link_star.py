"""Do label-only uploads link to the home? STAR human scene labels (objects, relations, actions per key frame) as the
best case of sending labels instead of pixels. Same 4-household test as the image engine, on the same STAR videos:
a video's upload vs the household's other STAR videos and the same number of videos from 3 other households.
Label similarity: TF-IDF cosine over label words; image similarity: SigLIP, most similar frame pair (reference).

    python -m src.linking.label_link_star
"""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

d = []
for sp in ("train", "val"):
    d += json.load(open(f"data/star/STAR_{sp}.json"))
meta = json.load(open("data/charades_link/frames.json"))
subj = {m["video"]: m["subject"] for m in meta}
rows = {}
for i, m in enumerate(meta):
    rows.setdefault(m["video"], []).append(i)
lab = {}
for q in d:
    v = q["video_id"]
    for fr in q.get("situations", {}).values():
        lab.setdefault(v, []).extend(fr.get("bbox_labels", []) + fr.get("rel_labels", []) + fr.get("actions", []))
vids = sorted(v for v in lab if v in subj and v in rows)
by_h = {}
for v in vids:
    by_h.setdefault(subj[v], []).append(v)
by_h = {h: vs for h, vs in by_h.items() if len(vs) >= 2}
vids = [v for vs in by_h.values() for v in vs]
ix = {v: i for i, v in enumerate(vids)}
X = TfidfVectorizer(token_pattern=r"\S+").fit_transform([" ".join(lab[v]) for v in vids])
S_lab = (X @ X.T).toarray()
E = np.load("data/charades_link/emb_siglip_blur.npy").astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
print(f"{len(vids)} STAR videos with labels in {len(by_h)} households with >= 2 such videos; "
      f"labels per video median {int(np.median([len(lab[v]) for v in vids]))}")
rng = np.random.default_rng(0)
H = list(by_h)


def test(sim, R=50):
    res = []
    for h in H:
        for v in by_h[h]:
            own = [u for u in by_h[h] if u != v]
            for _ in range(R // 10):
                others = rng.choice([x for x in H if x != h], 3, replace=False)
                n = min(len(own), *(len(by_h[o]) for o in others))
                s_own = max(sim(v, u) for u in rng.choice(own, n, replace=False))
                best = max(max(sim(v, u) for u in rng.choice(by_h[o], n, replace=False)) for o in others)
                res.append((s_own > best) + 0.5 * (s_own == best))
    return np.mean(res) * 100


img = lambda v, u: float((E[rows[v]] @ E[rows[u]].T).max())
labf = lambda v, u: float(S_lab[ix[v], ix[u]])
print(f"labels only (TF-IDF):  {test(labf):.1f}% linked   (chance 25%)")
print(f"images (SigLIP):       {test(img):.1f}% linked")
