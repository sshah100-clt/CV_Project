"""Charades: per-frame cross-video linking on frames where a person is visible (person mask > 2% of the frame).
Query, history and other households all use person-visible frames only. K = 4 (chance 25%), equal frame counts.
Results are split by whether a face was detected in the query frame.

Needs the person masks (python -m src.data.upload_variants masks) and the embeddings of each upload variant.
    python -m src.linking.person_link
"""
import os
import json

import cv2
import numpy as np

d = "data/charades_link/"
m = json.load(open(d + "frames.json"))
subj = np.array([x["subject"] for x in m])
vid = np.array([x["video"] for x in m])
face = np.array([x["faces"] > 0 for x in m])

area_file = d + "person_area.npy"  # share of each frame covered by the person mask
if not os.path.exists(area_file):
    area = [(cv2.imread(f"{d}mask/{x['video']}_{x['k']}.png", cv2.IMREAD_GRAYSCALE) > 0).mean() for x in m]
    np.save(area_file, np.array(area))
P = np.load(area_file) > 0.02

homes = {}
for i, s in enumerate(subj):
    if P[i]:
        homes.setdefault(s, []).append(i)
homes = {s: np.array(r) for s, r in homes.items() if len(set(vid[r])) >= 2 and len(r) >= 12}
H = list(homes)
rng = np.random.default_rng(0)
Q = [i for s in H for i in homes[s]]
Q = rng.choice(Q, 4000, replace=False)
draws = [(rng.choice([h for h in H if h != subj[q]], 3, replace=False), rng.integers(1 << 30)) for q in Q]


def run(name):
    E = np.load(d + f"emb_{name}.npy").astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    win = []
    for q, (oth, sd) in zip(Q, draws):
        r = np.random.default_rng(sd)
        own = homes[subj[q]][vid[homes[subj[q]]] != vid[q]]
        n = min(len(own), *(len(homes[h]) for h in oth), 30)
        so = (E[r.choice(own, n, replace=False)] @ E[q]).max()
        sb = max((E[r.choice(homes[h], n, replace=False)] @ E[q]).max() for h in oth)
        win.append((so > sb) + 0.5 * (so == sb))
    win = np.array(win)
    f = face[Q]
    print(f"{name:22s} all {100 * win.mean():5.1f}   face visible {100 * win[f].mean():5.1f} (n={f.sum()})   "
          f"no face {100 * win[~f].mean():5.1f} (n={(~f).sum()})", flush=True)


print(f"homes {len(H)}, query frames {len(Q)}, share with a face {100 * face[Q].mean():.1f}%")
for n in ["dinov2_raw", "dinov2_blur", "siglip_blur", "clip_blur", "siglip_personly", "dinov2_personly",
          "siglip_bgblur", "siglip_silhouette"]:
    run(n)
