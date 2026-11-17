"""Charades: linking as the provider holds more earlier uploads from a household (m = 1, 4, 8, 16).
Upload = all 3 frames of one video. The true household is represented by m of its OTHER videos; each of the K-1 = 3
other households by m of its videos (same number of frames). Linked if the most similar pair of frames belongs to the
true household. Chance 25%. Same households for every m (those with >= 17 videos).
    python -m src.linking.growth_link
"""
import json
import numpy as np

d = "data/charades_link/"
m = json.load(open(d + "frames.json"))
subj = np.array([x["subject"] for x in m]); vid = np.array([x["video"] for x in m])
rows = {}
for i, v in enumerate(vid):
    rows.setdefault(v, []).append(i)
vids_of = {}
for v in rows:
    vids_of.setdefault(subj[rows[v][0]], []).append(v)
MS, K, R = [1, 4, 8, 16], 4, 3000
homes = sorted(h for h, vs in vids_of.items() if len(vs) >= max(MS) + 1)
print(f"households with >= {max(MS) + 1} videos: {len(homes)} of {len(vids_of)}")
rng = np.random.default_rng(0)
plan = []
for _ in range(R):
    a = homes[rng.integers(len(homes))]
    q = vids_of[a][rng.integers(len(vids_of[a]))]
    own = rng.permutation([v for v in vids_of[a] if v != q])
    oth = rng.choice([h for h in homes if h != a], K - 1, replace=False)
    othv = [rng.permutation(vids_of[h]) for h in oth]
    plan.append((q, own, othv))
for name in ["siglip_blur", "clip_blur", "dinov2_blur"]:
    E = np.load(d + f"emb_{name}.npy").astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    out = []
    for mm in MS:
        w = []
        for q, own, othv in plan:
            Eq = E[rows[q]]
            sc = lambda vs: float((E[np.concatenate([rows[v] for v in vs[:mm]])] @ Eq.T).max())
            s = sc(own); b = max(sc(o) for o in othv)
            w.append((s > b) + 0.5 * (s == b))
        out.append(f"m={mm}: {100 * np.mean(w):.1f}")
    print(name.split("_")[0].ljust(7), "  ".join(out), flush=True)
