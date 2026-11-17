"""Charades: household linking against many households at once (rank-1 over a gallery), so chance = 1 / gallery size.
Query: one frame of household A (--upload: all 3 frames of one video, scored by the most similar pair). A's candidate set: n frames from A's OTHER videos. Every other household in the
gallery: n of its frames. Linked if the most similar frame belongs to A. Same n for every household (no size bias).
    python -m src.linking.gallery_link
"""
import sys
import json
import numpy as np
UPLOAD = "--upload" in sys.argv

d = "data/charades_link/"
m = json.load(open(d + "frames.json"))
subj = np.array([x["subject"] for x in m]); vid = np.array([x["video"] for x in m])
N = 6                                    # frames per household in the gallery (= 2 videos x 3 frames)
homes = {}
for i, s in enumerate(subj):
    homes.setdefault(s, []).append(i)
homes = {s: np.array(r) for s, r in homes.items()}
# households whose OTHER videos give >= N frames for any query video
ok = [s for s, r in homes.items() if all((vid[r] != v).sum() >= N for v in set(vid[r]))]
print(f"query = {'one upload (3 frames)' if UPLOAD else 'one frame'}; households usable: {len(ok)} of {len(homes)}; {N} frames per household in the gallery")
rng = np.random.default_rng(0)
Q = rng.choice(np.concatenate([homes[s] for s in ok]), 3000, replace=False)
if UPLOAD:                               # one query per video: all its frames
    vids = rng.choice(sorted({vid[i] for s in ok for i in homes[s]}), 3000, replace=False)
    Q = [np.where(vid == v)[0] for v in vids]
SIZES = [4, 10, 25, 50, 100, len(ok)]
plan = []
for q in Q:
    q0 = q[0] if UPLOAD else q
    a = subj[q0]
    own = homes[a][vid[homes[a]] != vid[q0]]
    others = [h for h in ok if h != a]
    perm = rng.permutation(len(others))
    plan.append((q, rng.choice(own, N, replace=False), [rng.choice(homes[others[j]], N, replace=False) for j in perm]))
for name in ["siglip_blur", "clip_blur", "dinov2_blur"]:
    E = np.load(d + f"emb_{name}.npy").astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    res = {k: [] for k in SIZES}; top5 = []
    for q, own, oth in plan:
        Eq = E[q] if UPLOAD else E[q][None]
        so = float((E[own] @ Eq.T).max())
        sb = np.array([float((E[o] @ Eq.T).max()) for o in oth])
        for k in SIZES:
            res[k].append(float(so > sb[:k - 1].max()))
        top5.append(float((sb > so).sum() < 5))
    print(name.split("_")[0].ljust(7), "  ".join(f"{k} homes: {100*np.mean(res[k]):5.1f}% (chance {100/k:.1f}%)" for k in SIZES),
          f" | top-5 of {len(ok)}: {100*np.mean(top5):.1f}% (chance {500/len(ok):.1f}%)", flush=True)
