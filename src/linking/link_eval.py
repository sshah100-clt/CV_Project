"""Cumulative linkage test on Charades-Link with an embedding linking model.

Episode: K subjects each have a history of m earlier uploads (frames from m
different videos, faces blurred). A new upload comes from a held-out video of
one of them. The linking model links it to the history with the highest cosine
similarity (nearest neighbour). Chance = 1/K. If accuracy rises with m, the
uploads leak cumulatively even though each single upload is face-redacted.

    python -m src.linking.link_eval dinov2 blur [cross_room]
"""
import sys
import json
import random
from collections import defaultdict

import numpy as np

OUT = "data/charades_link"
MS = (1, 2, 4, 8)


def build_index(frames):
    """subject -> video -> list of (row, scene); private scenes and frames are excluded."""
    idx = defaultdict(lambda: defaultdict(list))
    for i, f in enumerate(frames):
        if not f["private"]:
            idx[f["subject"]][f["video"]].append((i, f["scene"]))
    return idx


try:  # videos sharing a near-identical frame (DINOv2 cos > 0.90) are never query + history together
    NEAR_DUP = {k: set(v) for k, v in json.load(open(OUT + "/near_duplicates.json")).items()}
except FileNotFoundError:
    NEAR_DUP = {}


def episodes(idx, m, K, n, cross_room, rng):
    """Yield (query_row, [history_rows per candidate], true_index)."""
    eligible = [s for s, v in idx.items() if len(v) >= m + 1]
    out = []
    while len(out) < n:
        subs = rng.sample(eligible, K)
        target = subs[0]
        vids = list(idx[target])
        qv = rng.choice(vids)
        q_row, q_scene = rng.choice(idx[target][qv])
        pool = [v for v in vids if v != qv and v not in NEAR_DUP.get(qv, ())
                and (not cross_room or idx[target][v][0][1] != q_scene)]
        if len(pool) < m:
            continue
        hists = []
        for s in subs:
            vs = rng.sample(pool, m) if s == target else rng.sample(list(idx[s]), m)
            hists.append([rng.choice(idx[s][v])[0] for v in vs])
        order = list(range(K))
        rng.shuffle(order)
        out.append((q_row, [hists[o] for o in order], order.index(0)))
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def run(emb, frames, K=4, n=2000, cross_room=False, seed=0):
    idx = build_index(frames)
    rng = random.Random(seed)
    res = {}
    for m in MS:
        eps = episodes(idx, m, K, n, cross_room, rng)
        correct = 0
        for q, hists, t in eps:
            scores = [float((emb[h] @ emb[q]).max()) for h in hists]
            correct += int(np.argmax(scores) == t)
        lo, hi = wilson(correct, len(eps))
        res[m] = (correct / len(eps), lo, hi)
        print(f"m={m}: link accuracy {correct / len(eps):.3f}  [{lo:.3f}, {hi:.3f}]  chance {1 / K:.3f}", flush=True)
    return res


def dump(n=300, K=4, seed=1):
    """Write fixed episodes (as frame file names) for the VLM linking model."""
    frames = json.load(open(OUT + "/frames.json"))
    idx = build_index(frames)
    name = lambda r: f"{frames[r]['video']}_{frames[r]['k']}.jpg"
    out = []
    for cross in (False, True):
        rng = random.Random(seed)
        for m in MS:
            for q, hists, t in episodes(idx, m, K, n, cross, rng):
                out.append({"m": m, "cross_room": cross, "query": name(q),
                            "sets": [[name(h) for h in hs] for hs in hists], "answer": "ABCD"[t]})
    json.dump(out, open(OUT + "/episodes_vlm.json", "w"))
    print(len(out), "episodes written")


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump()
        sys.exit()
    name, variant = sys.argv[1], sys.argv[2]
    cross = len(sys.argv) > 3 and sys.argv[3] == "cross_room"
    frames = json.load(open(OUT + "/frames.json"))
    emb = np.load(f"{OUT}/emb_{name}_{variant}.npy").astype(np.float32)
    print(name, variant, "cross_room" if cross else "any_room")
    run(emb, frames, cross_room=cross)
