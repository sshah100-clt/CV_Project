"""HD-EPIC P01, 'always send' policy: link each upload using (a) all earlier uploads and (b) only earlier uploads from
OTHER videos, so shared or near-repeat frames of the same video cannot drive the result. K = 4, chance 25%.
Runs for full frames, objects-only images and detector labels.

    python -m src.linking.cross_video_link [--vqa data/hdepic/vqa]
"""
import sys
import types

import numpy as np

from src.linking import linking_engine as P

VQA = sys.argv[sys.argv.index("--vqa") + 1] if "--vqa" in sys.argv else "data/hdepic/vqa"


def go(send_emb, bank, cloud):
    a = types.SimpleNamespace(emb="data/hdepic/emb_siglip_blur_s2.npz", vqa=VQA, results="data/hdepic/results",
                              bank=bank, send_emb=send_emb, home="P01", cloud=cloud, K=4, R=20, cmp_n=400,
                              crowd_min=90, bank_meta="data/charades_link/frames.json", bank_homes=200, bank_min=20,
                              taus=[], seeds=1, out=None, plot=None)
    e = P.Engine(a)
    rng = np.random.default_rng(0)
    ups = [(P.spread(w, e.n_frames[q]), v) for v, _, q, w in e.ev]
    res = {"same+other": [], "other videos only": []}
    crowd = list(e.crowd)
    for j in range(1, len(ups)):
        q = e.E[ups[j][0]]
        for key in res:
            prev = [f for f, v in ups[:j] if key == "same+other" or v != ups[j][1]]
            if not prev:
                continue
            hist = np.unique(np.concatenate(prev))
            if len(hist) < 3:
                continue
            so = (q @ e.E[hist].T).max(0)
            w = 0
            for _ in range(20):
                oth = rng.choice(crowd, 3, replace=False)
                n = min(len(hist), min(len(e.crowd[h]) for h in oth), 400)
                s = so[rng.choice(len(hist), n, replace=False)].max()
                b = max(float((q @ e.E[rng.choice(e.crowd[h], n, replace=False)].T).max()) for h in oth)
                w += (s > b) + 0.5 * (s == b)
            res[key].append(w / 20)
    print(cloud, send_emb or "full frames", {k: f"{100 * np.mean(v):.1f}% (n={len(v)})" for k, v in res.items()},
          flush=True)


B = "data/charades_link"
go(None, f"{B}/emb_siglip_blur.npy", "7b")
go("data/hdepic/emb_siglip_objonly_s2.npz", f"{B}/emb_siglip_objonly_bank.npz", "7b_objonly")
go("data/hdepic/emb_labels_s2.npz", f"{B}/emb_labels_bank.npz", "7b_labels")
