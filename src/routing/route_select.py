"""Selective escalation on the HD-EPIC P01 pilot: which questions should the device send, in which form, and what does
that cost in accuracy and in household linking?  Uses only saved answers and saved features (CPU, minutes).

    python -m src.routing.route_select [--vqa data/hdepic/vqa] [--out results/routing/route_select.json]

Device knowledge at decision time: the question type (prototype) and the 3B's confidence (top option probability).
Rules (each tuned on training videos only, scored on held-out videos):
    never        3B answers everything.
    always       everything is sent.
    conf         send when confidence < c                      (c tuned on the training videos)
    type         send question types where, on the training videos, the cloud beat the 3B
    type+conf    send those types only when confidence < c     (c tuned on the training videos)
Evaluation: 5-fold split by video (P01 has 27 videos; a question belongs to its first input video), repeated over
10 random fold assignments; accuracy is averaged over repeats. Each question gets one held-out decision per repeat.
Accuracy change vs 'never': 95% bootstrap CI over videos (2,000 draws) and exact McNemar test (repeat 0).
Linking (repeat 0 decisions; K = 4, chance 25, equal frame counts, R = 20 draws, 3 seeds), computed on the features
of what is actually uploaded (full frame / centre crop / objects only):
    link        new upload vs all earlier uploads of P01
    link_xv     new upload vs earlier uploads from OTHER videos only (no near-repeat frames)
    rand        the same two numbers for random sending of the same number of questions (10 draws)
"""
import os
import sys
import json
import math
import types
import argparse
import numpy as np

from src.linking import linking_engine as P

R_ = "data/hdepic/results/qa_{}_f32_pt30_dense_dev.jsonl"
CLOUD = [("7B", "blur", "frames"), ("7B", "cropcenter", "crop"), ("7B", "objonly", "objonly"),
         ("32B", "blur", "frames"), ("GPT-4o", "blur", "frames"), ("GPT-4o", "cropcenter", "crop"),
         ("GPT-4o", "objonly", "objonly")]
MODEL = {"7B": "Qwen2.5-VL-7B-Instruct", "32B": "Qwen2.5-VL-32B-Instruct", "GPT-4o": "gpt-4o-2024-11-20"}
EMB = {"frames": "data/hdepic/emb_siglip_blur_s2.npz", "crop": "data/hdepic/emb_siglip_blur_center_s2.npz",
       "objonly": "data/hdepic/emb_siglip_objonly_s2.npz"}
CGRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def read(path):
    out = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            out[r["id"]] = r
    return out


def mcnemar(a, b):
    x, y = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(x, y) + 1)) / 2 ** n)


# --- rules
def tune(rule, tr, conf, typ, okl, okc):
    """Fit a rule on training questions tr (indices). Returns a decision function over indices."""
    if rule == "never":
        return lambda ix: np.zeros(len(ix), bool)
    if rule == "always":
        return lambda ix: np.ones(len(ix), bool)
    S = set()
    if rule in ("type", "type+conf"):
        for t in set(typ[tr]):
            m = tr[typ[tr] == t]
            if len(m) >= 3 and okc[m].mean() > okl[m].mean():
                S.add(t)
    if rule == "type":
        return lambda ix: np.array([typ[i] in S for i in ix])
    elig = (lambda ix: np.ones(len(ix), bool)) if rule == "conf" else (lambda ix: np.array([typ[i] in S for i in ix]))
    best = None
    for c in CGRID:                                    # accuracy on training; ties -> fewer sent
        go = elig(tr) & (conf[tr] < c)
        acc = np.where(go, okc[tr], okl[tr]).mean()
        key = (round(acc, 9), -go.sum())
        if best is None or key > best[0]:
            best = (key, c)
    c = best[1]
    return lambda ix: elig(ix) & (conf[ix] < c)


def heldout(rule, groups, conf, typ, okl, okc, seed, k=5):
    vids = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(vids)
    fold_of = {v: i % k for i, v in enumerate(order)}
    f = np.array([fold_of[g] for g in groups])
    send = np.zeros(len(groups), bool)
    for j in range(k):
        te, tr = np.where(f == j)[0], np.where(f != j)[0]
        send[te] = tune(rule, tr, conf, typ, okl, okc)(te)
    return send


# --- linking
def link(eng, uploads, vids, rng, R, other_video_only):
    out = []
    crowd = list(eng.crowd)
    for j in range(1, len(uploads)):
        prev = [u for u, v in zip(uploads[:j], vids[:j]) if not other_video_only or v != vids[j]]
        if not prev:
            continue
        hist = np.unique(np.concatenate(prev))
        q = eng.E[uploads[j]]
        so_all = (q @ eng.E[hist].T).max(0)
        w = 0.0
        for _ in range(R):
            oth = rng.choice(crowd, eng.a.K - 1, replace=False)
            n = min(len(hist), min(len(eng.crowd[h]) for h in oth), eng.a.cmp_n)
            s = so_all[rng.choice(len(hist), n, replace=False)].max()
            b = max(float((q @ eng.E[rng.choice(eng.crowd[h], n, replace=False)].T).max()) for h in oth)
            w += (s > b) + 0.5 * (s == b)
        out.append(w / R)
    return out


def link_of(eng, send, frames, groups, R, seeds):
    ix = np.where(send)[0]
    ups, vs = [frames[i] for i in ix], [groups[i] for i in ix]
    a, b = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        a += link(eng, ups, vs, rng, R, False)
        b += link(eng, ups, vs, rng, R, True)
    f = lambda x: (100 * float(np.mean(x)), len(x) // seeds) if x else (float("nan"), 0)
    return f(a), f(b)


# --- main
def main(a):
    base = types.SimpleNamespace(emb=EMB["frames"], vqa=a.vqa, results="data/hdepic/results", bank=a.bank,
                                 send_emb=None, home="P01", cloud="7b", K=4, R=a.R, cmp_n=400, crowd_min=90,
                                 bank_meta="data/charades_link/frames.json", bank_homes=200, bank_min=20)
    eng = P.Engine(base)
    E_of = {}
    for form, p in EMB.items():
        z = np.load(p, allow_pickle=True)
        assert (z["vid"].astype(str) == eng.vid).all(), p          # same frame order (as linking_engine --send_emb)
        assert "t" not in z.files or (z["t"].astype(int) == eng.t).all(), p
        e = z["e"].astype(np.float32)
        E_of[form] = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
    qids = [e[2] for e in eng.ev]                                  # time order
    groups = np.array([e[0] for e in eng.ev])                      # first input video
    frames = [P.spread(e[3], eng.n_frames[e[2]]) for e in eng.ev]
    loc = read(R_.format("Qwen2.5-VL-3B-Instruct_raw"))
    conf = np.array([max(loc[q]["probs"]) for q in qids])
    typ = np.array([loc[q]["prototype"] for q in qids])
    okl = np.array([loc[q]["pred"] == loc[q]["gold"] for q in qids])
    print(f"P01: {len(qids)} questions, {len(set(groups))} videos, {len(set(typ))} question types; "
          f"3B accuracy {100 * okl.mean():.1f}", flush=True)
    rules = ["never", "always", "conf", "type", "type+conf"]
    res = []
    for mdl, form, emb in CLOUD:
        path = R_.format(f"{MODEL[mdl]}_{form}")
        if not os.path.exists(path):
            print(f"-- {mdl} {form}: no saved answers, skipped", flush=True)
            continue
        cl = read(path)
        assert all(q in cl for q in qids), f"{path}: missing questions"
        okc = np.array([cl[q]["pred"] == cl[q]["gold"] for q in qids])
        eng.E = E_of[emb]
        print(f"\n== cloud {mdl} on {emb}: cloud-only accuracy {100 * okc.mean():.1f}", flush=True)
        for rule in rules:
            sends = [heldout(rule, groups, conf, typ, okl, okc, seed) for seed in range(a.repeats)]
            accs = [100 * np.where(s, okc, okl).mean() for s in sends]
            s0 = sends[0]
            ok0 = np.where(s0, okc, okl)
            # bootstrap over videos of the accuracy change (repeat 0)
            vids = np.unique(groups)
            d = ok0.astype(float) - okl
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(2000):
                pick = rng.choice(vids, len(vids), replace=True)
                ix = np.concatenate([np.where(groups == v)[0] for v in pick])
                bs.append(100 * d[ix].mean())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            p = mcnemar(ok0, okl)
            if s0.sum() >= 2:
                (lk, nl), (lx, nx) = link_of(eng, s0, frames, groups, a.R, a.seeds)
                rl, rx = [], []
                for r in range(a.rand):
                    rs = np.zeros(len(qids), bool)
                    rs[np.random.default_rng(100 + r).choice(len(qids), int(s0.sum()), replace=False)] = True
                    (x1, _), (x2, _) = link_of(eng, rs, frames, groups, a.R, 1)
                    rl.append(x1); rx.append(x2)
                rnd = (float(np.nanmean(rl)), float(np.nanmean(rx)))
            else:
                lk = lx = float("nan"); nl = nx = 0; rnd = (float("nan"), float("nan"))
            sent_types = sorted(set(typ[s0]))
            r = {"model": mdl, "form": emb, "rule": rule, "sent": int(s0.sum()),
                 "sent_mean": float(np.mean([s.sum() for s in sends])), "acc": float(np.mean(accs)),
                 "acc_sd": float(np.std(accs)), "delta": float(np.mean(accs) - 100 * okl.mean()),
                 "delta_ci_rep0": [float(lo), float(hi)], "mcnemar_p_rep0": p,
                 "link": lk, "n_link": nl, "link_xv": lx, "n_link_xv": nx, "rand_link": rnd[0], "rand_link_xv": rnd[1],
                 "sent_types_rep0": sent_types}
            res.append(r)
            print(f"  {rule:10s} sent {r['sent_mean']:5.1f}  acc {r['acc']:5.1f} (sd {r['acc_sd']:.1f})  "
                  f"change {r['delta']:+5.1f} [CI {lo:+.1f}, {hi:+.1f}] p={p:.3f}  "
                  f"link {lk:5.1f} (n={nl}) vs random {rnd[0]:5.1f} | other videos {lx:5.1f} (n={nx}) vs random {rnd[1]:5.1f}",
                  flush=True)
    # in-sample upper bounds (not achievable: chosen with the answers of the same questions)
    print("\nIn-sample references (NOT held out):")
    for mdl, form, emb in CLOUD:
        path = R_.format(f"{MODEL[mdl]}_{form}")
        if not os.path.exists(path):
            continue
        cl = read(path)
        okc = np.array([cl[q]["pred"] == cl[q]["gold"] for q in qids])
        best_t = np.zeros(len(qids), bool)
        for t in set(typ):
            m = typ == t
            if okc[m].mean() > okl[m].mean():
                best_t |= m
        print(f"  {mdl:6s} {emb:8s} best types chosen in-sample: sent {best_t.sum():3d}, acc "
              f"{100 * np.where(best_t, okc, okl).mean():.1f}; per-question oracle {100 * (okc | okl).mean():.1f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa", default="data/hdepic/vqa")
    ap.add_argument("--bank", default="data/charades_link/emb_siglip_blur.npy")
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rand", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out")
    main(ap.parse_args())
