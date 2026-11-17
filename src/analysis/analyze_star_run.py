"""Analysis of the GPU runs: the VLM linking model on Charades-Link, and STAR question answering.

  linking    Qwen2.5-VL-7B accuracy per variant / m / room split with bootstrap intervals
             over households, letter bias, and embedding linking models on the same episodes
             (zero-shot max cosine, and the trained head under 5-fold CV over households)
  star       STAR val accuracy per model and variant, paired differences to raw,
             per question type, letter bias, and a confidence router: the 3B edge model
             answers locally and sends its least-confident questions to the cloud model

    python -m src.analysis.analyze_star_run linking
    python -m src.analysis.analyze_star_run star
"""
import os
import sys
import csv
import json
import glob
import random

import numpy as np

STAR = "data/star/results"        # star_qa.py run outputs
RES = "results/star"
os.makedirs(RES, exist_ok=True)
OUT = "data/charades_link"
L = "ABCD"


def boot_groups(vals, groups, n=1000, seed=0):
    """95% interval of the mean, resampling groups (households or videos)."""
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    by = {}
    for v, g in zip(vals, groups):
        by.setdefault(g, []).append(v)
    S = [np.array(by[g]) for g in by]
    rng = np.random.default_rng(seed)
    bs = [np.concatenate([S[i] for i in rng.choice(len(S), len(S))]).mean() for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


def fmt(v, ci):
    return f"{100 * v:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def attacker():
    import torch
    import torch.nn.functional as F
    from src.linking.train_linker import train_head
    frames = json.load(open(OUT + "/frames.json"))
    fid = {f"{f['video']}_{f['k']}.jpg": i for i, f in enumerate(frames)}
    subj = lambda n: frames[fid[n]]["subject"]
    subs = sorted({f["subject"] for f in frames})
    random.Random(0).shuffle(subs)
    fold = {s: i % 5 for i, s in enumerate(subs)}
    res = {}
    for v in ("raw", "blur", "linkbreak"):
        rows = [json.loads(l) for f in sorted(glob.glob(f"{OUT}/vlm_{v}_*.jsonl")) for l in open(f)]
        # embedding linking models on the same episodes
        embs = {bb: np.load(f"{OUT}/emb_{bb}_{v}.npy").astype(np.float32) for bb in ("dinov2", "siglip", "clip")
                if os.path.exists(f"{OUT}/emb_{bb}_{v}.npy")}
        heads = {}
        for bb, A in embs.items():
            heads[bb] = []
            for k in range(5):  # head trained without the households of fold k; used for queries from fold k
                tr = [i for i, f in enumerate(frames) if fold[f["subject"]] != k and not f["private"]]
                sid = {s: j for j, s in enumerate(sorted({frames[i]["subject"] for i in tr}))}
                h = train_head(A[tr], np.array([sid[frames[i]["subject"]] for i in tr]))
                with torch.no_grad():
                    heads[bb].append(F.normalize(h(torch.tensor(A)), dim=-1).numpy())
        An = {bb: [A / np.linalg.norm(A, axis=1, keepdims=True)] * 5 for bb, A in embs.items()}

        def emb_hit(Es, r):
            E = Es[fold[subj(r["query"])]]
            q = E[fid[r["query"]]]
            s = [float((E[[fid[n] for n in st]] @ q).max()) for st in r["sets"]]
            return int(L[int(np.argmax(s))] == r["answer"])

        res[v] = {}
        print(f"== {v}  (n={len(rows)}; chance 25.0)")
        preds = "".join(r["pred"] for r in rows)
        print("   VLM letter shares  pred " + " ".join(f"{l}:{preds.count(l) / len(preds):.2f}" for l in L)
              + " | answer " + " ".join(f"{l}:{sum(r['answer'] == l for r in rows) / len(rows):.2f}" for l in L))
        for cross in (False, True):
            for m in (1, 2, 4, 8):
                rs = [r for r in rows if r["m"] == m and r["cross_room"] == cross]
                g = [subj(r["query"]) for r in rs]
                line = {}
                hit = [int(r["pred"] == r["answer"]) for r in rs]
                line["vlm"] = (np.mean(hit), boot_groups(hit, g))
                # letter-bias correction: divide by the mean probability of each letter in this cell
                pr = np.array([r["probs"] for r in rs])
                cal = [int(L[int(np.argmax(p / pr.mean(0)))] == r["answer"]) for p, r in zip(pr, rs)]
                line["vlm_cal"] = (np.mean(cal), boot_groups(cal, g))
                for bb in embs:
                    for kind, E in (("cos", An[bb]), ("head", heads[bb])):
                        h = [emb_hit(E, r) for r in rs]
                        line[f"{bb}_{kind}"] = (np.mean(h), boot_groups(h, g))
                res[v][f"{'cross' if cross else 'any'}_m{m}"] = {k: [a, *b] for k, (a, b) in line.items()}
                print(f"   {'cross_room' if cross else 'any_room  '} m={m}: " +
                      " | ".join(f"{k} {fmt(a, b)}" for k, (a, b) in line.items()), flush=True)
    json.dump(res, open(f"{RES}/linker_analysis.json", "w"), indent=1, default=float)


def star():
    meta = "data/charades"
    scene = {r["id"]: r["scene"] for s in ("train", "test")
             for r in csv.DictReader(open(f"{meta}/Charades_v1_{s}.csv"))}
    private = {"Bathroom", "Bedroom", "Closet / Walk-in closet / Spear closet"}
    R = {}
    for f in glob.glob(f"{STAR}/star_*.jsonl"):
        tag = os.path.basename(f)[5:].rsplit("_", 1)[0].replace("Qwen2.5-VL-", "").replace("-Instruct", "")
        if tag.count("_") != 1:  # only the main runs (model_variant); skip subset runs such as 7B_blur_cot
            continue
        for l in open(f):
            r = json.loads(l)
            R.setdefault(tag, {})[r["question_id"]] = r
    qids = sorted(R["3B_raw"])
    vid = [R["3B_raw"][q]["video_id"] for q in qids]
    qtype = np.array([q.split("_")[0] for q in qids])
    priv = np.array([scene.get(v) in private for v in vid])
    C = {t: np.array([R[t][q]["pred"] == R[t][q]["gold"] for q in qids], float) for t in R}
    P = {t: np.array([R[t][q]["probs"] for q in qids]) for t in R}
    gold = np.array([R["3B_raw"][q]["gold"] for q in qids])
    out = {}
    print(f"== STAR val, {len(qids)} questions, {len(set(vid))} videos (intervals: bootstrap over videos)")
    print("   gold letter shares " + " ".join(f"{L[i]}:{(gold == i).mean():.2f}" for i in range(4)))
    for mdl in ("3B", "7B", "32B"):
        for v in ("raw", "blur", "linkbreak"):
            t = f"{mdl}_{v}"
            c = C[t]
            pr = np.array([R[t][q]["pred"] for q in qids])
            d = c - C[f"{mdl}_raw"]
            types = " ".join(f"{ty[:4]} {100 * c[qtype == ty].mean():.1f}" for ty in sorted(set(qtype)))
            out[t] = dict(acc=c.mean(), ci=boot_groups(c, vid).tolist(), private=c[priv].mean(),
                          delta_raw=d.mean(), delta_ci=boot_groups(d, vid).tolist(),
                          types={ty: c[qtype == ty].mean() for ty in set(qtype)})
            print(f"   {t:14s} acc {fmt(c.mean(), out[t]['ci'])} | private {100 * c[priv].mean():.1f} | "
                  f"vs raw {100 * d.mean():+.1f} [{100 * out[t]['delta_ci'][0]:+.1f}, {100 * out[t]['delta_ci'][1]:+.1f}] | "
                  f"{types} | pred " + " ".join(f"{L[i]}:{(pr == i).mean():.2f}" for i in range(4)))
    # agreement / complementarity between models on blur uploads
    for a, b in (("3B_blur", "7B_blur"), ("3B_blur", "32B_blur"), ("7B_blur", "32B_blur")):
        print(f"   {a} vs {b}: both right {100 * (C[a] * C[b]).mean():.1f}, only {a} {100 * (C[a] * (1 - C[b])).mean():.1f}, "
              f"only {b} {100 * ((1 - C[a]) * C[b]).mean():.1f}, oracle pick {100 * np.maximum(C[a], C[b]).mean():.1f}")
    # confidence router: edge 3B answers on raw frames (they never leave the device);
    # the least-confident share s of questions goes to the cloud model on cleaned frames
    conf = P["3B_raw"].max(1)
    order = np.argsort(conf)
    print("   router (edge 3B on raw, cloud on the variant; sends the least-confident share):")
    out["router"] = {}
    for cloud in ("7B_blur", "32B_blur", "7B_linkbreak", "32B_linkbreak"):
        row = {}
        for s in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            send = np.zeros(len(qids), bool)
            send[order[: int(round(s * len(qids)))]] = True
            acc = np.where(send, C[cloud], C["3B_raw"]).mean()
            rnd = (1 - s) * C["3B_raw"].mean() + s * C[cloud].mean()
            row[s] = (acc, rnd)
        out["router"][cloud] = row
        print(f"     cloud {cloud:13s} " + " | ".join(f"send {int(100 * s)}%: {100 * a:.1f} (random {100 * r:.1f})" for s, (a, r) in row.items()))
    # how well does edge confidence predict that the cloud helps (U as a binary target)
    from sklearn.metrics import roc_auc_score
    for cloud in ("7B_blur", "32B_blur"):
        gain = (C[cloud] > C["3B_raw"]).astype(int)
        print(f"   AUC of (1 - edge confidence) for 'cloud {cloud} fixes an edge error': {roc_auc_score(gain, -conf):.3f} "
              f"(base rate {100 * gain.mean():.1f}%)")
    # calibration of the edge model
    bins = np.quantile(conf, np.linspace(0, 1, 6))
    print("   3B_raw calibration (confidence quintile -> accuracy): " + " ".join(
        f"{conf[(conf >= lo) & (conf <= hi)].mean():.2f}->{C['3B_raw'][(conf >= lo) & (conf <= hi)].mean():.2f}"
        for lo, hi in zip(bins[:-1], bins[1:])))
    json.dump(out, open(f"{RES}/star_analysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    {"linking": attacker, "star": star}[sys.argv[1]]()
