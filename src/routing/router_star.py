"""Can the device predict when the cloud helps? STAR, 7,098 questions, 3B (device, raw frames) vs 7B (cloud, face-blurred).
Features known on the device before sending: the 3B's option probabilities (top, margin, entropy, chosen letter) and
the question type/template. Target: the cloud answers right. 5-fold cross-validation over households (Charades
subjects), so test households are never seen in training.

    python -m src.routing.router_star
Prints accuracy at each share of questions sent for: random, 3B confidence, the learned predictor, and the oracle.
"""
import os
import json
import glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def load(p):
    r = {}
    for f in sorted(glob.glob(p)):
        for l in open(f):
            if l.strip():
                x = json.loads(l)
                r[x["question_id"]] = x
    return r


loc = load("data/star/results/star_Qwen2.5-VL-3B-Instruct_raw_[0-9].jsonl")
cld = load("data/star/results/star_Qwen2.5-VL-7B-Instruct_blur_[0-9].jsonl")
subj = {m["video"]: m["subject"] for m in json.load(open("data/charades_link/frames.json"))}
ids = sorted(q for q in loc if q in cld)
P = np.array([loc[q]["probs"] for q in ids]); P = P / P.sum(1, keepdims=True)
srt = np.sort(P, 1)
ent = -(P * np.log(P + 1e-9)).sum(1)
tmpl = [q.rsplit("_", 1)[0] for q in ids]                      # e.g. Interaction_T1
T = sorted(set(tmpl))
X = np.column_stack([srt[:, -1], srt[:, -1] - srt[:, -2], ent, np.eye(4)[[loc[q]["pred"] for q in ids]],
                     np.array([[t == u for u in T] for t in tmpl], float)])
okL = np.array([loc[q]["pred"] == loc[q]["gold"] for q in ids], float)
okC = np.array([cld[q]["pred"] == cld[q]["gold"] for q in ids], float)
g = np.array([subj[loc[q]["video_id"]] for q in ids])
print(f"{len(ids)} questions, {len(set(g))} households; 3B {okL.mean()*100:.1f}, 7B {okC.mean()*100:.1f}, "
      f"cloud fixes {((okC==1)&(okL==0)).mean()*100:.1f}%, cloud breaks {((okC==0)&(okL==1)).mean()*100:.1f}%")

# predicted gain of sending = P(cloud right) - P(device right), both learned out of fold
gain = np.zeros(len(ids))
for tr, te in GroupKFold(5).split(X, groups=g):
    mc = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], okC[tr])
    ml = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], okL[tr])
    gain[te] = mc.predict_proba(X[te])[:, 1] - ml.predict_proba(X[te])[:, 1]
conf = srt[:, -1]
rng = np.random.default_rng(0)


def acc_at(order, frac):
    n = int(round(frac * len(ids)))
    send = np.zeros(len(ids), bool); send[order[:n]] = True
    return np.where(send, okC, okL).mean() * 100


def boot_diff(order_a, order_b, frac, B=500):
    """95% CI of accuracy(a) - accuracy(b) at the same share sent, resampling households."""
    n = int(round(frac * len(ids)))
    sa = np.zeros(len(ids), bool); sa[order_a[:n]] = True
    sb = np.zeros(len(ids), bool); sb[order_b[:n]] = True
    d = np.where(sa, okC, okL) - np.where(sb, okC, okL)
    u = np.unique(g); idx = {h: np.where(g == h)[0] for h in u}
    bs = [np.concatenate([d[idx[h]] for h in rng.choice(u, len(u))]).mean() * 100 for _ in range(B)]
    return d.mean() * 100, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


o_conf = np.argsort(conf)                       # least confident first
o_gain = np.argsort(-gain)                      # highest predicted gain first
o_orac = np.argsort(-(okC - okL), kind="stable")
o_rand = rng.permutation(len(ids))
print(f"\n{'sent':>5s} {'random':>7s} {'3B conf':>8s} {'learned':>8s} {'oracle':>7s}   learned - 3B conf [95% CI]")
for f in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
    m, lo, hi = boot_diff(o_gain, o_conf, f)
    print(f"{f*100:4.0f}% {acc_at(o_rand,f):7.1f} {acc_at(o_conf,f):8.1f} {acc_at(o_gain,f):8.1f} {acc_at(o_orac,f):7.1f}"
          f"   {m:+5.1f} [{lo:+.1f}, {hi:+.1f}]")
print(f"\nlearned gain > 0 for {(gain>0).mean()*100:.1f}% of questions; sending exactly those: "
      f"{np.where(gain>0, okC, okL).mean()*100:.1f}%")
os.makedirs("results/routing", exist_ok=True)
np.save("results/routing/star_router_gain.npy", np.column_stack([np.array(ids, dtype=object), gain]))
