"""Privacy-B engine on STAR / Charades: 140 households with questions, every room of the home, real saved answers
(3B on device, 7B in the cloud). Same linking test and device-side check as linking_engine.py (HD-EPIC).

    python -m src.linking.linking_engine_star [--prior 8] [--send_view c25] [--plot out.png]

Setting
  * Households = Charades subjects. Test households: the 140 with STAR questions. The device's public reference
    population: the 127 subjects WITHOUT STAR questions (disjoint by construction).
  * Each test household has used the assistant before: the cloud already holds --prior earlier uploads from it
    (its other Charades videos, 3 frames each). Then its STAR videos arrive in random order; if the device escalates
    a video, its 3 frames are uploaded and its questions are answered by the 7B; otherwise by the 3B.
  * The cloud's test for each new upload: is it more similar to this household's earlier uploads than to K-1 random
    other test households (same number of frames each)? Chance = 1/K.
  * Policies as in linking_engine.py: never / always / conf / risk (device runs the same test against public homes).
"""
import os
import sys
import json
import glob
import argparse
import numpy as np

from src.linking import linking_engine as P


def load_star(pattern):
    rows = {}
    for f in sorted(glob.glob(pattern)):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                rows[r["question_id"]] = r
    return rows


class Star(P.Engine):
    def __init__(self, a):
        self.a = a
        d = a.data
        meta = json.load(open(os.path.join(d, "charades_link", "frames.json")))
        E = np.load(os.path.join(d, "charades_link", f"emb_siglip_blur{'_' + a.send_view if a.send_view else ''}.npy"))
        self.E = E.astype(np.float32)
        self.E /= np.linalg.norm(self.E, axis=1, keepdims=True) + 1e-8
        rows_of, subj_of = {}, {}
        for i, m in enumerate(meta):
            rows_of.setdefault(m["video"], []).append(i)
            subj_of[m["video"]] = m["subject"]
        rows_of = {v: np.array(r) for v, r in rows_of.items()}
        loc = load_star(os.path.join(a.star, "star_Qwen2.5-VL-3B-Instruct_raw_[0-9].jsonl"))
        cld = load_star(os.path.join(a.star, f"star_Qwen2.5-VL-7B-Instruct_{a.cloud_variant}_[0-9].jsonl"))
        qs = [q for q in loc if q in cld]
        by_video = {}
        for q in qs:
            by_video.setdefault(loc[q]["video_id"], []).append(q)
        test = sorted({subj_of[v] for v in by_video})
        rng = np.random.default_rng(a.seed)
        # streams: per test household, its STAR videos (random order), each with its questions
        self.streams = {}
        for s in test:
            vids = [v for v in by_video if subj_of[v] == s]
            rng.shuffle(vids)
            other = [v for v, ss in subj_of.items() if ss == s and v not in by_video]
            rng.shuffle(other)
            ev = []
            for v in vids:
                q = by_video[v]
                ev.append({"rows": rows_of[v], "conf": float(np.mean([max(loc[x]["probs"]) for x in q])),
                           "ok_local": [loc[x]["pred"] == loc[x]["gold"] for x in q],
                           "ok_cloud": [cld[x]["pred"] == cld[x]["gold"] for x in q]})
            self.streams[s] = {"prior": [rows_of[v] for v in other[:a.prior]], "ev": ev}
        # cloud's crowd: all frames of the other test households
        self.crowd = {s: np.concatenate([rows_of[v] for v, ss in subj_of.items() if ss == s]) for s in test}
        # device's public reference population: subjects without STAR questions
        pub = sorted(set(subj_of.values()) - set(test))
        prow = [np.concatenate([rows_of[v] for v, ss in subj_of.items() if ss == s]) for s in pub]
        prow = [r for r in prow if len(r) >= a.bank_min]
        self.bank = self.E[np.concatenate(prow)]
        sizes = np.array([len(r) for r in prow])
        self.bank_start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        self.pub_n = int(np.median(sizes))
        nq = sum(len(e["ok_local"]) for st in self.streams.values() for e in st["ev"])
        # the cloud may link with a different model than the device's check (default: the same)
        self.Ec = self.E
        if a.cloud_model != "siglip":
            Ec = np.load(os.path.join(d, "charades_link", f"emb_{a.cloud_model}_blur{'_' + a.send_view if a.send_view else ''}.npy")).astype(np.float32)
            self.Ec = Ec / (np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-8)
        # device check may combine several feature models: an upload passes only if it passes under every model
        self.dev = []
        for m in a.device_models:
            Em = self.E if m == "siglip" else np.load(os.path.join(d, "charades_link", f"emb_{m}_blur{'_' + a.send_view if a.send_view else ''}.npy")).astype(np.float32)
            Em = Em / (np.linalg.norm(Em, axis=1, keepdims=True) + 1e-8)
            self.dev.append((Em, Em[np.concatenate(prow)]))
        print(f"STAR: {len(test)} test households, {sum(len(st['ev']) for st in self.streams.values())} videos, "
              f"{nq} questions; prior uploads per household: median "
              f"{int(np.median([len(st['prior']) for st in self.streams.values()]))} (asked {a.prior}); public homes "
              f"{len(prow)}; view {a.send_view or 'full frame'}; device checks with {'+'.join(a.device_models)}, cloud links with {a.cloud_model}", flush=True)

    def risk(self, frames, sent_frames):
        sent_frames = np.unique(sent_frames)
        if len(sent_frames) > self.pub_n:
            sent_frames = np.random.default_rng(len(sent_frames)).choice(sent_frames, self.pub_n, replace=False)
        p = 0.0
        for Em, bank in self.dev:
            x = Em[frames]
            s_own = (x @ Em[sent_frames].T).max()
            s_pub = np.maximum.reduceat(x @ bank.T, self.bank_start, axis=1).max(0)
            p = max(p, float(np.mean(s_pub < s_own) ** (self.a.K - 1)))
        return p

    def link_one(self, s, hist, q, rng):
        """Cloud's K-household test for one upload of household s (others = other test households)."""
        others_all = [h for h in self.crowd if h != s]
        wins = 0.0
        for _ in range(self.a.R):
            others = rng.choice(others_all, self.a.K - 1, replace=False)
            n = min(len(hist), min(len(self.crowd[h]) for h in others), self.a.cmp_n)
            E = self.Ec
            s_own = (E[q] @ E[rng.choice(hist, n, replace=False)].T).max()
            best = max(float((E[q] @ E[rng.choice(self.crowd[h], n, replace=False)].T).max()) for h in others)
            wins += (s_own > best) + 0.5 * (s_own == best)
        return wins / self.a.R

    def run_all(self, policy, param, seed=0):
        rng = np.random.default_rng(seed)
        ok, links, sent, total = [], [], 0, 0
        for s, st in self.streams.items():
            hist = np.concatenate(st["prior"]) if st["prior"] else np.array([], int)
            for e in st["ev"]:
                total += 1
                if policy == "never":
                    go = False
                elif policy == "always":
                    go = True
                elif policy == "conf":
                    go = e["conf"] < param
                elif policy == "random":
                    go = rng.random() < param
                elif policy == "risk":
                    c, tau = param
                    go = e["conf"] < c and (len(hist) == 0 or self.risk(e["rows"], hist) <= tau)
                ok += e["ok_cloud"] if go else e["ok_local"]
                if go:
                    sent += 1
                    if len(hist) >= 3:
                        links.append(self.link_one(s, hist, e["rows"], rng))
                    hist = np.concatenate([hist, e["rows"]])
        return {"policy": policy, "param": param, "sent": sent, "videos": total, "acc": float(np.mean(ok) * 100),
                "link": float(np.mean(links) * 100) if links else float("nan"), "n_link": len(links)}


def main(a):
    st = Star(a)
    runs = [("never", None), ("always", None)] + [("conf", c) for c in (0.5, 0.7, 0.9)]
    runs += [("risk", (c, t)) for c in (0.7, 1.01) for t in a.taus]
    res = [st.run_all(p, x) for p, x in runs]
    for r in [r for r in res if r["policy"] == "risk"]:
        rr = [st.run_all("random", r["sent"] / r["videos"], seed=k) for k in range(a.seeds)]
        res.append({"policy": "random", "param": f"same rate as risk{r['param']}", "sent": int(np.mean([x['sent'] for x in rr])),
                    "videos": r["videos"], "acc": float(np.mean([x["acc"] for x in rr])),
                    "link": float(np.nanmean([x["link"] for x in rr])), "n_link": 0})
    print(f"\n{'policy':8s} {'param':32s} {'sent/videos':>12s} {'acc %':>6s} {'link %':>7s}")
    for r in res:
        print(f"{r['policy']:8s} {str(r['param'])[:32]:32s} {r['sent']:5d}/{r['videos']:<6d} {r['acc']:6.1f} {r['link']:7.1f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--star", default="data/star/results")
    ap.add_argument("--cloud_variant", default="blur")
    ap.add_argument("--send_view", default="")             # '', or c40 / c25 / c15 / low40 (embed_views.py charades)
    ap.add_argument("--cloud_model", default="siglip", choices=["siglip", "dinov2", "clip"])
    ap.add_argument("--device_models", nargs="+", default=["siglip"])
    ap.add_argument("--prior", type=int, default=8)        # uploads the cloud already holds from each household
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--cmp_n", type=int, default=400)
    ap.add_argument("--bank_min", type=int, default=9)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    main(ap.parse_args())
