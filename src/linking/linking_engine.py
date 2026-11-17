"""Linking engine (CPU, seconds): how much a home's cloud uploads can be linked, and what a device-side
check does about it.  Inputs are already computed: SigLIP features of every 2nd second of the HD-EPIC videos
(hdepic.py link/embed), the saved P01 pilot answers (3B on device; 7B / GPT-4o in the cloud), and SigLIP features
of Charades frames as a public set of other people's homes.

    python -m src.linking.linking_engine [--cloud 7b|gpt|7b_crop] [--out results/linking/run_7b.json] [--plot out.png]

Setting
  * Protected home: P01. Its upload stream = the 277 pilot questions in time order. If a question goes to the cloud,
    the upload is the frames of that question's time window (as many as the saved run used).
  * Other homes (P02-P09) upload random windows of their own videos at the same time; they are the crowd.
  * The cloud sees all uploads without accounts (relay). For each new P01 upload it compares it with the earlier
    uploads of P01 and of 3 random other homes and picks the most similar home (nearest neighbour on the most
    similar pair of frames). Chance = 25%. This is the cross-day linking test of hdepic.py link, made per upload.
  * Accuracy: if a question is not sent, the device's own 3B answer counts; if sent, the cloud answer counts.
    Both are the real saved answers.

Policies (what the device does)
  never      3B answers everything; nothing is sent.
  always     every question goes to the cloud.
  conf       send when the 3B's confidence (top option probability) < c.
  random     send a random share p of questions (same count as another policy -> fair comparison).
  first      send the first N questions, then stop (a plain upload budget).
  risk       send as 'conf', but first the device runs the cloud's linking test on itself: would this upload be
             matched to its own earlier uploads rather than to K-1 random PUBLIC homes (Charades subjects)?
             p_link = F(s_own)^(K-1), where s_own = best similarity to its own earlier uploads and F is the
             distribution of the best similarity to one public home. If p_link > tau the device answers itself.
             The device knows only what it sent and the public frames; it knows nothing about the test homes.
  risk_select  like 'risk', but when risk is too high it first tries to send the least risky frames of the same
             window (same number of frames up to what the window has); accuracy for these is the saved cloud
             answer (an assumption, marked in the output).
"""
import os
import re
import sys
import json
import glob
import argparse
import numpy as np


def secs(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def home_of(v):
    return v.split("-")[0]


# --- data
def load_emb(path):
    z = np.load(path, allow_pickle=True)
    vid, t, e = z["vid"].astype(str), z["t"].astype(int), z["e"].astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
    return vid, t, e


def load_questions(vqa_dir):
    q = {}
    for f in glob.glob(os.path.join(vqa_dir, "*.json")):
        for k, v in json.load(open(f)).items():
            q[k] = v
    return q


def load_answers(results, name):
    pats = {"3b": "qa_Qwen2.5-VL-3B-Instruct_raw_f32_pt30_dense_dev.jsonl",
            "7b": "qa_Qwen2.5-VL-7B-Instruct_blur_f32_pt30_dense_dev.jsonl",
            "7b_crop": "qa_Qwen2.5-VL-7B-Instruct_cropcenter_f32_pt30_dense_dev.jsonl",
            "gpt": "qa_gpt-4o-2024-11-20_blur_f32_pt30_dense_dev.jsonl",
            "7b_labels": "qa_Qwen2.5-VL-7B-Instruct_labels_f32_pt30_dense_dev.jsonl",
            "gpt_labels": "qa_gpt-4o-2024-11-20_labels_f32_pt30_dense_dev.jsonl",
            "7b_objonly": "qa_Qwen2.5-VL-7B-Instruct_objonly_f32_pt30_dense_dev.jsonl",
            "gpt_objonly": "qa_gpt-4o-2024-11-20_objonly_f32_pt30_dense_dev.jsonl"}
    rows = {}
    for l in open(os.path.join(results, pats[name])):
        if l.strip():
            r = json.loads(l)
            rows[r["id"]] = r
    return rows


def windows(q, vid_frames):
    """Frames (row indices into the embedding table) the question's window covers, per input video."""
    idx = []
    for inp in q["inputs"].values():
        v = inp["id"]
        if v not in vid_frames:
            continue
        t, rows = vid_frames[v]
        if "start_time" in inp:
            a, b = secs(inp["start_time"]), secs(inp["end_time"])
            sel = rows[(t >= a - 1) & (t <= b + 1)]
            if len(sel) == 0:                       # window shorter than the 2-s step: nearest frame
                sel = rows[[int(np.argmin(np.abs(t - (a + b) / 2)))]]
        else:
            sel = rows
        idx.extend(sel.tolist())
    return np.array(idx, dtype=int)


def spread(idx, n):
    """n frames spread evenly over a window (as the QA runs sample frames)."""
    if len(idx) <= n:
        return idx
    return idx[np.linspace(0, len(idx) - 1, n).round().astype(int)]


# --- engine
class Engine:
    def __init__(self, a):
        self.a = a
        self.vid, self.t, self.E = load_emb(a.emb)
        if a.send_emb:     # features of what is actually uploaded (e.g. centre crops), same frame order
            z = np.load(a.send_emb, allow_pickle=True)
            assert (z["vid"].astype(str) == self.vid).all(), "send_emb frame order differs from emb"
            self.E = z["e"].astype(np.float32)
            self.E /= np.linalg.norm(self.E, axis=1, keepdims=True) + 1e-8
            print(f"uploads are {a.send_emb}", flush=True)
        self.vid_frames = {}
        for v in np.unique(self.vid):
            rows = np.where(self.vid == v)[0]
            self.vid_frames[v] = (self.t[rows], rows)
        Q = load_questions(a.vqa)
        self.local = load_answers(a.results, "3b")
        self.cloud = load_answers(a.results, a.cloud)
        ids = [i for i in self.local if i in self.cloud and i in Q]
        ev = []
        for i in ids:
            w = windows(Q[i], self.vid_frames)
            if len(w) == 0:
                continue
            first = next(iter(Q[i]["inputs"].values()))
            start = secs(first["start_time"]) if "start_time" in first else 0.0
            ev.append((first["id"], start, i, w))
        ev.sort(key=lambda x: (x[0], x[1]))           # video names start with date-time: time order
        self.ev = ev
        self.n_frames = {i: int(self.cloud[i].get("n_frames") or 8) for i in ids}
        self.conf = np.array([max(self.local[e[2]]["probs"]) for e in ev])
        self.ok_local = np.array([self.local[e[2]]["pred"] == self.local[e[2]]["gold"] for e in ev], float)
        self.ok_cloud = np.array([self.cloud[e[2]]["pred"] == self.cloud[e[2]]["gold"] for e in ev], float)
        # crowd: other homes' frames, by home
        self.crowd = {}
        for v, (t, rows) in self.vid_frames.items():
            h = home_of(v)
            if h != a.home:
                self.crowd.setdefault(h, []).append(rows)
        self.crowd = {h: np.concatenate(r) for h, r in self.crowd.items()}
        small = [h for h, r in self.crowd.items() if len(r) < a.crowd_min]
        self.crowd = {h: r for h, r in self.crowd.items() if len(r) >= a.crowd_min}
        if small:
            print(f"crowd homes left out (fewer than {a.crowd_min} frames): {small}", flush=True)
        if a.bank.endswith(".npz"):                       # e.g. label vectors of public homes (object_labels.py make)
            zb = np.load(a.bank)
            B, subj = zb["e"], zb["subject"].astype(str)
        else:
            B = np.load(a.bank, mmap_mode="r")
            subj = np.array([x["subject"] for x in json.load(open(a.bank_meta))])
        rng = np.random.default_rng(0)
        us, cnt = np.unique(subj, return_counts=True)
        keep = us[cnt >= a.bank_min]
        keep = rng.choice(keep, min(a.bank_homes, len(keep)), replace=False)
        rows = np.concatenate([np.where(subj == h)[0] for h in keep])   # grouped by public home
        self.bank = np.asarray(B[rows], np.float32)
        self.bank /= np.linalg.norm(self.bank, axis=1, keepdims=True) + 1e-8
        sizes = np.array([np.sum(subj == h) for h in keep])
        self.bank_start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        self.pub_n = int(np.median(sizes))
        print(f"P01 events {len(ev)} (of {len(ids)} answered questions); frames {len(self.E)}; crowd homes "
              f"{ {h: len(r) for h, r in self.crowd.items()} }; public homes {len(self.bank_start)} ({len(self.bank)} frames); cloud = {a.cloud}",
              flush=True)

    # ---- the cloud's linking test for one new upload, given what was sent before
    def link_trials(self, sent, rng, R):
        """sent: list of frame-index arrays (P01 uploads in order). Returns per upload (from the 2nd on) the share of
        R draws in which the cloud picks P01 among 4 homes."""
        # Every home is compared on the same number of frames (a larger set wins a max-similarity test by size
        # alone). n = min(frames P01 has sent so far, frames of the smallest chosen other home, cap).
        out = []
        crowd_h = list(self.crowd)
        for j in range(1, len(sent)):
            q = self.E[sent[j]]
            hist = np.unique(np.concatenate(sent[:j]))
            sim_own = (q @ self.E[hist].T).max(0)                       # per history frame
            wins = 0
            for _ in range(R):
                others = rng.choice(crowd_h, self.a.K - 1, replace=False)
                n = min(len(hist), min(len(self.crowd[h]) for h in others), self.a.cmp_n)
                s_own = sim_own[rng.choice(len(hist), n, replace=False)].max()
                best = max(float((q @ self.E[rng.choice(self.crowd[h], n, replace=False)].T).max()) for h in others)
                wins += (s_own > best) + 0.5 * (s_own == best)
            out.append(wins / R)
        return np.array(out)

    def frame_scores(self, frames, sent_frames):
        """Per frame: best similarity to own earlier uploads, and best similarity to each public home."""
        x = self.E[frames]
        sent_frames = np.unique(sent_frames)
        if len(sent_frames) > self.pub_n:                                       # same set size as a public home
            sent_frames = np.random.default_rng(len(sent_frames)).choice(sent_frames, self.pub_n, replace=False)
        own = (x @ self.E[sent_frames].T).max(1)
        pub = np.maximum.reduceat(x @ self.bank.T, self.bank_start, axis=1)     # frames x public homes
        return own, pub

    def risk(self, frames, sent_frames):
        """Device-side estimate of the chance that the cloud links this upload to the home (K-home test)."""
        if len(sent_frames) == 0:
            return 0.0
        own, pub = self.frame_scores(frames, sent_frames)
        s_own, s_pub = own.max(), pub.max(0)                 # upload level: most similar pair of frames
        return float(np.mean(s_pub < s_own) ** (self.a.K - 1))

    def run(self, policy, param, seed=0):
        rng = np.random.default_rng(seed)
        n = len(self.ev)
        send = np.zeros(n, bool)
        uploads, sent_frames, assumed = [], np.array([], int), 0
        if policy == "random":
            send[rng.choice(n, int(round(param * n)), replace=False)] = True
        for i, (_, _, qid, w) in enumerate(self.ev):
            frames = spread(w, self.n_frames[qid])
            if policy == "never":
                go = False
            elif policy == "always":
                go = True
            elif policy == "conf":
                go = self.conf[i] < param
            elif policy == "random":
                go = send[i]
            elif policy == "first":
                go = i < param
            elif policy in ("risk", "risk_select"):
                c, tau = param
                go = self.conf[i] < c
                if go and self.risk(frames, sent_frames) > tau:
                    go = False
                    if policy == "risk_select" and len(w) > len(frames):
                        own, pub = self.frame_scores(w, sent_frames)
                        r = (pub < own[:, None]).mean(1)              # per frame: share of public homes it beats
                        alt = w[np.argsort(r, kind="stable")[:len(frames)]]
                        if self.risk(alt, sent_frames) <= tau:
                            frames, go = np.sort(alt), True
                            assumed += 1
            send[i] = go
            if go:
                uploads.append(frames)
                sent_frames = np.concatenate([sent_frames, frames])
        acc = float(np.where(send, self.ok_cloud, self.ok_local).mean() * 100)
        link = self.link_trials(uploads, rng, self.a.R) if len(uploads) > 1 else np.array([])
        return {"policy": policy, "param": param, "sent": int(send.sum()), "acc": acc,
                "link": float(link.mean() * 100) if len(link) else float("nan"),
                "link_last": float(link[-max(1, len(link) // 4):].mean() * 100) if len(link) else float("nan"),
                "linked_uploads": float(link.sum()) if len(link) else 0.0, "assumed": assumed}


def main(a):
    eng = Engine(a)
    runs = [("never", None), ("always", None)]
    runs += [("conf", c) for c in (0.4, 0.5, 0.6, 0.7, 0.8)]
    runs += [("first", nn) for nn in (25, 50, 100)]
    runs += [("risk", (c, tau)) for c in (0.6, 0.8, 1.01) for tau in a.taus]
    runs += [("risk_select", (c, tau)) for c in (0.6, 0.8, 1.01) for tau in a.taus]
    res = [eng.run(p, x) for p, x in runs]
    # random dropping with the same number of uploads as each risk policy: the fair comparison
    for r in [r for r in res if r["policy"].startswith("risk")]:
        rr = [eng.run("random", r["sent"] / len(eng.ev), seed=s) for s in range(a.seeds)]
        res.append({"policy": "random", "param": f"same count as {r['policy']}{r['param']}", "sent": r["sent"],
                    "acc": float(np.mean([x["acc"] for x in rr])), "link": float(np.nanmean([x["link"] for x in rr])),
                    "link_last": float(np.nanmean([x["link_last"] for x in rr])),
                    "linked_uploads": float(np.mean([x["linked_uploads"] for x in rr])), "assumed": 0})
    print(f"\n{'policy':12s} {'param':34s} {'sent':>5s} {'acc %':>6s} {'link %':>7s} {'link last25%':>12s} "
          f"{'linked uploads':>14s}")
    for r in res:
        print(f"{r['policy']:12s} {str(r['param'])[:34]:34s} {r['sent']:5d} {r['acc']:6.1f} {r['link']:7.1f} "
              f"{r['link_last']:12.1f} {r['linked_uploads']:14.1f}" + (f"  ({r['assumed']} re-chosen uploads: cloud answer assumed unchanged)" if r["assumed"] else ""))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        style = {"conf": ("o", "#1f6feb"), "first": ("s", "#8a8a8a"), "random": ("x", "#b0b0b0"),
                 "risk": ("^", "#d1242f"), "risk_select": ("v", "#9a6700"), "always": ("D", "#000000"),
                 "never": ("*", "#000000")}
        for pol, (mk, col) in style.items():
            pts = [(r["link"], r["acc"]) for r in res if r["policy"] == pol and r["link"] == r["link"]]
            if pol == "never":
                ax.axhline([r["acc"] for r in res if r["policy"] == "never"][0], color="#000", lw=1, ls=":",
                           label="never send (3B only)")
                continue
            if pts:
                ax.scatter(*zip(*pts), marker=mk, color=col, label=pol, s=36)
        ax.axvline(100 / a.K, color="#888", lw=1, ls="--", label="linking chance")
        ax.set_xlabel("cloud links P01's uploads to P01 (%)")
        ax.set_ylabel("question-answering accuracy (%)")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print("plot:", a.plot)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="data/hdepic/emb_siglip_blur_s2.npz")
    ap.add_argument("--vqa", default="data/hdepic/vqa")
    ap.add_argument("--results", default="data/hdepic/results")
    ap.add_argument("--bank", default="data/charades_link/emb_siglip_blur.npy")
    ap.add_argument("--send_emb")                           # e.g. emb_siglip_blur_center_s2.npz (centre crops)
    ap.add_argument("--home", default="P01")
    ap.add_argument("--cloud", default="7b", choices=["7b", "gpt", "7b_crop", "7b_labels", "gpt_labels", "7b_objonly", "gpt_objonly"])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--R", type=int, default=20)            # distractor draws per upload
    ap.add_argument("--cmp_n", type=int, default=400)       # at most this many frames per home in the cloud's test
    ap.add_argument("--crowd_min", type=int, default=90)    # other homes need this many frames (2-s step)
    ap.add_argument("--bank_meta", default="data/charades_link/frames.json")  # subject = public home
    ap.add_argument("--bank_homes", type=int, default=200)
    ap.add_argument("--bank_min", type=int, default=20)                      # frames per public home
    ap.add_argument("--taus", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out")
    ap.add_argument("--plot")
    main(ap.parse_args())
