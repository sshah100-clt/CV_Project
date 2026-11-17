"""Label-only uploads: an open-vocabulary object detector on the device lists the objects in each frame (no pixels
are sent). Vocabulary: the 300 noun classes of EPIC-KITCHENS-100 (a standard kitchen list, independent of the
HD-EPIC questions, so labels cannot copy answer options). Detector: OWLv2 (google/owlv2-base-patch16-ensemble).

    python -m src.qa.object_labels detect --data data [--shard 0 --nshards 2]   # GPU; resumes
    python -m src.qa.object_labels detect --data data --charades 60             # also 60 Charades homes (public set; optional)
    python -m src.qa.object_labels make   --data data                           # CPU
    python -m src.qa.object_labels objonly --data data                          # CPU: objects-only images (from the detections)
    python -m src.qa.object_labels objemb --data data                           # GPU: SigLIP features of objects-only images
Outputs (data/hdepic/):
    labels/det_owlv2*.jsonl                         detections per frame (label, score, box in frame pixels)
    results/labels_owlv2_<pilot tag>.jsonl          label text per pilot question (read by hdepic_pilot.py qa_text
                                                    --evidence_kind labels and gpt_qa.py --form labels)
    emb_labels_s2.npz                               label vector per 2-second frame, all homes (linking_engine --send_emb)
    data/charades_link/emb_labels_bank.npz          label vectors of the public Charades homes (linking_engine --bank)
Frames: the face-blurred frames the questions use (blurring changes faces only).
"""
import os
import re
import sys
import json
import argparse
import urllib.request
import numpy as np

from src.data import hdepic as H
from src.qa import hdepic_pilot as P

DET = "google/owlv2-base-patch16-ensemble"
NOUNS_URL = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_noun_classes.csv"
KEEP = 0.08      # stored detections: score above this (make applies --thr)
TOPK = 15


def vocab(r):
    path = os.path.join(r, "labels", "epic100_nouns.txt")
    if not os.path.exists(path):
        import csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = list(csv.DictReader(urllib.request.urlopen(NOUNS_URL, timeout=60).read().decode().splitlines()))
        words = []
        for x in rows:
            k = x["key"].strip()
            parts = k.split(":")
            words.append(" ".join(parts[1:] + parts[:1]) if len(parts) > 1 else k)   # board:chopping -> chopping board
        open(path, "w").write("\n".join(words))
    return [w for w in open(path).read().splitlines() if w.strip()]


BACKGROUND_CATS = ("furniture", "storage")   # EPIC-100 categories that are parts of the room (drawer, top, sink, wall,
                                             # floor, shelf, window, kitchen door, cupboard, drying rack, ...)
FIXED = {"tap", "fridge", "hob", "oven", "heat", "dishwasher", "microwave", "button", "alarm", "freezer", "light",
         "washing machine", "extractor fan", "plug", "knob", "wire", "heater", "tv"}
                                             # EPIC-100 "appliances" that are fixed parts of the room (portable ones:
                                             # kettle, blender, toaster, scale, phone, ... stay objects)
MAX_AREA = 0.25                              # boxes covering more of the frame count as background too


def categories(r):
    """noun (as in vocab) -> EPIC-100 category."""
    import csv
    path = os.path.join(r, "labels", "epic100_nouns.csv")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(urllib.request.urlopen(NOUNS_URL, timeout=60).read().decode())
    out = {}
    for x in csv.DictReader(open(path)):
        parts = x["key"].strip().split(":")
        out[" ".join(parts[1:] + parts[:1]) if len(parts) > 1 else parts[0]] = x["category"]
    return out


def keep_boxes(d, cat, w, h, thr):
    """Object boxes kept in an objects-only image: score >= thr, not a room part, at most MAX_AREA of the frame."""
    return [(x1, y1, x2, y2) for lab, s, x1, y1, x2, y2 in d
            if s >= thr and cat.get(lab) not in BACKGROUND_CATS and lab not in FIXED
            and (x2 - x1) * (y2 - y1) <= MAX_AREA * w * h]


def objonly(a):
    """Objects-only images: pixels inside kept object boxes, gray (128) elsewhere; from the blurred frames.
    HD-EPIC: frames/objonly/<video>/<frame> for every detected frame (pilot questions incl. dense frames, and the
    2-second frames of all homes). Charades (public homes, if detected): data/charades_link/objonly/<frame>."""
    from PIL import Image
    r = H.root(a)
    det, cat = read_dets(r), categories(r)
    blur = os.path.join(r, "frames", "blur") + os.sep
    cdir = os.path.join(a.data, "charades_link", "blur") + os.sep
    made, gray, area = 0, 0, []
    for f, d in det.items():
        if f.startswith(blur):
            dst = os.path.join(r, "frames", "objonly", f[len(blur):])
        elif f.startswith(cdir):
            dst = os.path.join(a.data, "charades_link", "objonly", f[len(cdir):])
        else:
            continue
        im = Image.open(f).convert("RGB")
        w, h = im.size
        boxes = keep_boxes(d, cat, w, h, a.thr)
        out = Image.new("RGB", (w, h), (128, 128, 128))
        mask = np.zeros((h, w), bool)
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
            if x2 > x1 and y2 > y1:
                out.paste(im.crop((x1, y1, x2, y2)), (x1, y1))
                mask[y1:y2, x1:x2] = True
        gray += not mask.any()
        area.append(mask.mean())
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            out.save(dst, quality=90)
            made += 1
    print(f"objects-only images: {made} written ({len(area)} detected frames); all gray {gray} "
          f"({gray / max(1, len(area)) * 100:.1f}%); kept area median {np.median(area) * 100:.1f}% of the frame", flush=True)
    # every frame of every pilot question must exist, or the QA run would silently fall back to other frames
    miss = [f for _, fs in pilot_frames(a, r) for _, f in fs
            if not os.path.exists(os.path.join(r, "frames", "objonly", f[len(blur):]))]
    if miss:
        sys.exit(f"STOP: {len(miss)} pilot frames have no objects-only image, e.g. {miss[0]}")
    print("all pilot frames have an objects-only image", flush=True)


def objemb(a):
    """SigLIP features of the objects-only 2-second frames (same order as emb_siglip_blur_s2.npz) and of the public
    Charades objects-only frames."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    r = H.root(a)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(H.BACKBONES["siglip"])
    model = AutoModel.from_pretrained(H.BACKBONES["siglip"]).to(dev).eval()

    def emb(paths):
        E = []
        for i in range(0, len(paths), 64):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + 64]]
            with torch.no_grad():
                e = model.get_image_features(pixel_values=proc(images=ims, return_tensors="pt")["pixel_values"].to(dev))
            E.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy().astype(np.float16))
        return np.concatenate(E)
    blur = os.path.join(r, "frames", "blur") + os.sep
    s2 = [os.path.join(r, "frames", "objonly", f[len(blur):]) for f in s2_frames(r)]
    miss = [p for p in s2 if not os.path.exists(p)]
    if miss:
        sys.exit(f"STOP: {len(miss)} 2-second frames have no objects-only image, e.g. {miss[0]}")
    z = np.load(os.path.join(r, "emb_siglip_blur_s2.npz"))
    np.savez(os.path.join(r, "emb_siglip_objonly_s2.npz"), vid=z["vid"], t=z["t"], e=emb(s2))
    print(f"objects-only features: {len(s2)} frames", flush=True)
    cdir = os.path.join(a.data, "charades_link")
    if os.path.exists(os.path.join(cdir, "frames.json")):
        meta = json.load(open(os.path.join(cdir, "frames.json")))
        rows = [(os.path.join(cdir, "objonly", f"{m['video']}_{m['k']}.jpg"), m["subject"]) for m in meta]
        rows = [x for x in rows if os.path.exists(x[0])]
        if rows:
            np.savez(os.path.join(cdir, "emb_siglip_objonly_bank.npz"), e=emb([p for p, _ in rows]),
                     subject=np.array([s for _, s in rows]))
            print(f"public objects-only features: {len(rows)} Charades frames", flush=True)


def pilot_args(a):
    b = argparse.Namespace(**vars(a))
    b.homes, b.split, b.per_type, b.dense, b.nframes, b.variant, b.limit = ["P01"], "dev", 30, True, 32, "blur", None
    return b


def pilot_frames(a, r):
    """(question id, [(tag, frame path)]) for the pilot questions, exactly the frames the QA runs used."""
    b = pilot_args(a)
    out = []
    for q in H.select(b, r):
        content, files = H.build(q, r, "blur", b.nframes, dense=True)
        tags = [c["text"] for c in content if c["type"] == "text" and re.match(r"^\[.+, .+\]$", c["text"])]
        assert len(tags) == len(files), q["id"]
        out.append((q["id"], list(zip(tags, files))))
    return out


def s2_frames(r):
    z = np.load(os.path.join(r, "emb_siglip_blur_s2.npz"))
    return [os.path.join(r, "frames", "blur", v, f"{t:06d}.jpg") for v, t in zip(z["vid"].astype(str), z["t"].astype(int))]


def charades_frames(a, n_homes):
    c = os.path.join(a.data, "charades_link")
    if not os.path.exists(os.path.join(c, "frames.json")):
        if n_homes and n_homes < 10 ** 6:
            print(f"no {c}/frames.json: public Charades homes skipped", flush=True)
        return []
    meta = json.load(open(os.path.join(c, "frames.json")))
    by = {}
    for m in meta:
        by.setdefault(m["subject"], []).append(os.path.join(c, "blur", f"{m['video']}_{m['k']}.jpg"))
    by = {s: [f for f in fs if os.path.exists(f)] for s, fs in by.items()}   # some frames may be missing
    subs = sorted(s for s, f in by.items() if len(f) >= 20)[:n_homes]
    out = [f for s in subs for f in by[s]]
    if n_homes < 10 ** 6:
        n_all = sum(1 for m in meta if m["subject"] in set(subs))
        print(f"public Charades homes: {len(subs)} with >= 20 frames on disk; {len(out)} frames "
              f"({n_all - len(out)} listed but missing on disk, skipped)", flush=True)
    return out


def text_feats(model, proc, queries, dev):
    """Label-name embeddings, computed once (the class head normalises them)."""
    import torch
    t = proc(text=queries, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        return model.owlv2.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"])


def logits_boxes(model, proc, tf, ims, dev):
    """OWLv2 detection head with fixed label embeddings (same computation as the model's forward)."""
    import torch
    pv = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
    with torch.no_grad():
        fmap = model.image_embedder(pixel_values=pv)[0]
        B, h, w, D = fmap.shape
        feats = fmap.reshape(B, h * w, D)
        q = tf[None].expand(B, -1, -1)
        logits, _ = model.class_predictor(feats, q, torch.ones(q.shape[:2], dtype=torch.bool, device=dev))
        boxes = model.box_predictor(feats, fmap)
    return logits, boxes


FAST = [True]   # fixed label embeddings; falls back to the model's standard forward if this library version differs


def run_det(model, proc, tf, words, ims, dev):
    """Per image: up to TOPK [label, score, x1, y1, x2, y2] (image pixels), best box per label, score >= KEEP."""
    import torch
    logits = None
    if FAST[0]:
        try:
            logits, boxes = logits_boxes(model, proc, tf, ims, dev)
        except (AttributeError, TypeError, ValueError) as e:
            print(f"fast detector path unavailable ({type(e).__name__}: {e}); using the standard forward", flush=True)
            FAST[0] = False
    if logits is None:
        x = proc(text=[[f"a photo of a {w}" for w in words]] * len(ims), images=ims, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**x)
        logits, boxes = o.logits, o.pred_boxes
    sc = torch.sigmoid(logits).float().cpu().numpy()                 # images x boxes x labels
    bx = boxes.float().cpu().numpy()                                 # cx cy w h, relative to the padded square
    out = []
    for im, s, b in zip(ims, sc, bx):
        best = s.argmax(0)                                           # best box per label
        score = s[best, np.arange(s.shape[1])]
        keep = [j for j in np.argsort(-score)[:TOPK] if score[j] >= KEEP]
        side = max(im.size)                                          # OWLv2 pads to a square (bottom / right)
        d = []
        for j in keep:
            cx, cy, w, h = b[best[j]] * side
            d.append([words[j], round(float(score[j]), 3), round(max(0, cx - w / 2)), round(max(0, cy - h / 2)),
                      round(min(im.size[0], cx + w / 2)), round(min(im.size[1], cy + h / 2))])
        out.append(d)
    return out


def detect(a):
    import torch
    from PIL import Image
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    r = H.root(a)
    words = vocab(r)
    queries = [f"a photo of a {w}" for w in words]
    files = sorted({f for _, fs in pilot_frames(a, r) for _, f in fs} | set(s2_frames(r)))
    if a.charades:
        files += charades_frames(a, a.charades)
    files = files[a.shard::a.nshards]
    out = os.path.join(r, "labels", f"det_owlv2{'' if a.nshards == 1 else f'_s{a.shard}of{a.nshards}'}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                done.add(json.loads(l)["f"])
            except json.JSONDecodeError:   # a line cut by a crash: that frame is redone
                pass
    todo = [f for f in files if f not in done]
    print(f"detector: {len(files)} frames in this shard, {len(todo)} to do, {len(words)} labels", flush=True)
    if not todo:
        return
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    proc = Owlv2Processor.from_pretrained(DET)
    model = Owlv2ForObjectDetection.from_pretrained(DET).to(dev).eval()
    tf = text_feats(model, proc, queries, dev)
    fo = open(out, "a")
    for i in range(0, len(todo), a.batch):
        batch = todo[i:i + a.batch]
        ims = [Image.open(f).convert("RGB") for f in batch]
        for f, d in zip(batch, run_det(model, proc, tf, words, ims, dev)):
            fo.write(json.dumps({"f": f, "d": d}) + "\n")
        fo.flush()
        if (i // a.batch) % 50 == 0:
            print(f"  {i + len(batch)}/{len(todo)}", flush=True)


def read_dets(r):
    import glob
    det = {}
    for p in sorted(glob.glob(os.path.join(r, "labels", "det_owlv2*.jsonl"))):
        for l in open(p):
            try:
                x = json.loads(l)
            except json.JSONDecodeError:
                continue
            det[x["f"]] = x["d"]
    return det


def frame_line(tag, d, thr, maxn):
    objs = [f"{w} [{x1} {y1} {x2} {y2}]" for w, s, x1, y1, x2, y2 in d if s >= thr][:maxn]
    return f"{tag} " + (", ".join(objs) if objs else "nothing detected")


def make(a):
    r = H.root(a)
    words = vocab(r)
    det = read_dets(r)
    # 1) label text per pilot question
    b = pilot_args(a)
    path = P.res_path(b, f"labels_owlv2_{P.tag(b)}.jsonl")
    n_miss, n_obj, rows = 0, [], []
    for qid, fs in pilot_frames(a, r):
        if any(f not in det for _, f in fs):
            n_miss += 1
            continue
        lines = [frame_line(t, det[f], a.thr, a.maxn) for t, f in fs]
        n_obj.append(sum(len([1 for x in det[f] if x[1] >= a.thr][:a.maxn]) for _, f in fs))
        rows.append({"id": qid, "evidence": "\n".join(lines), "words": sum(len(l.split()) for l in lines)})
    with open(path, "w") as fo:
        for x in rows:
            fo.write(json.dumps(x) + "\n")
    print(f"label text: {len(rows)} questions written ({n_miss} missing detections) -> {path}; objects per question "
          f"median {int(np.median(n_obj)) if n_obj else 0}; words median {int(np.median([x['words'] for x in rows])) if rows else 0}",
          flush=True)
    # 2) label vectors per 2-second frame (all homes) and for the public Charades homes
    ix = {w: i for i, w in enumerate(words)}

    def vec(fs):
        V = np.zeros((len(fs), len(words)), np.float32)
        ok = np.zeros(len(fs), bool)
        for i, f in enumerate(fs):
            if f in det:
                ok[i] = True
                for w, s, *_ in det[f]:
                    if s >= a.thr:
                        V[i, ix[w]] = 1.0
        return V, ok
    s2 = s2_frames(r)
    V, ok = vec(s2)
    if not ok.all():
        sys.exit(f"{(~ok).sum()} of {len(s2)} 2-second frames have no detections file entry: run detect first")
    idf = np.log((1 + len(V)) / (1 + V.sum(0))) + 1                       # idf from the HD-EPIC frames
    E = V * idf
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    z = np.load(os.path.join(r, "emb_siglip_blur_s2.npz"))
    np.savez(os.path.join(r, "emb_labels_s2.npz"), vid=z["vid"], t=z["t"], e=E)
    print(f"label vectors: {len(E)} frames; frames with no label {(V.sum(1) == 0).mean()*100:.1f}%; labels per frame "
          f"median {int(np.median(V.sum(1)))}", flush=True)
    cf = charades_frames(a, 10 ** 6)
    cf = [f for f in cf if f in det]
    if cf:
        meta = {os.path.join(a.data, "charades_link", "blur", f"{m['video']}_{m['k']}.jpg"): m["subject"]
                for m in json.load(open(os.path.join(a.data, "charades_link", "frames.json")))}
        Vb, _ = vec(cf)
        Eb = Vb * idf
        Eb /= np.linalg.norm(Eb, axis=1, keepdims=True) + 1e-8
        out = os.path.join(a.data, "charades_link", "emb_labels_bank.npz")
        np.savez(out, e=Eb, subject=np.array([meta[f] for f in cf]))
        print(f"public label vectors: {len(Eb)} Charades frames from {len(set(meta[f] for f in cf))} homes -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["detect", "make", "objonly", "objemb"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--charades", type=int, default=0)       # also detect on this many public Charades homes
    ap.add_argument("--thr", type=float, default=0.2)        # make: detections at or above this score are used
    ap.add_argument("--maxn", type=int, default=8)           # make: at most this many objects per frame in the text
    a = ap.parse_args()
    a.data = os.path.abspath(a.data)
    {"detect": detect, "make": make, "objonly": objonly, "objemb": objemb}[a.cmd](a)
