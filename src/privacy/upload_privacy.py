"""Privacy risk of what the edge device sends to the cloud, and the on-device gate that reduces it.

Threat model. The cloud provider (an honest-but-curious party, the "adversary" of the privacy literature) sees every
upload the edge agent sends: the question text plus one escalation form. Uploads travel through an Oblivious HTTP
relay, so the provider has no IP address or account; it can only use the CONTENT. Threats measured, named after the
LINDDUN privacy threat categories:
  Linking          are separate uploads from one home recognisable as one home?  (`link`, `vlmlink`)
  Data disclosure  does an upload show sensitive content (other people, screens, documents, cards, medicine,
                   alcohol, religious items, private rooms, outside view)?          (`judge`)
  Inference        what does the provider learn about the household when it gathers the uploads of one home
                   (children, pets, diet, alcohol, medicine, country, ...)?        (`profile`)
(Identifying: in HD-EPIC each home has one participant, so home linkage is also person linkage.)

Uploads: every selected question of all 9 homes gives one upload per form, built exactly as the agent would build it:
  frames        the question's frames, faces blurred                               (pilot form "frames")
  crop          the same frames cropped to the region the edge 3B picks (pilot `ground` prompt)
  notes         the edge 3B's timed observations, options hidden (pilot `notes` prompt), text only
  question      the question and options only (sent with every escalation; annotator-written)
  frames_gated, crop_gated   the same after the on-device gate: the 3B finds sensitive items in each raw
                frame; their boxes are pixelated; frames showing a private room or nudity are not sent.
All homes use the SAME source: 1-per-second 448-px frames (no dense frames; crops are cut from those frames, not from
the full-resolution video, because only P01 kept its videos; a different source for one home would itself be a cue).

Commands (each resumes; --mock gives random model outputs for plumbing tests):
    P="python -m src.privacy.upload_privacy --data data"     # all 9 homes by default
    $P select                                   # uploads.json: questions per home, balanced over days
    $P ground --model Qwen/Qwen2.5-VL-3B-Instruct ; $P notes --model Qwen/Qwen2.5-VL-3B-Instruct
    $P gate   --model Qwen/Qwen2.5-VL-3B-Instruct [--pilot]   # --pilot: frames of the P01 pilot questions (for QA)
    $P make                                     # gated frames, crops, forms.json
    $P embed                                    # SigLIP / DINOv2 image and BGE text embeddings
    $P link                                     # embedding linkers (untrained, trained), CPU
    $P vlmlink --model Qwen/Qwen2.5-VL-7B-Instruct
    $P judge   --model Qwen/Qwen2.5-VL-32B-Instruct
    $P profile --model Qwen/Qwen2.5-VL-32B-Instruct
    $P report
"""
import os
import re
import sys
import json
import math
import random
import argparse

import numpy as np

from src.data import hdepic as H
from src.qa import hdepic_pilot as P

IMAGE_FORMS = ("frames", "crop", "frames_gated", "crop_gated")
TEXT_FORMS = ("notes", "question")
CHUNK = 8  # the judge sees an upload in chunks of at most 8 images (an item counts if any chunk shows it)


def out_dir(args):
    d = os.path.join(H.root(args), "privacy")
    os.makedirs(d, exist_ok=True)
    return d


def opath(args, name):
    return os.path.join(out_dir(args), name)


def mname(args):
    return P.mname(args)


def load_uploads(args):
    p = opath(args, "uploads.json")
    if not os.path.exists(p):
        sys.exit("no uploads.json: run `select` first")
    return json.load(open(p))


def qindex(args):
    return {q["id"]: q for q in H.questions(H.root(args))}


class Model(P.VLM):
    """pilot.VLM plus mock replies in the formats the privacy prompts ask for."""

    def generate(self, content, files, max_new=200, kind=None):
        if not self.mock or kind is None:
            return super().generate(content, files, max_new)
        rng = self.rng
        if kind == "gate":
            if rng.random() < 0.7:
                return "[]"
            x, y = rng.integers(0, 300, 2)
            lab = ["screen", "paper", "person", "private_room", "medicine"][rng.integers(0, 5)]
            return f'```json\n[{{"bbox_2d": [{x}, {y}, {x + 80}, {y + 60}], "label": "{lab}"}}]\n```'
        if kind == "judge":
            d = {k: bool(rng.random() < 0.15) for k in JUDGE_ITEMS}
            if d["personal_info"]:
                d["read_text"] = "mock"
            return json.dumps(d)
        if kind == "profile":
            vals = {k: (v[rng.integers(0, len(v))] if isinstance(v, list) else "unknown") for k, v in PROFILE_VALUES.items()}
            vals["country"] = ["united kingdom", "unknown"][rng.integers(0, 2)]
            vals["evidence"] = {}
            return json.dumps(vals)
        raise ValueError(kind)


# --- uploads

def select(args):
    """Per home, up to --per_home single-day questions whose videos are extracted, round-robin over days so every
    day is represented (cross-day linkage needs >= 2 days per home). Seeded."""
    r = H.root(args)
    by = {}
    for q in H.questions(r):
        vids = [i["id"] for i in q["inputs"].values()]
        homes, days = {H.home_of(v) for v in vids}, {H.day_of(v) for v in vids}
        if len(homes) != 1 or len(days) != 1:
            continue
        h, d = homes.pop(), days.pop()
        if h not in args.homes:
            continue
        built = H.build(q, r, "blur", args.nframes)
        if built is None:
            continue
        content, files = built
        by.setdefault(h, {}).setdefault(d, []).append({
            "id": q["id"], "home": h, "day": d, "prototype": q["prototype"], "video": H.first_video(q),
            "frames": [os.path.relpath(f, os.path.join(r, "frames", "blur")) for f in files],
            "question": P.question_text(q, content)})
    out = []
    for h in sorted(by):
        days = sorted(by[h])
        for d in days:
            random.Random(f"{h}-{d}").shuffle(by[h][d])
        pick = []
        while len(pick) < args.per_home and any(by[h][d] for d in days):
            for d in days:
                if by[h][d] and len(pick) < args.per_home:
                    pick.append(by[h][d].pop())
        out += pick
        print(f"{h}: {len(pick)} uploads, days " + ", ".join(f"{d}:{sum(u['day'] == d for u in pick)}" for d in days)
              + f", {sum(len(u['frames']) for u in pick)} frames", flush=True)
    missing = sorted(set(args.homes) - set(by))
    if missing:
        print("homes with no extracted video:", " ".join(missing), flush=True)
    json.dump(out, open(opath(args, "uploads.json"), "w"), indent=0)
    print(f"{len(out)} uploads written", flush=True)


def ground(args):
    """Edge 3B picks the region for each upload (same prompt and parsing as the pilot's `ground`)."""
    vlm, qs, r = Model(args), qindex(args), H.root(args)
    fo, done, _l = P.open_out(opath(args, f"ground_{mname(args)}.jsonl"))
    for u in load_uploads(args):
        if u["id"] in done:
            continue
        reply = P.ground_reply(vlm, qs[u["id"]], r, args)
        box = P.parse_box(reply)
        fo.write(json.dumps({"id": u["id"], "reply": reply, "box": box or list(P.CENTER), "parsed": box is not None}) + "\n")
        fo.flush()


def notes(args):
    """Edge 3B's timed observations (same prompt as the pilot's `notes`, options hidden)."""
    vlm, qs, r = Model(args), qindex(args), H.root(args)
    fo, done, _l = P.open_out(opath(args, f"notes_{mname(args)}.jsonl"))
    for u in load_uploads(args):
        if u["id"] in done:
            continue
        fo.write(json.dumps({"id": u["id"], **P.notes_one(vlm, qs[u["id"]], r, args, P.NOTES_ASK)}) + "\n")
        fo.flush()


# --- on-device gate

GATE_LABELS = ("person", "screen", "paper", "card", "medicine", "photo", "private_room", "nudity")
GATE_BLOCK = ("private_room", "nudity")  # the whole frame is withheld
GATE_SYNONYMS = (  # the 3B does not always keep to the label list; first match wins, checked in this order
    ("nudity", r"nud|naked"), ("private_room", r"toilet|bath|shower|\bbed|bedroom|private"),
    ("card", r"card|passport|licen[cs]e"), ("photo", r"photo|picture|portrait"),
    ("screen", r"screen|phone|laptop|computer|monitor|tablet|\btv\b|television|display"),
    ("paper", r"paper|letter|document|mail|envelope|receipt|calendar|note|invoice|bill\b"),
    ("medicine", r"medic|pill|drug|inhaler|syringe|prescription"),
    ("person", r"person|people|face|man\b|woman|child|baby|kid|body|human"))
GATE_ASK = ("This frame comes from a camera in a private home and may be sent to a cloud service. Find every "
            "privacy-sensitive item in it. Use only these labels: person (the face or body of anyone; the camera "
            "wearer's own hands and arms do not count), screen (a phone, computer, tablet or TV screen that is on), "
            "paper (a letter, document, mail, receipt, calendar or note with writing on it), card (a payment card or "
            "ID card), medicine (pills, medicine packaging or medical equipment), photo (a photograph of people), "
            "private_room (a toilet, bath, shower or bed), nudity. Reply in JSON as a list of "
            '{"bbox_2d": [x1, y1, x2, y2], "label": "..."}. Reply [] if there is none.')


def qwen_size(h, w, max_pixels, min_pixels=64 * 28 * 28, f=28):
    """Image size after Qwen2.5-VL's smart_resize (its boxes are in these pixel coordinates)."""
    hb, wb = max(f, round(h / f) * f), max(f, round(w / f) * f)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = math.floor(h / beta / f) * f, math.floor(w / beta / f) * f
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
    return hb, wb


def parse_gate(reply, h, w, max_pixels):
    """-> (items [{label, box in frame pixels}], parsed). Unknown labels are kept (redacted) as 'other'."""
    t = re.sub(r"```(?:json)?", "", reply).strip()
    if re.fullmatch(r"\[\s*\]|none\.?|no(?:ne)? (?:sensitive )?items?\.?", t, re.I):
        return [], True
    objs = None
    try:
        objs = json.loads(t)
    except ValueError:
        found = re.findall(r"\{[^{}]*\}", t)
        if found:
            try:
                objs = [json.loads(o) for o in found]
            except ValueError:
                objs = None
    if objs is None:
        return [], False
    if isinstance(objs, dict):
        objs = [objs]
    hb, wb = qwen_size(h, w, max_pixels)
    items = []
    for o in objs:
        if not isinstance(o, dict):
            return [], False
        b = o.get("bbox_2d", o.get("bbox"))
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
            return [], False
        x1, y1, x2, y2 = min(b[0], b[2]) * w / wb, min(b[1], b[3]) * h / hb, max(b[0], b[2]) * w / wb, max(b[1], b[3]) * h / hb
        lab = str(o.get("label", "")).strip().lower().replace(" ", "_")
        if lab not in GATE_LABELS:
            lab = next((k for k, pat in GATE_SYNONYMS if re.search(pat, lab.replace("_", " "))), "other")
        items.append({"label": lab, "raw_label": str(o.get("label", "")),
                      "box": [round(max(0, x1)), round(max(0, y1)), round(min(w, x2)), round(min(h, y2))]})
    return items, True


def gate_frames(args):
    """Relative frame paths the gate must check: all upload frames, or (--pilot) the P01 pilot questions' frames
    (with dense frames), so the 7B can answer the pilot questions from gated frames."""
    r = H.root(args)
    if not args.pilot:
        return sorted({f for u in load_uploads(args) for f in u["frames"]})
    a = argparse.Namespace(**vars(args))
    a.homes, a.split, a.per_type, a.dense, a.variant, a.limit = ["P01"], "dev", 30, True, "blur", None
    fs = set()
    for q in H.select(a, r):
        _, files = H.build(q, r, "blur", a.nframes, dense=True)
        fs |= {os.path.relpath(f, os.path.join(r, "frames", "blur")) for f in files}
    return sorted(fs)


def read_gates(args, model=None):
    """All gate rows of one model (the run may be split into shard files and a pilot file)."""
    import glob
    rows = {}
    for f in sorted(glob.glob(opath(args, f"gate_{model or args.gate_model}*.jsonl"))):
        rows.update(P.read_jsonl(f))
    return rows


def gate(args):
    """Edge 3B checks each RAW frame for sensitive items (one frame per call, Qwen's native box format).
    A reply that cannot be parsed blocks the frame (the gate fails closed). Each process (shard, --pilot) writes its
    own file; frames already checked in any file are skipped."""
    import cv2
    r, vlm = H.root(args), Model(args)
    seen = set(read_gates(args, mname(args)))
    suffix = ("_pilot" if args.pilot else "") + (f"_s{args.shard}of{args.nshards}" if args.nshards > 1 else "")
    fo, done, _l = P.open_out(opath(args, f"gate_{mname(args)}{suffix}.jsonl"))
    todo = [f for f in gate_frames(args)[args.shard::args.nshards] if f not in seen]  # shard the full list first
    print(f"gate: {len(todo)} frames to check", flush=True)
    for i, f in enumerate(todo):
        raw = os.path.join(r, "frames", "raw", f)
        h, w = cv2.imread(raw).shape[:2]
        reply = vlm.generate([{"type": "image"}, {"type": "text", "text": GATE_ASK}], [raw], 200, kind="gate")
        items, ok = parse_gate(reply, h, w, args.max_pixels)
        fo.write(json.dumps({"id": f, "reply": reply, "items": items, "parsed": ok,
                             "block": (not ok) or any(it["label"] in GATE_BLOCK for it in items)}) + "\n")
        fo.flush()
        if i % 2000 == 0:
            print(f"  {i}/{len(todo)}", flush=True)


def redact(img, boxes):
    """Pixelate each box to 4 x 4 cells and blur (as hdepic.blur_boxes, without enlarging the box)."""
    import cv2
    out = img.copy()
    for x1, y1, x2, y2 in boxes:
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        small = cv2.resize(out[y1:y2, x1:x2], (4, 4), interpolation=cv2.INTER_AREA)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST), (0, 0), 3)
    return out


# --- forms

def make(args):
    """Gated frames (frames/gated/..., also used by `hdepic.py qa --variant gated`: a withheld frame becomes a gray
    image there), crops of the blurred and gated frames, and forms.json: form -> upload id -> files or text."""
    import cv2
    r = H.root(args)
    gates = read_gates(args)
    boxes = P.read_jsonl(opath(args, f"ground_{args.ground_model}.jsonl"))
    notes_ = P.read_jsonl(opath(args, f"notes_{args.notes_model}.jsonl"))
    made = 0
    for f, g in gates.items():  # every gated frame, including the pilot frames
        dst = os.path.join(r, "frames", "gated", f)
        if os.path.exists(dst):
            continue
        img = cv2.imread(os.path.join(r, "frames", "blur", f))
        img = np.full_like(img, 128) if g["block"] else redact(img, [it["box"] for it in g["items"]])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        made += 1
    print(f"gated frames: {made} written ({len(gates)} checked)", flush=True)
    forms = {k: {} for k in IMAGE_FORMS + TEXT_FORMS}
    miss = {"gate": 0, "ground": 0, "notes": 0}
    for u in load_uploads(args):
        forms["question"][u["id"]] = u["question"]
        if u["id"] in notes_:
            forms["notes"][u["id"]] = notes_[u["id"]]["evidence"]
        else:
            miss["notes"] += 1
        forms["frames"][u["id"]] = [os.path.join("frames", "blur", f) for f in u["frames"]]
        if not all(f in gates for f in u["frames"]):
            miss["gate"] += 1
            continue
        kept = [f for f in u["frames"] if not gates[f]["block"]]
        forms["frames_gated"][u["id"]] = [os.path.join("frames", "gated", f) for f in kept]
        if u["id"] not in boxes:
            miss["ground"] += 1
            continue
        x1, y1, x2, y2 = boxes[u["id"]]["box"]
        for form, src, fl in (("crop", "blur", u["frames"]), ("crop_gated", "gated", kept)):
            outs = []
            for f in fl:
                rel = os.path.join("privacy", "crop", form, u["id"], f.replace("/", "_"))
                dst = os.path.join(r, rel)
                if not os.path.exists(dst):
                    img = cv2.imread(os.path.join(r, "frames", src, f))
                    c = img[y1:y2, x1:x2]
                    k = 448 / min(c.shape[:2])
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    cv2.imwrite(dst, cv2.resize(c, (round(c.shape[1] * k), round(c.shape[0] * k)), interpolation=cv2.INTER_CUBIC),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                outs.append(rel)
            forms[form][u["id"]] = outs
    json.dump(forms, open(opath(args, "forms.json"), "w"))
    print("forms: " + ", ".join(f"{k} {len(v)}" for k, v in forms.items()) + f" | missing {miss}", flush=True)
    blocked = [g for g in gates.values() if g["block"]]
    print(f"gate: {len(blocked)}/{len(gates)} frames withheld ({sum(not g['parsed'] for g in gates.values())} unparsed "
          f"replies, withheld by fail-closed), {sum(len(g['items']) for g in gates.values())} boxes redacted", flush=True)


def load_forms(args):
    p = opath(args, "forms.json")
    if not os.path.exists(p):
        sys.exit("no forms.json: run `make` first")
    return json.load(open(p))


# --- embedding linkers

def embed(args):
    """Per image form: SigLIP and DINOv2 embeddings of every file. Text forms: BGE (CLS, normalised)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
    r, forms = H.root(args), load_forms(args)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    for bb in ("siglip", "dinov2"):
        proc = model = None
        for form in IMAGE_FORMS:
            out = opath(args, f"emb_{bb}_{form}.npz")
            files = sorted({f for fl in forms[form].values() for f in fl})
            if os.path.exists(out) and sorted(np.load(out)["files"].tolist()) == files:
                continue
            if model is None:
                proc = AutoImageProcessor.from_pretrained(H.BACKBONES[bb])
                model = AutoModel.from_pretrained(H.BACKBONES[bb]).to(dev).eval()
            E = []
            for i in range(0, len(files), 64):
                ims = [Image.open(os.path.join(r, f)).convert("RGB") for f in files[i:i + 64]]
                x = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
                with torch.no_grad():
                    if bb == "dinov2":
                        hdn = model(pixel_values=x).last_hidden_state
                        e = torch.cat([hdn[:, 0], hdn[:, 1:].mean(1)], 1)
                    else:
                        e = model.get_image_features(pixel_values=x)
                E.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
            np.savez(out, files=np.array(files), e=np.concatenate(E) if E else np.zeros((0, 1), np.float32))
            print(f"embedded {bb}/{form}: {len(files)} images", flush=True)
    tok = AutoTokenizer.from_pretrained(args.text_model)
    tm = AutoModel.from_pretrained(args.text_model).to(dev).eval()
    for form in TEXT_FORMS:
        ids = sorted(forms[form])
        if not ids:
            print(f"no {form} texts to embed", flush=True)
            continue
        E = []
        for i in range(0, len(ids), 64):
            x = tok([forms[form][k] for k in ids[i:i + 64]], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                e = tm(**x).last_hidden_state[:, 0]
            E.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        np.savez(opath(args, f"emb_bge_{form}.npz"), ids=np.array(ids), e=np.concatenate(E))
        print(f"embedded bge/{form}: {len(ids)} texts", flush=True)


def upload_sim(F, owner, n_up):
    """Upload-level similarity: U[i, j] = max cosine between any file of upload i and any file of upload j."""
    order = np.argsort(owner, kind="stable")
    F, owner = F[order], owner[order]
    starts = np.searchsorted(owner, np.arange(n_up))
    U = np.full((n_up, n_up), -np.inf, np.float32)
    present = np.unique(owner)
    for i in present:
        a, b = starts[i], np.searchsorted(owner, i, side="right")
        m = (F[a:b] @ F.T).max(0)
        U[i, present] = np.maximum.reduceat(m, starts[present])
    return U


def link_U(U, home, day, ok, K, episodes, ms, label, rng, quiet=False):
    """K-way and gallery rank-1 over uploads (rows of U). Query: an upload of home h on day d. True history: m uploads of
    h from OTHER days; every other candidate home: m of its uploads. Score: max similarity. Ties count 1/#tied.
    Only uploads with ok=True take part. 95% CI: bootstrap over query homes."""
    homes = sorted({home[i] for i in np.flatnonzero(ok)})
    res = {}
    for m in ms:
        pools = {h: np.flatnonzero(ok & (home == h)) for h in homes}
        qable = [h for h in homes if any((ok & (home == h) & (day != d)).sum() >= m for d in set(day[pools[h]]))]
        out = {"k": [], "g": [], "qh": [], "gsize": []}
        tries = 0
        while len(out["k"]) < episodes and qable and tries < 50 * episodes:
            tries += 1
            h = qable[rng.integers(len(qable))]
            q = pools[h][rng.integers(len(pools[h]))]
            true_pool = np.flatnonzero(ok & (home == h) & (day != day[q]))
            others = [g for g in homes if g != h and len(pools[g]) >= m]
            if len(true_pool) < m or len(others) < K - 1:
                continue
            score = lambda p: float(U[q, rng.choice(p, m, replace=False)].max())
            s_true = score(true_pool)
            s_oth = {g: score(pools[g]) for g in others}
            def r1(cands):
                s = np.array([s_true] + [s_oth[g] for g in cands])
                top = np.flatnonzero(s >= s.max() - 1e-9)
                return float(0 in top) / len(top)
            pick = [others[i] for i in rng.choice(len(others), K - 1, replace=False)]
            out["k"].append(r1(pick))
            out["g"].append(r1(others))
            out["gsize"].append(len(others) + 1)
            out["qh"].append(h)
        if not out["k"]:
            continue
        k = H.cluster_ci(out["k"], out["qh"])
        g = H.cluster_ci(out["g"], out["qh"])
        gs = float(np.mean(out["gsize"]))
        res[m] = {"k": [round(100 * x, 1) for x in k], "gallery": [round(100 * x, 1) for x in g], "gallery_size": round(gs, 1),
                  "n": len(out["k"]), "query_homes": len(set(out["qh"]))}
        if not quiet:
            print(f"  {label} m={m:2d}: K={K} {100 * k[0]:.1f} [{100 * k[1]:.1f}, {100 * k[2]:.1f}] (chance {100 / K:.0f}) | "
              f"gallery {100 * g[0]:.1f} [{100 * g[1]:.1f}, {100 * g[2]:.1f}] (chance {100 / gs:.0f}) | n={len(out['k'])}, "
                  f"{len(set(out['qh']))} query homes", flush=True)
    return res


def pair_auc(U, home, day, ok):
    """Same-home pairs from different days vs different-home pairs: ROC-AUC and TPR at 1% FPR."""
    from sklearn.metrics import roc_auc_score
    idx = np.flatnonzero(ok)
    i, j = np.triu_indices(len(idx), 1)
    a, b = idx[i], idx[j]
    same, diffday = home[a] == home[b], day[a] != day[b]
    pos, neg = U[a, b][same & diffday], U[a, b][~same]
    if len(pos) < 10 or len(neg) < 10:
        return None
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    thr = np.quantile(neg, 0.99)
    return {"auc": round(float(roc_auc_score(y, np.r_[pos, neg])), 3), "tpr_at_1fpr": round(100 * float((pos > thr).mean()), 1),
            "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def link(args):
    """Linkage of each form with embedding linkers -> privacy/link_embed.json.
    untrained: nearest neighbour on SigLIP / DINOv2 (images), TF-IDF / BGE (text).
    trained: a contrastive head on DINOv2 (train_linker.train_head) trained on the frames of the 5 homes that are NOT
    candidates, for --groups random groups of 4 candidate homes (all 4 are the candidates, K = 4)."""
    import torch
    import torch.nn.functional as Fn
    from src.linking.train_linker import train_head
    ups = load_uploads(args)
    forms = load_forms(args)
    ids = [u["id"] for u in ups]
    pos = {k: i for i, k in enumerate(ids)}
    home, day = np.array([u["home"] for u in ups]), np.array([u["day"] for u in ups])
    ms, res = (1, 2, 4, 8, 16), {}
    rng = np.random.default_rng(0)
    for form in IMAGE_FORMS:
        for bb in ("siglip", "dinov2"):
            z = np.load(opath(args, f"emb_{bb}_{form}.npz"))
            fidx = {f: i for i, f in enumerate(z["files"].tolist())}
            rows, owner = [], []
            for k, fl in forms[form].items():
                for f in fl:
                    rows.append(fidx[f])
                    owner.append(pos[k])
            if not rows:
                continue
            F, owner = z["e"][rows].astype(np.float32), np.array(owner)
            ok = np.isin(np.arange(len(ids)), owner)
            U = upload_sim(F, owner, len(ids))
            print(f"{form} / {bb} untrained ({ok.sum()} uploads)", flush=True)
            res[f"{form}|{bb}"] = {"rank1": link_U(U, home, day, ok, args.K, args.episodes, ms, f"{form}/{bb}", rng),
                                   "pairs": pair_auc(U, home, day, ok)}
            if bb != "dinov2":
                continue
            homes = sorted(set(home[ok]))
            per_group = max(1, args.episodes // args.groups)
            acc = {m: {"k": [], "qh": []} for m in ms}
            for gi in range(args.groups):
                cand = sorted(rng.choice(homes, args.K, replace=False).tolist())
                tr = np.isin(home[owner], [h for h in homes if h not in cand])
                lab = {h: i for i, h in enumerate(sorted(set(home[owner][tr])))}
                head = train_head(F[tr], np.array([lab[h] for h in home[owner][tr]]), seed=gi)
                with torch.no_grad():
                    Z = Fn.normalize(head(torch.tensor(F)), dim=-1).numpy()
                Ug = upload_sim(Z, owner, len(ids))
                okg = ok & np.isin(home, cand)
                r = link_U(Ug, home, day, okg, args.K, per_group, ms, f"{form}/trained group {gi}", rng, quiet=not args.verbose)
                for m, v in r.items():
                    acc[m]["k"] += [v["k"][0] / 100] * v["n"]
                    acc[m]["qh"] += [f"{gi}"] * v["n"]
            tr_res = {}
            for m in ms:
                if acc[m]["k"]:
                    mean = float(np.mean(acc[m]["k"]))
                    g = np.array([np.mean([x for x, q in zip(acc[m]["k"], acc[m]["qh"]) if q == key]) for key in sorted(set(acc[m]["qh"]))])
                    lo, hi = np.percentile([g[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)], [2.5, 97.5])
                    tr_res[m] = {"k": [round(100 * mean, 1), round(100 * lo, 1), round(100 * hi, 1)], "groups": len(g)}
                    print(f"  {form}/dinov2 TRAINED m={m:2d}: K={args.K} {100 * mean:.1f} [{100 * lo:.1f}, {100 * hi:.1f}] "
                          f"(CI over {len(g)} home groups)", flush=True)
            res[f"{form}|dinov2_trained"] = {"rank1": tr_res}
    for form in TEXT_FORMS:
        keys = [k for k in ids if k in forms[form]]
        if not keys:
            continue
        ok = np.isin(ids, keys)
        Xt = P.tfidf([forms[form].get(k, "") for k in ids])
        z = np.load(opath(args, f"emb_bge_{form}.npz"))
        bidx = {k: i for i, k in enumerate(z["ids"].tolist())}
        Xb = np.stack([z["e"][bidx[k]] if k in bidx else np.zeros(z["e"].shape[1], np.float32) for k in ids])
        for name, X in (("tfidf", Xt), ("bge", Xb)):
            U = (X @ X.T).astype(np.float32)
            print(f"{form} / {name} ({ok.sum()} uploads)", flush=True)
            res[f"{form}|{name}"] = {"rank1": link_U(U, home, day, ok, args.K, args.episodes, ms, f"{form}/{name}", rng),
                                     "pairs": pair_auc(U, home, day, ok)}
    json.dump(res, open(opath(args, "link_embed.json"), "w"), indent=1)


# --- VLM linker (the provider's own VLM)

LINK_ASK = ("Upload Q is a new upload to a cloud service from a camera in a private home. Sets A, B, C and D each hold "
            "earlier uploads, and each set comes from one household. Faces are blurred. Which set comes from the same "
            "household as upload Q? Answer with one letter: A, B, C or D.")
VLM_CONDITIONS = (("frames", "frames"), ("crop", "crop"), ("notes", "notes"), ("question", "question"),
                  ("frames_gated", "frames_gated"), ("crop_gated", "crop_gated"),
                  ("crop", "frames"), ("notes", "frames"))  # (query form, history form): the last two are mixed


def upload_content(form, item, n_img, r):
    """(content, files) showing one upload: up to n_img evenly spaced images, or its text."""
    if form in TEXT_FORMS:
        return [{"type": "text", "text": "\n" + item + "\n"}], []
    if not item:
        return None
    sel = [item[i] for i in sorted(set(np.linspace(0, len(item) - 1, min(n_img, len(item))).round().astype(int).tolist()))]
    return [{"type": "image"} for _ in sel], [os.path.join(r, f) for f in sel]


def vlmlink(args):
    """K = 4 episodes with the cloud VLM as the linker; letter probabilities from the next-token logits.
    Query upload: up to 4 images or its text; each history upload: 1 image (its middle frame) or its text."""
    r, ups, forms = H.root(args), load_uploads(args), load_forms(args)
    home = {u["id"]: u["home"] for u in ups}
    day = {u["id"]: u["day"] for u in ups}
    vlm = Model(args)
    fo, done, _l = P.open_out(opath(args, f"vlmlink_{mname(args)}.jsonl"))
    for qf, hf in VLM_CONDITIONS:
        for m in (1, 2, 4):
            rng = random.Random(f"{qf}|{hf}|{m}")
            qids = [k for k in forms[qf] if forms[qf][k]]
            hist = {}
            for k, v in forms[hf].items():
                if v:
                    hist.setdefault(home[k], []).append(k)
            n = made = 0
            while made < args.vlm_episodes and n < 50 * args.vlm_episodes:
                n += 1
                q = rng.choice(qids)
                h = home[q]
                true_pool = [k for k in hist.get(h, []) if day[k] != day[q]]
                others = [g for g in sorted(hist) if g != h and len(hist[g]) >= m]
                if len(true_pool) < m or len(others) < 3:
                    continue
                eid = f"{qf}|{hf}|{m}|{made}"
                made += 1
                cands = [h] + rng.sample(others, 3)
                rng.shuffle(cands)
                sets = [rng.sample(true_pool if g == h else hist[g], m) for g in cands]
                if eid in done:
                    continue
                c, files = upload_content(qf, forms[qf][q], 4, r)
                content = [{"type": "text", "text": "Upload Q:"}] + c
                for L, s in zip("ABCD", sets):
                    content.append({"type": "text", "text": f"Set {L}:"})
                    for k in s:
                        ci, fi = upload_content(hf, forms[hf][k], 1, r)
                        content += ci
                        files += fi
                content.append({"type": "text", "text": LINK_ASK})
                p, mass, ntok = vlm.letters(content, files)
                p4 = p[:4] / p[:4].sum()
                fo.write(json.dumps({"id": eid, "query_form": qf, "history_form": hf, "m": m, "query": q, "home": h,
                                     "answer": cands.index(h), "sets": sets, "probs": p4.round(4).tolist(),
                                     "pred": int(p4.argmax()), "letter_mass": round(mass, 4), "n_images": len(files),
                                     "n_tokens": ntok}) + "\n")
                fo.flush()
            print(f"vlmlink {qf}->{hf} m={m}: {made} episodes", flush=True)


# --- data disclosure (sensitive content)

JUDGE_ITEMS = {
    "person": "a person other than the camera wearer (face or body; the wearer's own hands and arms do not count)",
    "child": "a child or baby, or children's items such as toys, a high chair or a baby bottle",
    "screen": "a phone, computer, tablet or TV screen that is on",
    "readable_text": "handwritten or printed text that can be read, such as a letter, mail, a document, a note or a "
                     "calendar (product labels and packaging do not count)",
    "personal_info": "a name, address, phone number, email address or account number that can be read",
    "card": "a payment card, ID card or bank document",
    "health": "medicine, pills, medical equipment or health documents",
    "alcohol_tobacco": "alcohol or tobacco",
    "religion_politics": "a religious or political object or symbol",
    "photo": "a photograph or picture of people",
    "private_room": "a bathroom, toilet, shower or bed, or nudity",
    "outside_view": "a view outside through a window or door (street, garden or other buildings)",
}
JUDGE_HEAD = {"image": "These images were sent to a cloud service by a camera in a private home. For each item below, "
                       "answer true if it is visible in ANY of the images, otherwise false.",
              "text": "This text was sent to a cloud service by a camera assistant in a private home. For each item "
                      "below, answer true if the text mentions or describes it, otherwise false."}


def judge_prompt(kind):
    keys = "\n".join(f'"{k}": {v}' for k, v in JUDGE_ITEMS.items())
    return (f"{JUDGE_HEAD[kind]}\n{keys}\nReply with JSON only, with exactly these keys and true or false values. If "
            '"personal_info" is true, also add the key "read_text" with the text you can read.')


def parse_json(reply):
    t = re.sub(r"```(?:json)?", "", reply).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def parse_judge(reply):
    d = parse_json(reply)
    if d is None or not all(isinstance(d.get(k), bool) for k in JUDGE_ITEMS):
        return None
    return {k: d[k] for k in JUDGE_ITEMS}


GATED_OF = {"frames_gated": "frames", "crop_gated": "crop"}


def changed_chunks(args, forms):
    """(form, upload, chunk) of the gated forms whose images the gate changed (a redacted or withheld frame).
    Unchanged gated chunks show the same images as the ungated chunk (re-saved as JPEG)."""
    gates = read_gates(args)
    out = set()
    for k, fl in forms["frames_gated"].items():
        orig = [f[len("frames/gated/"):] for f in fl]
        for i, f in enumerate(orig):
            g = gates.get(f)
            if g and (g["block"] or g["items"]):
                out |= {("frames_gated", k, i // CHUNK), ("crop_gated", k, i // CHUNK)}
    for k in forms["frames"]:   # withheld frames shift the chunks of that upload: treat the whole upload as changed
        if len(forms["frames_gated"].get(k, [])) != len(forms["frames"][k]):
            n = (len(forms["frames"][k]) + CHUNK - 1) // CHUNK
            out |= {(fg, k, c) for fg in GATED_OF for c in range(n)}
    return out


def judge_sample(args, forms):
    """Upload ids judged for the crop / notes forms (and their gated versions): all, or a fixed random sample."""
    ids = sorted(forms["frames"])
    if not args.judge_uploads or args.judge_uploads >= len(ids):
        return set(ids)
    return set(random.Random(0).sample(ids, args.judge_uploads))


def judge(args):
    """The cloud model (stand-in: Qwen2.5-VL-32B) lists the sensitive content of every upload of every form, in chunks
    of at most CHUNK images; image forms also for the gated uploads, so the gate's effect is measured by a model
    that is not the gate.
    --judge_uploads N: crop, crop_gated and notes only for a fixed random sample of N uploads (frames: all).
    --gated_changed_only: gated chunks are judged only where the gate changed an image, plus --verify_unchanged
    unchanged chunks (agreement with the ungated verdict is reported); the report reuses the ungated verdict for
    the other unchanged chunks."""
    r, forms, vlm = H.root(args), load_forms(args), Model(args)
    fo, done, _l = P.open_out(opath(args, f"judge_{mname(args)}.jsonl"))
    sample = judge_sample(args, forms)
    changed = changed_chunks(args, forms) if args.gated_changed_only else None
    verify = set()
    if changed is not None:
        cand = sorted((fg, k, c) for fg in GATED_OF for k in forms[fg] if fg == "frames_gated" or k in sample
                      for c in range((len(forms[fg][k]) + CHUNK - 1) // CHUNK) if (fg, k, c) not in changed)
        verify = set(random.Random(1).sample(cand, min(args.verify_unchanged, len(cand))))
        print(f"judge: {len(changed)} gated chunks changed by the gate; {len(verify)} unchanged chunks judged to verify "
              f"reuse; crop/notes sample {len(sample)} uploads", flush=True)
    for form in ("frames", "frames_gated", "crop", "crop_gated", "notes"):
        for k in sorted(forms[form]):
            if form != "frames" and form != "frames_gated" and k not in sample:
                continue
            item = forms[form][k]
            chunks = [item] if form in TEXT_FORMS else [item[i:i + CHUNK] for i in range(0, len(item), CHUNK)]
            for c, part in enumerate(chunks):
                jid = f"{form}|{k}|{c}"
                if jid in done:
                    continue
                if changed is not None and form in GATED_OF and (form, k, c) not in changed and (form, k, c) not in verify:
                    continue
                if form in TEXT_FORMS:
                    content, files = [{"type": "text", "text": f"Text:\n{part}\n\n{judge_prompt('text')}"}], []
                else:
                    content = [{"type": "image"} for _ in part] + [{"type": "text", "text": judge_prompt("image")}]
                    files = [os.path.join(r, f) for f in part]
                reply = vlm.generate(content, files, 300, kind="judge")
                lab = parse_judge(reply)
                fo.write(json.dumps({"id": jid, "form": form, "upload": k, "chunk": c, "n_images": len(files),
                                     "reply": reply, "labels": lab, "parsed": lab is not None}) + "\n")
                fo.flush()
        print(f"judge {form}: done", flush=True)


# --- inference about the household (profile)

PROFILE_VALUES = {"country": None, "children": ["yes", "no", "unknown"], "pets": ["yes", "no", "unknown"],
                  "alcohol": ["yes", "no", "unknown"], "smoking": ["yes", "no", "unknown"],
                  "medication": ["yes", "no", "unknown"], "diet": ["vegetarian", "vegan", "meat_or_fish", "unknown"],
                  "religion": None, "income": ["low", "middle", "high", "unknown"],
                  "adults": ["1", "2", "3_or_more", "unknown"]}
PROFILE_ASK = ("All of these uploads come from the same household. Infer the following about the household. Use "
               '"unknown" when the uploads give no evidence. Reply with JSON only, with exactly these keys: '
               '"country" (the country where the home is), "children" (yes, no or unknown: a child lives there), '
               '"pets" (yes, no or unknown), "alcohol" (yes, no or unknown: alcohol is kept or drunk there), '
               '"smoking" (yes, no or unknown), "medication" (yes, no or unknown: someone there takes medicine), '
               '"diet" (vegetarian, vegan, meat_or_fish or unknown), "religion" (the religion, or unknown), '
               '"income" (low, middle, high or unknown), "adults" (1, 2, 3_or_more or unknown: adults living there), '
               '"evidence" (an object mapping each key whose value is not unknown to one short sentence of evidence).')
PROFILE_FORMS = ("frames", "crop", "notes", "question", "frames_gated", "crop_gated")
COUNTRY = {"uk": "united kingdom", "u.k.": "united kingdom", "england": "united kingdom", "britain": "united kingdom",
           "great britain": "united kingdom", "scotland": "united kingdom", "wales": "united kingdom",
           "usa": "united states", "us": "united states", "u.s.": "united states", "america": "united states"}


def norm_value(k, v):
    v = str(v).strip().lower().rstrip(".")
    if k == "country":
        return COUNTRY.get(v, v)
    if k == "religion" and v in ("none", "no religion", "not known", "unclear"):
        return "unknown"
    return v.replace(" ", "_")


def parse_profile(reply):
    d = parse_json(reply)
    if d is None or not all(k in d for k in PROFILE_VALUES):
        return None
    return {k: norm_value(k, d[k]) for k in PROFILE_VALUES}, d.get("evidence", {})


def reference_frames(args, h, n=64):
    """n raw frames of home h, evenly spaced over all of its extracted seconds (all videos, all days)."""
    r = H.root(args)
    import glob
    vids = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(r, "frames", "raw", f"{h}-*", "done.json")))
    secs = [(v, t) for v in vids for t in range(H.have(r, v))]
    pick = [secs[i] for i in np.linspace(0, len(secs) - 1, min(n, len(secs))).round().astype(int)]
    return [os.path.join("frames", "raw", v, f"{t:06d}.jpg") for v, t in pick]


def profile(args):
    """(1) Reference profile of each home from 64 RAW frames over all its footage (4 calls of 16 frames; a key's value
    is the most frequent non-unknown answer): what the home shows to a party that sees everything. Checked by a
    person afterwards (privacy/profile_check.csv). (2) The same questions from m uploads of one form (m = 1, 4, 16;
    3 random draws per home): what the provider can infer from what the agent sends."""
    r, ups, forms, vlm = H.root(args), load_uploads(args), load_forms(args), Model(args)
    fo, done, _l = P.open_out(opath(args, f"profile_{mname(args)}.jsonl"))
    homes = sorted({u["home"] for u in ups})
    for h in homes:
        fr = reference_frames(args, h)
        for c in range(0, len(fr), 16):
            pid = f"reference|{h}|{c // 16}"
            if pid in done:
                continue
            part = fr[c:c + 16]
            content = [{"type": "image"} for _ in part] + [{"type": "text", "text": PROFILE_ASK}]
            reply = vlm.generate(content, [os.path.join(r, f) for f in part], 400, kind="profile")
            got = parse_profile(reply)
            fo.write(json.dumps({"id": pid, "kind": "reference", "home": h, "files": part, "reply": reply,
                                 "profile": got[0] if got else None, "evidence": got[1] if got else None}) + "\n")
            fo.flush()
    by_home = {}
    for u in ups:
        by_home.setdefault(u["home"], []).append(u["id"])
    for form in args.profile_forms:
        for m in (1, 4, 16):
            for h in homes:
                pool = [k for k in by_home[h] if forms[form].get(k)]
                if len(pool) < m:
                    continue
                rng = random.Random(f"{form}|{m}|{h}")
                for d in range(3):
                    pid = f"{form}|{m}|{h}|{d}"
                    s = rng.sample(pool, m)  # drawn before the resume check: a resumed run makes the same draws
                    if pid in done:
                        continue
                    content, files = [], []
                    for i, k in enumerate(s):
                        ci, fi = upload_content(form, forms[form][k], 1, r)
                        content += [{"type": "text", "text": f"Upload {i + 1}:"}] + ci
                        files += fi
                    content.append({"type": "text", "text": PROFILE_ASK})
                    reply = vlm.generate(content, files, 400, kind="profile")
                    got = parse_profile(reply)
                    fo.write(json.dumps({"id": pid, "kind": form, "m": m, "home": h, "uploads": s, "reply": reply,
                                         "profile": got[0] if got else None}) + "\n")
                    fo.flush()
        print(f"profile {form}: done", flush=True)


# --- report

def ci_homes(vals, homes):
    m, lo, hi = H.cluster_ci(vals, homes)
    return f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def reference_profiles(rows):
    ref = {}
    for x in rows:
        if x["kind"] != "reference" or not x["profile"]:
            continue
        for k, v in x["profile"].items():
            ref.setdefault(x["home"], {}).setdefault(k, []).append(v)
    out = {}
    for h, d in ref.items():
        out[h] = {}
        for k, vs in d.items():
            known = [v for v in vs if v != "unknown"]
            out[h][k] = max(sorted(set(known)), key=known.count) if known else "unknown"
    return out


def specific(k, v):
    """Values that disclose something: not unknown, and not a plain 'no' (absence of evidence)."""
    return v not in ("unknown", "no", "", "none")


def report(args):
    lines = ["# Privacy of device-to-cloud uploads", ""]
    ups = load_uploads(args)
    home_of = {u["id"]: u["home"] for u in ups}
    lines.append(f"Uploads: {len(ups)} from {len({u['home'] for u in ups})} homes "
                 + ", ".join(f"{h} {sum(u['home'] == h for u in ups)}" for h in sorted({u['home'] for u in ups})))
    # gate
    g = read_gates(args)
    if g:
        labs = {}
        for x in g.values():
            for it in x["items"]:
                labs[it["label"]] = labs.get(it["label"], 0) + 1
        lines += ["", "## On-device gate (edge 3B, per raw frame)",
                  f"{len(g)} frames checked; {sum(x['block'] for x in g.values())} withheld; "
                  f"{sum(not x['parsed'] for x in g.values())} unparsed (withheld); boxes by label: {labs}"]
    # linkage (embeddings)
    p = opath(args, "link_embed.json")
    if os.path.exists(p):
        L = json.load(open(p))
        lines += ["", "## Linking: embedding linkers (cross-day, K = 4, chance 25; gallery = all homes)",
                  "| form / linker | m=1 | m=4 | m=8 | m=16 | gallery m=8 | pair AUC | TPR@1%FPR |", "|---|---|---|---|---|---|---|---|"]
        for key, v in L.items():
            rk = v["rank1"]
            cell = lambda m: (f"{rk[str(m)]['k'][0]:.1f} [{rk[str(m)]['k'][1]:.0f}, {rk[str(m)]['k'][2]:.0f}]" if str(m) in rk else "-")
            gal = rk.get("8", {}).get("gallery")
            pr = v.get("pairs") or {}
            lines.append(f"| {key} | {cell(1)} | {cell(4)} | {cell(8)} | {cell(16)} | "
                         f"{(str(gal[0]) + ' (chance ' + str(round(100 / rk['8']['gallery_size'])) + ')') if gal else '-'} | "
                         f"{pr.get('auc', '-')} | {pr.get('tpr_at_1fpr', '-')} |")
    # VLM linker
    v = P.read_jsonl(opath(args, f"vlmlink_{args.link_model}.jsonl"))
    if v:
        lines += ["", f"## Linking: cloud VLM as linker ({args.link_model}, K = 4, chance 25; CI over query homes)",
                  "| query form -> history form | m=1 | m=2 | m=4 |", "|---|---|---|---|"]
        for qf, hf in VLM_CONDITIONS:
            row = []
            for m in (1, 2, 4):
                rs = [x for x in v.values() if x["query_form"] == qf and x["history_form"] == hf and x["m"] == m]
                row.append(ci_homes([x["pred"] == x["answer"] for x in rs], [x["home"] for x in rs]) + f" n={len(rs)}" if rs else "-")
            lines.append(f"| {qf} -> {hf} | " + " | ".join(row) + " |")
    # disclosure
    j = P.read_jsonl(opath(args, f"judge_{args.judge_model}.jsonl"))
    if j:
        up = {}
        bad = 0
        rows = {x["id"]: x for x in j.values()}
        # unchanged gated chunks not judged: same images as the ungated chunk -> its verdict (checked below)
        fms = load_forms(args)
        changed = changed_chunks(args, fms)
        agree, nver, reused = 0, 0, 0
        for fg, fu in GATED_OF.items():
            for k in fms[fg]:
                for c in range((len(fms[fg][k]) + CHUNK - 1) // CHUNK):
                    gid, uid = f"{fg}|{k}|{c}", f"{fu}|{k}|{c}"
                    if (fg, k, c) in changed or uid not in rows:
                        continue
                    if gid in rows:
                        if rows[gid]["parsed"] and rows[uid]["parsed"]:
                            nver += 1
                            agree += rows[gid]["labels"] == rows[uid]["labels"]
                    else:
                        rows[gid] = dict(rows[uid], id=gid, form=fg)
                        reused += 1
        for x in rows.values():
            if not x["parsed"]:
                bad += 1
                continue
            d = up.setdefault((x["form"], x["upload"]), {k: False for k in JUDGE_ITEMS})
            for k in JUDGE_ITEMS:
                d[k] |= x["labels"][k]
        common = set.intersection(*[{u for (ff, u) in up if ff == f} for f in ("frames", "crop", "notes")]) \
            if all(any(ff == f for (ff, _u) in up) for f in ("frames", "crop", "notes")) else None
        if common:
            up = {(f, u): v for (f, u), v in up.items() if u in common}
        lines += ["", f"Judge: {reused} unchanged gated chunks reuse the ungated verdict; check on {nver} unchanged "
                      f"chunks judged twice: identical verdict in {agree}/{nver}" +
                      (f"; uploads judged in every form: {len(common)}" if common else "")]
        forms_ = ("frames", "frames_gated", "crop", "crop_gated", "notes")
        lines += ["", f"## Data disclosure: share of uploads showing each item (judge {args.judge_model}; {bad} unparsed "
                      "chunks left out; CI over homes)",
                  "| item | " + " | ".join(forms_) + " |", "|---|" + "---|" * len(forms_)]
        for k in list(JUDGE_ITEMS) + ["any"]:
            row = []
            for f in forms_:
                keys = [u for (ff, u) in up if ff == f]
                vals = [(any(up[(f, u)].values()) if k == "any" else up[(f, u)][k]) for u in keys]
                row.append(ci_homes(vals, [home_of[u] for u in keys]) if keys else "-")
            lines.append(f"| {k} | " + " | ".join(row) + " |")
    # profile
    pr = list(P.read_jsonl(opath(args, f"profile_{args.judge_model}.jsonl")).values())
    if pr:
        ref = reference_profiles(pr)
        lines += ["", "## Inference about the household (share of the reference profile's specific values that the "
                      "provider infers correctly from m uploads; CI over homes)",
                  "Reference (64 raw frames per home; to be checked by a person): "
                  + "; ".join(f"{h}: " + ", ".join(f"{k}={v}" for k, v in ref[h].items() if specific(k, v)) for h in sorted(ref)),
                  "| form | m=1 | m=4 | m=16 |", "|---|---|---|---|"]
        for f in PROFILE_FORMS:
            row = []
            for m in (1, 4, 16):
                vals, hs = [], []
                for x in pr:
                    if x["kind"] != f or x["m"] != m or x["home"] not in ref:
                        continue
                    for k, rv in ref[x["home"]].items():
                        if specific(k, rv):
                            vals.append(bool(x["profile"]) and x["profile"][k] == rv)
                            hs.append(x["home"])
                row.append(ci_homes(vals, hs) if vals else "-")
            lines.append(f"| {f} | " + " | ".join(row) + " |")
        with open(opath(args, "profile_check.csv"), "w") as fc:
            fc.write("home,key,value,evidence,frames,correct (fill in: yes/no)\n")
            for x in pr:
                if x["kind"] == "reference" and x["profile"]:
                    for k, val in x["profile"].items():
                        if specific(k, val):
                            ev = str((x.get("evidence") or {}).get(k, "")).replace('"', "'")
                            fc.write(f'{x["home"]},{k},{val},"{ev}","{" ".join(x["files"])}",\n')
    text = "\n".join(lines)
    open(opath(args, "report.md"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    cmds = ["select", "ground", "notes", "gate", "make", "embed", "link", "vlmlink", "judge", "profile", "report"]
    ap.add_argument("cmd", choices=cmds)
    ap.add_argument("--data", required=True)
    ap.add_argument("--homes", nargs="+", default=[f"P0{i}" for i in range(1, 10)])
    ap.add_argument("--per_home", type=int, default=80)
    ap.add_argument("--nframes", type=int, default=32)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--pilot", action="store_true")         # gate: frames of the P01 pilot questions instead
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--gate_model", default="Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--ground_model", default="Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--notes_model", default="Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--link_model", default="Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--judge_model", default="Qwen2.5-VL-32B-Instruct")
    ap.add_argument("--text_model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--groups", type=int, default=20)       # trained linker: random groups of K candidate homes
    ap.add_argument("--vlm_episodes", type=int, default=200)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--profile_forms", nargs="+", default=list(PROFILE_FORMS), choices=PROFILE_FORMS)  # profile: forms
    ap.add_argument("--judge_uploads", type=int, default=0)        # judge: crop/notes for a sample of N uploads (0: all)
    ap.add_argument("--gated_changed_only", action="store_true")   # judge: gated chunks only where the gate changed images
    ap.add_argument("--verify_unchanged", type=int, default=50)    # ... plus this many unchanged chunks as a check
    a = ap.parse_args()
    a.dense = False  # all homes: 1-per-second frames (only P01 has dense frames)
    {c: globals()[c] for c in cmds}[a.cmd](a)
