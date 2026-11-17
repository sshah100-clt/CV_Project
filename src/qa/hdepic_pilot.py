"""HD-EPIC pilot (P01 dev split): what each way of asking the cloud costs and gives.

Ways to escalate (what leaves the home):
  frames  the question's frames, faces blurred                       -> hdepic.py qa (7B / 32B, variant blur)
  notes   timed observations written by the device 3B from raw frames,
          answer options hidden; text only                           -> `notes`, then `qa_text`
  crop    each frame cropped to one region the device 3B picks,
          cut from the full-resolution video, faces blurred          -> `ground`, `crops`, then `qa_crop`
  full    whole frames at FULL_SIDE px (resolution control)          -> `crops`, then `qa_crop --crop_mode full`
  blind   question and options only, no images                       -> `qa_blind`
  local   nothing leaves (3B on raw frames)                          -> hdepic.py qa (3B, variant raw)

`describe` + `linkage`: cross-day household linking (K homes, 95% CI over query homes) for full frames, centre crops,
device-written scene text and the question text alone (every escalation sends the question).
`plan`: can the 3B choose WHEN to look? Time windows scored against random windows of the same lengths.

All model steps accept --mock (random outputs, no model) to test the pipeline without a GPU.

    S="python -m src.qa.hdepic_pilot --data data --homes P01 --split dev --per_type 30 --dense"
    $S notes  --model Qwen/Qwen2.5-VL-3B-Instruct ; $S qa_text --model Qwen/Qwen2.5-VL-7B-Instruct
    $S ground --model Qwen/Qwen2.5-VL-3B-Instruct ; $S crops ; $S qa_crop --model Qwen/Qwen2.5-VL-7B-Instruct
    $S qa_blind --model Qwen/Qwen2.5-VL-7B-Instruct
    $S plan   --model Qwen/Qwen2.5-VL-3B-Instruct
    python -m src.qa.hdepic_pilot --data data --homes P01 P02 P03 P04 P05 P06 P07 P08 P09 describe --model Qwen/Qwen2.5-VL-3B-Instruct
    python -m src.qa.hdepic_pilot --data data --homes P01 P02 P03 P04 P05 P06 P07 P08 P09 linkage
"""
import os
import re
import sys
import json
import glob
import fcntl
import random
import argparse

import numpy as np

from src.data import hdepic as H

CENTER = (90, 90, 358, 358)  # centre crop in 448-px frame coordinates (middle 60%)
FULL_SIDE = 756  # high-resolution full-frame control: about the pixel density of the centre crops (448/268 = 1.67x;
                 # 756/448 = 1.69x), a multiple of Qwen's 28-px patch grid
STOP = set("a an the of to in on at and or is are was were be been it its this that these those with for from by as "
           "i my me you your he she they we our their there here what which who when where how did do does done "
           "video image frame frames time person participant one two three".split())


# --- helpers

def res_path(args, name):
    os.makedirs(os.path.join(H.root(args), "results"), exist_ok=True)
    return os.path.join(H.root(args), "results", name)


def open_out(path):
    """Append-only output with a lock (one writer) and resume by id."""
    lock = open(path + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"{path} is already being written by another process")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                done.add(json.loads(l)["id"])
            except (ValueError, KeyError):
                pass
    return open(path, "a"), done, lock


def read_jsonl(path):
    rows = {}
    if os.path.exists(path):
        for l in open(path):
            x = json.loads(l)
            rows[x["id"]] = x
    return rows


class VLM:
    """Qwen2.5-VL with the same loading and letter read-out as hdepic.qa / star_qa.py; --mock gives random outputs."""

    def __init__(self, args):
        self.mock = args.mock
        self.rng = np.random.default_rng(0)
        if self.mock:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa").eval()
        self.proc = AutoProcessor.from_pretrained(args.model, min_pixels=64 * 28 * 28, max_pixels=args.max_pixels)
        self.ids = [self.proc.tokenizer.encode(l, add_special_tokens=False)[0] for l in H.LETTERS]

    def _inputs(self, content, files):
        from PIL import Image
        ims = [Image.open(f).convert("RGB") for f in files]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        return self.proc(text=[text], images=ims or None, return_tensors="pt").to(self.model.device)

    def letters(self, content, files):
        """(probabilities of A-E, letter mass, input tokens)"""
        if self.mock:
            p = self.rng.dirichlet(np.ones(5))
            return p, 1.0, 100 + 256 * len(files)
        x = self._inputs(content, files)
        with self.torch.no_grad():
            full = self.torch.softmax(self.model(**x).logits[0, -1].float(), 0)
        p = (full[self.ids] / full[self.ids].sum()).cpu().numpy()
        return p, float(full[self.ids].sum()), int(x["input_ids"].shape[1])

    def generate(self, content, files, max_new=200):
        if self.mock:
            if '"windows"' in content[-1].get("text", ""):  # plan prompt: random windows in the frames' time format
                t = sorted(self.rng.integers(0, 1500, 4).tolist())
                return json.dumps({"windows": [[H.hms(t[0]), H.hms(t[1])], [H.hms(t[2]), H.hms(t[3])]], "looking_for": "mock"})
            return "[" + ", ".join(str(int(v)) for v in sorted(self.rng.integers(0, 448, 4))) + "] mock text kettle sink"
        x = self._inputs(content, files)
        with self.torch.no_grad():
            y = self.model.generate(**x, max_new_tokens=max_new, do_sample=False)
        return self.proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def tag(args):
    return f"f{args.nframes}_pt{args.per_type}{'_dense' if args.dense else ''}_{args.split}"


def mname(args):
    return "mock" if args.mock else os.path.basename(args.model)


def question_text(q, content):
    """Question + options exactly as the frame prompt shows them (BBOX already rescaled by hdepic.build)."""
    last = content[-1]["text"]
    return last[last.index("Question:"):last.rindex("\n")]


def frame_time(path):
    """Video time (s) of a saved frame: NNNNNN.jpg is second N (frame at N + 0.5 s), dNNNNNNNNN.jpg is ms."""
    b = os.path.basename(path)[:-4]
    return int(b[1:]) / 1000 if b.startswith("d") else int(b) + 0.5


# --- text escalation

def strip_options(q, content, ask):
    """The QA prompt of q (same frames, time labels and question) with the answer options removed and `ask` in place
    of the answer instruction, so the edge model can neither name nor copy an option."""
    last = content[-1]["text"]
    tail = "\n" + "\n".join(f"{l}. {c}" for l, c in zip(H.LETTERS, q["choices"])) + "\n" + H.ASK
    if not last.endswith(tail):
        raise ValueError(f"{q['id']}: prompt does not end with the options")
    return content[:-1] + [{"type": "text", "text": last[: -len(tail)] + "\n" + ask}]


# The options are hidden so the notes cannot name or copy an answer; the line format has no example content, so the
# model cannot copy an example into its notes.
NOTES_ASK = ("Do not answer the question. Another assistant will answer it from your notes alone, without seeing "
             "the frames. Write what you see in these frames that is relevant to the question, as short lines in "
             "the form HH:MM:SS <what is visible>, using the time labels of the frames. Describe only what is "
             "visible and do not repeat a line. At most 120 words.")
LEAK_WORDS = re.compile(r"\b(?:option|choice|answer)s?\b", re.I)
LEAK_LETTER = re.compile(r"(?:^|\s)\(?[A-E][.):](?:\s|$)", re.M)


def notes_one(vlm, q, r, args, ask):
    """Timed observations by the device 3B from RAW frames, options hidden (also used by upload_privacy.py)."""
    content, files = H.build(q, r, "raw", args.nframes, dense=args.dense)
    text = vlm.generate(strip_options(q, content, ask), files, 200)
    return {"evidence": text, "n_frames_seen": len(files), "words": len(text.split()),
            "mentions_answer_or_letter": bool(LEAK_WORDS.search(text) or LEAK_LETTER.search(text))}


def notes(args):
    """Device 3B writes timed observations from RAW frames, without seeing the answer options (only text leaves)."""
    r, vlm = H.root(args), VLM(args)
    fo, done, _l = open_out(res_path(args, f"notes_{mname(args)}_{tag(args)}.jsonl"))
    for q in H.select(args, r):
        if q["id"] in done:
            continue
        fo.write(json.dumps({"id": q["id"], **notes_one(vlm, q, r, args, NOTES_ASK)}) + "\n")
        fo.flush()


LABELS_HEAD = ("An object detector on a home camera listed the objects it found in frames of the video. Each line is "
               "one frame: [input, time] then objects with their boxes as [x1 y1 x2 y2] pixel coordinates in that "
               "frame (the same coordinates as <BBOX> in the question):\n")


def text_inputs(args, r, q, ev):
    """Text-only prompt: the device's notes (or detector labels) + question + options (shared by qa_text and gpt_qa.py)."""
    content, _ = H.build(q, r, "blur", args.nframes, dense=args.dense)
    if getattr(args, "evidence_kind", "") == "labels":
        return [{"type": "text", "text": LABELS_HEAD + f"{ev[q['id']]['evidence']}\n\n{question_text(q, content)}\n{H.ASK}"}]
    return [{"type": "text", "text": "A home camera assistant watched the video and wrote these notes (times are video "
                                     f"times):\n{ev[q['id']]['evidence']}\n\n{question_text(q, content)}\n{H.ASK}"}]


def qa_text(args):
    """Cloud model answers from the question and the device's notes (or detector labels) only, no pixels."""
    r, vlm = H.root(args), VLM(args)
    ev = read_jsonl(res_path(args, f"{args.evidence_kind}_{args.evidence_model}_{tag(args)}.jsonl"))
    if not ev:
        sys.exit(f"no {args.evidence_kind}_{args.evidence_model}_{tag(args)}.jsonl: run `{args.evidence_kind}` first")
    form = args.evidence_kind
    fo, done, _l = open_out(res_path(args, f"qa_{mname(args)}_{form}_{tag(args)}.jsonl"))
    for q in H.select(args, r):
        if q["id"] in done or q["id"] not in ev:
            continue
        p, mass, ntok = vlm.letters(text_inputs(args, r, q, ev), [])
        fo.write(json.dumps({"id": q["id"], "prototype": q["prototype"], "video": H.first_video(q), "form": form,
                             "pred": int(p.argmax()), "gold": q["correct_idx"], "probs": p.round(4).tolist(),
                             "letter_mass": round(mass, 4), "n_frames": 0, "n_tokens": ntok,
                             "text_words": len(ev[q["id"]]["evidence"].split())}) + "\n")
        fo.flush()


# --- blind baseline (no video)

BLIND_HEAD = "No video or images are provided. Answer from the question and the options alone.\n"


def blind_inputs(args, r, q):
    """Question + options only (shared by qa_blind and gpt_qa.py --form blind)."""
    content, _ = H.build(q, r, "blur", args.nframes, dense=args.dense)
    return [{"type": "text", "text": BLIND_HEAD + question_text(q, content) + "\n" + H.ASK}]


def qa_blind(args):
    """How much is answerable from the question text alone (language prior; standard blind baseline)."""
    r, vlm = H.root(args), VLM(args)
    fo, done, _l = open_out(res_path(args, f"qa_{mname(args)}_blind_{tag(args)}.jsonl"))
    for q in H.select(args, r):
        if q["id"] in done:
            continue
        p, mass, ntok = vlm.letters(blind_inputs(args, r, q), [])
        fo.write(json.dumps({"id": q["id"], "prototype": q["prototype"], "video": H.first_video(q), "form": "blind",
                             "pred": int(p.argmax()), "gold": q["correct_idx"], "probs": p.round(4).tolist(),
                             "letter_mass": round(mass, 4), "n_frames": 0, "n_tokens": ntok}) + "\n")
        fo.flush()


# --- crop escalation

GROUND_ASK = ("Do not answer. Which image region is needed to answer this question? Give one box [x1, y1, x2, y2] "
              "in pixel coordinates of these 448 x 448 frames that covers the hands, objects and places involved. "
              "Reply with the box only.")


def parse_box(text, min_side=0.4, pad=0.15, size=448):
    """First 4 numbers -> box, padded, at least min_side of the frame, clipped; None if unusable."""
    v = [float(x) for x in re.findall(r"-?\d+\.?\d*", text)[:4]]
    if len(v) < 4:
        return None
    x1, y1, x2, y2 = min(v[0], v[2]), min(v[1], v[3]), max(v[0], v[2]), max(v[1], v[3])
    if x2 - x1 < 4 or y2 - y1 < 4 or x2 > 2 * size or y2 > 2 * size:
        return None
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = max((x2 - x1) * (1 + 2 * pad), min_side * size)
    h = max((y2 - y1) * (1 + 2 * pad), min_side * size)
    w, h = min(w, size), min(h, size)
    x1, y1 = min(max(0, cx - w / 2), size - w), min(max(0, cy - h / 2), size - h)
    return [round(x1), round(y1), round(x1 + w), round(y1 + h)]


def ground_reply(vlm, q, r, args):
    """The device 3B's raw reply naming one region for question q (4 of its raw frames + the question).
    Shared by `ground` and upload_privacy.py."""
    content, files = H.build(q, r, "raw", args.nframes, dense=args.dense, ask=GROUND_ASK)
    keep = set(np.linspace(0, len(files) - 1, min(4, len(files))).round().astype(int).tolist())
    c2, f2, i = [], [], 0  # keep 4 frames (with their time labels) + the question text
    for item in content[:-1]:
        if item.get("type") == "image":
            if i in keep:
                c2 += [last_label, item]
                f2.append(files[i])
            i += 1
        elif item["text"].startswith("["):
            last_label = item
        else:
            c2.append(item)
    return vlm.generate(c2 + [content[-1]], f2, 40)


def ground(args):
    """Device 3B picks ONE region per question from 4 of its raw frames; the same box is used for all its frames."""
    r, vlm = H.root(args), VLM(args)
    fo, done, _l = open_out(res_path(args, f"ground_{mname(args)}_{tag(args)}.jsonl"))
    for q in H.select(args, r):
        if q["id"] in done:
            continue
        reply = ground_reply(vlm, q, r, args)
        box = parse_box(reply)
        fo.write(json.dumps({"id": q["id"], "reply": reply, "box": box or list(CENTER), "parsed": box is not None}) + "\n")
        fo.flush()


def crops(args):
    """Cut each question's frames from the FULL-RESOLUTION video at the chosen box (guided) and at the centre box,
    resize to 448 px short side, blur faces; also the whole frame at FULL_SIDE px (high-resolution control, mode
    `full`). One sequential pass per video."""
    import cv2
    r, det = H.root(args), H.face_detector(args)
    boxes = read_jsonl(res_path(args, f"ground_{args.ground_model}_{tag(args)}.jsonl"))
    jobs = {}  # video -> frame index -> [(output path, box)]
    for q in H.select(args, r):
        if q["id"] not in boxes:
            continue
        _, files = H.build(q, r, "blur", args.nframes, dense=args.dense)
        for mode, box, side in (("guided", boxes[q["id"]]["box"], 448), ("center", list(CENTER), 448), ("full", None, FULL_SIDE)):
            for j, f in enumerate(files):
                out = os.path.join(r, "crops", mode, q["id"], f"{j:03d}.jpg")
                if not os.path.exists(out):
                    v = os.path.basename(os.path.dirname(f))
                    jobs.setdefault(v, {}).setdefault(frame_time(f), []).append((out, box, side))
    made = missing = 0
    for v, by_t in sorted(jobs.items()):
        path = os.path.join(r, "videos", H.home_of(v), v + ".mp4")
        if not os.path.exists(path):
            missing += sum(len(x) for x in by_t.values())
            continue
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        idx = {}
        for t, outs in by_t.items():
            idx.setdefault(int(round(t * fps)), []).extend(outs)
        i, last = 0, max(idx)
        while i <= last and cap.grab():
            if i in idx:
                ok, img = cap.retrieve()
                for out, box, side in idx.pop(i):
                    if not ok:
                        missing += 1
                        continue
                    if box is None:  # whole frame
                        crop = img
                    else:
                        s = min(img.shape[:2]) / 448  # box is in 448-px frame coordinates
                        x1, y1, x2, y2 = [int(round(b * s)) for b in box]
                        crop = img[y1:y2, x1:x2]
                    det.setInputSize((crop.shape[1], crop.shape[0]))
                    faces = det.detect(crop)[1]
                    k = side / min(crop.shape[:2])
                    small = cv2.resize(crop, (round(crop.shape[1] * k), round(crop.shape[0] * k)), interpolation=cv2.INTER_AREA)
                    small, _ = H.blur_boxes(small, [] if faces is None else faces[:, :4] * k, args.face_max)
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    cv2.imwrite(out, small, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    made += 1
            i += 1
        missing += sum(len(x) for x in idx.values())
    print(f"crops: {made} written, {missing} missing (video not kept or unreadable)", flush=True)


def crop_inputs(args, r, q, boxes, mode):
    """(content, crop files, box) of the crop prompt for question q, or None if a crop is missing.
    Shared by qa_crop (Qwen) and gpt_qa.py so every model sees the same crop prompt."""
    content, files = H.build(q, r, "blur", args.nframes, dense=args.dense)
    cfiles = [os.path.join(r, "crops", mode, q["id"], f"{j:03d}.jpg") for j in range(len(files))]
    if not all(os.path.exists(f) for f in cfiles):
        return None
    if mode == "full":  # whole frame at FULL_SIDE px: same wording, BBOX scaled from 448-px to FULL_SIDE-px coordinates
        k = FULL_SIDE / 448
        last = re.sub(r"<BBOX ([\d. ]+)>", lambda m: "<BBOX " + " ".join(
            f"{float(v) * k:.0f}" for v in m.group(1).split()) + ">", content[-1]["text"])
        return content[:-1] + [{"type": "text", "text": last}], cfiles, (0, 0, 448, 448)
    x1, y1, x2, y2 = boxes[q["id"]]["box"] if mode == "guided" else CENTER
    k = 448 / min(x2 - x1, y2 - y1)
    last = content[-1]["text"].replace(
        "<BBOX x1 y1 x2 y2> gives pixel coordinates in the frame at that time.",
        "Each image is a close-up crop of the frame. <BBOX x1 y1 x2 y2> gives pixel coordinates in the crop.")
    last = re.sub(r"<BBOX ([\d. ]+)>", lambda m: "<BBOX " + " ".join(
        f"{(float(v) - (x1 if i % 2 == 0 else y1)) * k:.0f}" for i, v in enumerate(m.group(1).split())) + ">", last)
    return content[:-1] + [{"type": "text", "text": last}], cfiles, (x1, y1, x2, y2)


def qa_crop(args):
    """Cloud model answers from the cropped frames (same time labels; BBOX moved into crop coordinates).
    --crop_mode full: the whole frames at FULL_SIDE px (high-resolution control; needs about 3x the image tokens)."""
    if args.crop_mode == "full" and args.max_pixels < FULL_SIDE ** 2:
        print(f"crop_mode full: max_pixels raised from {args.max_pixels} to {FULL_SIDE}^2 so the frames are not "
              "downscaled", flush=True)
        args.max_pixels = FULL_SIDE ** 2
    r, vlm = H.root(args), VLM(args)
    boxes = read_jsonl(res_path(args, f"ground_{args.ground_model}_{tag(args)}.jsonl"))
    fo, done, _l = open_out(res_path(args, f"qa_{mname(args)}_crop{args.crop_mode}_{tag(args)}.jsonl"))
    skipped = 0
    for q in H.select(args, r):
        if q["id"] in done or q["id"] not in boxes:
            continue
        got = crop_inputs(args, r, q, boxes, args.crop_mode)
        if got is None:
            skipped += 1
            continue
        content, cfiles, (x1, y1, x2, y2) = got
        p, mass, ntok = vlm.letters(content, cfiles)
        fo.write(json.dumps({"id": q["id"], "prototype": q["prototype"], "video": H.first_video(q),
                             "form": f"crop{args.crop_mode}", "pred": int(p.argmax()), "gold": q["correct_idx"],
                             "probs": p.round(4).tolist(), "letter_mass": round(mass, 4), "n_frames": len(cfiles),
                             "n_tokens": ntok, "crop_area": round((x2 - x1) * (y2 - y1) / 448 ** 2, 3),
                             "max_pixels": args.max_pixels}) + "\n")
        fo.flush()
    if skipped:
        print(f"{skipped} questions skipped: crops missing (run `crops` first)", flush=True)


# --- household linking per form

DESCRIBE_ASK = "Describe these frames from a kitchen video in at most 60 words: the actions and the objects you see."


def describe(args):
    """Device-written scene text for EVERY home (same prompt everywhere), to measure how linkable text uploads are:
    n_clips per home, spread over its days, 4 raw frames 2 s apart."""
    r, vlm = H.root(args), VLM(args)
    rng = random.Random(0)
    fo, done, _l = open_out(res_path(args, f"describe_{mname(args)}.jsonl"))
    for h in args.homes:
        vids = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(r, "frames", "raw", f"{h}-*", "done.json")))
        clips = [(v, t) for v in vids for t in range(0, max(0, H.have(r, v) - 8), 8)]
        rng.shuffle(clips)
        by_day = {}
        for v, t in clips:  # round-robin over days so every day is represented
            by_day.setdefault(H.day_of(v), []).append((v, t))
        pick, days = [], sorted(by_day)
        while len(pick) < args.n_clips and any(by_day.values()):
            for d in days:
                if by_day[d] and len(pick) < args.n_clips:
                    pick.append(by_day[d].pop())
        for v, t in pick:
            cid = f"{v}_{t}"
            if cid in done:
                continue
            files = [os.path.join(r, "frames", "raw", v, f"{t + 2 * j:06d}.jpg") for j in range(4)]
            content = [{"type": "image"} for _ in files] + [{"type": "text", "text": DESCRIBE_ASK}]
            fo.write(json.dumps({"id": cid, "home": h, "day": H.day_of(v), "text": vlm.generate(content, files, 100)}) + "\n")
            fo.flush()


def tfidf(texts):
    """L2-normalised TF-IDF rows (dense; vocabulary = words in >= 2 texts)."""
    toks = [[w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP and len(w) > 2] for t in texts]
    df = {}
    for ts in toks:
        for w in set(ts):
            df[w] = df.get(w, 0) + 1
    vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= 2))}
    X = np.zeros((len(texts), max(1, len(vocab))), np.float32)
    for i, ts in enumerate(toks):
        for w in ts:
            if w in vocab:
                X[i, vocab[w]] += 1
    X *= np.log(len(texts) / np.array([df[w] for w in sorted(vocab)] or [1], np.float32))[None]
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def center_embed(args):
    """SigLIP embeddings of centre crops (448-px blurred frames, same frames/step as hdepic.embed)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    r = H.root(args)
    out = os.path.join(r, f"emb_siglip_blur_center_s{args.step}.npz")
    vids = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(r, "frames", "raw", "*", "done.json")))
    if os.path.exists(out):
        z = np.load(out)
        if sorted(set(z["vid"])) == vids:
            return z["vid"], z["e"]
    keys = [(v, t) for v in vids for t in range(0, H.have(r, v), args.step)]
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(H.BACKBONES["siglip"])
    model = AutoModel.from_pretrained(H.BACKBONES["siglip"]).to(dev).eval()
    E = []
    for i in range(0, len(keys), 64):
        ims = [Image.open(os.path.join(r, "frames", "blur", v, f"{t:06d}.jpg")).convert("RGB").crop(CENTER) for v, t in keys[i:i + 64]]
        with torch.no_grad():
            e = model.get_image_features(pixel_values=proc(images=ims, return_tensors="pt")["pixel_values"].to(dev))
        E.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    vid, e = np.array([k[0] for k in keys]), np.concatenate(E)
    np.savez(out, vid=vid, e=e)
    return vid, e


def linkage(args):
    """Cross-day K-way linkage for each form of upload, all homes, CI over query homes -> results/linkage_forms.json"""
    r = H.root(args)
    ms = (1, 2, 4, 8, 16, 32)
    out = {}
    args.backbone, args.variant = "siglip", "blur"
    vid, _, E = H.embed(args)
    out["frames"] = H.link_eval(E, np.array([H.home_of(v) for v in vid]), np.array([H.day_of(v) for v in vid]),
                                args.K, args.episodes, ms, "full frames (SigLIP)")
    vid, E = center_embed(args)
    out["center_crop"] = H.link_eval(E, np.array([H.home_of(v) for v in vid]), np.array([H.day_of(v) for v in vid]),
                                     args.K, args.episodes, ms, "centre crops (SigLIP)")
    d = list(read_jsonl(res_path(args, f"describe_{args.describe_model}.jsonl")).values())
    if d:
        out["edge_text"] = H.link_eval(tfidf([x["text"] for x in d]), np.array([x["home"] for x in d]),
                                       np.array([x["day"] for x in d]), args.K, args.episodes, ms, "edge-written text (TF-IDF)")
    else:
        print("no describe_*.jsonl: run `describe` for text linkage", flush=True)
    # question text alone: every escalation sends it. All homes' questions are public, no video needed.
    rng = random.Random(0)
    qs = [q for q in H.questions(r) if len({H.home_of(i["id"]) for i in q["inputs"].values()}) == 1]
    by = {}
    for q in qs:
        by.setdefault(H.home_of(H.first_video(q)), []).append(q)
    pick = [q for h in sorted(by) for q in rng.sample(by[h], min(400, len(by[h])))]
    out["question_text"] = H.link_eval(
        tfidf([q["question"] + " " + " ".join(map(str, q["choices"])) for q in pick]),
        np.array([H.home_of(H.first_video(q)) for q in pick]),
        np.array([max(H.day_of(i["id"]) for i in q["inputs"].values()) for q in pick]),
        args.K, args.episodes, ms, "question text (TF-IDF)")
    json.dump(out, open(res_path(args, "linkage_forms.json"), "w"), indent=1)


# --- agent planner feasibility

# Options are hidden (their times would give the answer away); the JSON format has no example windows to copy.
PLAN_ASK = ("Do not answer the question. These {n} frames are spread evenly over {span}. You can look again in more "
            "detail at up to {w} time windows of at most {s} seconds each. Choose the windows most likely to show "
            "what the question asks about, using the time labels of the frames. Reply with JSON only, in this form: "
            '{{"windows": [["HH:MM:SS", "HH:MM:SS"], ...], "looking_for": "<what to look for>"}}')


def parse_time(x):
    """Seconds from 'HH:MM:SS(.f)', 'MM:SS' or a number; None if unreadable."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.fullmatch(r"\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\s*", str(x))
    if m:
        return int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    try:
        return float(x)
    except ValueError:
        return None


def parse_windows(reply, a, b, w, s):
    """Windows [[t0, t1], ...] in seconds, at most w, each at most s long (centred cut), at least 1 s, clipped to
    [a, b]; None if the reply has no JSON object with a "windows" list."""
    m = re.search(r"\{.*\}", reply, re.S)
    pl = None
    for txt in ([m.group(0), m.group(0).replace("'", '"')] if m else []):
        try:
            pl = json.loads(txt)
            break
        except ValueError:
            continue
    if isinstance(pl, list) and len(pl) == 1 and isinstance(pl[0], dict):
        pl = pl[0]  # the object wrapped in a list
    if not isinstance(pl, dict) or not isinstance(pl.get("windows"), list):
        # not valid JSON (e.g. times without quotes, or cut off at the token limit): read the [time, time] pairs
        T_ = r'"?(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)"?'
        pairs = re.findall(r"\[\s*" + T_ + r"\s*,\s*" + T_ + r"\s*\]", reply)
        if '"windows"' not in reply or not pairs:
            return None
        pl = {"windows": [list(x) for x in pairs]}
    out = []
    for x in pl["windows"][:w]:
        if not (isinstance(x, (list, tuple)) and len(x) == 2):
            continue
        t0, t1 = parse_time(x[0]), parse_time(x[1])
        if t0 is None or t1 is None:
            continue
        t0, t1 = min(t0, t1), max(t0, t1)
        c, L = (t0 + t1) / 2, min(max(t1 - t0, 1.0), s)
        t0, t1 = max(a, c - L / 2), min(b, c + L / 2)
        if t1 > t0:
            out.append([round(t0, 2), round(t1, 2)])
    return out


def gold_spans(q):
    """Time spans (s) of the correct choice: 'X to Y' ranges, else single times (zero-length spans)."""
    c = str(q["choices"][q["correct_idx"]])
    spans = [(H.secs(x), H.secs(y)) for x, kx, y, ky in re.findall(H.TIME + r" to " + H.TIME, c)]
    return spans or [(H.secs(x), H.secs(x)) for x, _ in re.findall(H.TIME, c)]


def hits(wins, spans):
    return any(w0 <= g1 and g0 <= w1 for w0, w1 in wins for g0, g1 in spans)


def plan(args):
    """Can the 3B choose WHEN to look? Questions whose answer is a time (single-video, dev split), 16 evenly
    spaced raw overview frames, options hidden. Scored: do its windows contain the correct time, against the same
    window lengths placed at random in the same span (chance)."""
    import copy
    r, vlm = H.root(args), VLM(args)
    a2 = copy.copy(args)
    a2.per_type, a2.limit = args.plan_per_type, None
    qs = [q for q in H.select(a2, r) if gold_spans(q)  # every type whose answer is a time (the *_localization types)
          and sum(k.startswith("video") for k in q["inputs"]) == 1 and not any(k.startswith("image") for k in q["inputs"])]
    path = res_path(args, f"plan_{mname(args)}_pt{args.plan_per_type}_{args.split}.jsonl")
    fo, done, _l = open_out(path)
    rng = np.random.default_rng(0)
    for q in qs:
        if q["id"] in done:
            continue
        k, inp = next((k, i) for k, i in q["inputs"].items() if k.startswith("video"))
        n = H.have(r, inp["id"])
        a = H.secs(inp["start_time"]) if "start_time" in inp else 0.0
        b = H.secs(inp["end_time"]) if "end_time" in inp else float(n)
        span = (f"{k} from {H.hms(a)} to {H.hms(b)}" if "start_time" in inp
                else f"the whole of {k}, from {H.hms(0)} to {H.hms(b)}")
        content, files = H.build(dict(q, choices=[]), r, "raw", 16)  # no options: their times would give the answer
        ask = PLAN_ASK.format(n=len(files), span=span, w=args.plan_windows, s=args.plan_window_s)
        last = content[-1]["text"]
        assert last.endswith("\n\n" + H.ASK), q["id"]
        content = content[:-1] + [{"type": "text", "text": last[: -len(H.ASK) - 1] + ask}]
        reply = vlm.generate(content, files, 200)
        wins = parse_windows(reply, a, b, args.plan_windows, args.plan_window_s)
        gold = gold_spans(q)
        p_rand = 0.0
        if wins:  # same window lengths, placed uniformly at random inside [a, b]
            draws = [[(t0, t0 + L) for L in [w1 - w0 for w0, w1 in wins] for t0 in [rng.uniform(a, max(a, b - L))]]
                     for _ in range(2000)]
            p_rand = float(np.mean([hits(d, gold) for d in draws]))
        fo.write(json.dumps({"id": q["id"], "prototype": q["prototype"], "video": inp["id"], "span": [a, b],
                             "n_frames": len(files), "reply": reply, "valid": wins is not None,
                             "windows": wins or [], "gold": gold, "hit": bool(wins) and hits(wins, gold),
                             "n_windows": len(wins or []), "p_random": round(p_rand, 4),
                             "looked_s": round(sum(w1 - w0 for w0, w1 in wins or []), 1)}) + "\n")
        fo.flush()
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        return
    print(f"\nplanner: {len(rows)} questions; valid JSON {sum(x['valid'] for x in rows)}; with at least one usable "
          f"window inside the span {sum(x['n_windows'] > 0 for x in rows)}; "
          f"mean seconds looked at {np.mean([x['looked_s'] for x in rows]):.0f} "
          f"({100 * np.mean([x['looked_s'] / max(1e-9, x['span'][1] - x['span'][0]) for x in rows]):.1f}% of the span)")
    d = [float(x["hit"]) - x["p_random"] for x in rows]
    m, lo, hi = H.cluster_ci(d, [x["video"] for x in rows])
    print(f"  windows contain the correct time: {100 * np.mean([x['hit'] for x in rows]):.1f}%  vs random windows "
          f"of the same lengths {100 * np.mean([x['p_random'] for x in rows]):.1f}%  -> difference {100 * m:+.1f} "
          f"[{100 * lo:+.1f}, {100 * hi:+.1f}] (95% CI over videos)")
    for p in sorted({x["prototype"] for x in rows}):
        R = [x for x in rows if x["prototype"] == p]
        print(f"  {p:48s} n={len(R):3d}  hit {100 * np.mean([x['hit'] for x in R]):5.1f}  "
              f"random {100 * np.mean([x['p_random'] for x in R]):5.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["notes", "qa_text", "qa_blind", "ground", "crops", "qa_crop", "describe", "linkage",
                                   "plan"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--homes", nargs="+", default=["P01"])
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--per_type", type=int, default=30)
    ap.add_argument("--nframes", type=int, default=32)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variant", default="blur")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--mock", action="store_true")             # random model outputs: plumbing test only
    ap.add_argument("--evidence_model", default="Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--evidence_kind", default="notes", choices=["notes", "labels"])  # qa_text input (labels: --evidence_model owlv2)
    ap.add_argument("--ground_model", default="Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--describe_model", default="Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--crop_mode", default="guided", choices=["guided", "center", "full"])
    ap.add_argument("--face_score", type=float, default=0.8)
    ap.add_argument("--face_max", type=float, default=0.25)
    ap.add_argument("--n_clips", type=int, default=40)         # describe: clips per home
    ap.add_argument("--plan_per_type", type=int, default=300)   # plan: questions per localization type
    ap.add_argument("--plan_windows", type=int, default=3)      # plan: at most this many windows
    ap.add_argument("--plan_window_s", type=float, default=30)  # plan: each at most this many seconds
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=2000)
    names = ap._actions[1].choices  # the command may appear anywhere, even after --homes P01 P02 ...:
    argv = sys.argv[1:]             # move it to the front so --homes cannot swallow it
    argv = [x for x in argv if x in names][:1] + [x for i, x in enumerate(argv) if not (x in names and i == next(j for j, y in enumerate(argv) if y in names))]
    a = ap.parse_args(argv)
    globals()[a.cmd](a)
