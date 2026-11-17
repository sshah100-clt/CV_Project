"""Cloud QA through an OpenAI-compatible chat API (a GPT model), on the same questions, frames and prompt
text as the Qwen pilot runs, so every row is paired with the 3B / 7B / 32B rows of the same question.

Forms (what the cloud receives):
  frames       the face-blurred frames + prompt of `hdepic.py qa --variant blur`      -> qa_<model>_blur_<tag>.jsonl
  blind        question + options only, no images (prior / contamination check)     -> qa_<model>_blind_<tag>.jsonl
  crop_guided  the crops + prompt of `hdepic_pilot.py qa_crop --crop_mode guided`    -> qa_<model>_cropguided_<tag>.jsonl
  crop_center  the crops + prompt of `hdepic_pilot.py qa_crop --crop_mode center`    -> qa_<model>_cropcenter_<tag>.jsonl
  notes        the device 3B's notes (`hdepic_pilot.py notes`) + question, no images -> qa_<model>_notes_<tag>.jsonl
  objonly      objects-only images of the same frames (`object_labels.py objonly`) -> qa_<model>_objonly_<tag>.jsonl
  labels       detector labels (`object_labels.py make`; --evidence_model owlv2)  -> qa_<model>_labels_<tag>.jsonl
  box_size     box questions only: frames prompt + the frame size stated            -> qa_<model>_boxsize_<tag>.jsonl
  box_draw     box questions only: boxes drawn on their frames, named by colour     -> qa_<model>_boxdraw_<tag>.jsonl

Key: environment variable OPENAI_API_KEY (never on the command line). Endpoint: --base_url, else OPENAI_BASE_URL,
else https://api.openai.com/v1 (any server that implements /chat/completions).

Safety of the results:
  * A run's settings (model, form, image detail, decoding, selection, annotations) are stored in <out>.meta.json.
    Appending to an existing file with different settings is refused, so one file never mixes configurations.
  * Only successful calls become rows; failures go to <out>.errors.jsonl and are retried on the next run
    (content-policy blocks are not retried unless --retry_blocked). Rerunning resumes by question id.
  * An unparseable reply is kept as a row with pred = -1 (counts as wrong) and parsed = false.
  * --max_usd stops before the estimated spend passes the cap (needs --price_in/--price_out, $ per 1M tokens).

    S="python -m src.qa.gpt_qa --data data --homes P01 --split dev --per_type 30 --dense --model <model>"
    $S --form frames --dry                 # one request printed (images elided), no API call
    $S --form frames --limit 20            # cost and format check, then rerun without --limit to finish
    $S --form blind ; $S --form crop_center ; $S --form crop_guided
"""
import os
import re
import sys
import json
import time
import base64
import random
import hashlib
import glob
import argparse
import datetime
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data import hdepic as H
from src.qa import hdepic_pilot as P

SUFFIX = {"frames": "blur", "blind": "blind", "crop_guided": "cropguided", "crop_center": "cropcenter", "notes": "notes", "labels": "labels", "objonly": "objonly",
          "box_size": "boxsize", "box_draw": "boxdraw"}
# Box-format check (questions with a <BBOX> only): the frames form gives boxes as pixel coordinates of the 448-px
# frames without the image size (Qwen's native format). box_size adds the image size to the same sentence; box_draw
# draws each box on the frame at the time it refers to and names it by colour instead of coordinates.
BOX_FORMS = ("box_size", "box_draw")
FRAME_SENT = "<BBOX x1 y1 x2 y2> gives pixel coordinates in the frame at that time."
BOX_COLOURS = [("red", (255, 0, 0)), ("blue", (0, 90, 255)), ("green", (0, 200, 0)), ("yellow", (255, 220, 0))]
TRANSIENT = {408, 409, 429, 500, 502, 503, 504}
# System prompt of the official HD-EPIC eval (hd-epic/hd-epic-vqa-eval, configs/gemini/gemini_pro_final.json, "sys"),
# also hard-coded in lmms-eval's hd_epic task. Its last sentence (boxes as ymin,xmin,ymax,xmax on 1000x1000) is left
# out because our question text already defines boxes as x1 y1 x2 y2 pixels. The user message stays byte-identical
# to the Qwen runs. Without a system prompt GPT-4o explains before answering and many replies cannot be parsed.
SYSTEM = ("You are an expert video analyzer, and your job is to answer the multiple choice question by giving only "
          "the letter identifying the answer. Do not give any other information. For example, acceptable answers are "
          "'A' or 'B' or 'C' etc.. You must give an answer, even if you are not sure.")


# --- inputs (identical to the Qwen runs)

def inputs(args, r, q, boxes):
    """(content, image files) exactly as the Qwen run of this form saw them, or None (crop missing)."""
    if args.form == "blind":
        return P.blind_inputs(args, r, q), []
    if args.form in ("notes", "labels"):
        return (P.text_inputs(args, r, q, boxes), []) if q["id"] in boxes else None
    if args.form == "frames":
        return H.build(q, r, "blur", args.nframes, dense=args.dense)
    if args.form == "objonly":     # objects-only images (object_labels.py objonly), same frames and prompt as `frames`
        return H.build(q, r, "objonly", args.nframes, dense=args.dense)
    if args.form in BOX_FORMS:
        return box_inputs(args, r, q)
    if q["id"] not in boxes:
        return None
    got = P.crop_inputs(args, r, q, boxes, args.form.split("_")[1])
    return None if got is None else got[:2]


def box_inputs(args, r, q):
    """The frames prompt of q with the box format changed (see BOX_FORMS); None if a box's frame is not shown."""
    from PIL import Image, ImageDraw
    content, files = H.build(q, r, "blur", args.nframes, dense=args.dense)
    last = content[-1]["text"]
    if FRAME_SENT not in last or "<BBOX " not in last.replace(FRAME_SENT, ""):
        return None
    if args.form == "box_size":
        w, h = Image.open(files[0]).size
        return content[:-1] + [{"type": "text", "text": last.replace(
            FRAME_SENT, FRAME_SENT[:-1] + f"; every frame is {w} x {h} pixels, x from the left edge, y from the top edge.")}], files
    # box_draw: pair each box with the first <TIME> after it (else the last one before it), draw it on that frame
    head, qpart = last.split(FRAME_SENT, 1)
    boxes = list(re.finditer(r"<BBOX ([\d. ]+)>", qpart))
    times = list(re.finditer(H.TIME, qpart))
    files, drawn = list(files), {}
    for j, m in enumerate(boxes):
        after = [t for t in times if t.start() > m.end()]
        before = [t for t in times if t.end() < m.start()]
        t = after[0] if after else (before[-1] if before else None)
        if t is None or t.group(2) not in q["inputs"]:
            return None
        vid = q["inputs"][t.group(2)]["id"]
        sec = min(int(H.secs(t.group(1))), H.have(r, vid) - 1)
        f = os.path.join(r, "frames", "blur", vid, f"{sec:06d}.jpg")
        if f not in files:
            return None
        drawn.setdefault(f, []).append((j, [float(v) for v in m.group(1).split()]))
    if len(boxes) > len(BOX_COLOURS):
        return None
    for f, bs in drawn.items():
        im = Image.open(f).convert("RGB")
        d = ImageDraw.Draw(im)
        for j, (x1, y1, x2, y2) in bs:
            d.rectangle([x1, y1, x2, y2], outline=BOX_COLOURS[j][1], width=3)
        out = os.path.join(r, "boxdraw", q["id"], os.path.basename(os.path.dirname(f)) + "_" + os.path.basename(f))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        im.save(out, quality=95)
        files = [out if x == f else x for x in files]
    k = iter(range(len(boxes)))
    qpart = re.sub(r"<BBOX ([\d. ]+)>", lambda m: f"<{BOX_COLOURS[next(k)][0]} box>", qpart)
    sent = "<red box>, <blue box> etc. name the box drawn in that colour on the frame at the time given with it."
    return content[:-1] + [{"type": "text", "text": head + sent + qpart}], files


def messages(content, files, detail):
    """Qwen content list ({"type": "text"} / {"type": "image"} placeholders, images in order) -> chat message."""
    parts, it = [], iter(files)
    for c in content:
        if c["type"] == "text":
            parts.append({"type": "text", "text": c["text"]})
        else:
            b64 = base64.b64encode(open(next(it), "rb").read()).decode()
            parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64, "detail": detail}})
    if next(it, None) is not None:
        raise ValueError("more image files than image placeholders")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": parts}]


# --- reply parsing

def parse_letter(text):
    """Index 0-4 of the chosen option, or -1 (counted as wrong, flagged parsed=false). Accepted: a reply that starts
    with the letter ('B', '(B)', 'B.', '**B.**', 'B) ...'), or a sentence that names exactly one letter as the answer
    ('Answer: B', 'The correct answer is **B**.', 'The best description is: **E**.'). Replies that name two letters
    ('A or B') or none give -1."""
    if not text:
        return -1
    if re.search(r"\b[A-E]\**\)?\s+(?:or|and|/)\s+\**\(?[A-E]\b", text):
        return -1
    t = text.strip().strip("*_`").strip()
    m = re.match(r"^\(?([A-E])\)?(?:\s*$|[.:)\]*])", t)
    if m:
        return "ABCDE".index(m.group(1))
    found = set(re.findall(r"(?i:answer|option|description)\s*(?:is|:)?\s*:?\s*\**\s*\(?([A-E])\)?(?![A-Za-z])", text))
    found |= set(re.findall(r"\*\*\s*\(?([A-E])\)?(?:[.:)]|\*\*)", text))
    return "ABCDE".index(found.pop()) if len(found) == 1 else -1


def letter_probs(choice):
    """Probabilities of A-E from the first output token's top logprobs (None if not returned)."""
    lp = (choice.get("logprobs") or {}).get("content") or []
    if not lp:
        return None, None
    import math
    p = [0.0] * 5
    for c in lp[0].get("top_logprobs", []):
        tok = c["token"].strip().strip("(").strip(")").strip(".")
        if tok in "ABCDE" and len(tok) == 1:
            p["ABCDE".index(tok)] += math.exp(c["logprob"])
    mass = sum(p)
    return ([round(x / mass, 4) for x in p] if mass > 0 else None), round(mass, 4)


# --- API

class Api:
    def __init__(self, args):
        self.url = (args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key and not args.dry:
            sys.exit("set OPENAI_API_KEY in the environment (not on the command line)")
        self.params = {"model": args.model, "temperature": 0, "max_completion_tokens": args.max_out}
        if args.logprobs:
            self.params.update(logprobs=True, top_logprobs=5)
        if args.reasoning_effort:
            self.params["reasoning_effort"] = args.reasoning_effort
        self.lock = threading.Lock()
        self.timeout = args.timeout

    def _post(self, body):
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(req, timeout=self.timeout) as f:
            return json.loads(f.read())

    def negotiate(self, err, sent):
        """Adapt to a server that rejects a parameter (older models, reasoning models, gateways). True if the
        rejected parameter is now gone, also when another thread removed it first."""
        msg = err.lower()
        with self.lock:
            for name in ("max_completion_tokens", "temperature", "logprobs", "reasoning_effort"):
                if name in msg and name in sent and name not in self.params:
                    return True
            if "max_completion_tokens" in msg and "max_completion_tokens" in self.params:
                self.params["max_tokens"] = self.params.pop("max_completion_tokens")
            elif "temperature" in msg and "temperature" in self.params:
                self.params.pop("temperature")
            elif "logprobs" in msg and "logprobs" in self.params:
                self.params.pop("logprobs"), self.params.pop("top_logprobs", None)
            elif "reasoning_effort" in msg and "reasoning_effort" in self.params:
                self.params.pop("reasoning_effort")
            else:
                return False
            print("server rejected a parameter; now using", {k: v for k, v in self.params.items() if k != "model"}, flush=True)
            return True

    def call(self, msgs, attempts=6):
        """(response, None) or (None, (kind, message)). kind: fatal | blocked | bad_request | transient."""
        for a in range(attempts):
            with self.lock:
                body = dict(self.params, messages=msgs)
            try:
                return self._post(body), None
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")[:500]
                if e.code in (401, 403):
                    return None, ("fatal", f"HTTP {e.code}: {text}")
                if e.code == 400:
                    if "content_policy" in text or "content_filter" in text or "safety" in text.lower():
                        return None, ("blocked", text)
                    if self.negotiate(text, body):
                        continue
                    return None, ("bad_request", text)
                if e.code in TRANSIENT and a < attempts - 1:
                    wait = float(e.headers.get("Retry-After") or 0) or min(60, 2 ** a + random.random())
                    time.sleep(wait)
                    continue
                return None, ("transient", f"HTTP {e.code}: {text}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if a < attempts - 1:
                    time.sleep(min(60, 2 ** a + random.random()))
                    continue
                return None, ("transient", repr(e))
        return None, ("transient", "retries exhausted")


# --- run

def file_sha(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def main(args):
    r = H.root(args)
    args.variant = "blur"
    qs = H.select(args, r)
    if args.form in BOX_FORMS:
        qs = [q for q in qs if "<BBOX" in q["question"]]
        print(f"{len(qs)} questions with a box", flush=True)
    ev_kind = "labels" if args.form == "labels" else "notes"
    args.evidence_kind = ev_kind              # text_inputs picks the labels wording from this
    ev_path = P.res_path(args, f"{ev_kind}_{args.evidence_model}_{P.tag(args)}.jsonl")
    if args.form == "crop_guided":
        boxes = P.read_jsonl(P.res_path(args, f"ground_{args.ground_model}_{P.tag(args)}.jsonl"))
    elif args.form in ("notes", "labels"):  # the device's notes play the role of the boxes: per-question side input
        boxes = P.read_jsonl(ev_path)
        if not boxes:
            sys.exit(f"no {ev_path}: run `hdepic_pilot.py {ev_kind}` first")
    else:
        boxes = {q["id"]: None for q in qs}
    model_tag = args.model.replace("/", "_")
    out = P.res_path(args, f"qa_{model_tag}_{SUFFIX[args.form]}_{P.tag(args)}.jsonl")
    meta = {"model": args.model, "form": args.form, "system": SYSTEM, "detail": args.detail, "max_out": args.max_out,
            "logprobs": args.logprobs, "reasoning_effort": args.reasoning_effort,
            "selection": {"homes": args.homes, "split": args.split, "per_type": args.per_type, "nframes": args.nframes,
                          "dense": args.dense},
            "annotations_sha": file_sha(glob.glob(os.path.join(r, "vqa", "*.json"))),
            "ground_sha": file_sha([P.res_path(args, f"ground_{args.ground_model}_{P.tag(args)}.jsonl")]) if args.form == "crop_guided" else None,
            "evidence_sha": file_sha([ev_path]) if args.form in ("notes", "labels") else None}

    if args.dry:
        q = next(q for q in qs if inputs(args, r, q, boxes))
        content, files = inputs(args, r, q, boxes)
        it = iter(files)
        shown = [c if c["type"] == "text" else {"type": "image", "file": os.path.relpath(next(it), r)} for c in content]
        print(json.dumps({"would_write": out, "settings": meta, "question": q["id"], "n_images": len(files),
                          "content": shown}, indent=1))
        return

    mpath = out + ".meta.json"
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
        if diff:
            sys.exit(f"{out} was made with other settings {diff}; use a different --model/--form or move the old file")
    else:
        json.dump(dict(meta, started=datetime.datetime.now().isoformat(timespec="seconds")), open(mpath, "w"), indent=1)

    fo, done, _lock = P.open_out(out)
    epath = out + ".errors.jsonl"
    blocked = set()
    if os.path.exists(epath) and not args.retry_blocked:
        blocked = {json.loads(l)["id"] for l in open(epath) if json.loads(l)["kind"] == "blocked"}
    todo = [q for q in qs if q["id"] not in done and q["id"] not in blocked][: args.limit]
    api = Api(args)
    wlock = threading.Lock()
    stats = {"rows": 0, "in": 0, "out": 0, "err": 0, "unparsed": 0, "usd": 0.0, "stop": None}
    priced = args.price_in is not None and args.price_out is not None
    print(f"{len(todo)} questions to send ({len(done)} already done, {len(blocked)} blocked) -> {out}", flush=True)

    def one(q):
        if stats["stop"]:
            return
        got = inputs(args, r, q, boxes)
        if got is None:
            with wlock, open(epath, "a") as fe:
                fe.write(json.dumps({"id": q["id"], "kind": "missing_input", "msg": "crop, box or evidence missing"}) + "\n")
            return
        content, files = got
        resp, err = api.call(messages(content, files, args.detail))
        with wlock:
            if err:
                stats["err"] += 1
                with open(epath, "a") as fe:
                    fe.write(json.dumps({"id": q["id"], "kind": err[0], "msg": err[1], "time": time.time()}) + "\n")
                if err[0] == "fatal":
                    stats["stop"] = err[1]
                elif stats["err"] >= 5 and stats["rows"] == 0:
                    stats["stop"] = "the first 5 calls all failed: " + err[1][:200]
                return
            ch = resp["choices"][0]
            text = (ch.get("message") or {}).get("content") or ""
            if not text and ch.get("finish_reason") == "length":
                stats["err"] += 1
                with open(epath, "a") as fe:
                    fe.write(json.dumps({"id": q["id"], "kind": "empty_length",
                                         "msg": "no answer within --max_out tokens (reasoning model? raise --max_out)"}) + "\n")
                stats["stop"] = "empty replies: raise --max_out"
                return
            pred = parse_letter(text)
            probs, mass = letter_probs(ch)
            u = resp.get("usage") or {}
            tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            stats["rows"] += 1; stats["in"] += tin; stats["out"] += tout; stats["unparsed"] += pred < 0
            if priced:
                stats["usd"] += (tin * args.price_in + tout * args.price_out) / 1e6
            fo.write(json.dumps({
                "id": q["id"], "prototype": q["prototype"], "video": H.first_video(q),
                "home": sorted({H.home_of(i["id"]) for i in q["inputs"].values()}), "form": args.form,
                "pred": pred, "gold": q["correct_idx"], "parsed": pred >= 0, "probs": probs, "letter_mass": mass,
                "reply": text[:300], "finish_reason": ch.get("finish_reason"), "model": resp.get("model"),
                "n_frames": len(files), "n_tokens": tin, "out_tokens": tout,
                "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "dense_frames": sum(os.path.basename(f).startswith("d") for f in files),
                "time": datetime.datetime.now().isoformat(timespec="seconds")}) + "\n")
            fo.flush()
            n = stats["rows"]
            if priced and args.max_usd is not None and stats["usd"] >= args.max_usd:
                stats["stop"] = f"spend cap ${args.max_usd} reached"
            if n in (1, 5) or n % 25 == 0:
                avg = (stats["in"] + stats["out"]) / n
                est = f"; ${stats['usd']:.2f} so far, about ${stats['usd'] / n * (len(todo) - n):.2f} more" if priced else ""
                print(f"{n}/{len(todo)} done, {avg:.0f} tokens/question, {stats['err']} errors, "
                      f"{stats['unparsed']} unparsed{est}", flush=True)

    if todo:
        one(todo[0])  # first call alone: the server's accepted parameters are settled before going parallel
    with ThreadPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(one, q) for q in todo[1:]]):
            f.result()
    meta_now = json.load(open(mpath))
    meta_now["effective_params"] = {k: v for k, v in api.params.items() if k != "model"}
    json.dump(meta_now, open(mpath, "w"), indent=1)
    print(f"finished: {stats['rows']} rows written, {stats['err']} errors (see {epath}), {stats['unparsed']} unparsed, "
          f"{stats['in']} input + {stats['out']} output tokens" + (f", about ${stats['usd']:.2f}" if priced else ""), flush=True)
    if stats["stop"]:
        sys.exit("STOPPED: " + stats["stop"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)                      # exactly as the API names it
    ap.add_argument("--form", required=True, choices=list(SUFFIX))
    ap.add_argument("--homes", nargs="+", default=["P01"])
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--per_type", type=int, default=30)
    ap.add_argument("--nframes", type=int, default=32)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--limit", type=int, default=None)            # send at most N new questions this run
    ap.add_argument("--ground_model", default="Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--evidence_model", default="Qwen2.5-VL-3B-Instruct")   # notes: whose notes
    # 'low' sends each image at up to 512 px; our frames and crops are 448 px, so nothing is downscaled
    ap.add_argument("--detail", default="low", choices=["low", "high", "auto"])
    ap.add_argument("--max_out", type=int, default=32)            # official eval: 1 (Gemini), lmms-eval: 10; reasoning models need ~2000
    ap.add_argument("--reasoning_effort", default=None)           # reasoning models only, e.g. low
    ap.add_argument("--no_logprobs", dest="logprobs", action="store_false")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--price_in", type=float, default=None)       # $ per 1M input tokens
    ap.add_argument("--price_out", type=float, default=None)      # $ per 1M output tokens
    ap.add_argument("--max_usd", type=float, default=None)
    ap.add_argument("--retry_blocked", action="store_true")
    ap.add_argument("--dry", action="store_true")
    main(ap.parse_args())
