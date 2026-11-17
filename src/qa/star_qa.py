"""STAR question answering with a VLM on each upload variant (real utility U).

For every STAR val question, F frames are sampled uniformly in [start, end] of
its Charades video, cleaned with the variant (raw / blur / linkbreak, the same
operators as Charades-Link), and shown to Qwen2.5-VL with the 4 choices. The
probability of each answer letter is read from the next-token logits.
Small models (3B) stand in for the local edge model; larger ones for the cloud.

    python -m src.qa.star_qa frames --data data                      # extract + clean frames (CPU)
    CUDA_VISIBLE_DEVICES=0 python -m src.qa.star_qa run --data data --model Qwen/Qwen2.5-VL-7B-Instruct --variant blur --shard 0 --nshards 4
    python -m src.qa.star_qa merge --data data
"""
import os
import re
import sys
import fcntl
import json
import glob
import argparse

import numpy as np

F = 4
PRIVATE_SCENES = {"Bathroom", "Bedroom", "Closet / Walk-in closet / Spear closet"}
VARIANTS = ("raw", "blur", "linkbreak")


def frames(args):
    import cv2
    from src.data import charades_link
    from src.data import upload_variants as operators
    from ultralytics import YOLO
    charades_link.YUNET = os.path.join(args.data, "models/face_detection_yunet_2023mar.onnx")
    seg = YOLO(os.path.join(args.data, "models/yolo11n-seg.pt"))
    qs = json.load(open(os.path.join(args.data, "star/STAR_val.json")))
    out = os.path.join(args.data, "star", "frames")
    for v in VARIANTS:
        os.makedirs(os.path.join(out, v), exist_ok=True)
    for i, q in enumerate(qs):
        name = q["question_id"]
        if os.path.exists(os.path.join(out, "linkbreak", f"{name}_{F - 1}.jpg")):
            continue
        cap = cv2.VideoCapture(os.path.join(args.data, "charades", "videos", q["video_id"] + ".mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        last = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1  # some STAR segments end after the video does
        for j, t in enumerate(np.linspace(q["start"], q["end"], F + 2)[1:-1]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(t * fps), last))
            ok, img = cap.read()
            if not ok:  # keep the black frame so every question has F images, but record it
                img = np.zeros((480, 640, 3), np.uint8)
                with open(os.path.join(out, "failed_frames.txt"), "a") as fl:
                    fl.write(f"{name}_{j}\n")
            blurred, _ = charades_link.blur_faces(img)
            r = seg(img, classes=[0], conf=0.25, verbose=False, retina_masks=True)[0]
            m = np.zeros(img.shape[:2], np.uint8)
            if r.masks is not None:
                m = (r.masks.data.cpu().numpy().max(0) > 0.5).astype(np.uint8) * 255
                m = cv2.dilate(cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST), np.ones((9, 9), np.uint8))
            for v, im in (("raw", img), ("blur", blurred), ("linkbreak", operators.apply("linkbreak", blurred, m))):
                cv2.imwrite(os.path.join(out, v, f"{name}_{j}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if i % 500 == 0:
            print(i, "/", len(qs), flush=True)


def parse_letter(ans, choices):
    """Answer index from a free-text reply, or -1. Avoids reading the article 'A' as an answer."""
    last = re.findall(r"Answer:\s*\(?([ABCD])\b", ans)  # reasoning replies end with 'Answer: X'
    if last:
        return "ABCD".index(last[-1])
    for pat in (r"^\s*\(?([ABCD])\)?\s*(?:[.:)]|$)", r"(?:answer|option)\s*(?:is|:)?\s*\(?([ABCD])\b"):
        hit = re.search(pat, ans, re.I if "answer" in pat else 0)
        if hit:
            return "ABCD".index(hit.group(1).upper())
    found = [i for i, c in enumerate(choices) if c.lower().rstrip(".") in ans.lower()]
    return found[0] if len(found) == 1 else -1


def run(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    qs = json.load(open(os.path.join(args.data, "star/STAR_val.json")))[args.shard::args.nshards][: args.limit]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa").eval()
    proc = AutoProcessor.from_pretrained(args.model, min_pixels=64 * 28 * 28, max_pixels=args.max_pixels)
    letters = ["A", "B", "C", "D"]
    ids = [proc.tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]
    args.generate = args.generate or args.cot
    tag = f"{os.path.basename(args.model)}_{args.variant}" + ("_cot" if args.cot else "_gen" if args.generate else "")
    path = os.path.join(args.data, "star", "results", f"star_{tag}_{args.shard}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = open(path + ".lock", "w")  # a second copy of the same run exits instead of writing duplicates
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"{path} is already being written by another process")
    done = set()  # resume by question id, so duplicate or partial lines never skip a question
    if os.path.exists(path):
        for l in open(path):
            try:
                done.add(json.loads(l)["question_id"])
            except (ValueError, KeyError):
                pass
    with open(path, "a") as fo:
        for q in qs:
            if q["question_id"] in done:
                continue
            ims = [Image.open(os.path.join(args.data, "star", "frames", args.variant, f"{q['question_id']}_{j}.jpg")).convert("RGB") for j in range(F)]
            opts = "\n".join(f"{l}. {c['choice']}" for l, c in zip(letters, q["choices"]))
            ask = ("Think step by step in a few sentences, then give the final answer on the last line as 'Answer: X'."
                   if args.cot else "Answer with one letter: A, B, C or D.")
            content = [{"type": "image"} for _ in ims] + [{"type": "text", "text":
                       f"These are frames from a home video, in time order.\nQuestion: {q['question']}\n{opts}\n{ask}"}]
            text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
            x = proc(text=[text], images=ims, return_tensors="pt").to(model.device)
            gold = [c["choice"] for c in q["choices"]].index(q["answer"])
            with torch.no_grad():
                if args.generate:  # check of the logit read-out: free answer + the first-step letter probabilities
                    y = model.generate(**x, max_new_tokens=512 if args.cot else 128, do_sample=False,
                                       output_scores=True, return_dict_in_generate=True)
                    ans = proc.batch_decode(y.sequences[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                    full = torch.softmax(y.scores[0][0].float(), 0)
                    p = (full[ids] / full[ids].sum()).cpu().numpy()
                    # first generated token only: a valid answer read-out for --generate, NOT a confidence for --cot
                    fo.write(json.dumps({"question_id": q["question_id"], "video_id": q["video_id"], "answer_text": ans,
                                         "pred": parse_letter(ans, [c["choice"] for c in q["choices"]]), "gold": gold,
                                         "first_token_probs": p.round(4).tolist(),
                                         "first_token_letter_mass": round(float(full[ids].sum()), 4)}) + "\n")
                    fo.flush()
                    continue
                full = torch.softmax(model(**x).logits[0, -1].float(), 0)
                p = (full[ids] / full[ids].sum()).cpu().numpy()
            fo.write(json.dumps({"question_id": q["question_id"], "video_id": q["video_id"], "probs": p.round(4).tolist(),
                                 "pred": int(p.argmax()), "gold": gold, "letter_mass": round(float(full[ids].sum()), 4)}) + "\n")
            fo.flush()


def merge(args):
    import csv
    scene = {r["id"]: r["scene"] for s in ("train", "test") for r in csv.DictReader(open(os.path.join(args.data, f"charades/Charades_v1_{s}.csv")))}
    tags = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(os.path.join(args.data, "star", "results", "star_*.jsonl"))})
    for t in tags:
        rows = {}  # shard files only (t_0.jsonl, ...), not t_gen_0 / t_cot_0; identical duplicates collapse
        for f in glob.glob(os.path.join(args.data, "star", "results", t + "_[0-9]*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                old = rows.get(r["question_id"])
                if old is not None and (old["pred"], old["gold"]) != (r["pred"], r["gold"]):
                    sys.exit(f"{t}: conflicting answers for {r['question_id']}")
                rows[r["question_id"]] = r
        rows = list(rows.values())
        acc = np.mean([r["pred"] == r["gold"] for r in rows])
        priv = [r["pred"] == r["gold"] for r in rows if scene.get(r["video_id"]) in PRIVATE_SCENES]
        print(f"{t[5:]}: accuracy {acc:.3f} (n={len(rows)}); private-room questions {np.mean(priv):.3f} (n={len(priv)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frames", "run", "merge"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--variant", default="blur", choices=VARIANTS)
    ap.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)       # first N questions of the shard only
    ap.add_argument("--generate", action="store_true")       # free answer instead of letter logits
    ap.add_argument("--cot", action="store_true")            # short reasoning, then 'Answer: X
    a = ap.parse_args()
    {"frames": frames, "run": run, "merge": merge}[a.cmd](a)
