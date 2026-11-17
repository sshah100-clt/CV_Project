"""Code-validity checks for the Qwen2.5-VL pipeline used in star_qa.py (same loader, same letter read-out).

  mmstar   MMStar val (1,500 image questions, answers A-D). Published (model cards): 7B 63.9, 32B 69.5.
           If our pipeline reproduces these, model loading, image processing and the letter read-out are sound.
  star     The 507-question STAR subset (every 14th question, blur frames) at a chosen resolution, with the
           4 frames given either as 4 images (as in star_qa.py) or as ONE video with its real frame rate,
           which uses Qwen2.5-VL's time encoding. Tests whether our STAR protocol penalises the larger model.

Resolution presets (visual tokens per image / frame):
  low  64-256   (star_qa.py setting)
  rec  256-1280 (Qwen model-card recommendation)

    pip install pyarrow                                   # once, to read the MMStar parquet
    python -m src.qa.check_vlm mmstar --data data --model Qwen/Qwen2.5-VL-7B-Instruct --res rec
    python -m src.qa.check_vlm star   --data data --model Qwen/Qwen2.5-VL-7B-Instruct --res rec --as_video
    python -m src.qa.check_vlm report --data data
"""
import os
import io
import sys
import json
import glob
import fcntl
import argparse

import numpy as np

RES = {"low": (64 * 28 * 28, 256 * 28 * 28), "rec": (256 * 28 * 28, 1280 * 28 * 28)}
LETTERS = ["A", "B", "C", "D"]
F = 4  # frames per STAR question, as in star_qa.py


def out_path(args):
    tag = f"{args.cmd}_{os.path.basename(args.model)}_{args.res}" + ("_video" if getattr(args, "as_video", False) else "")
    return os.path.join(args.data, "check_vlm", f"sanity_{tag}.jsonl")


def open_output(path):
    """Lock the output (one writer only) and return (file, set of question ids already done)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def load_model(args):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa").eval()
    lo, hi = RES[args.res]
    proc = AutoProcessor.from_pretrained(args.model, min_pixels=lo, max_pixels=hi)
    ids = [proc.tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    return model, proc, ids


def letter_probs(model, x, ids):
    import torch
    with torch.no_grad():
        full = torch.softmax(model(**x).logits[0, -1].float(), 0)
    p = (full[ids] / full[ids].sum()).cpu().numpy()
    return p, float(full[ids].sum())


def mmstar(args):
    import pyarrow.parquet as pq
    from PIL import Image
    from huggingface_hub import snapshot_download
    root = snapshot_download("Lin-Chen/MMStar", repo_type="dataset", allow_patterns=["*.parquet"])
    files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    rows = [r for f in files for r in pq.read_table(f).to_pylist()]
    print(len(rows), "MMStar questions from", files, flush=True)
    model, proc, ids = load_model(args)
    fo, done, _lock = open_output(out_path(args))
    for r in rows:
        if r["index"] in done:
            continue
        img = r["image"]
        img = Image.open(io.BytesIO(img["bytes"] if isinstance(img, dict) else img)).convert("RGB")
        content = [{"type": "image"}, {"type": "text", "text":
                   r["question"] + "\nAnswer with the option's letter from the given choices directly."}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        x = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        p, mass = letter_probs(model, x, ids)
        fo.write(json.dumps({"id": r["index"], "category": r["category"], "pred": LETTERS[int(p.argmax())],
                             "gold": r["answer"].strip(), "letter_mass": round(mass, 4)}) + "\n")
        fo.flush()


def star(args):
    from PIL import Image
    qs = json.load(open(os.path.join(args.data, "star/STAR_val.json")))[0::14]  # the 507-question check subset
    model, proc, ids = load_model(args)
    fo, done, _lock = open_output(out_path(args))
    shown = False
    for q in qs:
        if q["question_id"] in done:
            continue
        ims = [Image.open(os.path.join(args.data, "star", "frames", "blur", f"{q['question_id']}_{j}.jpg")).convert("RGB")
               for j in range(F)]
        opts = "\n".join(f"{l}. {c['choice']}" for l, c in zip(LETTERS, q["choices"]))
        prompt = (f"These are frames from a home video, in time order.\nQuestion: {q['question']}\n{opts}\n"
                  "Answer with one letter: A, B, C or D.")
        if args.as_video:
            # a video needs one frame size; a black placeholder frame (unreadable frame) can differ
            size = max({im.size for im in ims}, key=[im.size for im in ims].count)
            ims = [im if im.size == size else im.resize(size) for im in ims]
            # frames sit at 1/5 ... 4/5 of the segment, so they are (end - start) / 5 seconds apart
            fps = 5.0 / max(q["end"] - q["start"], 0.5)
            content = [{"type": "video"}, {"type": "text", "text": prompt}]
            text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
            x = proc(text=[text], videos=[ims], fps=[fps], return_tensors="pt")
            if not shown:
                print("video inputs:", {k: tuple(v.shape) if hasattr(v, "shape") else v for k, v in x.items()
                                        if k != "input_ids"}, flush=True)
                if "second_per_grid_ts" not in x:
                    sys.exit("processor did not return second_per_grid_ts: the frame rate was not used")
                shown = True
            x = x.to(model.device)
        else:
            content = [{"type": "image"} for _ in ims] + [{"type": "text", "text": prompt}]
            text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
            x = proc(text=[text], images=ims, return_tensors="pt").to(model.device)
        p, mass = letter_probs(model, x, ids)
        gold = [c["choice"] for c in q["choices"]].index(q["answer"])
        fo.write(json.dumps({"id": q["question_id"], "video_id": q["video_id"], "pred": int(p.argmax()), "gold": gold,
                             "probs": p.round(4).tolist(), "letter_mass": round(mass, 4)}) + "\n")
        fo.flush()


def report(args):
    for f in sorted(glob.glob(os.path.join(args.data, "check_vlm", "sanity_*.jsonl"))):
        rows = {}
        for l in open(f):
            r = json.loads(l)
            rows[r["id"]] = r
        acc = np.mean([str(r["pred"]) == str(r["gold"]) for r in rows.values()])
        print(f"{os.path.basename(f)[7:-6]:50s} n={len(rows):5d} accuracy {100 * acc:.1f} "
              f"letter mass {np.mean([r['letter_mass'] for r in rows.values()]):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mmstar", "star", "report"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--res", default="rec", choices=RES)
    ap.add_argument("--as_video", action="store_true")
    a = ap.parse_args()
    {"mmstar": mmstar, "star": star, "report": report}[a.cmd](a)
