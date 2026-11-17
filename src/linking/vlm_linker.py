"""VLM linking model for Charades-Link (GPU).

Shows Qwen2.5-VL the new upload (Q) and K upload histories (A-D), asks which
history shows the same person or home, and reads the probability of each
letter from the next-token logits. Faces are already blurred in the frames.

Needs data/charades_link/{raw,blur,linkbreak}/ and episodes_vlm.json (python -m src.linking.link_eval dump).
Run (one process per GPU, sharded):
    CUDA_VISIBLE_DEVICES=0 python -m src.linking.vlm_linker --data data/charades_link --variant blur --shard 0 --nshards 4
    ...
    python -m src.linking.vlm_linker --data data/charades_link --variant blur --merge
"""
import os
import json
import glob
import argparse

import numpy as np

PROMPT = ("Image Q is a new photo from a home camera. Sets A, B, C and D each contain earlier photos "
          "from one household. Faces are blurred. Which set comes from the same person or the same home "
          "as image Q? Answer with one letter: A, B, C or D.")


def run(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    eps = json.load(open(os.path.join(args.data, "episodes_vlm.json")))[args.shard::args.nshards]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    proc = AutoProcessor.from_pretrained(args.model, min_pixels=64 * 28 * 28, max_pixels=args.max_pixels)
    letters = ["A", "B", "C", "D"]
    letter_ids = [proc.tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]
    img = lambda n: Image.open(os.path.join(args.data, args.variant, n)).convert("RGB")
    out_path = os.path.join(args.data, f"vlm_{args.variant}_{os.path.basename(args.model)}_{args.shard}.jsonl")
    done = sum(1 for _ in open(out_path)) if os.path.exists(out_path) else 0
    with open(out_path, "a") as fo:
        for e in eps[done:]:
            content, images = [{"type": "text", "text": "Image Q:"}, {"type": "image"}], [img(e["query"])]
            for l, s in zip(letters, e["sets"]):
                content.append({"type": "text", "text": f"Set {l}:"})
                for n in s:
                    content.append({"type": "image"})
                    images.append(img(n))
            content.append({"type": "text", "text": PROMPT})
            text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
            x = proc(text=[text], images=images, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**x).logits[0, -1]
            p = torch.softmax(logits[letter_ids].float(), 0).cpu().numpy()
            fo.write(json.dumps({**e, "probs": p.round(4).tolist(), "pred": letters[int(p.argmax())]}) + "\n")
            fo.flush()


def merge(args):
    rows = []
    for f in glob.glob(os.path.join(args.data, f"vlm_{args.variant}_{os.path.basename(args.model)}_*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    for cross in (False, True):
        for m in sorted({r["m"] for r in rows}):
            rs = [r for r in rows if r["m"] == m and r["cross_room"] == cross]
            if rs:
                acc = np.mean([r["pred"] == r["answer"] for r in rs])
                print(f"{'cross_room' if cross else 'any_room'} m={m}: accuracy {acc:.3f} (n={len(rs)}, chance 0.25)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", default="blur")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge(a) if a.merge else run(a)
