"""Token and cost audit of every GPT run, from the saved rows (the API's own usage counts per call).

    python -m src.qa.gpt_cost data/hdepic/results --price_in 2.5 --price_out 10

Counts only calls that returned an answer (rows). Failed calls are listed separately; they are normally not billed.
The cost is an estimate from list prices: it ignores cached-input discounts, so it is an upper bound. The exact
bill is on the OpenAI usage dashboard.
"""
import os
import sys
import json
import glob
import argparse


def main(a):
    files = sorted(f for f in glob.glob(os.path.join(a.dir, "qa_*.jsonl"))
                   if not f.endswith(".errors.jsonl") and os.path.exists(f + ".meta.json"))  # gpt_qa writes <out>.meta.json
    if not files:
        sys.exit(f"no GPT result files (qa_*.jsonl with a .meta.json) in {a.dir}")
    tot = {"rows": 0, "in": 0, "out": 0, "err": 0}
    print(f"{'file':60s} {'rows':>5s} {'input tok':>10s} {'output tok':>10s} {'failed':>6s} {'est. $':>8s}")
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        tin = sum(r.get("n_tokens") or 0 for r in rows)
        tout = sum(r.get("out_tokens") or 0 for r in rows)
        ef = f + ".errors.jsonl"
        err = sum(1 for l in open(ef) if l.strip()) if os.path.exists(ef) else 0
        usd = (tin * a.price_in + tout * a.price_out) / 1e6
        print(f"{os.path.basename(f)[:60]:60s} {len(rows):5d} {tin:10d} {tout:10d} {err:6d} {usd:8.2f}")
        tot["rows"] += len(rows); tot["in"] += tin; tot["out"] += tout; tot["err"] += err
    usd = (tot["in"] * a.price_in + tot["out"] * a.price_out) / 1e6
    print(f"{'TOTAL':60s} {tot['rows']:5d} {tot['in']:10d} {tot['out']:10d} {tot['err']:6d} {usd:8.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--price_in", type=float, required=True)   # $ per 1M input tokens
    ap.add_argument("--price_out", type=float, required=True)  # $ per 1M output tokens
    main(ap.parse_args())
