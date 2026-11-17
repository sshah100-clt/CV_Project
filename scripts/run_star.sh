#!/usr/bin/env bash
# GPU runs on Charades / STAR: the VLM linking model on Charades-Link episodes, and STAR question answering with
# Qwen2.5-VL 3B / 7B / 32B on raw, face-blurred and link-break frames. Every job resumes where it stopped.
# Needs: scripts/run_charades.sh (frames + episodes_vlm.json), data/star/STAR_val.json, data/charades/videos/.
# Start inside tmux:  tmux new -s star 'bash scripts/run_star.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"      # about 90 GB of model weights
mkdir -p logs results
NG=$(python -c "import torch; print(torch.cuda.device_count())")

shards () {  # shards <log name> <command...>: one shard per GPU, wait for all
  name=$1; shift
  for g in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=$g "$@" --shard $g --nshards $NG > logs/${name}_$g.log 2>&1 &
  done
  wait
}

echo "[1/4] STAR frames: sample, blur faces, link-break ($(date))"
python -m src.qa.star_qa frames --data data > logs/star_frames.log 2>&1

echo "[2/4] VLM linking model (Qwen2.5-VL-7B) on Charades-Link ($(date))"
for v in blur linkbreak raw; do
  shards vlm_link_$v python -m src.linking.vlm_linker --data data/charades_link --variant $v
done

echo "[3/4] STAR QA: 3B (edge) and 7B, three variants ($(date))"
for m in Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2.5-VL-7B-Instruct; do
  for v in raw blur linkbreak; do
    shards star_$(basename $m)_$v python -m src.qa.star_qa run --data data --model $m --variant $v
  done
done

echo "[4/4] STAR QA: 32B (cloud), split over all GPUs ($(date))"
for v in raw blur linkbreak; do
  python -m src.qa.star_qa run --data data --model Qwen/Qwen2.5-VL-32B-Instruct --variant $v \
    --shard 0 --nshards 1 > logs/star_32B_$v.log 2>&1
done

{
  echo "== VLM linking model (chance 0.25)"
  for v in blur linkbreak raw; do echo "-- $v"; python -m src.linking.vlm_linker --data data/charades_link --variant $v --merge; done
  echo "== STAR val accuracy"
  python -m src.qa.star_qa merge --data data
} | tee results/star_summary.txt
echo "done $(date)"
