#!/usr/bin/env bash
# Privacy of device-to-cloud uploads over all 9 homes, and the on-device gate. No API key needed.
#   linking          are uploads of one home recognisable as one home?        (embedding linkers, 7B as linker)
#   data disclosure  does an upload show sensitive content?                   (32B judge)
#   inference        what does the provider learn about the household?        (32B profile)
# Needs scripts/run_hdepic.sh. Resumes; logs: logs/privacy_*.log; report: data/hdepic/privacy/report.md
#   GPU=0 GPUS_32B=0,1,2 bash scripts/run_upload_privacy.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:-0}; GPUS_32B=${GPUS_32B:-0,1,2}
P="python -m src.privacy.upload_privacy --data data"     # all 9 homes by default
M3=Qwen/Qwen2.5-VL-3B-Instruct; M7=Qwen/Qwen2.5-VL-7B-Instruct; M32=Qwen/Qwen2.5-VL-32B-Instruct
mkdir -p logs
run() { name=$1; shift; echo "== $name ($(date +%H:%M))"; "$@" > logs/privacy_$name.log 2>&1; }

run select $P select
export CUDA_VISIBLE_DEVICES=$GPU
run ground     $P ground --model $M3
run notes      $P notes  --model $M3
run gate       $P gate   --model $M3
run gate_pilot $P gate   --model $M3 --pilot      # gate the P01 pilot frames too, so the 7B can answer from them
run make       $P make
run embed      $P embed
run link       $P link
run vlmlink    $P vlmlink --model $M7
run qa_gated   python -m src.data.hdepic qa --data data --model $M7 --homes P01 --split dev --per_type 30 --dense --variant gated
export CUDA_VISIBLE_DEVICES=$GPUS_32B
run judge      $P judge   --model $M32
run profile    $P profile --model $M32
run report     $P report
echo "report: data/hdepic/privacy/report.md"
