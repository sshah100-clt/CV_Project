#!/usr/bin/env bash
# HD-EPIC pilot on the P01 dev questions (277 questions, at most 30 per type), plus cross-day linking over 9 homes.
# Downloads: all 27 videos of P01 (kept: crops and dense frames are cut from them) and the 2 smallest videos of each
# other home (deleted after frame extraction). Every step resumes if interrupted.
#   GPU=0 GPUS_32B=0,1,2 bash scripts/run_hdepic.sh        (32B needs about 3 x 24 GB)
# Logs: logs/hdepic_*.log    Results: data/hdepic/results/
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:-0}; GPUS_32B=${GPUS_32B:-0,1,2}
OTHERS="P02 P03 P04 P05 P06 P07 P08 P09"; ALL="P01 $OTHERS"
SEL="--homes P01 --split dev --per_type 30 --dense"
H="python -m src.data.hdepic --data data"
P="python -m src.qa.hdepic_pilot --data data"
M3=Qwen/Qwen2.5-VL-3B-Instruct; M7=Qwen/Qwen2.5-VL-7B-Instruct; M32=Qwen/Qwen2.5-VL-32B-Instruct
mkdir -p logs
run() { name=$1; shift; echo "== $name ($(date +%H:%M))"; "$@" > logs/hdepic_$name.log 2>&1; }

# 1. data: questions, videos, 1-per-second frames (raw + face-blurred), exact-time frames for short windows
run download_p01    $H download --homes P01
run download_others $H download --homes $OTHERS --per_home 2
run frames_p01      $H frames --homes P01
run frames_others   $H frames --homes $OTHERS --delete
run dense           $H dense $SEL

# 2. frame features (every 2nd second, all homes) and cross-day linking
for bb in siglip dinov2 clip; do
  CUDA_VISIBLE_DEVICES=$GPU run link_$bb $H link --homes $ALL --backbone $bb
done
CUDA_VISIBLE_DEVICES=$GPU run views python -m src.data.embed_views hdepic --data data

# 3. question answering: 3B on the device (raw frames), 7B / 32B in the cloud (face-blurred frames)
CUDA_VISIBLE_DEVICES=$GPU      run qa3   $H qa $SEL --model $M3 --variant raw
CUDA_VISIBLE_DEVICES=$GPU      run qa7   $H qa $SEL --model $M7 --variant blur
CUDA_VISIBLE_DEVICES=$GPUS_32B run qa32  $H qa $SEL --model $M32 --variant blur

# 4. other ways to escalate: device notes, crops (guided, centre, full frames), question only
export CUDA_VISIBLE_DEVICES=$GPU
run notes     $P notes  $SEL --model $M3
run qa_notes  $P qa_text $SEL --model $M7 --evidence_kind notes
run ground    $P ground $SEL --model $M3
run crops     $P crops  $SEL
for mode in guided center full; do
  run qa_crop_$mode $P qa_crop $SEL --model $M7 --crop_mode $mode
done
run blind3    $P qa_blind $SEL --model $M3
run blind7    $P qa_blind $SEL --model $M7
CUDA_VISIBLE_DEVICES=$GPUS_32B run blind32 $P qa_blind $SEL --model $M32

# 5. linking per form (frames, centre crops, device-written text, question text) and the planner check
run describe  $P describe --homes $ALL --model $M3
run linkage   $P linkage  --homes $ALL
run plan      $P plan --homes P01 --split dev --model $M3

# 6. summary
run report $H report
python -m src.analysis.analyze_pilot | tee logs/hdepic_summary.log
