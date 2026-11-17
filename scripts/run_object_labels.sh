#!/usr/bin/env bash
# Label-only and objects-only uploads. An OWLv2 detector lists the objects in every frame. From its detections:
# label text per pilot question and label vectors per frame (all homes); objects-only images and their features.
# The 3B and 7B answer the pilot questions from the labels, the 7B also from the objects-only images.
# Needs scripts/run_hdepic.sh (frames and data/hdepic/emb_siglip_blur_s2.npz). Resumes; logs: logs/labels_*.log
#   GPU=0 bash scripts/run_object_labels.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}
SEL="--homes P01 --split dev --per_type 30 --dense"
L="python -m src.qa.object_labels --data data"
mkdir -p logs
run() { name=$1; shift; echo "== $name ($(date +%H:%M))"; "$@" > logs/labels_$name.log 2>&1; }
[ -f data/hdepic/emb_siglip_blur_s2.npz ] || { echo "run scripts/run_hdepic.sh first"; exit 1; }

# 1. detector on all HD-EPIC frames, plus 60 public Charades homes (reference set for the linking engine)
run detect  $L detect --charades 60
# 2. label text and label vectors; the 3B and 7B answer from the labels
run make    $L make
for m in Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2.5-VL-7B-Instruct; do
  run qa_$(basename $m) python -m src.qa.hdepic_pilot qa_text --data data $SEL --model $m \
    --evidence_kind labels --evidence_model owlv2
done
# 3. objects-only images, their SigLIP features, and the 7B answering from them
run objonly $L objonly
run objemb  $L objemb
run qa_obj7 python -m src.data.hdepic qa --data data $SEL --model Qwen/Qwen2.5-VL-7B-Instruct --variant objonly
echo "done"
