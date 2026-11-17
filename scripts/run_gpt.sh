#!/usr/bin/env bash
# GPT-4o on the same P01 dev questions, frames and prompts as the Qwen runs (paid API calls).
# Needs OPENAI_API_KEY in the environment and the frames, crops and labels from run_hdepic.sh / run_object_labels.sh.
# Each form stops before its estimated cost passes MAX_USD (list prices below; check them before running).
#   MAX_USD=3 bash scripts/run_gpt.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
[ -n "${OPENAI_API_KEY:-}" ] || { echo "set OPENAI_API_KEY first"; exit 1; }
MODEL=gpt-4o-2024-11-20     # pinned snapshot
G="python -m src.qa.gpt_qa --data data --homes P01 --split dev --per_type 30 --dense --model $MODEL \
   --price_in 2.5 --price_out 10 --max_usd ${MAX_USD:-3}"
mkdir -p logs

$G --form frames --dry > logs/gpt_dry.log     # prints one request (images elided), no API call
for form in frames blind crop_center crop_guided notes labels objonly box_size box_draw; do
  echo "== $form"
  $G --form $form > logs/gpt_$form.log 2>&1
done
python -m src.qa.gpt_cost data/hdepic/results --price_in 2.5 --price_out 10
