#!/usr/bin/env bash
# CPU analyses behind the result tables. They need only the saved answers and features, so they run from the data
# release alone (see DATA.md); no video, no GPU, no API key. Output: results/, logs/analysis_*.log
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p logs results/linking results/routing
run() { name=$1; shift; echo "== $name"; "$@" > logs/analysis_$name.log 2>&1; }

# HD-EPIC
run pilot          python -m src.analysis.analyze_pilot
run route_select   python -m src.routing.route_select --out results/routing/route_select.json
run engine_7b      python -m src.linking.linking_engine --R 20 --seeds 3 --out results/linking/run_7b.json
run engine_objonly python -m src.linking.linking_engine --cloud 7b_objonly --R 20 --seeds 3 \
                     --send_emb data/hdepic/emb_siglip_objonly_s2.npz \
                     --bank data/charades_link/emb_siglip_objonly_bank.npz --out results/linking/run_7b_objonly.json
run engine_labels  python -m src.linking.linking_engine --cloud 7b_labels --R 20 --seeds 3 \
                     --send_emb data/hdepic/emb_labels_s2.npz \
                     --bank data/charades_link/emb_labels_bank.npz --out results/linking/run_7b_labels.json
run cross_video    python -m src.linking.cross_video_link

# Charades / STAR
run gallery        python -m src.linking.gallery_link --upload
run growth         python -m src.linking.growth_link
run label_link     python -m src.linking.label_link_star
run router_star    python -m src.routing.router_star
for p in 0 8; do
  run star_prior$p python -m src.linking.linking_engine_star --prior $p --out results/linking/star_prior$p.json
  for m in dinov2 clip; do
    run star_${m}_prior$p python -m src.linking.linking_engine_star --prior $p --cloud_model $m \
      --out results/linking/star_${m}_prior$p.json
  done
  for v in c40 c25 c15 low40; do
    run star_${v}_prior$p python -m src.linking.linking_engine_star --prior $p --send_view $v --taus 0.25 --seeds 2 \
      --out results/linking/star_view_${v}_prior$p.json
  done
done
run star_run       python -m src.analysis.analyze_star_run star
echo "done; tables in logs/analysis_*.log"
