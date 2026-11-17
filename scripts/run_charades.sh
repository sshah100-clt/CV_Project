#!/usr/bin/env bash
# Charades-Link: household linking on face-blurred Charades frames (CPU or one GPU / Apple MPS).
# Needs data/charades/Charades_v1_{train,test,classes}.csv/.txt and data/charades/videos/<id>.mp4 (480p).
# Logs: logs/charades_*.log    Results: results/linking/
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p logs
run() { name=$1; shift; echo "== $name"; "$@" > logs/charades_$name.log 2>&1; }

# 1. frames (3 per video, raw + face-blurred), embeddings of the raw frames, near-duplicate videos
run extract    python -m src.data.charades_link extract
run emb_raw    python -m src.data.embed dinov2 raw
run near_dups  python -m src.data.charades_link near_dups

# 2. person masks and the upload variants
run masks      python -m src.data.upload_variants masks
for v in gray down bgblur silhouette linkbreak personly; do
  run make_$v  python -m src.data.upload_variants make $v
done

# 3. embeddings of every variant
for v in blur gray down bgblur silhouette linkbreak personly; do
  run emb_dinov2_$v python -m src.data.embed dinov2 $v
  run emb_siglip_$v python -m src.data.embed siglip $v
done
run emb_siglip_raw python -m src.data.embed siglip raw
run emb_clip_blur  python -m src.data.embed clip blur
run views          python -m src.data.embed_views charades --data data

# 4. linking experiments
run link_eval      python -m src.linking.link_eval dinov2 blur
run link_eval_room python -m src.linking.link_eval dinov2 blur cross_room
run link_cv        python -m src.linking.link_cv blur linkbreak
run link_metrics   python -m src.linking.link_metrics blur silhouette personly bgblur linkbreak
run link_person    python -m src.linking.link_metrics blur silhouette personly bgblur --person
run link_face      python -m src.linking.link_metrics blur personly --face
run leak_eval      python -m src.linking.leak_eval dinov2 blur
run paired         python -m src.linking.paired_history dinov2 blur
run selector       python -m src.linking.selector_eval dinov2 blur
run action_proxy   python -m src.qa.action_proxy blur bgblur linkbreak
run gallery        python -m src.linking.gallery_link --upload
run growth         python -m src.linking.growth_link
run person         python -m src.linking.person_link
run blur_stats     python -m src.data.blur_stats charades --data data

# 5. fixed episodes for the VLM linking model (run on a GPU with scripts/run_star.sh)
run episodes       python -m src.linking.link_eval dump
echo "done"
