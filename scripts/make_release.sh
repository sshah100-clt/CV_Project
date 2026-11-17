#!/usr/bin/env bash
# Pack the derived data (model answers, features, episode lists) into archives for a GitHub release.
# No video frames and no dataset annotations are included: those come from the official dataset pages.
# The 32B household-profile outputs (guesses about real participants) are left out on purpose.
#   bash scripts/make_release.sh [root with data/, default: repo root] [output dir, default: release/]
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(cd "${1:-.}" && pwd)
OUT=$(mkdir -p "${2:-release}" && cd "${2:-release}" && pwd)
pack() {  # pack <archive name> <paths relative to ROOT...>
  name=$1; shift
  (cd "$ROOT" && tar -czhf "$OUT/$name" --exclude "._*" "$@")
  echo "$name  $(du -h "$OUT/$name" | cut -f1)"
}

cd "$ROOT"
pack hdepic_results.tar.gz \
  data/hdepic/results \
  $(ls data/hdepic/privacy/*.json data/hdepic/privacy/emb_*.npz data/hdepic/privacy/{gate,ground,notes,vlmlink}_*.jsonl)
pack hdepic_features.tar.gz data/hdepic/emb_*.npz
pack charades_link.tar.gz \
  data/charades_link/{frames.json,near_duplicates.json,person_present.json,person_area.npy,episodes_vlm.json} \
  data/charades_link/emb_*.npy data/charades_link/emb_*_bank.npz data/charades_link/vlm_*.jsonl
pack star_results.tar.gz data/star/results/star_*.jsonl
(cd "$OUT" && shasum -a 256 *.tar.gz > SHA256SUMS.txt)
echo "written to $OUT"
