#!/usr/bin/env bash
# One-time setup: Python environment and the two small detector models.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

mkdir -p data/models
[ -f data/models/face_detection_yunet_2023mar.onnx ] || curl -fsSL -o data/models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
[ -f data/models/yolo11n-seg.pt ] || curl -fsSL -o data/models/yolo11n-seg.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
