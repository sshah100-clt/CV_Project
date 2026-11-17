# Data

All code runs from the repository root and reads and writes under `data/` and `results/` (both git-ignored).

## 1. Datasets (download from the official pages)

| dataset | what we use | where it goes |
|---|---|---|
| [Charades](https://prior.allenai.org/projects/charades) | 480p videos, `Charades_v1_{train,test}.csv`, `Charades_v1_classes.txt` | `data/charades/videos/<id>.mp4`, `data/charades/*.csv` |
| [STAR](https://bobbywu.com/STAR/) | `STAR_train.json`, `STAR_val.json` (questions on Charades videos) | `data/star/` |
| [HD-EPIC](https://hd-epic.github.io/) | videos of P01-P09 and the VQA benchmark questions | `python -m src.data.hdepic --data data download` (videos + questions) or `... questions` (questions only) puts them in `data/hdepic/` |

Each dataset has its own license and terms of use; read them before downloading. We do not redistribute any video,
frame or annotation file.

Detector models (`scripts/setup.sh` downloads them): YuNet face detector and YOLO11n-seg, in `data/models/`.

## 2. Derived data (GitHub release)

The release holds what our runs produced: model answers, image and text features, and episode lists. Unpack every
archive in the repository root; the paths inside match the layout below. With the release alone,
`scripts/run_analysis.sh` reproduces the result tables on a CPU.

| archive | contents |
|---|---|
| `hdepic_results.tar.gz` | `data/hdepic/results/`: one JSON line per question for every QA run (`qa_<model>_<form>_<tag>.jsonl`, with `.meta.json` settings for GPT runs), device notes, grounding boxes, detector labels, planner replies, scene descriptions, linking per form. `data/hdepic/privacy/`: upload lists, gate decisions, crop boxes, notes, features of every upload form, VLM-linker replies, linking results |
| `hdepic_features.tar.gz` | `data/hdepic/emb_*.npz`: SigLIP / DINOv2 / CLIP features of every 2nd second of the 43 videos (face-blurred frames), of narrower views, of objects-only images, and label vectors |
| `charades_link.tar.gz` | `data/charades_link/`: frame index (`frames.json`: video, frame, subject, scene, faces found), near-duplicate videos, person-visible flags and areas, VLM episodes, features of every upload variant, VLM-linker replies |
| `star_results.tar.gz` | `data/star/results/`: STAR answers of Qwen2.5-VL 3B / 7B / 32B on raw, face-blurred and link-break frames |

`SHA256SUMS.txt` lists the checksum of each archive.

Not included on purpose: video frames and crops (dataset licenses), dataset annotations (download them from the
official pages), and the 32B model's per-household profile guesses and per-upload sensitive-content judgements
(they are guesses about real participants).

### File names

`<tag>` = `f32_pt30_dense_dev`: at most 32 frames per question, at most 30 questions per type, exact-time frames in
short windows, dev split (a fixed 30% of the videos, chosen by a hash of the video name).

QA forms: `raw` (device, raw frames), `blur` (face-blurred frames), `cropguided` / `cropcenter` (crops),
`cropfull` (whole frames at 756 px), `notes` (device notes, text only), `labels` (detector labels, text only),
`objonly` (objects-only images), `gated` (frames after the on-device gate), `blind` (question only),
`boxsize` / `boxdraw` (GPT box-format checks, box questions only).

## 3. Layout

```
data/
  models/            face_detection_yunet_2023mar.onnx, yolo11n-seg.pt
  charades/          Charades_v1_*.csv, videos/
  charades_link/     frames.json, raw/ blur/ ... (frames per upload variant), mask/, emb_*.npy, vlm_*.jsonl
  star/              STAR_val.json, frames/<variant>/, results/
  hdepic/            vqa/, videos/, frames/{raw,blur,...}/, emb_*.npz, results/, privacy/, labels/
results/             analysis outputs (linking/, routing/, star/)
```
