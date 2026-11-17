# Household linking in edge-to-cloud video question answering

A small vision-language model (VLM) on a home device answers questions about long home videos. When it is unsure,
it can send part of the video to a larger cloud model. This code measures what that costs in privacy. Even with faces
blurred and no account or IP address (for example behind an Oblivious HTTP relay), can the cloud provider tell that
different uploads come from the same home? And does sending less (crops, text, object labels) avoid that and still
help accuracy?

Course project, ITCS 6169/8169 (Computer Vision), UNC Charlotte, Fall 2026.
Group 1: Omckar Savlani, Kevin Richard, Swapn Shah.

## Setting

- **Device**: Qwen2.5-VL-3B, a size meant for an 8 GB edge device such as a Jetson Orin Nano. It sees the raw frames.
- **Cloud**: Qwen2.5-VL-7B / 32B or GPT-4o. It sees only what the device sends: face-blurred frames, crops, the
  device's text notes, object labels or objects-only images, always with the question.
- **Linking test**: the provider compares a new upload with earlier uploads it holds and picks the most similar
  household (nearest neighbour on image features: SigLIP, CLIP, DINOv2, or a trained head). Chance is 1/K for K
  candidate households.
- **Data**: Charades (267 households, every room; STAR questions for 140 of them) and HD-EPIC (9 kitchens, long
  egocentric videos, 5-choice questions).

## Main results

Charades, face-blurred uploads. Rank-1 linking against a gallery of k households (SigLIP; chance 1/k). The provider
holds 6 frames from each household's other videos:

| households | 4 | 10 | 25 | 50 | 100 | 175 |
|---|---|---|---|---|---|---|
| chance | 25.0 | 10.0 | 4.0 | 2.0 | 1.0 | 0.6 |
| linked | 82.6 | 66.1 | 50.9 | 41.6 | 33.9 | 28.6 |

Linking grows as the provider holds more uploads from a household (K = 4, SigLIP): 65.5% with 1 earlier upload,
79.6% with 4, 84.3% with 8, 89.2% with 16. Face blurring lowers linking by about 2 points (DINOv2: 78.8% raw,
76.7% blurred). Both the home alone (person greyed out) and the person alone (home greyed out) link well above chance.

HD-EPIC pilot (P01, 277 dev questions from 11 videos; 95% CI by bootstrap over videos):

| who answers, from what | accuracy |
|---|---|
| 3B on the device, raw frames | 39.0 [34.1, 46.7] |
| 7B, face-blurred frames | 42.6 [38.5, 48.1] |
| 32B, face-blurred frames | 39.4 [32.9, 45.8] |
| 7B, centre crops | 46.6 [40.2, 53.2] |
| 7B, the device's text notes | 31.8 [27.8, 37.0] |
| 7B, objects-only images | 34.7 [30.3, 41.0] |
| GPT-4o (gpt-4o-2024-11-20), face-blurred frames | 38.6 [33.7, 43.1] |
| GPT-4o, question only (no video) | 19.1 [13.2, 25.7] |

In this pilot, no upload form was both useful and hard to link. Accuracy needs room context (crops, frames), and room
context is what identifies the home. Face blurring changed only 1.6% of HD-EPIC frames, and the 12 largest changes were all
false detections. Details and the selective-escalation analysis: `src/routing/route_select.py`.

## Repository

```
src/
  data/      dataset preparation: Charades-Link frames and upload variants, HD-EPIC download / frames / QA,
             image features
  qa/        question answering: STAR, HD-EPIC pilot forms (notes, crops, planner), GPT models, object labels
  linking/   household linking: nearest-neighbour and trained linking models, VLM linking model, gallery and
             growth tests, the linking engine used to replay escalation policies
  routing/   which questions the device should send (held-out rules on HD-EPIC, learned router on STAR)
  privacy/   linking, sensitive-content and household-profile risk of each upload form; on-device gate
  analysis/  summaries of the GPU runs
scripts/     setup and one run script per experiment group
DATA.md      datasets, the data release, and the folder layout
```

Every script runs as a module from the repository root, for example `python -m src.linking.gallery_link`. Each
file starts with a short description and its commands.

## Setup

```
bash scripts/setup.sh          # .venv, packages, face and person detectors
```

Python 3.12+, PyTorch 2.4+. numpy, scikit-learn and scipy are pinned in `requirements.txt` so the CPU analyses give the exact reported numbers. The GPU runs used 4 x 24 GB cards (the 32B model needs about 3 of them).

## Reproduce

From the data release (CPU only): download the archives from the GitHub release and unpack them in the repository
root. Add three small files from the official dataset pages (see DATA.md): the HD-EPIC questions, the Charades
`Charades_v1_{train,test}.csv` (in `data/charades/`) and STAR `STAR_{train,val}.json` (in `data/star/`). Then

```
python -m src.data.hdepic --data data questions    # HD-EPIC questions only (small)
bash scripts/run_analysis.sh
```

From the videos (see DATA.md for the downloads):

```
bash scripts/run_charades.sh        # Charades-Link frames, variants, features, linking tests (CPU / one GPU)
bash scripts/run_star.sh            # VLM linking model and STAR QA with Qwen2.5-VL 3B / 7B / 32B (GPUs)
bash scripts/run_hdepic.sh          # HD-EPIC download, frames, QA forms, linking per form (GPUs)
bash scripts/run_object_labels.sh   # label-only and objects-only uploads (OWLv2)
bash scripts/run_upload_privacy.sh  # privacy of each upload form over 9 homes, on-device gate
bash scripts/run_gpt.sh             # GPT-4o on the same questions (needs OPENAI_API_KEY; paid)
```

All model steps resume where they stopped. The HD-EPIC pilot commands accept `--mock` (random model outputs) to
test the pipeline without a GPU.

## Notes

- HD-EPIC boxes in the questions are `(ymin, xmin, ymax, xmax)` in original pixels, as in the official evaluation
  (hd-epic/hd-epic-vqa-eval). `src/data/hdepic.py` rewrites them to `x1 y1 x2 y2` in the pixels of the frames shown.
  `src/data/check_bbox_order.py` draws both readings for a visual check.
- GPT-4o runs use the official HD-EPIC system prompt (letter only).
- The HD-EPIC pilot is small: one home's dev split. Results on the full test split are future work.

## License

No license has been chosen yet; all rights are reserved by the authors. The datasets keep their own licenses.
