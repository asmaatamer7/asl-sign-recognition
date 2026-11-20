# Isolated Sign-Language Recognition on ASL Citizen (pose-based)

**Course:** Deep Learning AI403, Pharos University in Alexandria, Fall 2026
**Team:** <TEAM NAME> — <member 1 (ID)>, <member 2 (ID)>, <member 3 (ID)>, <member 4 (ID)>
**Idea chosen:** #15 Isolated sign-language recognition (video / pose sequence → word)

## 1. Task and best result
Classify a short video of one ASL sign into one of **100 signs** (random subset of ASL Citizen, official
signer-independent split). Videos → MediaPipe Holistic keypoints (7 body + 2×21 hand joints) → sequence models. **Framework: TensorFlow / Keras.**
Best result: **<MODEL> — top-1 XX.X ± X.X %, top-5 XX.X %, macro-F1 XX.X %** (mean ± SD, 3 seeds, test set). *(fill in)*

## 2. Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
## 3. Data (no raw data is stored in the repo)
Download `ASL_Citizen.zip` (42.8 GB, v1.0) from https://www.microsoft.com/en-us/download/details.aspx?id=105253
```bash
python data/make_subset.py --zip ASL_Citizen.zip --n-signs 100          # -> subset.csv + videos_subset/
python data/extract_keypoints.py --workers 4                            # -> keypoints/*.npy + subset_kp.csv
```
Shared drive with the extracted `keypoints/` and `subset_kp.csv`: **<LINK>**.  See `data/DATASET.md`.

## 4. Training (every model, one seed)
```bash
python -m src.train --config configs/baseline.yaml    --seed 0 --out experiments/main/baseline/seed0
python -m src.train --config configs/gru.yaml         --seed 0 --out experiments/main/gru/seed0
python -m src.train --config configs/transformer.yaml --seed 0 --out experiments/main/transformer/seed0
python -m src.train --config configs/stgcn.yaml       --seed 0 --out experiments/main/stgcn/seed0
```
### Full experimental pipeline (what produced the report)
```bash
for m in baseline gru transformer stgcn; do python -m src.tune --model $m --trials 6; done   # hyper-parameter study
python -m src.run_all --study main --seeds 0 1 2             # final comparison (uses configs/best_*.yaml)
python -m src.run_all --study aug repr ablation --seeds 0 1 2
python -m src.pretrain --config configs/best_transformer.yaml # self-supervised pre-training (train split only)
python -m src.run_all --study pretrain --seeds 0 1 2
python -m src.error_analysis --model stgcn --seed 0           # failure analysis
python -m src.aggregate                                       # tables 4,5,7 + hypotheses + figures
```
## 5. Evaluation
`python -m src.evaluate --run experiments/main/stgcn/seed0` (the same `src/evaluate.py` is used by all models).
## 6. Demo
`python demo/app.py --run experiments/main/stgcn/seed0 experiments/main/transformer/seed0`
## 7. Hardware and expected runtimes
Keypoint extraction ≈ 20–40 min (CPU, 4 workers) · each deep-model run ≈ 2–6 min on a T4 GPU · full pipeline ≈ 3–4 h.
*Fill in the real hardware and times from `experiments/tables/table7_computational_cost.md`.*
## 8. Credits
ASL Citizen dataset (Desai et al., NeurIPS 2023 D&B) · MediaPipe Holistic (Google) · ST-GCN idea (Yan et al., AAAI 2018) · TensorFlow/Keras ·
Transformer (Vaswani et al., 2017). **Written by us:** multi-head self-attention (`src/models/transformer.py`),
graph convolution (`src/models/stgcn.py`), all training/evaluation code. No external AI APIs are used for predictions.
## Repository layout
`configs/` one config per model (+ `best_*.yaml` from tuning) · `data/` subset + extraction scripts · `src/datasets/` loaders, normalisation,
augmentation · `src/models/` one file per model family · `src/train.py`, `src/evaluate.py` shared · `experiments/` logs/tables/figures ·
`demo/` Gradio app · `report/` report + slides · `contributions/` individual logs · `tests/` synthetic smoke test.
