# <Title>: Pose-Based Isolated Sign-Language Recognition on ASL Citizen
Team <name> · Deep Learning AI403 · Fall 2026   (target 15–25 pages; every claim backed by a paper or by OUR tables)

> How to use: sections marked **[AUTO]** take their numbers from `experiments/tables/*.md` after `python -m src.aggregate`.
> Sections marked **[WRITE]** need your own words. Delete these hints. Cite reliable sources for every technical decision.

## 1. Abstract [WRITE, last]
Task, data, 4 models, best result (mean ± SD), main finding on each hypothesis, one limitation. ≤200 words.
## 2. Introduction [WRITE]
Sign languages, why isolated sign recognition, why pose keypoints instead of raw video, contributions (bullet list).
## 3. Problem Statement [WRITE]
Input: video of one sign → 64×49×C keypoint tensor. Output: one of 100 classes. Metrics. **Hypotheses (state before results):**
- H1: Transformer outperforms BiGRU. · H2: ST-GCN reaches similar accuracy with fewer parameters.
- H3: Pose augmentation improves generalisation to unseen signers. · H4: Hand joints carry more information than body joints.
## 4. Motivation [WRITE]  Accessibility, communication barriers, low-cost pose input, privacy (no raw faces stored).
## 5. Related Work [WRITE]
ASL Citizen (Desai et al., NeurIPS D&B 2023); WLASL (Li et al., WACV 2020, Pose-TGCN); SPOTER (Boháček & Hrúz, 2022); ST-GCN (Yan et al., AAAI 2018);
Transformer (Vaswani et al., 2017); GRU (Cho et al., 2014); MediaPipe (Lugaresi et al., 2019); masked-modelling pre-training (e.g. BERT/MAE). *Verify every reference before citing.*
## 6. Dataset [AUTO + WRITE]  → **Table 1** (see data/DATASET.md): dataset, source, size, format, classes, licence, role.
Signer-independent split, #videos/#signers per split, leakage check, why a 100-sign subset (compute budget).
## 7. Data Preprocessing [WRITE]  MediaPipe Holistic, frame step 2, 64-frame resampling, 49 joints, shoulder normalisation, missing-joint handling, augmentation list with parameters.
## 8. Methodology [WRITE]  Protocol: same split/preprocessing/tuning budget (6 random-search trials, val only), 3 seeds, test used once; metrics; bootstrap CI.
## 9. Baseline [WRITE]  Mean/std/min/max + temporal-thirds features → standardised logistic regression.
## 10. Proposed / Advanced Models [WRITE] → **Table 2** (model, family, input, main layers, parameters, pretrained?)
Classic: BiGRU. Modern: Transformer (self-attention written by us) + masked-frame SSL pre-training. Advanced: ST-GCN (graph convolution written by us). Equations/diagrams.
## 11. Experimental Setup [WRITE]  Hardware, framework versions, seeds, early stopping, scheduler.
## 12. Hyperparameters [AUTO] → **Table 3** + `figures/tuning_curves.png` (search spaces from `src/tune.py`).
## 13. Results [AUTO] → **Table 4** (`table4_main_results.md`, mean ± SD) + per-class F1 (`per_class_f1_*.csv`) + `figures/main_comparison.png`. Also data-strategy studies (augmentation, representation).
## 14. Model Comparison [AUTO + WRITE]  `hypotheses.md` + **Table 7** (`table7_computational_cost.md`). Verdict on each hypothesis; remember n=3 seeds → say "inconclusive" when p ≥ 0.05. Pretrained-vs-scratch result.
## 15. Ablation Study [AUTO + WRITE] → **Table 5** (`table5_ablation_and_studies.md`): no attention, hands-only, body-only, no augmentation, no pre-training. Interpret each Δ.
## 16. Error Analysis [AUTO + WRITE] → **Table 6** (`error_analysis/<model>/table6_error_types.csv`): fill "likely cause" after WATCHING the videos in `failure_examples.csv`. Include the confusion figure, per-signer and hand-tracking-quality results.
## 17. Discussion [WRITE]  Why did the winner win? Link to results, not guesses.
## 18. Limitations [WRITE]  100-sign subset, 1 dataset, MediaPipe failures, 3 seeds, webcam domain gap in the demo, no facial/non-manual features.
## 19. Ethical Considerations [WRITE]  Deaf community involvement, not a replacement for interpreters, dataset licence/consent, bias (signer demographics), privacy.
## 20. Conclusion [WRITE]
## 21. Future Work [WRITE]  Full 2,731 signs, RGB+pose fusion, continuous signing, face keypoints, larger pre-training.
## 22. References [WRITE]
