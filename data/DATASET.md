# Dataset documentation (fill the <...> after running make_subset.py)
| Field | Value |
|---|---|
| Name / version | ASL Citizen v1.0 |
| Source | https://www.microsoft.com/en-us/download/details.aspx?id=105253 (Microsoft Download Center; project page on Microsoft Research) |
| Published | 15/10/2024 (download page) |
| Size | 42.8 GB zip; ≈83k videos, 2,731 signs, ≈52 signers (full dataset) |
| Licence | <COPY THE LICENCE TEXT FROM THE PROJECT PAGE / DOWNLOAD AGREEMENT> |
| Paper | Desai et al., "ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition", NeurIPS 2023 Datasets & Benchmarks |
| Our subset | 100 signs sampled at random (seed 0) among signs present in train/val/test; all their videos |
| Split | Official ASL Citizen train/val/test CSVs = **signer-independent**; verified: 0 shared signers (make_subset.py asserts it) |
| Subset sizes | train <N> videos/<S> signers · val <N>/<S> · test <N>/<S> |
| Preprocessing | MediaPipe Holistic (model_complexity=0), every 2nd frame, resampled to 64 frames; 7 pose + 21+21 hand joints; centre = shoulder midpoint, scale = shoulder width; undetected joints = 0 |
| Test-set policy | Test used only for the final evaluation of the best-on-validation checkpoint; tuning uses val only |
