"""Creates a tiny synthetic dataset in the same format as the real one, to smoke-test the code
without downloading anything:  python tests/make_fake_data.py && bash tests/smoke_test.sh"""
from pathlib import Path
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
out = Path("keypoints"); out.mkdir(exist_ok=True)
K, rows = 12, []
protos = rng.normal(0, 1, (K, 4, 75, 3)).astype(np.float32)               # per-class trajectory anchors
signers = {"train": range(0, 8), "val": range(8, 10), "test": range(10, 13)}
for split, sg in signers.items():
    for s in sg:
        for k in range(K):
            for rep in range(2 if split == "train" else 1):
                t = np.linspace(0, 3, 64); base = np.stack([np.interp(t, [0, 1, 2, 3], protos[k, :, j, 0]) for j in range(75)], 1)
                arr = np.zeros((64, 75, 3), np.float32)
                arr[..., 0] = 0.5 + 0.05 * base + rng.normal(0, 0.01, (64, 75)); arr[..., 1] = 0.5 + 0.05 * base[:, ::-1]
                arr[:, 11, :2] = [0.6, 0.5]; arr[:, 12, :2] = [0.4, 0.5]          # shoulders
                if rng.random() < 0.1: arr[:, 54:75] = 0                           # missing hand
                f = out / f"v_{split}_{s}_{k}_{rep}.npy"; np.save(f, arr)
                rows.append(dict(split=split, signer=f"P{s}", gloss=f"SIGN{k:02d}", video_file=f.stem + ".mp4", kp_path=str(f)))
pd.DataFrame(rows).to_csv("subset_kp.csv", index=False); print("fake data:", len(rows), "samples")
