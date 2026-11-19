"""subset.csv (video_path) -> keypoints/*.npy + subset_kp.csv (kp_path).
Resumable: already extracted files are skipped. To split the work between several
Colab/Kaggle accounts use --part i --parts n, copy all keypoints/ folders into one,
then run once with --merge to build subset_kp.csv.
Usage: python data/extract_keypoints.py --workers 4 [--part 0 --parts 3]"""
import argparse, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.keypoints import make_holistic, video_to_keypoints

OUT = Path("keypoints"); OUT.mkdir(exist_ok=True)
_H = None

def _init():
    global _H
    _H = make_holistic()

def work(video_path):
    f = OUT / (Path(video_path).stem + ".npy")
    if f.exists():
        return str(f)
    arr = video_to_keypoints(video_path, _H)
    if arr is None:
        return None
    np.save(f, arr)
    return str(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="only build subset_kp.csv from existing .npy")
    a = ap.parse_args()
    df = pd.read_csv("subset.csv")
    todo = df.iloc[a.part::a.parts]
    if not a.merge:
        with Pool(a.workers, initializer=_init) as p:
            for i, _ in enumerate(p.imap(work, todo["video_path"], chunksize=8)):
                if i % 100 == 0:
                    print(f"{i}/{len(todo)}", flush=True)
    df["kp_path"] = [str(OUT / (Path(v).stem + ".npy")) for v in df["video_path"]]
    df = df[[Path(k).exists() for k in df["kp_path"]]]
    df.to_csv("subset_kp.csv", index=False)
    print("kept", len(df), "videos -> subset_kp.csv")
