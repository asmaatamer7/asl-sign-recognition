"""Pick N_SIGNS glosses from ASL Citizen and pull ONLY their videos out of the big zip.
Uses the official signer-independent train/val/test splits (no leakage by design).

Usage:  python data/make_subset.py --zip ASL_Citizen.zip --n-signs 100
"""
import argparse, zipfile
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--zip", default="ASL_Citizen.zip")
ap.add_argument("--n-signs", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out-dir", default="videos_subset")
a = ap.parse_args()

out = Path(a.out_dir); out.mkdir(exist_ok=True)
z = zipfile.ZipFile(a.zip)
names = z.namelist()
print("first members:", names[:5])

def find_col(df, *cands):
    low = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"none of {cands} in {list(df.columns)}")

dfs = []
for s in ["train", "val", "test"]:
    member = next(n for n in names if n.lower().endswith(f"{s}.csv"))
    d = pd.read_csv(z.open(member)); d["split"] = s; dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
print(df.columns.tolist()); print(df.head())

GLOSS = find_col(df, "Gloss"); FILE = find_col(df, "Video file", "video_file")
PID = find_col(df, "Participant ID", "participant_id", "signer")
df = df.rename(columns={GLOSS: "gloss", FILE: "video_file", PID: "signer"})

# glosses present in all 3 splits, then a seeded random sample
n_splits = df.groupby("gloss")["split"].nunique()
cand = n_splits[n_splits == 3].index.to_series()
chosen = cand.sample(a.n_signs, random_state=a.seed)
sub = df[df["gloss"].isin(chosen)].copy()

by_name = {Path(n).name: n for n in names if n.lower().endswith(".mp4")}
paths = []
for f in sub["video_file"]:
    dst = out / f
    if not dst.exists():
        dst.write_bytes(z.read(by_name[f]))
    paths.append(str(dst))
sub["video_path"] = paths
sub.to_csv("subset.csv", index=False)

# leakage check (goes into the dataset documentation)
for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
    shared = set(sub[sub.split == x].signer) & set(sub[sub.split == y].signer)
    print(f"shared signers {x}/{y}: {len(shared)}")
    assert not shared, "signer leakage!"
print(sub.groupby("split").agg(videos=("video_file", "count"), signers=("signer", "nunique")))
print("classes:", sub.gloss.nunique())
