"""Failure analysis of one model: confusion pairs, per-class / per-signer accuracy, hand-detection
quality, and a table of concrete failure examples (open the listed videos and write the likely cause).
  python -m src.error_analysis --model stgcn --seed 0"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from src.utils import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="stgcn"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--study", default="main"); ap.add_argument("--csv", default="subset_kp.csv")
a = ap.parse_args()
run = ROOT / "experiments" / a.study / a.model / f"seed{a.seed}"
out = ROOT / "experiments" / "error_analysis" / a.model; out.mkdir(parents=True, exist_ok=True)
z = np.load(run / "test_preds.npz"); P, y = z["probs"], z["y"]
df = pd.read_csv(a.csv); df = df[df.split == "test"].reset_index(drop=True)
classes = sorted(pd.read_csv(a.csv).query("split=='train'").gloss.unique())
assert len(df) == len(y), "test csv does not match saved predictions"
pred = P.argmax(1); ok = pred == y
df["true"] = [classes[i] for i in y]; df["pred"] = [classes[i] for i in pred]
df["p_pred"] = P.max(1); df["p_true"] = P[np.arange(len(y)), y]
df["rank_true"] = (P > df.p_true.to_numpy()[:, None]).sum(1) + 1
# hand-detection quality straight from the raw keypoints (are the hands even tracked?)
def det(p):
    k = np.load(p); return (np.abs(k[:, 33:54]).sum((1, 2)) > 0).mean(), (np.abs(k[:, 54:75]).sum((1, 2)) > 0).mean()
d = np.array([det(p) for p in df.kp_path]); df["left_det"], df["right_det"] = d[:, 0], d[:, 1]
df["min_hand_det"] = np.maximum(d[:, 0], d[:, 1])   # detection rate of the best-tracked hand
df["correct"] = ok

# 1. confusion pairs
err = df[~ok]
pairs = err.groupby(["true", "pred"]).size().sort_values(ascending=False).reset_index(name="count")
pairs.to_csv(out / "top_confused_pairs.csv", index=False)
# 2. per class / per signer / per hand-quality
df.groupby("true").correct.mean().sort_values().to_csv(out / "per_class_accuracy.csv")
df.groupby("signer").correct.agg(["mean", "count"]).sort_values("mean").to_csv(out / "per_signer_accuracy.csv")
df["hand_quality"] = pd.cut(df.min_hand_det, [-0.01, 0.5, 0.9, 1.0], labels=["<50% frames", "50-90%", ">90%"])
q = df.groupby("hand_quality", observed=True).correct.agg(["mean", "count"]); q.to_csv(out / "accuracy_by_hand_tracking.csv"); print(q)
# 3. rule-based error typing -> Table 6
pair_n = {(r.true, r.pred): r["count"] for _, r in pairs.iterrows()}
def etype(r):
    if r.min_hand_det < 0.5: return "hand-tracking failure (hands detected in <50% frames)"
    if r.rank_true <= 5 and pair_n.get((r.true, r.pred), 0) >= 2: return "recurrent confusable-sign pair"
    if r.rank_true <= 5: return "near miss (true class in top-5)"
    return "other / unexplained (inspect video)"
err = df[~ok].copy(); err["error_type"] = err.apply(etype, axis=1)
t6 = err.groupby("error_type").agg(count=("true", "size"), example_video=("video_file", "first"),
                                    example=("true", lambda s: ""), ).reset_index()
t6["example"] = [f"{err[err.error_type == e].true.iloc[0]} → {err[err.error_type == e].pred.iloc[0]}" for e in t6.error_type]
t6["likely_cause (WRITE after watching videos)"] = ""
t6.to_csv(out / "table6_error_types.csv", index=False); print(t6.to_markdown(index=False))
err.sort_values("p_pred", ascending=False).head(40)[["video_file", "signer", "true", "pred", "p_pred", "p_true",
    "rank_true", "left_det", "right_det", "error_type"]].to_csv(out / "failure_examples.csv", index=False)
# 4. confusion matrix figure (most confused classes only, readable)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
worst = list(dict.fromkeys(list(pairs.true[:12]) + list(pairs.pred[:12])))[:20]
cm = pd.crosstab(df.true, df.pred).reindex(index=worst, columns=worst, fill_value=0)
plt.figure(figsize=(8, 7)); plt.imshow(cm.values, cmap="Blues")
plt.xticks(range(len(worst)), worst, rotation=90, fontsize=7); plt.yticks(range(len(worst)), worst, fontsize=7)
plt.xlabel("predicted"); plt.ylabel("true"); plt.colorbar(); plt.title(f"Confusions among most-confused signs ({a.model})")
plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=200)
print("top confused pairs:\n", pairs.head(10).to_string(index=False)); print("saved to", out)
