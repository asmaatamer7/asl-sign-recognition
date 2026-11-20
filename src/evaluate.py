"""Shared evaluation for ALL models (fair-comparison requirement).
Metrics: top-1, top-5, macro-F1, per-class precision/recall/F1, bootstrap 95% CI.
CLI re-evaluates a finished run:  python -m src.evaluate --run experiments/main/stgcn/seed0"""
import argparse, json
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


def compute_metrics(y, probs, n_boot=500, seed=0):
    K = probs.shape[1]
    pred = probs.argmax(1)
    top5 = np.argsort(-probs, 1)[:, :5]
    hit1, hit5 = pred == y, (top5 == y[:, None]).any(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(K), zero_division=0)
    rng = np.random.default_rng(seed)
    b1, bf = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        b1.append(hit1[i].mean()); bf.append(f1_score(y[i], pred[i], average="macro", zero_division=0))
    return {"top1": float(hit1.mean()), "top5": float(hit5.mean()),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "top1_ci95": [float(np.percentile(b1, 2.5)), float(np.percentile(b1, 97.5))],
            "macro_f1_ci95": [float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))],
            "per_class": {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist()}, "n": int(len(y))}


def predict(model, ds, batch_size=128):
    """Softmax probabilities for a SignDataset (no augmentation, original order)."""
    import tensorflow as tf
    P, Y = [], []
    for x, y in ds.batches(batch_size):
        P.append(tf.nn.softmax(model(x, training=False)).numpy()); Y.append(y)
    return np.concatenate(P), np.concatenate(Y)


if __name__ == "__main__":
    from src.train import evaluate_run
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--split", default="test")
    a = ap.parse_args()
    m = evaluate_run(a.run, a.split)
    print(json.dumps({k: v for k, v in m.items() if k != "per_class"}, indent=1))
