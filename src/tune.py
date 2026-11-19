"""Hyper-parameter study with an IDENTICAL budget for every model: N random-search trials,
same RNG seed, trained with seed 0, selected on VALIDATION top-1 (test never used here).
  python -m src.tune --model transformer --trials 6
Writes experiments/tuning/<model>.csv and configs/best_<model>.yaml."""
import argparse
import numpy as np, pandas as pd, yaml
from src.utils import ROOT, load_cfg, deep_update
from src.train import run

SPACES = {
    "baseline":    {"model.C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "gru":         {"train.lr": [3e-4, 1e-3, 3e-3], "model.hidden": [64, 128, 256], "model.layers": [1, 2],
                    "model.dropout": [0.2, 0.3, 0.5], "train.weight_decay": [0.0, 0.01, 0.05]},
    "transformer": {"train.lr": [3e-4, 1e-3, 3e-3], "model.d_model": [64, 128, 192], "model.layers": [2, 3, 4],
                    "model.dropout": [0.1, 0.2, 0.3], "train.weight_decay": [0.0, 0.01, 0.05]},
    "stgcn":       {"train.lr": [3e-4, 1e-3, 3e-3], "model.width": [32, 64, 96], "model.blocks": [4, 6],
                    "model.dropout": [0.1, 0.2, 0.3], "train.weight_decay": [0.0, 0.01, 0.05]},
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=SPACES); ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=None, help="override epochs for tuning only")
    a = ap.parse_args()
    rng = np.random.default_rng(1234)                       # same stream => same budget for all models
    space, rows = SPACES[a.model], []
    for t in range(a.trials):
        hp = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        ov = dict(hp)
        if a.epochs: ov["train.epochs"] = a.epochs
        cfg = load_cfg(ROOT / "configs" / f"{a.model}.yaml", ov)
        r = run(cfg, 0, ROOT / "experiments" / "tuning" / a.model / f"trial{t}", eval_test=False)
        rows.append({**hp, "val_top1": r["val"]["top1"], "val_macro_f1": r["val"]["macro_f1"],
                     "train_time_s": r["cost"]["train_time_s"]})
        pd.DataFrame(rows).to_csv(ROOT / "experiments" / "tuning" / f"{a.model}.csv", index=False)
    best = max(rows, key=lambda r: r["val_top1"])
    best_hp = {k: best[k] for k in space}
    cfg = load_cfg(ROOT / "configs" / f"{a.model}.yaml", best_hp)
    yaml.safe_dump({"data": cfg["data"], "train": cfg["train"], "model": cfg["model"]},
                   open(ROOT / "configs" / f"best_{a.model}.yaml", "w"))
    print("best:", best_hp, "-> configs/best_%s.yaml" % a.model)
