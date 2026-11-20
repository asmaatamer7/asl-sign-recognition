
"""Hyper-parameter tuning using validation performance only.
The test set is never used during tuning.
"""

import argparse
import numpy as np
import pandas as pd
import yaml

from src.utils import ROOT, load_cfg
from src.train import run


SPACES = {
    "baseline": {
        "model.C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    },
    "gru": {
        "train.lr": [3e-4, 1e-3, 3e-3],
        "model.hidden": [64, 128, 256],
        "model.layers": [1, 2],
        "model.dropout": [0.2, 0.3, 0.5],
        "train.weight_decay": [0.0, 0.01, 0.05]
    },
    "transformer": {
        "train.lr": [3e-4, 1e-3, 3e-3],
        "model.d_model": [64, 128, 192],
        "model.layers": [2, 3, 4],
        "model.dropout": [0.1, 0.2, 0.3],
        "train.weight_decay": [0.0, 0.01, 0.05]
    },
    "stgcn": {
        "train.lr": [3e-4, 1e-3, 3e-3],
        "model.width": [32, 64, 96],
        "model.blocks": [4, 6],
        "model.dropout": [0.1, 0.2, 0.3],
        "train.weight_decay": [0.0, 0.01, 0.05]
    }
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SPACES)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(1234)
    space = SPACES[args.model]
    rows = []

    # Test all six Baseline C values without repetition.
    if args.model == "baseline":
        if args.trials > len(space["model.C"]):
            raise ValueError("Baseline supports at most 6 unique trials.")

        baseline_values = rng.permutation(space["model.C"])
    else:
        baseline_values = None

    for trial in range(args.trials):

        if baseline_values is not None:
            hp = {"model.C": float(baseline_values[trial])}
        else:
            hp = {
                key: values[int(rng.integers(len(values)))]
                for key, values in space.items()
            }

        overrides = dict(hp)

        if args.epochs is not None:
            overrides["train.epochs"] = args.epochs

        cfg = load_cfg(
            ROOT / "configs" / f"{args.model}.yaml",
            overrides
        )

        output_dir = (
            ROOT / "experiments" / "tuning" /
            args.model / f"trial{trial}"
        )

        result = run(
            cfg,
            0,
            output_dir,
            eval_test=False
        )

        rows.append({
            **hp,
            "val_top1": result["val"]["top1"],
            "val_macro_f1": result["val"]["macro_f1"],
            "train_time_s": result["cost"]["train_time_s"]
        })

        pd.DataFrame(rows).to_csv(
            ROOT / "experiments" / "tuning" /
            f"{args.model}.csv",
            index=False
        )

    best = max(rows, key=lambda row: row["val_top1"])

    best_hp = {
        key: best[key]
        for key in space
    }

    best_cfg = load_cfg(
        ROOT / "configs" / f"{args.model}.yaml",
        best_hp
    )

    with open(
        ROOT / "configs" / f"best_{args.model}.yaml",
        "w"
    ) as file:
        yaml.safe_dump(
            {
                "data": best_cfg["data"],
                "train": best_cfg["train"],
                "model": best_cfg["model"]
            },
            file,
            sort_keys=False
        )

    print("\nAll trials completed.")
    print("Best hyperparameters:", best_hp)
    print("Best validation top1:", best["val_top1"])
    print("Best validation macro F1:", best["val_macro_f1"])
    print(f"Saved: configs/best_{args.model}.yaml")
