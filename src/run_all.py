"""Run studies for several seeds. Resumable (skips runs that already have metrics.json), so the
work can be split between machines:  --only transformer   or   --study aug --only stgcn
  python -m src.run_all --study main --seeds 0 1 2"""
import argparse
from src.utils import ROOT, load_cfg
from src.studies import STUDIES, cfg_path
from src.train import run

ap = argparse.ArgumentParser()
ap.add_argument("--study", nargs="+", default=["main"], choices=list(STUDIES))
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--only", nargs="*", help="substring filter on run names")
ap.add_argument("--epochs", type=int, default=None, help="smoke tests only")
a = ap.parse_args()
for study in a.study:
    for name, model, ov in STUDIES[study]:
        if a.only and not any(o in name for o in a.only):
            continue
        for seed in a.seeds:
            out = ROOT / "experiments" / study / name / f"seed{seed}"
            if (out / "metrics.json").exists():
                continue
            ov2 = dict(ov)
            if a.epochs: ov2["train.epochs"] = a.epochs
            run(load_cfg(cfg_path(model), ov2), seed, out)
