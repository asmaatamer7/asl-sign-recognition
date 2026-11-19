"""Definition of every experiment of the project. Each entry: study -> list of runs.
A run = (name, base model config, overrides). Output: experiments/<study>/<name>/seed<k>/"""
from src.utils import ROOT

MODELS = ["baseline", "gru", "transformer", "stgcn"]
NONE_AUG = {"data.aug": {"rotate": 0, "scale": 0, "shift": 0, "noise": 0, "tcrop": 0}}
GEOM = {"data.aug": {"rotate": 15, "scale": 0.15, "shift": 0.1, "noise": 0.01, "tcrop": 0}}
TEMP = {"data.aug": {"rotate": 0, "scale": 0, "shift": 0, "noise": 0, "tcrop": 0.7}}


def cfg_path(model):
    best = ROOT / "configs" / f"best_{model}.yaml"
    return best if best.exists() else ROOT / "configs" / f"{model}.yaml"   # tuned config if tuning was run


def _runs():
    S = {"main": [(m, m, {}) for m in MODELS]}
    deep = ["gru", "transformer", "stgcn"]
    S["aug"] = [(f"{m}_{v}", m, o) for m in deep for v, o in
                [("none", NONE_AUG), ("geometric", GEOM), ("temporal", TEMP), ("all", {})]]
    S["repr"] = [(f"{m}_{v}", m, o) for m in ["transformer", "stgcn"] for v, o in
                 [("xy", {}), ("xyz", {"data.use_z": True}), ("xy_vel", {"data.use_vel": True})]]
    S["ablation"] = ([("transformer_no_attention", "transformer", {"model.attn": False})] +
                     [(f"{m}_{v}", m, {"data.joints": v}) for m in ["transformer", "stgcn"] for v in ["hands", "body"]])
    S["pretrain"] = [("transformer_ssl", "transformer", {"train.init_from": "checkpoints/transformer_ssl.pt"})]
    return S


STUDIES = _runs()
