import copy, random
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[1]


def deep_update(base, new):
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_cfg(path, overrides=None):
    """base.yaml  <-  experiment yaml  <-  overrides {'train.lr': 0.003, ...}"""
    base = yaml.safe_load(open(ROOT / "configs" / "base.yaml"))
    cfg = deep_update(base, yaml.safe_load(open(path)) or {})
    for k, v in (overrides or {}).items():
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return cfg


def parse_set(items):
    """['train.lr=0.003', 'model.attn=false'] -> dict with yaml-typed values"""
    return {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in (items or []))}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)          # python + numpy + TensorFlow


def save_weights(model, path, cfg=None, classes=None, in_channels=None):
    """Framework-version-proof checkpoint: plain numpy arrays (+ config) in one .npz file."""
    import json
    extra = {}
    if cfg is not None:
        extra = {"cfg": json.dumps(cfg), "classes": json.dumps(classes), "in_channels": in_channels}
    np.savez(path, **{f"w{i}": w for i, w in enumerate(model.get_weights())}, **extra)


def load_weights(model, path):
    z = np.load(path, allow_pickle=False)
    ws = [z[f"w{i}"] for i in range(sum(k.startswith("w") and k[1:].isdigit() for k in z.files))]
    model.set_weights(ws)
