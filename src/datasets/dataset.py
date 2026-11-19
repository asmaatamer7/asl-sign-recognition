"""Loading, normalisation, augmentation and feature building for pose sequences.

Raw keypoints (T, 75, 3) -> keep 49 joints (7 upper-body pose + 21 left hand + 21 right hand)
-> normalise by shoulder centre / shoulder width -> (optional) augment -> (T, 49, C) features.
"""
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

POSE_KEEP = [0, 11, 12, 13, 14, 15, 16]          # nose, shoulders, elbows, wrists
RAW_IDX = POSE_KEEP + list(range(33, 54)) + list(range(54, 75))
J = len(RAW_IDX)                                  # 49
BODY, LHAND, RHAND = slice(0, 7), slice(7, 28), slice(28, 49)
JOINT_GROUPS = {"all": [BODY, LHAND, RHAND], "hands": [LHAND, RHAND], "body": [BODY]}


def preprocess(raw):
    """raw (T,75,3) -> pre (T,49,3) normalised, present (T,49) bool (joint was detected)."""
    kp = raw[:, RAW_IDX, :].astype(np.float32)
    present = np.abs(kp).sum(-1) > 0
    ls, rs = kp[:, 1, :2], kp[:, 2, :2]
    ok = present[:, 1] & present[:, 2]
    if ok.any():
        center = np.median((ls[ok] + rs[ok]) / 2, axis=0)
        scale = float(np.median(np.linalg.norm(ls[ok] - rs[ok], axis=1)))
    else:
        center, scale = np.array([0.5, 0.5], np.float32), 0.25
    scale = max(scale, 1e-3)
    kp[..., :2] = (kp[..., :2] - center) / scale
    kp[..., 2] = kp[..., 2] / scale
    kp *= present[..., None]
    return kp.astype(np.float32), present


def augment(pre, present, aug, rng):
    """Train-time augmentation. aug: dict(rotate deg, scale, shift, noise, tcrop). 0/None = off."""
    pre, present = pre.copy(), present.copy()
    T = pre.shape[0]
    tcrop = aug.get("tcrop") or 0
    if tcrop:                                      # random temporal crop, resampled back to T
        frac = rng.uniform(tcrop, 1.0)
        L = max(8, int(T * frac)); s = rng.integers(0, T - L + 1)
        idx = np.linspace(s, s + L - 1, T).astype(int)
        pre, present = pre[idx], present[idx]
    rot = aug.get("rotate") or 0
    if rot:
        th = np.deg2rad(rng.uniform(-rot, rot)); c, s_ = np.cos(th), np.sin(th)
        x, y = pre[..., 0].copy(), pre[..., 1].copy()
        pre[..., 0], pre[..., 1] = c * x - s_ * y, s_ * x + c * y
    sc = aug.get("scale") or 0
    if sc:
        pre *= rng.uniform(1 - sc, 1 + sc)
    sh = aug.get("shift") or 0
    if sh:
        pre[..., :2] += rng.uniform(-sh, sh, size=(1, 1, 2)).astype(np.float32)
    nz = aug.get("noise") or 0
    if nz:
        pre += rng.normal(0, nz, pre.shape).astype(np.float32)
    pre *= present[..., None]                      # undetected joints stay exactly zero
    return pre, present


def build_features(pre, present, dcfg):
    """-> (T, J, C) float32. Applies joint-group masking (ablation) and representation options."""
    keep = np.zeros(J, bool)
    for sl in JOINT_GROUPS[dcfg.get("joints", "all")]:
        keep[sl] = True
    f = pre[..., :3 if dcfg.get("use_z", False) else 2] * (present & keep)[..., None]
    if dcfg.get("use_vel", False):
        v = np.zeros_like(f); v[1:] = f[1:] - f[:-1]
        v *= (present & keep)[..., None]
        f = np.concatenate([f, 5.0 * v], axis=-1)
    return f.astype(np.float32)


def n_channels(dcfg):
    c = 3 if dcfg.get("use_z", False) else 2
    return c * 2 if dcfg.get("use_vel", False) else c


def sequence_to_tensor(raw, dcfg):
    """Used by the demo: raw keypoints (T,75,3) -> (1,T,J,C) tensor."""
    pre, present = preprocess(raw)
    return torch.from_numpy(build_features(pre, present, dcfg))[None]


def make_label_map(csv):
    df = pd.read_csv(csv)
    return {g: i for i, g in enumerate(sorted(df[df.split == "train"].gloss.unique()))}


class SignDataset(Dataset):
    def __init__(self, csv, split, dcfg, train=False, label_map=None, seed=0, limit=None):
        df = pd.read_csv(csv)
        df = df[df.split == split].reset_index(drop=True)
        if limit:
            df = df.iloc[:limit]
        self.df, self.dcfg, self.train = df, dcfg, train
        self.label_map = label_map or make_label_map(csv)
        self.y = df.gloss.map(self.label_map).to_numpy().astype(np.int64)
        self.pre, self.present = [], []
        for p in df.kp_path:
            pre, pr = preprocess(np.load(p)); self.pre.append(pre); self.present.append(pr)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        pre, pr = self.pre[i], self.present[i]
        if self.train and self.dcfg.get("aug"):
            pre, pr = augment(pre, pr, self.dcfg["aug"], self.rng)
        return torch.from_numpy(build_features(pre, pr, self.dcfg)), int(self.y[i])

    def tabular(self):
        """Hand-crafted features for the classical baseline: mean/std/min/max over time and the
        mean of 3 temporal thirds, per joint coordinate (no augmentation)."""
        X = []
        for pre, pr in zip(self.pre, self.present):
            f = build_features(pre, pr, self.dcfg).reshape(pre.shape[0], -1)
            T = f.shape[0]
            thirds = [f[k * T // 3:(k + 1) * T // 3].mean(0) for k in range(3)]
            X.append(np.concatenate([f.mean(0), f.std(0), f.min(0), f.max(0)] + thirds))
        return np.stack(X)
