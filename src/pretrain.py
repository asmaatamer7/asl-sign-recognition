"""Self-supervised pre-training of the Transformer encoder (masked-frame reconstruction).
Uses ONLY the training split (no labels, no val/test) -> no leakage.
  python -m src.pretrain --config configs/best_transformer.yaml --out checkpoints/transformer_ssl.pt
Then: python -m src.train --config ... --set train.init_from=checkpoints/transformer_ssl.pt"""
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import load_cfg, set_seed
from src.datasets.dataset import SignDataset, make_label_map, n_channels, J
from src.models import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--out", default="checkpoints/transformer_ssl.pt")
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--mask", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
cfg = load_cfg(a.config); set_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
lm = make_label_map(cfg["data"]["csv"]); C = n_channels(cfg["data"])
ds = SignDataset(cfg["data"]["csv"], "train", cfg["data"], True, lm, a.seed)
dl = DataLoader(ds, 64, shuffle=True, drop_last=True)
model = build_model(cfg, len(lm), C).to(dev)
d = cfg["model"]["d_model"]; dec = nn.Linear(d, J * C).to(dev)
opt = torch.optim.AdamW(list(model.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=0.01)
for ep in range(a.epochs):
    tot = 0
    for x, _ in dl:
        x = x.to(dev); B, T = x.shape[:2]
        # mask contiguous chunks of frames (8-frame blocks) so the task is not trivial interpolation
        blocks = torch.rand(B, T // 8, device=dev) < a.mask
        fm = blocks.repeat_interleave(8, 1)
        rec = dec(model.encode(x, fm))
        loss = ((rec - x.flatten(2)) ** 2).mean(-1)[fm].mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    print(f"ep {ep} recon loss {tot / len(dl):.4f}")
torch.save({k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("head")}, a.out)
print("saved", a.out)
