"""Self-supervised pre-training of the Transformer encoder (masked-frame reconstruction).
Uses ONLY the training split (no labels, no val/test) -> no leakage.
  python -m src.pretrain --config configs/best_transformer.yaml --out checkpoints/transformer_ssl.npz
Then the 'pretrain' study fine-tunes from it (train.init_from=checkpoints/transformer_ssl.npz)."""
import argparse
import numpy as np, tensorflow as tf
from src.utils import load_cfg, set_seed, save_weights
from src.datasets.dataset import SignDataset, make_label_map, n_channels, J
from src.models import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--out", default="checkpoints/transformer_ssl.npz")
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--mask", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
cfg = load_cfg(a.config); set_seed(a.seed)
lm = make_label_map(cfg["data"]["csv"]); C = n_channels(cfg["data"])
ds = SignDataset(cfg["data"]["csv"], "train", cfg["data"], True, lm, a.seed)
model = build_model(cfg, len(lm), C)
dec = tf.keras.layers.Dense(J * C); dec(tf.zeros((1, 64, cfg["model"]["d_model"])))
opt = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.01)


@tf.function
def step(x):
    B, T = tf.shape(x)[0], tf.shape(x)[1]
    # mask contiguous 8-frame blocks so the task is not trivial interpolation
    fm = tf.repeat(tf.random.uniform((B, T // 8)) < a.mask, 8, axis=1)
    with tf.GradientTape() as tape:
        rec = dec(model.encode(x, fm, training=True))
        err = tf.reduce_mean((rec - tf.reshape(x, (B, T, -1))) ** 2, -1)
        loss = tf.reduce_sum(err * tf.cast(fm, tf.float32)) / tf.maximum(tf.reduce_sum(tf.cast(fm, tf.float32)), 1.0)
    vs = model.trainable_variables + dec.trainable_variables
    gv = [(g, v) for g, v in zip(tape.gradient(loss, vs), vs) if g is not None]
    opt.apply_gradients(gv)
    return loss


for ep in range(a.epochs):
    ls = [float(step(tf.constant(x))) for x, _ in ds.batches(64, shuffle=True, drop_last=True, augment=True)]
    print(f"ep {ep} recon loss {np.mean(ls):.4f}")
save_weights(model, a.out)
print("saved", a.out)
