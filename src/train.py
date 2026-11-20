"""Single entry point to train + evaluate one configuration with one seed (TensorFlow / Keras).
  python -m src.train --config configs/transformer.yaml --seed 0 --out experiments/main/transformer/seed0
The TEST split is touched exactly once, at the end, with the weights that were best on validation."""
import argparse, json, os, platform, time
from pathlib import Path
import joblib, numpy as np, tensorflow as tf
from src.utils import load_cfg, parse_set, set_seed, save_weights, load_weights
from src.datasets.dataset import SignDataset, make_label_map, n_channels
from src.models import build_model
from src.evaluate import compute_metrics, predict


def hardware():
    gpus = tf.config.list_physical_devices("GPU")
    cpu = platform.processor() or platform.machine()
    if gpus:
        try: name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
        except Exception: name = "GPU"
        return f"{name} (CUDA) + {cpu} | TF {tf.__version__}"
    return f"CPU {cpu} | TF {tf.__version__}"


def time_inference(model, x1, reps=50):
    """ms per single sample (batch=1): CPU, and GPU if one is available."""
    f = tf.function(lambda a: model(a, training=False))
    out, x1 = {}, tf.constant(x1[:1])
    with tf.device("/CPU:0"):
        for _ in range(5): f(x1).numpy()
        t = time.perf_counter()
        for _ in range(reps): f(x1).numpy()
        out["infer_ms_cpu"] = (time.perf_counter() - t) / reps * 1000
    if tf.config.list_physical_devices("GPU"):
        g = tf.function(lambda a: model(a, training=False))
        with tf.device("/GPU:0"):
            for _ in range(10): g(x1).numpy()
            t = time.perf_counter()
            for _ in range(reps): g(x1).numpy()
            out["infer_ms_gpu"] = (time.perf_counter() - t) / reps * 1000
    return out


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, warm, total):
        super().__init__(); self.lr, self.warm, self.total = float(lr), float(max(warm, 1)), float(max(total, 1))

    def __call__(self, step):
        s = tf.cast(step, tf.float32)
        prog = tf.clip_by_value((s - self.warm) / max(1.0, self.total - self.warm), 0.0, 1.0)
        return self.lr * tf.where(s < self.warm, (s + 1) / self.warm, 0.5 * (1 + tf.cos(np.pi * prog)))

    def get_config(self):
        return {"lr": self.lr, "warm": self.warm, "total": self.total}


def run_baseline(cfg, seed, out, dtr, dva, dte, eval_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=cfg["model"].get("C", 1.0), max_iter=300, random_state=seed))
    Xtr, Xva = dtr.tabular(), dva.tabular()
    clf.fit(Xtr, dtr.y)
    train_time = time.time() - t0
    K = len(dtr.label_map)
    def probs(X):
        p = np.zeros((len(X), K)); p[:, clf.classes_] = clf.predict_proba(X); return p
    res = {"val": compute_metrics(dva.y, probs(Xva), n_boot=50)}
    joblib.dump({"clf": clf, "cfg": cfg, "classes": list(dtr.label_map)}, out / "model.joblib")
    n_par = int(np.size(clf[-1].coef_) + np.size(clf[-1].intercept_))
    t = time.perf_counter()
    for _ in range(50): clf.predict_proba(Xva[:1])
    cost = {"params": n_par, "size_mb": os.path.getsize(out / "model.joblib") / 1e6, "train_time_s": train_time,
            "infer_ms_cpu": (time.perf_counter() - t) / 50 * 1000, "hardware": hardware(), "epochs_run": 0}
    if eval_test:
        pt = probs(dte.tabular())
        res["test"] = compute_metrics(dte.y, pt); np.savez(out / "test_preds.npz", probs=pt, y=dte.y)
    return res, cost


def run_tf(cfg, seed, out, dtr, dva, dte, eval_test):
    tc = cfg["train"]; K = len(dtr.label_map); C = n_channels(cfg["data"])
    model = build_model(cfg, K, C)
    if tc.get("init_from"):
        load_weights(model, tc["init_from"]); print("loaded SSL-pretrained weights (classifier head is re-initialised)")
        for v in model.head.trainable_variables: v.assign(tf.random.normal(v.shape, stddev=0.02))
    steps_per_epoch = len(dtr) // tc["batch_size"]
    sched = WarmupCosine(tc["lr"], tc["warmup_epochs"] * steps_per_epoch, tc["epochs"] * steps_per_epoch)
    opt = tf.keras.optimizers.AdamW(learning_rate=sched, weight_decay=tc["weight_decay"])
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=tc["label_smoothing"])

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            loss = loss_fn(tf.one_hot(y, K), model(x, training=True))
        vs = model.trainable_variables
        grads = tape.gradient(loss, vs)
        gv = [(g, v) for g, v in zip(grads, vs) if g is not None]
        clipped, _ = tf.clip_by_global_norm([g for g, _ in gv], 1.0)
        opt.apply_gradients(zip(clipped, [v for _, v in gv]))
        return loss

    best, best_w, bad, hist = -1.0, None, 0, []
    gpu = bool(tf.config.list_physical_devices("GPU"))
    if gpu:
        try: tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception: pass
    t0 = time.time()
    for ep in range(tc["epochs"]):
        tl, n = 0.0, 0
        for x, y in dtr.batches(tc["batch_size"], shuffle=True, drop_last=True, augment=True):
            tl += float(train_step(tf.constant(x), tf.constant(y))); n += 1
        pv, yv = predict(model, dva); va = float((pv.argmax(1) == yv).mean())
        hist.append({"epoch": ep, "train_loss": tl / max(n, 1), "val_top1": va})
        if va > best:
            best, bad, best_w = va, 0, model.get_weights()
        else:
            bad += 1
            if bad >= tc["patience"]: break
    train_time = time.time() - t0
    model.set_weights(best_w)
    save_weights(model, out / "model_weights.npz", cfg, list(dtr.label_map), C)
    import pandas as pd; pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    pv, yv = predict(model, dva)
    res = {"val": compute_metrics(yv, pv, n_boot=50)}
    peak = None
    if gpu:
        try: peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e6
        except Exception: pass
    cost = {"params": int(model.count_params()), "size_mb": os.path.getsize(out / "model_weights.npz") / 1e6,
            "train_time_s": train_time, "epochs_run": len(hist), "hardware": hardware(), "peak_gpu_mem_mb": peak}
    cost.update(time_inference(model, dva.arrays()[0][:1]))
    if eval_test:
        pt, yt = predict(model, dte)
        res["test"] = compute_metrics(yt, pt); np.savez(out / "test_preds.npz", probs=pt, y=yt)
    return res, cost


def run(cfg, seed, out, eval_test=True):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    csv = cfg["data"]["csv"]; lm = make_label_map(csv)
    dtr = SignDataset(csv, "train", cfg["data"], True, lm, seed)
    dva = SignDataset(csv, "val", cfg["data"], False, lm)
    dte = SignDataset(csv, "test", cfg["data"], False, lm) if eval_test else None
    fn = run_baseline if cfg["model"]["name"] == "baseline" else run_tf
    res, cost = fn(cfg, seed, out, dtr, dva, dte, eval_test)
    res.update(cost=cost, seed=seed, cfg=cfg)
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    t = res.get("test") or res["val"]
    print(f"[{out}] val top1={res['val']['top1']:.3f}" + (f" | test top1={t['top1']:.3f} top5={t['top5']:.3f} "
          f"macroF1={t['macro_f1']:.3f}" if eval_test else "") + f" | {cost['train_time_s']:.0f}s")
    return res


def load_run(run_dir):
    """Rebuild a finished run: returns dict(cfg, classes, model) or the sklearn bundle for the baseline."""
    run_dir = Path(run_dir)
    if (run_dir / "model.joblib").exists():
        return joblib.load(run_dir / "model.joblib")
    z = np.load(run_dir / "model_weights.npz", allow_pickle=False)
    cfg, classes = json.loads(str(z["cfg"])), json.loads(str(z["classes"]))
    model = build_model(cfg, len(classes), int(z["in_channels"]))
    load_weights(model, run_dir / "model_weights.npz")
    return {"cfg": cfg, "classes": classes, "model": model}


def evaluate_run(run_dir, split="test"):
    ck = load_run(run_dir); cfg = ck["cfg"]
    lm = {g: i for i, g in enumerate(ck["classes"])}
    ds = SignDataset(cfg["data"]["csv"], split, cfg["data"], False, lm)
    if "clf" in ck:
        p = ck["clf"].predict_proba(ds.tabular()); full = np.zeros((len(ds), len(lm))); full[:, ck["clf"].classes_] = p
        return compute_metrics(ds.y, full)
    p, y = predict(ck["model"], ds)
    return compute_metrics(y, p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--set", nargs="*", help="key=value overrides")
    ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()
    run(load_cfg(a.config, parse_set(a.set)), a.seed, a.out, eval_test=not a.no_test)
