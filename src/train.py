"""Single entry point to train + evaluate one configuration with one seed.
  python -m src.train --config configs/transformer.yaml --seed 0 --out experiments/main/transformer/seed0
The TEST split is touched exactly once, at the end, with the checkpoint that was best on validation."""
import argparse, copy, json, os, platform, time
from pathlib import Path
import joblib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import load_cfg, parse_set, set_seed, count_params
from src.datasets.dataset import SignDataset, make_label_map, n_channels, J
from src.models import build_model
from src.evaluate import compute_metrics, predict


def hardware():
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)} (CUDA) + {platform.processor() or platform.machine()}"
    return f"CPU {platform.processor() or platform.machine()}"


@torch.no_grad()
def time_inference(model, x, reps=50):
    """ms per single sample (batch=1) on CPU, and on GPU if available."""
    out = {}
    m = copy.deepcopy(model).cpu().eval(); xs = x[:1].cpu()
    for _ in range(5): m(xs)
    t = time.perf_counter()
    for _ in range(reps): m(xs)
    out["infer_ms_cpu"] = (time.perf_counter() - t) / reps * 1000
    if torch.cuda.is_available():
        m = m.cuda(); xs = xs.cuda()
        for _ in range(10): m(xs)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(reps): m(xs)
        torch.cuda.synchronize(); out["infer_ms_gpu"] = (time.perf_counter() - t) / reps * 1000
    return out


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
    n_par = int(sum(np.size(s.coef_) + np.size(s.intercept_) for s in [clf[-1]]))
    t = time.perf_counter()
    for _ in range(50): clf.predict_proba(Xva[:1])
    cost = {"params": n_par, "size_mb": os.path.getsize(out / "model.joblib") / 1e6, "train_time_s": train_time,
            "infer_ms_cpu": (time.perf_counter() - t) / 50 * 1000, "hardware": hardware(), "epochs_run": 0}
    if eval_test:
        Xte = dte.tabular(); pt = probs(Xte)
        res["test"] = compute_metrics(dte.y, pt); np.savez(out / "test_preds.npz", probs=pt, y=dte.y)
    return res, cost


def run_torch(cfg, seed, out, dtr, dva, dte, eval_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc = cfg["train"]; K = len(dtr.label_map); C = n_channels(cfg["data"])
    model = build_model(cfg, K, C).to(device)
    if tc.get("init_from"):
        sd = torch.load(tc["init_from"], map_location=device)
        missing = model.load_state_dict(sd, strict=False)
        print("loaded pretrained encoder; missing keys:", len(missing.missing_keys))
    g = torch.Generator().manual_seed(seed)
    ltr = DataLoader(dtr, tc["batch_size"], shuffle=True, drop_last=True, generator=g)
    lva = DataLoader(dva, 128); lte = DataLoader(dte, 128)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps = tc["epochs"] * len(ltr); warm = tc["warmup_epochs"] * len(ltr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])
    best, best_state, bad, hist = -1, None, 0, []
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(tc["epochs"]):
        model.train(); tl = 0
        for x, y in ltr:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tl += loss.item()
        pv, yv = predict(model, lva, device); va = float((pv.argmax(1) == yv).mean())
        hist.append({"epoch": ep, "train_loss": tl / len(ltr), "val_top1": va})
        if va > best:
            best, bad = va, 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]: break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    torch.save({"cfg": cfg, "state": best_state, "classes": list(dtr.label_map), "in_channels": C}, out / "model.pt")
    import pandas as pd; pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    pv, yv = predict(model, lva, device)
    res = {"val": compute_metrics(yv, pv, n_boot=50)}
    cost = {"params": count_params(model), "size_mb": os.path.getsize(out / "model.pt") / 1e6,
            "train_time_s": train_time, "epochs_run": len(hist), "hardware": hardware(),
            "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None}
    cost.update(time_inference(model, dva[0][0][None]))
    if eval_test:
        pt, yt = predict(model, lte, device)
        res["test"] = compute_metrics(yt, pt); np.savez(out / "test_preds.npz", probs=pt, y=yt)
    return res, cost


def run(cfg, seed, out, eval_test=True):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    csv = cfg["data"]["csv"]; lm = make_label_map(csv)
    dtr = SignDataset(csv, "train", cfg["data"], True, lm, seed)
    dva = SignDataset(csv, "val", cfg["data"], False, lm)
    dte = SignDataset(csv, "test", cfg["data"], False, lm) if eval_test else None
    fn = run_baseline if cfg["model"]["name"] == "baseline" else run_torch
    res, cost = fn(cfg, seed, out, dtr, dva, dte, eval_test)
    res.update(cost=cost, seed=seed, cfg=cfg)
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    t = res.get("test") or res["val"]
    print(f"[{out}] val top1={res['val']['top1']:.3f}" + (f" | test top1={t['top1']:.3f} top5={t['top5']:.3f} "
          f"macroF1={t['macro_f1']:.3f}" if eval_test else "") + f" | {cost['train_time_s']:.0f}s")
    return res


def load_run(run_dir):
    run_dir = Path(run_dir)
    if (run_dir / "model.joblib").exists():
        return joblib.load(run_dir / "model.joblib")
    ck = torch.load(run_dir / "model.pt", map_location="cpu")
    model = build_model(ck["cfg"], len(ck["classes"]), ck["in_channels"]); model.load_state_dict(ck["state"])
    ck["model"] = model.eval()
    return ck


def evaluate_run(run_dir, split="test"):
    ck = load_run(run_dir); cfg = ck["cfg"]
    lm = {g: i for i, g in enumerate(ck["classes"])}
    ds = SignDataset(cfg["data"]["csv"], split, cfg["data"], False, lm)
    if "clf" in ck:
        p = ck["clf"].predict_proba(ds.tabular()); full = np.zeros((len(ds), len(lm))); full[:, ck["clf"].classes_] = p
        return compute_metrics(ds.y, full)
    p, y = predict(ck["model"], DataLoader(ds, 128), "cpu")
    return compute_metrics(y, p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--set", nargs="*", help="key=value overrides")
    ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()
    run(load_cfg(a.config, parse_set(a.set)), a.seed, a.out, eval_test=not a.no_test)
