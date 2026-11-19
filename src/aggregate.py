"""Collect every metrics.json -> tables (CSV + Markdown) + figures + hypothesis verdicts.
  python -m src.aggregate        ->  experiments/tables/*.md|csv, experiments/figures/*.png"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from src.utils import ROOT

EXP, TAB, FIG = ROOT / "experiments", ROOT / "experiments" / "tables", ROOT / "experiments" / "figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)


def collect(study):
    rows = []
    for f in sorted((EXP / study).glob("*/seed*/metrics.json")):
        m = json.load(open(f)); t = m.get("test")
        if not t: continue
        rows.append({"name": f.parent.parent.name, "seed": m["seed"], "top1": t["top1"], "top5": t["top5"],
                     "macro_f1": t["macro_f1"], "ci_lo": t["top1_ci95"][0], "ci_hi": t["top1_ci95"][1],
                     **{k: m["cost"].get(k) for k in ["params", "size_mb", "train_time_s", "infer_ms_cpu",
                                                      "infer_ms_gpu", "peak_gpu_mem_mb", "hardware", "epochs_run"]},
                     "f1_per_class": t["per_class"]["f1"]})
    return pd.DataFrame(rows)


def ms(x, p=1, scale=100):
    return f"{np.mean(x) * scale:.{p}f} ± {np.std(x, ddof=1) * scale if len(x) > 1 else 0:.{p}f}"


def summary(df):
    out = []
    for n, g in df.groupby("name", sort=False):
        out.append({"Model": n, "Seeds": len(g), "Top-1 (%)": ms(g.top1), "Top-5 (%)": ms(g.top5),
                    "Macro-F1 (%)": ms(g.macro_f1), "Top-1 95% CI (mean, bootstrap)":
                    f"[{g.ci_lo.mean() * 100:.1f}, {g.ci_hi.mean() * 100:.1f}]"})
    return pd.DataFrame(out)


def save(df, name):
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(df.to_markdown(index=False)); print(f"\n== {name} ==\n{df.to_markdown(index=False)}")


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return a.mean() - b.mean(), float(p)


def main():
    data = {s: collect(s) for s in ["main", "aug", "repr", "ablation", "pretrain"] if (EXP / s).exists()}
    main_df = data.get("main", pd.DataFrame())
    if len(main_df):
        save(summary(main_df), "table4_main_results")
        for n, g in main_df.groupby("name"):                       # per-class F1 (mean over seeds)
            pc = np.mean(np.stack(g.f1_per_class.to_list()), 0)
            pd.DataFrame({"class_id": range(len(pc)), "f1": pc}).to_csv(TAB / f"per_class_f1_{n}.csv", index=False)
        cost = main_df.groupby("name", sort=False).agg(params=("params", "first"), size_mb=("size_mb", "mean"),
              train_s=("train_time_s", "mean"), infer_cpu_ms=("infer_ms_cpu", "mean"),
              infer_gpu_ms=("infer_ms_gpu", "mean"), gpu_mem_mb=("peak_gpu_mem_mb", "mean"),
              hardware=("hardware", "first")).round(2).reset_index()
        save(cost, "table7_computational_cost")
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        g = main_df.groupby("name", sort=False)
        plt.figure(figsize=(6, 3.5)); plt.bar(g.top1.mean().index, g.top1.mean() * 100, yerr=g.top1.std() * 100, capsize=4)
        plt.ylabel("Test top-1 (%)"); plt.title("Main comparison (mean ± SD, 3 seeds)"); plt.tight_layout()
        plt.savefig(FIG / "main_comparison.png", dpi=200); plt.close()
    # ---- ablation-like tables: delta vs the full model of the same family
    rows = []
    for study in ["aug", "repr", "ablation", "pretrain"]:
        df = data.get(study)
        if df is None or not len(df): continue
        for n, g in df.groupby("name", sort=False):
            fam = n.split("_")[0]
            full = main_df[main_df.name == fam].top1 if len(main_df) else []
            d = (g.top1.mean() - np.mean(full)) * 100 if len(full) else np.nan
            rows.append({"Study": study, "Variant": n, "Top-1 (%)": ms(g.top1), "Macro-F1 (%)": ms(g.macro_f1),
                         "Δ top-1 vs full model (pts)": f"{d:+.2f}"})
    if rows: save(pd.DataFrame(rows), "table5_ablation_and_studies")
    # ---- hypothesis verdicts (edit the list to match YOUR hypotheses)
    def get(study, name):
        df = data.get(study); return df[df.name == name].top1.to_numpy() if df is not None and len(df) else np.array([])
    H = []
    def add(h, text, a, b, extra=""):
        if len(a) > 1 and len(b) > 1:
            d, p = welch(a, b)
            v = "SUPPORTED" if d > 0 and p < 0.05 else ("REJECTED (opposite direction)" if d < 0 and p < 0.05 else "INCONCLUSIVE")
            H.append({"Hypothesis": h, "Claim": text, "Δ top-1 (pts)": f"{d * 100:+.2f}", "Welch p": f"{p:.3f}",
                      "Verdict (n=3 seeds, α=0.05)": v, "Notes": extra})
    add("H1", "Transformer > BiGRU", get("main", "transformer"), get("main", "gru"))
    add("H3a", "Augmentation helps Transformer (all vs none)", get("aug", "transformer_all"), get("aug", "transformer_none"))
    add("H3b", "Augmentation helps ST-GCN (all vs none)", get("aug", "stgcn_all"), get("aug", "stgcn_none"))
    add("H4", "Hands matter more than body (ST-GCN hands-only vs body-only)", get("ablation", "stgcn_hands"), get("ablation", "stgcn_body"))
    if len(main_df):                                              # H2 is about size vs accuracy
        p = main_df.groupby("name").params.first(); a, b = get("main", "stgcn"), get("main", "transformer")
        if len(a) and len(b):
            H.append({"Hypothesis": "H2", "Claim": "ST-GCN reaches similar accuracy with fewer parameters",
                      "Δ top-1 (pts)": f"{(a.mean() - b.mean()) * 100:+.2f}", "Welch p": f"{welch(a, b)[1]:.3f}",
                      "Verdict (n=3 seeds, α=0.05)": "see params →", "Notes": f"params stgcn={p.get('stgcn')} vs transformer={p.get('transformer')}"})
    if H: save(pd.DataFrame(H), "hypotheses")
    # ---- tuning figure
    tune = sorted((EXP / "tuning").glob("*.csv"))
    if tune:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3.5))
        for f in tune:
            d = pd.read_csv(f); plt.plot(range(1, len(d) + 1), d.val_top1.cummax() * 100, marker="o", label=f.stem)
        plt.xlabel("random-search trial"); plt.ylabel("best val top-1 so far (%)"); plt.legend(); plt.tight_layout()
        plt.savefig(FIG / "tuning_curves.png", dpi=200); plt.close()


if __name__ == "__main__":
    main()
