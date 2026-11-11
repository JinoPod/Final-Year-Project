"""
Stability check for the fraud detection models.

Answers two questions for the report:

1. How much do the results depend on the particular train/test split and
   random seed?  -> Repeat the whole pipeline on N different stratified splits.
   For each split: fit the scaler on the training part, re-tune XGBoost with
   Optuna (CV on the training part only), pick each model's threshold from
   out-of-fold F1, refit, and evaluate once on that split's test part.
   XGBoost is re-tuned on every split; reusing the hyperparameters from the
   main run would leak information, because the main run's training data
   overlaps the other splits' test sets.

2. How precise are the headline numbers from the main run?  -> Bootstrap the
   main run's test set (resample transactions with replacement) to get 95%
   confidence intervals for the saved models' metrics.

Run:  python src/models/stability.py [--seeds 5] [--trials 20] [--folds 3]
Output: models/results/stability.json, models/visualizations/stability_*.png
"""

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.data.preprocess import FEATURE_COLS, TARGET_COL, clean_data, create_preprocessor, load_data
from src.models import train_model as tm

METRICS = ["precision", "recall", "f1_score", "auc_roc", "avg_precision"]


def run_split(df, seed, n_trials, n_folds):
    """Full pipeline on one train/test split. Returns {model_name: test metrics}."""
    X, y = df[FEATURE_COLS], df[TARGET_COL].astype(int).to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    pre = create_preprocessor()
    X_tr, X_te = pre.fit_transform(X_tr), pre.transform(X_te)

    # Seed everything in train_model for this run (SMOTE, models, CV folds, Optuna).
    tm.RANDOM_STATE = seed
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    out = {}
    study = tm.tune_xgboost(X_tr, y_tr, n_trials, cv)
    best = dict(study.best_params)
    smote_ratio = best.pop("smote_ratio")
    threshold = study.best_trial.user_attrs["threshold"]
    model = tm.make_pipeline(tm.xgb_estimator(best), smote_ratio).fit(X_tr, y_tr).named_steps["model"]
    out["xgboost"] = tm.compute_metrics(y_te, model.predict_proba(X_te)[:, 1], threshold)
    out["xgboost"]["smote_ratio"] = smote_ratio

    for name, est in tm.baseline_estimators().items():
        oof = tm.out_of_fold_proba(tm.make_pipeline(est, smote_ratio), X_tr, y_tr, cv)
        thr, _ = tm.best_f1_threshold(y_tr, oof)
        fitted = tm.make_pipeline(est, smote_ratio).fit(X_tr, y_tr).named_steps["model"]
        out[name] = tm.compute_metrics(y_te, fitted.predict_proba(X_te)[:, 1], thr)
    return out, int(y_te.sum())


def bootstrap_ci(y_true, y_proba, threshold, n_boot=1000, seed=0):
    """95% percentile bootstrap CIs on a fixed test set."""
    rng = np.random.default_rng(seed)
    y_pred = (y_proba >= threshold).astype(int)
    n = len(y_true)
    samples = {m: [] for m in ["precision", "recall", "f1_score", "auc_roc"]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, pr = y_true[idx], y_pred[idx], y_proba[idx]
        samples["precision"].append(precision_score(yt, yp, zero_division=0))
        samples["recall"].append(recall_score(yt, yp))
        samples["f1_score"].append(f1_score(yt, yp))
        samples["auc_roc"].append(roc_auc_score(yt, pr))
    return {m: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))}
            for m, v in samples.items()}


def main(n_seeds, n_trials, n_folds):
    results_dir = os.path.join(BASE_DIR, "models", "results")
    viz_dir = os.path.join(BASE_DIR, "models", "visualizations")
    os.makedirs(results_dir, exist_ok=True)

    # ---- Part 1: repeated splits -----------------------------------------
    df = clean_data(load_data(os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")))
    seeds = [101 + i for i in range(n_seeds)]  # distinct from the main run's seed (42)
    rows = []
    for i, seed in enumerate(seeds, 1):
        t0 = time.time()
        print(f"\n=== Split {i}/{n_seeds} (seed {seed}) ===", flush=True)
        split_results, n_fraud = run_split(df, seed, n_trials, n_folds)
        for name, m in split_results.items():
            rows.append({"seed": seed, "model": tm.DISPLAY_NAMES[name], "test_fraud": n_fraud,
                         **{k: m[k] for k in METRICS + ["threshold"]},
                         "fp": m["confusion_matrix"]["fp"], "fn": m["confusion_matrix"]["fn"]})
        done = pd.DataFrame(rows)
        print(done[done.seed == seed][["model", "precision", "recall", "f1_score", "auc_roc", "fp", "fn"]]
              .round(4).to_string(index=False))
        print(f"split took {time.time() - t0:.0f}s", flush=True)

    runs = pd.DataFrame(rows)
    summary = runs.groupby("model")[METRICS + ["fp", "fn"]].agg(["mean", "std", "min", "max"])
    order = [tm.DISPLAY_NAMES[n] for n in ["xgboost", "random_forest", "lightgbm", "logistic_regression"]]
    summary = summary.reindex(order)

    # Paired comparison: XGBoost minus each baseline, per split.
    pivot = runs.pivot(index="seed", columns="model", values="f1_score")
    paired = {b: {"mean_diff": float((pivot["XGBoost"] - pivot[b]).mean()),
                  "xgboost_wins": int((pivot["XGBoost"] > pivot[b]).sum()),
                  "ties": int((pivot["XGBoost"] == pivot[b]).sum()),
                  "splits": int(len(pivot))}
              for b in order[1:]}

    # ---- Part 2: bootstrap CIs on the main run's test set ----------------
    X_test = np.load(os.path.join(BASE_DIR, "data", "processed", "X_test.npy"))
    y_test = np.load(os.path.join(BASE_DIR, "data", "processed", "y_test.npy")).astype(int)
    boot = {}
    for name in ["xgboost", "random_forest", "lightgbm", "logistic_regression"]:
        d = tm.load_model(os.path.join(BASE_DIR, "models", "models"), name)
        proba = d["model"].predict_proba(X_test)[:, 1]
        boot[name] = {"point": {k: d["metadata"]["test_metrics"][k]
                                for k in ["precision", "recall", "f1_score", "auc_roc"]},
                      "ci95": bootstrap_ci(y_test, proba, d["metadata"]["threshold"])}

    # ---- Save ------------------------------------------------------------
    flat = {m: {f"{metric}_{stat}": float(summary.loc[m, (metric, stat)])
                for metric in METRICS + ["fp", "fn"] for stat in ["mean", "std", "min", "max"]}
            for m in order}
    json_path = os.path.join(results_dir, "stability.json")
    with open(json_path, "w") as f:
        json.dump({"settings": {"seeds": seeds, "optuna_trials_per_split": n_trials,
                                "cv_folds": n_folds, "test_size": 0.2},
                   "per_split": rows, "summary": flat,
                   "paired_f1_vs_xgboost": paired, "bootstrap_main_test_set": boot}, f, indent=2)
    report(json_path, viz_dir)


def report(json_path, viz_dir):
    """Plot and print a saved stability.json (can be re-run without retraining)."""
    with open(json_path) as f:
        data = json.load(f)
    runs = pd.DataFrame(data["per_split"])
    flat, paired, boot = data["summary"], data["paired_f1_vs_xgboost"], data["bootstrap_main_test_set"]
    order = list(flat.keys())
    n_seeds = len(data["settings"]["seeds"])

    plt.figure(figsize=(10, 5.5))
    long = runs.melt(id_vars=["seed", "model"], value_vars=["precision", "recall", "f1_score", "auc_roc"],
                     var_name="Metric", value_name="Score")
    long["Metric"] = long["Metric"].map({"precision": "Precision", "recall": "Recall",
                                         "f1_score": "F1", "auc_roc": "ROC-AUC"})
    ax = sns.boxplot(data=long, x="Metric", y="Score", hue="model", hue_order=order, fliersize=0)
    sns.stripplot(data=long, x="Metric", y="Score", hue="model", hue_order=order, dodge=True,
                  palette=["black"] * len(order), size=3, legend=False, ax=ax)
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"Stability across {n_seeds} independent train/test splits")
    sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(order), title=None)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "stability_boxplot.png"), dpi=120)
    plt.close()

    print("\n=== Across splits: mean ± std (min-max) ===")
    for m in order:
        s = flat[m]
        print(f"{m:20s} " + "  ".join(
            f"{k}={s[k + '_mean']:.3f}±{s[k + '_std']:.3f} ({s[k + '_min']:.3f}-{s[k + '_max']:.3f})"
            for k in ["precision", "recall", "f1_score", "auc_roc"]))
    print("\nPaired F1, XGBoost minus baseline:", json.dumps(paired, indent=1))
    print("\nBootstrap 95% CI on main test set:")
    for name, b in boot.items():
        print(f"{tm.DISPLAY_NAMES[name]:20s} " + "  ".join(
            f"{k}={b['point'][k]:.3f} [{b['ci95'][k]['low']:.3f}, {b['ci95'][k]['high']:.3f}]"
            for k in b["point"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stability check across splits and seeds")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--report-only", action="store_true",
                        help="Re-plot/print an existing stability.json without retraining")
    args = parser.parse_args()
    if args.report_only:
        report(os.path.join(BASE_DIR, "models", "results", "stability.json"),
               os.path.join(BASE_DIR, "models", "visualizations"))
    else:
        main(args.seeds, args.trials, args.folds)
