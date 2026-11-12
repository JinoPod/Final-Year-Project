"""
Figures and supporting numbers for the dissertation (Chapter 4: Results and Discussion).

Print-ready versions (white background, Times New Roman, 200 dpi) built from the saved
models, the held-out test set and the results files, so every value matches
models/results/metrics.json and stability.json.

Outputs (models/visualizations/dissertation/):
  fig_confusion_matrix.png     XGBoost confusion matrix (test set)
  fig_model_comparison.png     precision / recall / F1 / PR-AUC for the four models
  fig_roc_pr.png               ROC and precision-recall curves, operating point marked
  fig_threshold.png            precision, recall and F1 against the decision threshold
  fig_score_distribution.png   fraud scores of legitimate vs fraudulent test transactions
  fig_stability.png            F1, precision and recall over the 5 repeated splits
  fig_shap_bar.png             global importance (mean |SHAP|)
  fig_shap_beeswarm.png        SHAP summary plot
  fig_feature_guide.png        normal range vs typical fraud value, top 12 V features
  fig_waterfall_{tp,fp,fn}.png local explanations of the three report cases
  dissertation_numbers.json    score-band counts and the SHAP-LIME agreement test

Usage:
  python src/models/dissertation_figures.py
"""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import (auc, average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_curve)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from src.explanations import load_reference, feature_guide, REVIEW_FLOOR  # noqa: E402
from src.models.report_figures import pick_cases, unscale  # noqa: E402

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models", "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "models", "results")
OUT_DIR = os.path.join(PROJECT_DIR, "models", "visualizations", "dissertation")

MODELS = [("XGBoost", "xgboost"), ("Random Forest", "random_forest"),
          ("LightGBM", "lightgbm"), ("Logistic Regression", "logistic_regression")]

# XGBoost is the highlighted series; baselines are greys told apart by line style and hatch,
# so the charts still read in black-and-white print.
BLUE, CRIMSON, INK, GRID = "#2E64A0", "#BF4B3C", "#1E2A3A", "#D9D9D9"
STYLE = {
    "XGBoost": dict(color=BLUE, ls="-", lw=2.2, hatch=""),
    "Random Forest": dict(color="#555555", ls="--", lw=1.5, hatch="//"),
    "LightGBM": dict(color="#8C8C8C", ls="-.", lw=1.5, hatch=".."),
    "Logistic Regression": dict(color="#B5B5B5", ls=":", lw=1.8, hatch="xx"),
}
DPI = 200

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load():
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy")).astype(int)
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    with open(os.path.join(PROCESSED_DIR, "feature_names.txt")) as f:
        names = f.read().splitlines()
    preprocessor = joblib.load(os.path.join(PROCESSED_DIR, "preprocessor.pkl"))
    models = {}
    for label, key in MODELS:
        d = joblib.load(os.path.join(MODELS_DIR, f"{key}.pkl"))
        models[label] = (d["model"], float(d["metadata"]["threshold"]))
    return X_train, X_test, y_test, names, preprocessor, models


def fig_confusion(y, proba, thr):
    tn, fp, fn, tp = confusion_matrix(y, proba >= thr).ravel()
    cells = np.array([[tn, fp], [fn, tp]])
    labels = np.array([["True negative\n(correctly passed)", "False positive\n(false alarm)"],
                       ["False negative\n(missed fraud)", "True positive\n(fraud caught)"]])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    shade = np.array([[0.06, 0.30], [0.30, 0.06]])  # errors darker; colour is not used for size
    ax.imshow(shade, cmap="Greys", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i - 0.12, f"{cells[i, j]:,}", ha="center", va="center", fontsize=15,
                    fontweight="bold", color=INK)
            ax.text(j, i + 0.22, labels[i, j], ha="center", va="center", fontsize=8.5, color=INK)
    ax.set_xticks([0, 1], ["Predicted legitimate", "Predicted fraud"])
    ax.set_yticks([0, 1], ["Actual\nlegitimate", "Actual\nfraud"])
    ax.grid(False)
    ax.set_title(f"XGBoost on the test set (threshold {thr:.3f})")
    save(fig, "fig_confusion_matrix.png")
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def fig_comparison(metrics):
    keys = [("precision", "Precision"), ("recall", "Recall"), ("f1_score", "F1-score"),
            ("avg_precision", "PR-AUC")]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    w = 0.2
    x = np.arange(len(keys))
    for i, (label, _) in enumerate(MODELS):
        vals = [metrics[label][k] for k, _ in keys]
        st = STYLE[label]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=st["color"] if label == "XGBoost" else "white",
                      edgecolor=st["color"] if label == "XGBoost" else "#444444", hatch=st["hatch"], lw=0.8)
        if label == "XGBoost":
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center",
                        va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, [k[1] for k in keys])
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Score (test set)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_model_comparison.png")


def fig_roc_pr(y, probas, thresholds):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.1))
    for label, _ in MODELS:
        p, st = probas[label], STYLE[label]
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=st["color"], ls=st["ls"], lw=st["lw"],
                label=f"{label} ({auc(fpr, tpr):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        a2.plot(rec, prec, color=st["color"], ls=st["ls"], lw=st["lw"],
                label=f"{label} ({average_precision_score(y, p):.3f})")
    p, t = probas["XGBoost"], thresholds["XGBoost"]
    flagged = p >= t
    r_op = recall_score(y, flagged)
    a2.plot([r_op], [precision_score(y, flagged)], "o", ms=7, color=BLUE, mec="white", mew=1.5, zorder=5)
    a2.annotate("XGBoost operating point", (r_op, precision_score(y, flagged)), xytext=(0.12, 0.55),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8))
    a1.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls=":")
    a1.set(xlabel="False positive rate (log scale)", ylabel="True positive rate (recall)", title="ROC curves (AUC)")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (AP)")
    a1.set_xscale("symlog", linthresh=0.01)
    a1.set_xlim(0, 1)
    for a in (a1, a2):
        a.legend(loc="lower right" if a is a1 else "lower left", fontsize=7.2, frameon=True)
        a.set_ylim(0, 1.02)
    fig.tight_layout()
    save(fig, "fig_roc_pr.png")


def fig_threshold(y, p, thr):
    ts = np.linspace(0.05, 0.995, 200)
    prec = [precision_score(y, p >= t, zero_division=1) for t in ts]
    rec = [recall_score(y, p >= t) for t in ts]
    f1 = [f1_score(y, p >= t) for t in ts]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(ts, prec, color="#555555", ls="--", lw=1.5, label="Precision")
    ax.plot(ts, rec, color="#8C8C8C", ls=":", lw=1.8, label="Recall")
    ax.plot(ts, f1, color=BLUE, lw=2.2, label="F1-score")
    ax.axvline(0.5, color="#999999", lw=0.9, ls="-.")
    ax.axvline(thr, color=CRIMSON, lw=1.2)
    ax.text(0.5, 0.52, " default 0.5", fontsize=8, color="#555555")
    ax.text(thr, 0.52, f" tuned {thr:.3f}", fontsize=8, color=CRIMSON)
    ax.set(xlabel="Decision threshold", ylabel="Score (test set)", ylim=(0.5, 1.01), xlim=(0.05, 1.0))
    ax.legend(loc="lower left", frameon=True)
    save(fig, "fig_threshold.png")


def fig_scores(y, p, thr):
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.hist(p[y == 0], bins=bins, color="#B5B5B5", edgecolor="white", label="Legitimate (56,651)")
    ax.hist(p[y == 1], bins=bins, color=CRIMSON, edgecolor="white", alpha=0.9, label="Fraud (95)")
    ax.set_yscale("log")
    ax.axvspan(REVIEW_FLOOR, thr, color="#F2E3B3", alpha=0.6, lw=0, label="Needs review band", zorder=0)
    ax.axvline(thr, color=INK, lw=1.0)
    ax.text(thr + 0.005, 3e4, "alert\nthreshold", fontsize=8)
    ax.set(xlabel="Fraud score (predicted probability)", ylabel="Transactions (log scale)")
    ax.legend(loc="upper center", frameon=True)
    save(fig, "fig_score_distribution.png")
    flagged = p >= thr
    review = (p >= REVIEW_FLOOR) & ~flagged
    missed = (y == 1) & ~flagged
    return {
        "review_band": {"n": int(review.sum()), "fraud": int((review & (y == 1)).sum())},
        "missed_fraud_scores": {"n": int(missed.sum()),
                                "below_0.1": int((missed & (p < 0.1)).sum()),
                                "below_0.5": int((missed & (p < 0.5)).sum())},
        "legit_below_0.01": float(((y == 0) & (p < 0.01)).sum() / (y == 0).sum()),
    }


def fig_stability(stab):
    rows = stab["per_split"]
    metrics = [("f1_score", "F1-score"), ("precision", "Precision"), ("recall", "Recall")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.8), sharey=True)
    for ax, (k, title) in zip(axes, metrics):
        data = [[r[k] for r in rows if r["model"] == label] for label, _ in MODELS]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, medianprops=dict(color=INK))
        for patch, (label, _) in zip(bp["boxes"], MODELS):
            patch.set(facecolor=BLUE if label == "XGBoost" else "white", edgecolor="#444444",
                      hatch=STYLE[label]["hatch"], alpha=0.85 if label == "XGBoost" else 1)
        for i, d in enumerate(data, 1):
            ax.plot(np.full(len(d), i), d, "o", ms=3, color=INK, alpha=0.7)
        ax.set_xticks(range(1, 5), ["XGB", "RF", "LGBM", "LR"])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test-set score (5 splits)")
    fig.tight_layout()
    save(fig, "fig_stability.png")


def shap_figures(model, X_test, y, names, preprocessor, proba, thr):
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_test)
    raw = unscale(X_test, preprocessor, names)
    exp = shap.Explanation(values=sv.values, base_values=sv.base_values, data=raw, feature_names=names)

    shap.plots.bar(exp, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(5.6, 3.6)
    plt.xlabel("Mean |SHAP value| (log-odds), test set")
    save(fig, "fig_shap_bar.png")

    shap.plots.beeswarm(exp, max_display=12, show=False, plot_size=(6.0, 4.2))
    save(plt.gcf(), "fig_shap_beeswarm.png")

    titles = {"true_positive": ("tp", "Caught fraud (true positive)"),
              "false_positive": ("fp", "False alarm (false positive)"),
              "false_negative": ("fn", "Missed fraud (false negative)")}
    for kind, idx in pick_cases(y, proba, thr).items():
        short, title = titles[kind]
        shap.plots.waterfall(exp[idx], max_display=9, show=False)
        fig = plt.gcf()
        fig.set_size_inches(6.0, 3.4)
        plt.title(f"{title}: fraud score {proba[idx]:.3f}, threshold {thr:.3f}", fontsize=10)
        save(fig, f"fig_waterfall_{short}.png")
    return sv.values


def fig_feature_guide():
    rows = [r for r in feature_guide(load_reference(PROJECT_DIR)) if r["feature"].startswith("V")][:12][::-1]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    y = np.arange(len(rows))
    ax.barh(y, [r["normal_high"] - r["normal_low"] for r in rows], left=[r["normal_low"] for r in rows],
            height=0.5, color="#C9D8EA", edgecolor=BLUE, lw=0.8, label="Legitimate: middle 90% (5th-95th percentile)")
    ax.plot([r["fraud_median"] for r in rows], y, "o", ms=6.5, color=CRIMSON, mec="white", mew=1.2,
            label="Fraud: median value")
    ax.set_yticks(y, [f"{r['feature']} (#{r['rank']})" for r in rows])
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_xlabel("Feature value (training data)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_feature_guide.png")


def lime_agreement(model, X_train, X_test, y, names, shap_values, proba, thr, n_cases=30, k=5):
    """For caught frauds, how many of LIME's top-k features are also in SHAP's top-k."""
    import lime.lime_tabular
    rng = np.random.default_rng(42)
    tp = np.where((y == 1) & (proba >= thr))[0]
    cases = rng.choice(tp, size=min(n_cases, len(tp)), replace=False)
    background = X_train[rng.choice(len(X_train), size=20_000, replace=False)]
    explainer = lime.lime_tabular.LimeTabularExplainer(
        background, feature_names=names, class_names=["Normal", "Fraud"],
        mode="classification", random_state=42, discretize_continuous=True)
    overlaps, top1 = [], []
    for idx in cases:
        e = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=k, labels=(1,))
        lime_top = [i for i, _ in sorted(e.as_map()[1], key=lambda t: -abs(t[1]))[:k]]
        shap_top = list(np.argsort(-np.abs(shap_values[idx]))[:k])
        overlaps.append(len(set(lime_top) & set(shap_top)))
        top1.append(lime_top[0] == shap_top[0])
    overlaps = np.array(overlaps)
    return {"n_cases": int(len(cases)), "k": k, "mean_overlap": float(overlaps.mean()),
            "share_at_least_3": float((overlaps >= 3).mean()),
            "share_same_top1": float(np.mean(top1)),
            "overlap_counts": {int(v): int((overlaps == v).sum()) for v in range(k + 1)}}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, X_test, y, names, preprocessor, models = load()
    probas = {label: m.predict_proba(X_test)[:, 1] for label, (m, _) in models.items()}
    thresholds = {label: t for label, (_, t) in models.items()}
    with open(os.path.join(RESULTS_DIR, "metrics.json")) as f:
        results = json.load(f)["results"]
    metrics = {label: results[key]["test"] for label, key in MODELS}
    with open(os.path.join(RESULTS_DIR, "stability.json")) as f:
        stab = json.load(f)

    xgb, thr = models["XGBoost"]
    p = probas["XGBoost"]
    numbers = {"confusion_matrix": fig_confusion(y, p, thr)}
    fig_comparison(metrics)
    fig_roc_pr(y, probas, thresholds)
    fig_threshold(y, p, thr)
    numbers["score_bands"] = fig_scores(y, p, thr)
    fig_stability(stab)
    shap_values = shap_figures(xgb, X_test, y, names, preprocessor, p, thr)
    fig_feature_guide()
    numbers["lime_vs_shap"] = lime_agreement(xgb, X_train, X_test, y, names, shap_values, p, thr)
    with open(os.path.join(OUT_DIR, "dissertation_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=2)
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
