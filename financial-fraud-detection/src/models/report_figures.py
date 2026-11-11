"""
Generate the explainability figures for the report (objective ii).

Uses the saved XGBoost model and the held-out test set produced by
preprocess.py, so every number matches models/results/metrics.json.

Outputs (models/visualizations/report/):
  shap_global_beeswarm.png   - SHAP summary over the whole test set
  shap_global_importance.png - mean |SHAP| per feature (global importance)
  waterfall_true_positive.png   - a fraud the model caught
  waterfall_false_positive.png  - a legitimate transaction it flagged
  waterfall_false_negative.png  - a fraud it missed
  report_cases.json          - details of the three cases + importance ranking

Usage:
  python src/models/report_figures.py
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

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from src.inference import top_factors  # noqa: E402

PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "models", "xgboost.pkl")
OUT_DIR = os.path.join(PROJECT_DIR, "models", "visualizations", "report")


def unscale(X, preprocessor, feature_names):
    """Undo the Time/Amount scaling so plots show the original values."""
    raw = X.copy()
    for name in ("time", "amount"):
        scaler = preprocessor.named_transformers_[name]
        col = feature_names.index(name.capitalize())
        raw[:, col] = scaler.inverse_transform(X[:, [col]]).ravel()
    return raw


def pick_cases(y, proba, threshold):
    """
    One example of each outcome, chosen by a fixed rule so the figure is reproducible:
      TP - the caught fraud with the median score among caught frauds (a typical catch)
      FP - the legitimate transaction with the highest score (the worst false alarm)
      FN - the missed fraud with the highest score (the nearest miss)
    """
    flagged = proba >= threshold
    tp = np.where((y == 1) & flagged)[0]
    fp = np.where((y == 0) & flagged)[0]
    fn = np.where((y == 1) & ~flagged)[0]
    cases = {}
    if len(tp):
        cases["true_positive"] = int(tp[np.argsort(proba[tp])[len(tp) // 2]])
    if len(fp):
        cases["false_positive"] = int(fp[np.argmax(proba[fp])])
    if len(fn):
        cases["false_negative"] = int(fn[np.argmax(proba[fn])])
    return cases


TITLES = {
    "true_positive": "Caught fraud (true positive)",
    "false_positive": "False alarm (false positive)",
    "false_negative": "Missed fraud (false negative)",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y_test.npy")).astype(int)
    with open(os.path.join(PROCESSED_DIR, "feature_names.txt")) as f:
        feature_names = f.read().splitlines()
    preprocessor = joblib.load(os.path.join(PROCESSED_DIR, "preprocessor.pkl"))
    model_dict = joblib.load(MODEL_PATH)
    model, threshold = model_dict["model"], float(model_dict["metadata"]["threshold"])

    proba = model.predict_proba(X)[:, 1]
    print(f"Scoring {len(X):,} test transactions, threshold {threshold:.4f}")

    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    raw = unscale(X, preprocessor, feature_names)
    exp = shap.Explanation(values=sv.values, base_values=sv.base_values,
                           data=raw, feature_names=feature_names)

    # Global: beeswarm
    plt.figure()
    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.title("SHAP summary: test set (56,746 transactions)")
    plt.gcf().savefig(os.path.join(OUT_DIR, "shap_global_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    # Global: mean |SHAP|
    plt.figure()
    shap.plots.bar(exp, max_display=15, show=False)
    plt.title("Global feature importance (mean |SHAP|, log-odds)")
    plt.gcf().savefig(os.path.join(OUT_DIR, "shap_global_importance.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    mean_abs = np.abs(sv.values).mean(axis=0)
    fraud_mean_abs = np.abs(sv.values[y == 1]).mean(axis=0)
    ranking = [
        {"feature": feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 4),
         "mean_abs_shap_fraud_only": round(float(fraud_mean_abs[i]), 4)}
        for i in np.argsort(-mean_abs)
    ]

    # Local: one waterfall per outcome
    cases = pick_cases(y, proba, threshold)
    case_info = {}
    for kind, idx in cases.items():
        plt.figure()
        shap.plots.waterfall(exp[idx], max_display=12, show=False)
        plt.title(f"{TITLES[kind]}: score {proba[idx]:.3f} (threshold {threshold:.3f})")
        plt.gcf().savefig(os.path.join(OUT_DIR, f"waterfall_{kind}.png"), dpi=150, bbox_inches="tight")
        plt.close("all")
        case_info[kind] = {
            "test_index": idx,
            "actual_class": int(y[idx]),
            "fraud_probability": round(float(proba[idx]), 4),
            "amount": round(float(raw[idx, feature_names.index("Amount")]), 2),
            "top_factors": top_factors(sv.values[idx], raw[idx], feature_names, n=5),
        }
        print(f"{kind:15s} idx={idx:6d}  p={proba[idx]:.3f}  {case_info[kind]['top_factors']}")

    flagged = proba >= threshold
    summary = {
        "threshold": threshold,
        "base_value_log_odds": float(np.ravel(sv.base_values)[0]),
        "counts": {
            "tp": int(((y == 1) & flagged).sum()), "fp": int(((y == 0) & flagged).sum()),
            "fn": int(((y == 1) & ~flagged).sum()), "tn": int(((y == 0) & ~flagged).sum()),
        },
        "case_selection_rule": pick_cases.__doc__.strip(),
        "cases": case_info,
        "global_importance": ranking,
    }
    with open(os.path.join(OUT_DIR, "report_cases.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTop 10 features by mean |SHAP|:")
    for r in ranking[:10]:
        print(f"  {r['feature']:7s} {r['mean_abs_shap']:.4f}  (fraud only: {r['mean_abs_shap_fraud_only']:.4f})")
    print(f"\nFigures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
