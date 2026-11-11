"""
Model training for the XAI fraud detection project (report Section 3.4).

Primary model : XGBoost, hyperparameters tuned with Optuna to maximise F1.
Baselines     : Logistic Regression, Random Forest, LightGBM (same protocol,
                default hyperparameters) for the comparison in objective iii.

Protocol (no test-set leakage):
  1. Stratified k-fold cross-validation on the TRAINING split only.
  2. SMOTE sits inside an imbalanced-learn Pipeline, so it is applied to the
     training part of each fold only; validation folds stay real data with the
     true class ratio.
  3. Imbalance is handled once, by SMOTE. Models do not also use
     scale_pos_weight / class_weight (doing both over-corrects and hurts
     precision).
  4. The decision threshold is the one that maximises F1 on the out-of-fold
     predictions of the training split.
  5. The final model is refitted on the full (SMOTE-balanced) training split
     and evaluated exactly once on the untouched test split.

Run:  python src/models/train_model.py [--trials 40] [--folds 3]
"""

import argparse
import json
import os
import time

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import sklearn
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

RANDOM_STATE = 42
SMOTE_RATIOS = [0.1, 0.25, 0.5, 1.0]  # minority:majority ratio after SMOTE

DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
}


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def make_pipeline(estimator, smote_ratio):
    """SMOTE + model. SMOTE only runs during fit, never at predict time."""
    return Pipeline([
        ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=RANDOM_STATE)),
        ("model", estimator),
    ])


def make_cv(n_folds):
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)


def out_of_fold_proba(pipeline, X, y, cv):
    """Fraud probability for every training row, predicted by a model that never saw it."""
    return cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]


def best_f1_threshold(y_true, y_proba):
    """Return (threshold, f1) maximising F1 over all candidate thresholds."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall have one more entry than thresholds; drop the last point.
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "f1_at_0.5": float(f1_score(y_true, (y_proba >= 0.5).astype(int))),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def xgb_estimator(params=None):
    base = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    return xgb.XGBClassifier(**base, **(params or {}))


def baseline_estimators():
    """Default-hyperparameter baselines. No class weighting: SMOTE handles imbalance."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "lightgbm": lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    }


def tune_xgboost(X, y, n_trials, cv):
    """Optuna search over XGBoost hyperparameters and the SMOTE ratio, maximising CV F1."""

    def objective(trial):
        smote_ratio = trial.suggest_categorical("smote_ratio", SMOTE_RATIOS)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        proba = out_of_fold_proba(make_pipeline(xgb_estimator(params), smote_ratio), X, y, cv)
        threshold, f1 = best_f1_threshold(y, proba)
        trial.set_user_attr("threshold", threshold)
        trial.set_user_attr("auc_roc", float(roc_auc_score(y, proba)))
        return f1

    def report(study, trial):
        print(f"  trial {trial.number + 1:>3}/{n_trials}: CV F1={trial.value:.4f} "
              f"(best {study.best_value:.4f})", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name="xgboost_f1",
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[report])
    return study


# --------------------------------------------------------------------------
# Saving / loading
# --------------------------------------------------------------------------

def save_model(model, model_dir, name, metadata):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{name}.pkl")
    joblib.dump({"model": model, "metadata": metadata,
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, path)
    print(f"Saved {path}")


def load_model(model_path, model_name):
    """Load a saved model dict: {'model', 'metadata', 'timestamp'}."""
    return joblib.load(os.path.join(model_path, f"{model_name}.pkl"))


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_results(results, probas, y_test, viz_dir):
    os.makedirs(viz_dir, exist_ok=True)

    for name, res in results.items():
        cm = res["test"]["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        plt.title(f"{DISPLAY_NAMES[name]} (threshold={res['test']['threshold']:.3f})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{DISPLAY_NAMES[name]}_confusion_matrix.png"), dpi=120)
        plt.close()

    _, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
    for name, proba in probas.items():
        label = DISPLAY_NAMES[name]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax_roc.plot(fpr, tpr, label=f"{label} (AUC={results[name]['test']['auc_roc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax_pr.plot(rec, prec, label=f"{label} (AP={results[name]['test']['avg_precision']:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax_roc.set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    ax_pr.set(title="Precision-recall curve (test set)", xlabel="Recall", ylabel="Precision")
    ax_roc.legend(loc="lower right")
    ax_pr.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "roc_pr_curves.png"), dpi=120)
    plt.close()

    rows = [{"Model": DISPLAY_NAMES[n], "Precision": r["test"]["precision"],
             "Recall": r["test"]["recall"], "F1": r["test"]["f1_score"],
             "ROC-AUC": r["test"]["auc_roc"]} for n, r in results.items()]
    df = pd.DataFrame(rows).melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=df, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1)
    plt.title("Model comparison on the test set")
    sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(rows), title=None)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "model_comparison.png"), dpi=120)
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(n_trials, n_folds):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    model_dir = os.path.join(models_dir, "models")
    results_dir = os.path.join(models_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy")).astype(int)
    y_test = np.load(os.path.join(processed_dir, "y_test.npy")).astype(int)
    with open(os.path.join(processed_dir, "feature_names.txt")) as f:
        feature_names = f.read().splitlines()
    print(f"Train {X_train.shape} fraud={y_train.sum()} | Test {X_test.shape} fraud={y_test.sum()}")

    cv = make_cv(n_folds)
    versions = {"scikit-learn": sklearn.__version__, "xgboost": xgb.__version__,
                "lightgbm": lgb.__version__, "numpy": np.__version__}
    results, test_probas = {}, {}

    # ---- Primary model: XGBoost -------------------------------------------
    t0 = time.time()
    print(f"\nTuning XGBoost: {n_trials} trials x {n_folds}-fold CV (SMOTE inside each fold)")
    study = tune_xgboost(X_train, y_train, n_trials, cv)
    best = dict(study.best_params)
    smote_ratio = best.pop("smote_ratio")
    threshold = study.best_trial.user_attrs["threshold"]
    print(f"Best CV F1={study.best_value:.4f}, threshold={threshold:.4f}, "
          f"SMOTE ratio={smote_ratio}, params={best}")
    study.trials_dataframe().to_csv(os.path.join(results_dir, "xgboost_optuna_trials.csv"), index=False)

    pipe = make_pipeline(xgb_estimator(best), smote_ratio).fit(X_train, y_train)
    model = pipe.named_steps["model"]
    proba = model.predict_proba(X_test)[:, 1]
    results["xgboost"] = {
        "role": "primary",
        "cv": {"f1_score": study.best_value, "auc_roc": study.best_trial.user_attrs["auc_roc"]},
        "test": compute_metrics(y_test, proba, threshold),
        "train_seconds": round(time.time() - t0, 1),
    }
    test_probas["xgboost"] = proba
    xgb_meta = {"role": "primary", "params": best, "smote_ratio": smote_ratio}

    # ---- Baselines --------------------------------------------------------
    baseline_meta = {}
    for name, est in baseline_estimators().items():
        t0 = time.time()
        print(f"\nBaseline {DISPLAY_NAMES[name]}: {n_folds}-fold CV for threshold, then refit")
        oof = out_of_fold_proba(make_pipeline(est, smote_ratio), X_train, y_train, cv)
        thr, cv_f1 = best_f1_threshold(y_train, oof)
        fitted = make_pipeline(est, smote_ratio).fit(X_train, y_train).named_steps["model"]
        proba = fitted.predict_proba(X_test)[:, 1]
        results[name] = {
            "role": "baseline",
            "cv": {"f1_score": cv_f1, "auc_roc": float(roc_auc_score(y_train, oof))},
            "test": compute_metrics(y_test, proba, thr),
            "train_seconds": round(time.time() - t0, 1),
        }
        test_probas[name] = proba
        baseline_meta[name] = (fitted, {"role": "baseline", "params": "defaults",
                                        "smote_ratio": smote_ratio})

    # ---- Save -------------------------------------------------------------
    common = {"feature_names": feature_names, "cv_folds": n_folds,
              "threshold_source": "max F1 on out-of-fold predictions (training split)",
              "versions": versions}
    save_model(model, model_dir, "xgboost",
               {**xgb_meta, **common, "threshold": results["xgboost"]["test"]["threshold"],
                "cv": results["xgboost"]["cv"], "test_metrics": results["xgboost"]["test"]})
    for name, (fitted, meta) in baseline_meta.items():
        save_model(fitted, model_dir, name,
                   {**meta, **common, "threshold": results[name]["test"]["threshold"],
                    "cv": results[name]["cv"], "test_metrics": results[name]["test"]})

    summary = {"n_trials": n_trials, "cv_folds": n_folds, "smote_ratio": smote_ratio,
               "xgboost_params": best, "versions": versions, "results": results}
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    table = pd.DataFrame([
        {"model": DISPLAY_NAMES[n], "role": r["role"], "cv_f1": r["cv"]["f1_score"],
         **{k: v for k, v in r["test"].items() if k != "confusion_matrix"},
         **r["test"]["confusion_matrix"]}
        for n, r in results.items()
    ])
    table.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    plot_results(results, test_probas, y_test, os.path.join(models_dir, "visualizations"))

    pd.set_option("display.width", 200)
    print("\nTest-set results:")
    print(table[["model", "threshold", "precision", "recall", "f1_score", "auc_roc",
                 "avg_precision", "cv_f1", "tp", "fp", "fn"]].round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials for XGBoost")
    parser.add_argument("--folds", type=int, default=3, help="Cross-validation folds")
    args = parser.parse_args()
    main(args.trials, args.folds)
