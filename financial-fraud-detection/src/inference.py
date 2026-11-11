"""
Inference service: score uploaded transactions and explain each decision.

Implements the Data Preprocessing, Model Services and Explainability Services
layers of the system architecture (report Sections 3.6.2 - 3.6.4):

  validate schema  ->  reject incomplete rows  ->  scale with the saved
  preprocessor (same transformation as training)  ->  XGBoost fraud
  probability  ->  tuned decision threshold  ->  exact TreeSHAP values.

SMOTE is a training-time step only and is not applied here.
"""

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import shap

from src.data.preprocess import FEATURE_COLS, TARGET_COL, validate_schema

PRIMARY_MODEL = "xgboost"
MAX_ROWS = 20_000  # keeps scoring fast and the audit log a manageable size


@dataclass
class PreparedUpload:
    features: pd.DataFrame      # rows that can be scored, original (unscaled) values
    labels: pd.Series | None    # 'Class' column if the file had one, else None
    row_numbers: np.ndarray     # 1-based row numbers in the uploaded file
    rejected: pd.DataFrame      # rows that could not be scored, with a reason
    total_rows: int


@dataclass
class ScoredBatch:
    results: pd.DataFrame       # row, fraud_probability, flagged (+ actual if labelled)
    shap_values: np.ndarray     # (n_rows, n_features), log-odds contributions
    base_value: float           # model's average log-odds output (SHAP expected value)
    raw_features: pd.DataFrame  # unscaled inputs, as uploaded
    threshold: float
    model_name: str
    model_version: str


def prepare_upload(df):
    """
    Check an uploaded DataFrame and split it into scorable and rejected rows.

    Raises ValueError (readable message) if the file cannot be used at all:
    missing columns, non-numeric columns, no rows, or too many rows.
    """
    if len(df) == 0:
        raise ValueError("The file has no transactions.")
    if len(df) > MAX_ROWS:
        raise ValueError(f"The file has {len(df):,} rows; the limit is {MAX_ROWS:,}. "
                         "Split it into smaller files.")
    validate_schema(df)

    row_numbers = np.arange(1, len(df) + 1)
    missing = df[FEATURE_COLS].isna().any(axis=1).to_numpy()
    rejected = pd.DataFrame({
        "row": row_numbers[missing],
        "reason": [f"missing values in: {', '.join(df.columns[df.iloc[i].isna()].tolist())}"
                   for i in np.where(missing)[0]],
    })

    keep = ~missing
    labels = None
    if TARGET_COL in df.columns:
        labels = df.loc[keep, TARGET_COL]
        # Only use labels if they are clean 0/1 values.
        if labels.isna().any() or not labels.isin([0, 1]).all():
            labels = None
        else:
            labels = labels.astype(int).reset_index(drop=True)

    return PreparedUpload(
        features=df.loc[keep, FEATURE_COLS].reset_index(drop=True),
        labels=labels,
        row_numbers=row_numbers[keep],
        rejected=rejected,
        total_rows=len(df),
    )


class FraudScorer:
    """Wraps the saved preprocessor, the deployed model and its SHAP explainer."""

    def __init__(self, model_dict, preprocessor, model_name=PRIMARY_MODEL):
        self.model = model_dict["model"]
        self.metadata = model_dict.get("metadata", {})
        if "threshold" not in self.metadata:
            raise ValueError(f"Model '{model_name}' has no tuned threshold; retrain it first.")
        self.threshold = float(self.metadata["threshold"])
        self.model_name = model_name
        self.model_version = model_dict.get("timestamp", "unknown")
        self.preprocessor = preprocessor
        self.explainer = shap.TreeExplainer(self.model)

    @classmethod
    def from_disk(cls, base_dir, model_name=PRIMARY_MODEL):
        model_dict = joblib.load(os.path.join(base_dir, "models", "models", f"{model_name}.pkl"))
        preprocessor = joblib.load(os.path.join(base_dir, "data", "processed", "preprocessor.pkl"))
        return cls(model_dict, preprocessor, model_name)

    def score(self, prepared):
        """Score and explain every row of a PreparedUpload."""
        X = self.preprocessor.transform(prepared.features)
        proba = self.model.predict_proba(X)[:, 1]
        explanation = self.explainer(X)

        results = pd.DataFrame({
            "row": prepared.row_numbers,
            "fraud_probability": proba,
            "flagged": proba >= self.threshold,
        })
        if prepared.labels is not None:
            results["actual"] = prepared.labels.to_numpy()

        return ScoredBatch(
            results=results,
            shap_values=explanation.values,
            base_value=float(np.ravel(explanation.base_values)[0]),
            raw_features=prepared.features,
            threshold=self.threshold,
            model_name=self.model_name,
            model_version=self.model_version,
        )


def top_factors(shap_row, feature_values, feature_names=FEATURE_COLS, n=3):
    """
    Plain-language summary of the strongest contributions for one decision,
    e.g. "V14 = -5.21 (raises risk), Amount = 1.00 (lowers risk)".
    """
    order = np.argsort(-np.abs(shap_row))[:n]
    parts = []
    for i in order:
        direction = "raises risk" if shap_row[i] > 0 else "lowers risk"
        parts.append(f"{feature_names[i]} = {feature_values[i]:.2f} ({direction})")
    return ", ".join(parts)
