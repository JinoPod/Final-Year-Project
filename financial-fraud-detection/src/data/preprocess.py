"""
Data preprocessing for the European Credit Card Fraud dataset (Kaggle / ULB).

Implements Section 3.3 of the report:
  i)   Data cleaning  - check missing values and duplicates, scale Time and Amount
                        so they sit in a comparable range to the PCA features V1-V28.
  iii) Train/test split - stratified split; the test set keeps the original,
                        imbalanced class distribution.
SMOTE (step ii) is applied later to the training split only, in the training
pipeline, never here.

The scaler is fitted on the training split only (no information from the test
set leaks into it) and saved as preprocessor.pkl, so the dashboard can apply the
exact same transformation to uploaded transactions at inference time.
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "Class"
PCA_COLS = [f"V{i}" for i in range(1, 29)]
SCALED_COLS = ["Time", "Amount"]
# Column order the models expect (same order as the original dataset).
FEATURE_COLS = ["Time"] + PCA_COLS + ["Amount"]


def load_data(file_path):
    """Load the dataset and check it has the expected schema."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    validate_schema(df, require_target=True)
    return df


def validate_schema(df, require_target=False):
    """
    Check that a DataFrame has the columns the model needs.

    Used for both the training CSV and transactions uploaded to the dashboard
    (Section 3.6.2 i). Raises ValueError with a readable message if not.
    """
    required = FEATURE_COLS + ([TARGET_COL] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    non_numeric = [c for c in FEATURE_COLS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Columns must be numeric: {non_numeric}")


def analyze_data(df):
    """Print and return basic dataset statistics."""
    fraud_count = int(df[TARGET_COL].sum())
    total = len(df)
    stats = {
        "rows": total,
        "columns": df.shape[1],
        "fraud": fraud_count,
        "legitimate": total - fraud_count,
        "fraud_rate": fraud_count / total,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud: {stats['fraud']} ({stats['fraud_rate']:.4%}), "
          f"legitimate: {stats['legitimate']}")
    print(f"Missing values: {stats['missing_values']}, "
          f"duplicate rows: {stats['duplicate_rows']}")
    return stats


def clean_data(df):
    """
    Remove duplicate rows and rows with missing values.

    The raw dataset contains ~1,081 exact duplicate transactions. Leaving them
    in lets identical rows land in both train and test sets, which inflates the
    test scores, so they are dropped before splitting.
    """
    before = len(df)
    df = df.drop_duplicates()
    dropped_dupes = before - len(df)

    before = len(df)
    df = df.dropna()
    dropped_missing = before - len(df)

    print(f"Removed {dropped_dupes} duplicate rows and {dropped_missing} rows with missing values")
    return df.reset_index(drop=True)


def create_preprocessor():
    """
    Scale Time and Amount; pass V1-V28 through unchanged (already PCA-scaled).

    The transformers are listed in the original column order so the output
    keeps the order Time, V1..V28, Amount.
    """
    return ColumnTransformer(
        transformers=[
            ("time", StandardScaler(), ["Time"]),
            ("pca", "passthrough", PCA_COLS),
            ("amount", StandardScaler(), ["Amount"]),
        ],
        verbose_feature_names_out=False,
    )


def preprocess_data(file_path, test_size=0.2, random_state=42, output_dir=None):
    """
    Load, clean, split and scale the dataset.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor, feature_names)
               X arrays are scaled numpy arrays; y arrays are numpy int arrays.
    """
    df = load_data(file_path)
    raw_stats = analyze_data(df)

    df = clean_data(df)
    clean_stats = analyze_data(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int).to_numpy()

    # Stratified split keeps the same fraud rate in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit the scaler on training data only, then apply it to both splits.
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())
    assert feature_names == FEATURE_COLS, feature_names

    print(f"Train: {X_train_processed.shape}, fraud={y_train.sum()} ({y_train.mean():.4%})")
    print(f"Test:  {X_test_processed.shape}, fraud={y_test.sum()} ({y_test.mean():.4%})")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))
        np.save(os.path.join(output_dir, "X_train.npy"), X_train_processed)
        np.save(os.path.join(output_dir, "X_test.npy"), X_test_processed)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test)
        with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
            f.write("\n".join(feature_names))
        # Record what was done, for the report (dataset section) and reproducibility.
        summary = {
            "source_file": os.path.basename(file_path),
            "raw": raw_stats,
            "after_cleaning": clean_stats,
            "test_size": test_size,
            "random_state": random_state,
            "train_rows": int(len(y_train)),
            "train_fraud": int(y_train.sum()),
            "test_rows": int(len(y_test)),
            "test_fraud": int(y_test.sum()),
            "scaled_columns": SCALED_COLS,
        }
        with open(os.path.join(output_dir, "preprocessing_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved processed data to {output_dir}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_file = os.path.join(base_dir, "data", "raw", "creditcard.csv")
    output_dir = os.path.join(base_dir, "data", "processed")

    if not os.path.exists(data_file):
        raise SystemExit(
            f"Dataset not found at {data_file}\n"
            "Download the European Credit Card Fraud dataset (creditcard.csv) from Kaggle "
            "and place it in data/raw/."
        )

    preprocess_data(data_file, output_dir=output_dir)
