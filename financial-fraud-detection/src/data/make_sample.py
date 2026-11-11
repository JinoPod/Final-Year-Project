"""
Create demo CSV files for the dashboard's Upload & Score page.

Rows are drawn only from the TEST split (same cleaning and split as
preprocess.py), so the deployed model has never seen them during training.

Writes to data/sample/:
  sample_transactions.csv            - with the 'Class' label, so the dashboard
                                       can also show how many it got right
  sample_transactions_unlabelled.csv - same rows without 'Class', as a bank
                                       would actually send them

Run:  python src/data/make_sample.py [--rows 1000] [--fraud 30]
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.data.preprocess import FEATURE_COLS, TARGET_COL, clean_data, load_data


def main(n_rows, n_fraud, seed=7):
    df = clean_data(load_data(os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")))
    X, y = df[FEATURE_COLS], df[TARGET_COL].astype(int).to_numpy()
    # Identical call to preprocess.py, so this reproduces the exact test split.
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Check: scaling these rows must give exactly the saved X_test.npy.
    pre = joblib.load(os.path.join(BASE_DIR, "data", "processed", "preprocessor.pkl"))
    saved = np.load(os.path.join(BASE_DIR, "data", "processed", "X_test.npy"))
    if not np.allclose(pre.transform(X_test), saved):
        raise SystemExit("Test split does not match data/processed; re-run preprocess.py first.")

    test = X_test.assign(**{TARGET_COL: y_test})
    rng = np.random.default_rng(seed)
    fraud = test[test[TARGET_COL] == 1]
    legit = test[test[TARGET_COL] == 0]
    n_fraud = min(n_fraud, len(fraud))
    sample = pd.concat([
        fraud.iloc[rng.choice(len(fraud), n_fraud, replace=False)],
        legit.iloc[rng.choice(len(legit), n_rows - n_fraud, replace=False)],
    ]).sample(frac=1, random_state=seed)  # shuffle

    out_dir = os.path.join(BASE_DIR, "data", "sample")
    os.makedirs(out_dir, exist_ok=True)
    sample.to_csv(os.path.join(out_dir, "sample_transactions.csv"), index=False)
    sample.drop(columns=[TARGET_COL]).to_csv(
        os.path.join(out_dir, "sample_transactions_unlabelled.csv"), index=False)
    print(f"Wrote {len(sample)} test-split transactions ({n_fraud} fraud) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create demo upload files from the test split")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--fraud", type=int, default=30)
    args = parser.parse_args()
    main(args.rows, args.fraud)
