# XAI-Based Fraud Detection

An interpretable machine-learning system for credit card transaction monitoring.
A tuned XGBoost model (trained with SMOTE) flags fraudulent transactions, SHAP explains
each decision, and a Streamlit dashboard lets an analyst upload transactions, see the
explanations and review an append-only audit log.

Dataset: [European Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 492 fraud; features `Time`, `V1`–`V28` (PCA), `Amount`, label `Class`).

## Results (held-out test set, 56,746 transactions, 95 fraud)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **XGBoost (primary)** | **0.960** | 0.758 | **0.847** | 0.978 |
| Random Forest | 0.947 | 0.758 | 0.842 | 0.961 |
| LightGBM | 0.889 | 0.758 | 0.818 | 0.972 |
| Logistic Regression | 0.826 | 0.747 | 0.785 | 0.967 |

Stability over 5 independently re-tuned splits: XGBoost F1 **0.832 ± 0.031**,
precision 0.922 ± 0.023, recall 0.760 ± 0.055 (`models/results/stability.json`).
Full numbers: `models/results/metrics.json`.

## Setup

Requires Python 3.11 (the saved models were pickled with scikit-learn 1.9.0).

```bash
conda create -n fyp python=3.11
conda activate fyp
pip install -r requirements.txt
```

Place `creditcard.csv` in `data/raw/`.

## Pipeline

Run from this folder (`financial-fraud-detection/`), in order:

```bash
python src/data/preprocess.py          # clean, scale, stratified 80/20 split -> data/processed/
python src/models/train_model.py       # Optuna-tuned XGBoost + baselines -> models/models/, models/results/
python src/models/stability.py         # 5 re-tuned splits + bootstrap CIs (slow, ~1h)
python src/models/report_figures.py    # SHAP summary + example waterfalls -> models/visualizations/report/
python src/data/make_sample.py         # demo CSVs for the upload page -> data/sample/
```

Training is leak-free: SMOTE is applied only inside cross-validation training folds, the
decision threshold is chosen on out-of-fold predictions, and the test set is used once.

## Dashboard

```bash
streamlit run dashboard/app.py
```

Then open http://localhost:8501. Pages:

- **Home**: project overview
- **Upload & Score**: upload a CSV, get fraud scores and a SHAP explanation per row; each upload is written to the audit log
- **Audit Log**: browse past batches and decisions; stored input hashes are re-checked to detect tampering
- **Model Performance**: XGBoost test-set metrics, confusion matrix, ROC/PR curves and global SHAP charts
- **Transaction Explanation**: SHAP (and LIME) explanation for any test transaction

The dashboard shows only the primary XGBoost model. The baseline comparison is in
`models/results/` and `documentation/results_for_report.md`.

Every decision is shown as a risk level (🔴 High risk, 🟡 Needs review, 🟢 Low risk) with a
suggested action and plain-language reasons (`src/explanations.py`). The reasons compare each
value with normal training transactions using `models/models/feature_reference.json`, which is
rebuilt automatically if missing (delete it after retraining).

Try the upload page with `data/sample/sample_transactions.csv` (1,000 test transactions, 30 fraud).
The audit log is stored in `logs/audit.db` (SQLite, append-only).

## Project structure

```
financial-fraud-detection/
├── dashboard/app.py            # Streamlit dashboard
├── data/
│   ├── raw/                    # creditcard.csv (not in git)
│   ├── processed/              # scaled train/test arrays + preprocessor.pkl
│   └── sample/                 # demo upload files
├── logs/audit.db               # audit log (created on first upload, not in git)
├── models/
│   ├── models/                 # saved models (*.pkl)
│   ├── results/                # metrics, stability results, logs
│   └── visualizations/         # charts, incl. report/ figures
└── src/
    ├── data/                   # preprocess.py, make_sample.py
    ├── models/                 # train_model.py, stability.py, report_figures.py
    ├── inference.py            # scoring + SHAP for uploaded rows
    └── audit_log.py            # append-only SQLite audit log
```
