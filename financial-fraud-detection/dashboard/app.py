"""
Streamlit dashboard for financial fraud detection with explainable AI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import sys
import hashlib
import joblib
import shap
import lime
import lime.lime_tabular

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data.preprocess import preprocess_data
from src.models.report_figures import unscale
from src.inference import FraudScorer, prepare_upload, top_factors, MAX_ROWS, PRIMARY_MODEL
from src.audit_log import AuditLog
from src.visualization.visualization import (
    create_interactive_roc_curve, create_interactive_precision_recall_curve,
    create_interactive_confusion_matrix
)

# Set page config
st.set_page_config(
    page_title="Financial Fraud Detection Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "raw", "creditcard.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "models")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_DB_PATH = os.path.join(PROJECT_DIR, "logs", "audit.db")
SAMPLE_DIR = os.path.join(PROJECT_DIR, "data", "sample")


@st.cache_resource
def get_scorer():
    """Deployed model + preprocessor + SHAP explainer (loaded once)."""
    return FraudScorer.from_disk(PROJECT_DIR)


@st.cache_resource
def get_audit_log():
    return AuditLog(AUDIT_DB_PATH)


def plot_waterfall(shap_values, base_value, feature_values, feature_names, max_display=12):
    """SHAP waterfall from stored values (no model call needed)."""
    explanation = shap.Explanation(values=np.asarray(shap_values), base_values=base_value,
                                   data=np.asarray(feature_values), feature_names=list(feature_names))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Bars show how much each feature pushed this transaction towards fraud (red) or "
               "legitimate (blue), in log-odds. E[f(x)] is the model's average output; f(x) is "
               "this transaction's output before it is converted to a probability. Feature values "
               "are shown as uploaded (Time and Amount unscaled).")

# Function to load data
@st.cache_data
def load_data():
    """Load and preprocess the data."""
    if os.path.exists(os.path.join(PROCESSED_DIR, "X_test.npy")):
        # Load preprocessed data
        X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
        X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
        y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
        y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
        
        # Load feature names
        with open(os.path.join(PROCESSED_DIR, "feature_names.txt"), "r") as f:
            feature_names = f.read().splitlines()
        
        return X_train, X_test, y_train, y_test, feature_names
    else:
        # Preprocess data
        X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(
            DATA_PATH, output_dir=PROCESSED_DIR
        )
        return X_train, X_test, y_train, y_test, feature_names

@st.cache_resource
def load_model():
    """The deployed XGBoost model: {'model', 'metadata', 'timestamp'}."""
    return joblib.load(os.path.join(MODEL_DIR, "models", f"{PRIMARY_MODEL}.pkl"))


@st.cache_data
def test_predictions():
    """Fraud probability and decision for every test transaction, at the tuned threshold."""
    _, X_test, _, _, _ = load_data()
    model_dict = load_model()
    threshold = float(model_dict["metadata"]["threshold"])
    proba = model_dict["model"].predict_proba(X_test)[:, 1]
    return (proba >= threshold).astype(int), proba, threshold


@st.cache_data(show_spinner="Computing SHAP values for the test set...")
def test_shap_values():
    """Exact TreeSHAP values for the whole test set (same explainer as the upload page)."""
    _, X_test, _, _, _ = load_data()
    sv = get_scorer().explainer(X_test)
    return sv.values, float(np.ravel(sv.base_values)[0])


def show_metrics(y_true, y_pred, y_pred_proba):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
    cols[1].metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
    cols[2].metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
    cols[3].metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
    cols[4].metric("ROC-AUC", f"{roc_auc_score(y_true, y_pred_proba):.4f}")


def explain_transaction(X_test, feature_names, idx, proba, threshold):
    """Decision, SHAP waterfall and LIME explanation for one test transaction."""
    st.subheader("Prediction")
    p = proba[idx]
    if p >= threshold:
        st.error(f"Flagged as fraud (fraud probability {p:.4f}, threshold {threshold:.4f})")
    else:
        st.success(f"Not flagged (fraud probability {p:.4f}, threshold {threshold:.4f})")

    raw = unscale(X_test[idx:idx + 1], get_scorer().preprocessor, feature_names)[0]

    st.subheader("SHAP Explanation")
    shap_values, base_value = test_shap_values()
    plot_waterfall(shap_values[idx], base_value, raw, feature_names)

    st.subheader("LIME Explanation")
    st.caption("LIME fits a simple local model around this transaction. It is shown for "
               "comparison; SHAP is the primary explanation method.")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_test, feature_names=feature_names, class_names=["Normal", "Fraud"],
        mode="classification", random_state=42)
    lime_exp = lime_explainer.explain_instance(
        X_test[idx], load_model()["model"].predict_proba, num_features=10)
    fig = lime_exp.as_pyplot_figure()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature Values")
    st.dataframe(pd.DataFrame({"Feature": feature_names, "Value": raw}))


def page_upload():
    """Upload & Score page (report Sections 3.6.1 - 3.6.4)."""
    st.title("Upload & Score Transactions")
    st.write(
        "Upload a CSV of transactions. Each one is scored by the deployed XGBoost model, "
        "explained with SHAP, and written to the audit log."
    )
    with st.expander("File format"):
        st.markdown(
            "- Required columns: `Time`, `V1` … `V28`, `Amount` (numeric, any order).\n"
            "- Optional `Class` column (0 = legitimate, 1 = fraud): if present, the page also shows "
            "how many frauds were caught.\n"
            "- Other columns are ignored. Rows with missing values are not scored and are listed.\n"
            f"- Up to {MAX_ROWS:,} rows per file.\n\n"
            f"Demo files made from the held-out test set are in `{os.path.relpath(SAMPLE_DIR, PROJECT_DIR)}` "
            "(create them with `python src/data/make_sample.py`)."
        )

    uploaded = st.file_uploader("Transactions CSV", type=["csv"])
    if uploaded is None:
        st.info("Choose a CSV file to begin.")
        return

    file_bytes = uploaded.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Could not read the file as CSV: {e}")
        return
    try:
        prepared = prepare_upload(df)
    except ValueError as e:
        st.error(f"File rejected: {e}")
        return
    if len(prepared.features) == 0:
        st.error("None of the rows could be scored (all have missing values).")
        st.dataframe(prepared.rejected, hide_index=True)
        return

    try:
        scorer = get_scorer()
    except Exception as e:
        st.error(f"Could not load the deployed model: {e}")
        return

    # Score and log once per file per session (Streamlit re-runs the script on every click).
    file_key = hashlib.sha256(file_bytes).hexdigest()
    cache = st.session_state.setdefault("scored_uploads", {})
    if file_key not in cache:
        with st.spinner(f"Scoring {len(prepared.features):,} transactions..."):
            batch = scorer.score(prepared)
            batch_id = get_audit_log().log_batch(
                batch, source_file=uploaded.name, file_bytes=file_bytes,
                total_rows=prepared.total_rows, rejected_rows=len(prepared.rejected))
            raw = batch.raw_features.to_numpy(dtype=float)
            batch.results["top_factors"] = [
                top_factors(batch.shap_values[i], raw[i], list(batch.raw_features.columns))
                for i in range(len(batch.results))
            ]
        cache[file_key] = (batch, batch_id)
    batch, batch_id = cache[file_key]
    results = batch.results

    st.success(f"Scored and saved to the audit log as batch #{batch_id}.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions in file", f"{prepared.total_rows:,}")
    c2.metric("Scored", f"{len(results):,}")
    c3.metric("Not scored", f"{len(prepared.rejected):,}")
    c4.metric("Flagged as fraud", f"{int(results['flagged'].sum()):,}")
    st.caption(f"Model: {batch.model_name} (trained {batch.model_version}); "
               f"a transaction is flagged when its fraud probability is ≥ {batch.threshold:.4f}.")

    if len(prepared.rejected):
        with st.expander(f"{len(prepared.rejected)} rows not scored"):
            st.dataframe(prepared.rejected, hide_index=True)

    if "actual" in results.columns:
        actual_fraud = int(results["actual"].sum())
        caught = int((results["flagged"] & (results["actual"] == 1)).sum())
        false_alarms = int((results["flagged"] & (results["actual"] == 0)).sum())
        st.info(f"The file is labelled: caught **{caught} of {actual_fraud}** frauds, "
                f"with **{false_alarms}** false alarms.")

    st.subheader("Results")
    only_flagged = st.checkbox("Show only flagged transactions", value=bool(results["flagged"].any()))
    shown = results[results["flagged"]] if only_flagged else results
    shown = shown.sort_values("fraud_probability", ascending=False)
    st.dataframe(
        shown, hide_index=True, width="stretch",
        column_config={
            "row": st.column_config.NumberColumn("Row in file"),
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud probability", min_value=0.0, max_value=1.0, format="%.4f"),
            "flagged": st.column_config.CheckboxColumn("Flagged"),
            "actual": st.column_config.NumberColumn("Actual (Class)"),
            "top_factors": st.column_config.TextColumn("Main reasons", width="large"),
        },
    )
    st.download_button("Download results (CSV)", results.to_csv(index=False).encode(),
                       file_name=f"scored_batch_{batch_id}.csv", mime="text/csv")

    st.subheader("Explain a transaction")
    if shown.empty:
        st.info("No transactions to explain with the current filter.")
        return
    row = st.selectbox("Row in file", shown["row"].tolist())
    i = int(np.where(results["row"].to_numpy() == row)[0][0])
    prob = float(results["fraud_probability"].iat[i])
    if results["flagged"].iat[i]:
        st.error(f"Flagged as fraud — probability {prob:.4f} (threshold {batch.threshold:.4f})")
    else:
        st.success(f"Not flagged — probability {prob:.4f} (threshold {batch.threshold:.4f})")
    st.write(f"**Main reasons:** {results['top_factors'].iat[i]}")
    plot_waterfall(batch.shap_values[i], batch.base_value,
                   batch.raw_features.iloc[i].to_numpy(), batch.raw_features.columns)


def page_audit():
    """Audit Log page (report Section 3.6.5)."""
    st.title("Audit Log")
    st.write(
        "Every decision made on the Upload & Score page, with the explanation stored at the time. "
        "Explanations here are read from the log; the model is not re-run. The log is append-only."
    )
    log = get_audit_log()
    batches = log.list_batches()
    if batches.empty:
        st.info("No decisions logged yet. Score a file on the Upload & Score page first.")
        return

    st.subheader("Batches")
    st.dataframe(
        batches[["id", "created_at", "source_file", "model_name", "model_version", "threshold",
                 "total_rows", "scored_rows", "rejected_rows", "flagged_rows"]],
        hide_index=True, width="stretch",
        column_config={"created_at": "Logged at (UTC)", "id": "Batch"},
    )

    st.subheader("Decisions")
    c1, c2 = st.columns([2, 1])
    batch_choice = c1.selectbox(
        "Batch", ["All batches"] + [f"#{b} — {f}" for b, f in zip(batches["id"], batches["source_file"])])
    flagged_only = c2.checkbox("Flagged only", value=True)
    batch_id = None if batch_choice == "All batches" else int(batch_choice.split()[0][1:])
    decisions = log.get_decisions(batch_id=batch_id, flagged_only=flagged_only, limit=5000)
    if decisions.empty:
        st.info("No decisions match this filter.")
        return
    st.dataframe(
        decisions.drop(columns=["input_sha256"]), hide_index=True, width="stretch",
        column_config={
            "id": "Decision", "batch_id": "Batch", "created_at": "Logged at (UTC)",
            "row_number": "Row in file",
            "fraud_probability": st.column_config.NumberColumn("Fraud probability", format="%.4f"),
            "top_factors": st.column_config.TextColumn("Main reasons", width="large"),
        },
    )
    st.download_button("Download decision log (CSV)", decisions.to_csv(index=False).encode(),
                       file_name="decision_log.csv", mime="text/csv")

    st.subheader("Review a decision")
    decision_id = st.selectbox("Decision", decisions["id"].tolist())
    d = log.get_decision(decision_id)
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud probability", f"{d['fraud_probability']:.4f}")
    c2.metric("Decision", "Fraud" if d["flagged"] else "Legitimate")
    c3.metric("Threshold", f"{d['threshold']:.4f}")
    st.markdown(
        f"- **Logged:** {d['created_at']} UTC, batch #{d['batch_id']}, row {d['row_number']} "
        f"of `{d['source_file']}`\n"
        f"- **Model:** {d['model_name']} (trained {d['model_version']})\n"
        + (f"- **Actual label in file:** {'fraud' if d['actual'] == 1 else 'legitimate'}\n"
           if d["actual"] is not None else "")
        + f"- **Main reasons:** {d['top_factors']}"
    )
    if d["hash_ok"]:
        st.caption(f"Input integrity check passed (SHA-256 {d['input_sha256'][:16]}…).")
    else:
        st.error("Input integrity check FAILED: the stored input does not match its hash.")
    plot_waterfall(d["shap_values"], d["base_value"], d["features"], d["feature_names"])


# Main function
def main():
    """Main function for the dashboard."""
    st.sidebar.title("Financial Fraud Detection")
    page = st.sidebar.selectbox(
        "Navigation",
        ["Home", "Upload & Score", "Audit Log", "Model Performance", "Transaction Explanation"]
    )

    if page == "Home":
        st.title("Financial Fraud Detection Dashboard")
        st.write("""
        This dashboard detects fraudulent credit card transactions with a tuned XGBoost model
        and explains every decision with SHAP.

        ### Pages:
        - **Upload & Score**: Upload a CSV of transactions, get fraud decisions and a SHAP explanation for each
        - **Audit Log**: Review past decisions and their stored explanations
        - **Model Performance**: How well the model performs on the held-out test set
        - **Transaction Explanation**: Understand why a test-set transaction was or was not flagged
        """)

        st.subheader("Project Information")
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **Dataset**: European Credit Card Fraud (Kaggle / ULB), 284,807 transactions,
            0.17% fraud; features Time, Amount and PCA components V1-V28

            **Model**: XGBoost, tuned with Optuna

            **Explainability**: SHAP (exact TreeSHAP), with LIME for comparison
            """)
        with col2:
            st.info("""
            **Imbalance handling**: SMOTE on the training data only

            **Decision threshold**: tuned for F1 with cross-validation on the training
            data; the test set is used only for the final evaluation

            **Audit**: every uploaded decision is logged with its SHAP explanation
            """)

        X_train, X_test, y_train, y_test, _ = load_data()
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Samples", f"{X_train.shape[0]:,}")
        col2.metric("Test Samples", f"{X_test.shape[0]:,}")
        col3.metric("Features", f"{X_train.shape[1]:,}")

        st.subheader("Class Distribution")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for a, y, title in ((ax[0], y_train, "Training Set"), (ax[1], y_test, "Test Set")):
            counts = np.bincount(y.astype(int))
            a.bar(["Normal", "Fraud"], counts)
            a.set_title(title)
            for i, count in enumerate(counts):
                a.text(i, count + 10, f"{count:,}", ha="center")
        st.pyplot(fig)
        plt.close(fig)

    elif page == "Upload & Score":
        page_upload()

    elif page == "Audit Log":
        page_audit()

    elif page == "Model Performance":
        st.title("Model Performance")
        st.caption("XGBoost on the held-out test set (56,746 transactions, never used in training "
                   "or tuning).")
        _, X_test, _, y_test, feature_names = load_data()
        y_pred, y_pred_proba, threshold = test_predictions()
        st.info(f"Decision threshold: {threshold:.4f} (tuned on the training data)")

        st.subheader("Performance Metrics")
        show_metrics(y_test, y_pred, y_pred_proba)

        st.subheader("Confusion Matrix")
        st.plotly_chart(create_interactive_confusion_matrix(y_test, y_pred, "XGBoost"))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ROC Curve")
            st.plotly_chart(create_interactive_roc_curve(y_test, y_pred_proba, "XGBoost"))
        with col2:
            st.subheader("Precision-Recall Curve")
            st.plotly_chart(create_interactive_precision_recall_curve(y_test, y_pred_proba, "XGBoost"))

        st.subheader("Global Explanation (SHAP)")
        shap_values, base_value = test_shap_values()
        raw = unscale(X_test, get_scorer().preprocessor, feature_names)
        exp = shap.Explanation(values=shap_values, base_values=np.full(len(shap_values), base_value),
                               data=raw, feature_names=feature_names)
        col1, col2 = st.columns(2)
        with col1:
            shap.plots.bar(exp, max_display=15, show=False)
            st.pyplot(plt.gcf())
            plt.close("all")
            st.caption("Average impact of each feature on the model's output across the test set.")
        with col2:
            shap.plots.beeswarm(exp, max_display=15, show=False)
            st.pyplot(plt.gcf())
            plt.close("all")
            st.caption("Each dot is one transaction. Position shows whether the feature pushed it "
                       "towards fraud (right) or legitimate (left); colour shows the feature value.")

    elif page == "Transaction Explanation":
        st.title("Transaction Explanation")
        _, X_test, _, y_test, feature_names = load_data()
        y_pred, y_pred_proba, threshold = test_predictions()

        groups = {
            "Fraudulent (True Positive)": np.where((y_test == 1) & (y_pred == 1))[0],
            "Normal (True Negative)": np.where((y_test == 0) & (y_pred == 0))[0],
            "False Alarm (False Positive)": np.where((y_test == 0) & (y_pred == 1))[0],
            "Missed Fraud (False Negative)": np.where((y_test == 1) & (y_pred == 0))[0],
        }
        transaction_type = st.selectbox("Select Transaction Type", list(groups))
        indices = groups[transaction_type]
        if len(indices) == 0:
            st.warning("No transactions of this type in the test set.")
            return
        transaction_idx = st.selectbox("Select Transaction", indices)
        explain_transaction(X_test, feature_names, int(transaction_idx), y_pred_proba, threshold)


if __name__ == "__main__":
    main()
