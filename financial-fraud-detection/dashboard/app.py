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
from src.inference import FraudScorer, prepare_upload, MAX_ROWS, PRIMARY_MODEL
from src.audit_log import AuditLog
from src.explanations import (
    load_reference, verdict, reasons, short_reasons, summary_sentence, risk_key, risk_label,
    global_takeaways, fmt_prob, REVIEW_FLOOR, RISK_LEVELS
)
import plotly.graph_objects as go
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


@st.cache_resource
def get_reference():
    """Training-set summary used to phrase explanations (see src/explanations.py)."""
    return load_reference(PROJECT_DIR)


ANONYMISED_NOTE = (
    "V1–V28 were anonymised by the data provider, so their business meaning (e.g. location or "
    "merchant) is unknown. Reasons therefore compare each value with normal transactions from "
    "the training data."
)

HOW_TO_READ = {
    "reasons": (
        "Each bar is one of the strongest factors behind this decision. **Red bars push towards "
        "fraud, blue bars push towards legitimate.** A longer bar means a stronger influence. "
        "The label shows the feature and its value in this transaction."
    ),
    "waterfall": (
        "Read from the bottom up. The chart starts at the model's **average score** across all "
        "transactions (E[f(X)], bottom). Each bar then adds (red) or subtracts (blue) one "
        "feature's contribution, ending at **this transaction's score** (f(x), top). The grey "
        "numbers on the left are the transaction's values.\n\n"
        "Scores are on SHAP's internal *log-odds* scale: 0 means a 50% fraud score, +2 is about "
        "88%, +4 is about 98%, −2 is about 12%."
    ),
    "lime": (
        "LIME fits a simple model around this one transaction. Bars to the right support "
        "**fraud**, bars to the left support **normal**. LIME is approximate and can disagree "
        "with SHAP; SHAP is this system's primary explanation, and LIME is shown for comparison."
    ),
    "confusion": (
        "Rows are what really happened; columns are what the model decided.\n"
        "- **Top-left**: legitimate transactions correctly let through.\n"
        "- **Bottom-right**: frauds caught.\n"
        "- **Top-right**: false alarms (legitimate transactions flagged), i.e. wasted review time.\n"
        "- **Bottom-left**: missed frauds, the most costly mistake."
    ),
    "roc": (
        "Each point is a possible alert threshold: how many frauds are caught (up) against how "
        "many legitimate transactions are wrongly flagged (right). A curve hugging the "
        "**top-left corner** is good; the diagonal is random guessing. With so few frauds this "
        "chart looks optimistic for almost any model, so read the precision-recall curve too."
    ),
    "pr": (
        "Each point is a possible alert threshold. **Recall** (across) is the share of frauds "
        "caught; **precision** (up) is the share of alerts that are real fraud. A curve near the "
        "**top-right** is good. Moving the threshold trades one for the other: catching more "
        "fraud means more false alarms."
    ),
    "shap_bar": (
        "How much each feature influences the model's decisions **on average**, across every "
        "test transaction. Longer bar = more influence. It does not show direction; the chart "
        "on the right does."
    ),
    "beeswarm": (
        "Each dot is one test transaction. Dots **right of the centre line** were pushed "
        "towards fraud by that feature, dots on the left towards legitimate. Colour is the "
        "feature's value: **red = high, blue = low**.\n\n"
        "Example: for V14, blue dots sit far to the right, so **low V14 values push strongly "
        "towards fraud**."
    ),
}


def how_to_read(key):
    with st.expander("ℹ️ How to read this chart"):
        st.markdown(HOW_TO_READ[key])


def plot_waterfall(shap_values, base_value, feature_values, feature_names, max_display=12):
    """SHAP waterfall from stored values (no model call needed)."""
    explanation = shap.Explanation(values=np.asarray(shap_values), base_values=base_value,
                                   data=np.asarray(feature_values), feature_names=list(feature_names))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)


def plot_reasons(reason_list):
    """Simple horizontal bar chart of the strongest factors (red = towards fraud)."""
    rs = list(reversed(reason_list))  # strongest at the top
    labels = [f"{r['feature']} = {r['value']:,.2f}" for r in rs]
    fig = go.Figure(go.Bar(
        x=[r["shap"] for r in rs], y=labels, orientation="h",
        marker_color=["#d62728" if r["raises_risk"] else "#1f77b4" for r in rs],
        hovertext=[("raises" if r["raises_risk"] else "lowers") + f" risk ({r['strength']})" for r in rs],
        hoverinfo="text",
    ))
    fig.add_vline(x=0, line_color="grey", line_width=1)
    fig.update_layout(
        height=60 + 45 * len(rs), margin=dict(l=10, r=10, t=10, b=40), showlegend=False,
        xaxis=dict(title="← towards legitimate     towards fraud →", showticklabels=False,
                   zeroline=False),
    )
    st.plotly_chart(fig, width="stretch")


def risk_level_table(threshold):
    """Markdown table explaining the three risk levels."""
    rows = [
        ("high", f"Score at or above the alert threshold ({threshold:.1%})"),
        ("review", f"Score between {REVIEW_FLOOR:.0%} and the alert threshold: the model leans "
                   "towards fraud but is not confident enough to raise an alert"),
        ("low", f"Score below {REVIEW_FLOOR:.0%}"),
    ]
    lines = ["| Risk level | When | Suggested action |", "|---|---|---|"]
    for key, when in rows:
        lvl = RISK_LEVELS[key]
        lines.append(f"| {lvl['icon']} **{lvl['label']}** | {when} | {lvl['action']} |")
    return "\n".join(lines) + (
        "\n\nThe **fraud score** is the model's estimated chance that a transaction is fraud. "
        "The alert threshold was tuned on training data to balance catching fraud against "
        "false alarms.")


def show_explanation(probability, threshold, shap_row, values, feature_names, base_value,
                     actual=None):
    """Verdict, suggested action and plain-language reasons for one transaction."""
    v = verdict(probability, threshold)
    box = {"high": st.error, "review": st.warning, "low": st.success}[v["key"]]
    box(f"#### {v['icon']} {v['label']}: {v['headline']}\n{v['detail']}\n\n"
        f"**Suggested action:** {v['action']}")
    if v["borderline"]:
        st.info(f"⚠️ {v['borderline_note']}")
    if actual is not None:
        st.caption(f"Actual label in the data: **{'fraud' if actual == 1 else 'legitimate'}**.")

    reason_list = reasons(shap_row, values, list(feature_names), get_reference())
    st.markdown("#### Why")
    st.markdown(summary_sentence(reason_list, probability, threshold))
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("\n".join(f"{'🔺' if r['raises_risk'] else '🔻'} {r['sentence']}\n"
                              for r in reason_list))
    with col2:
        plot_reasons(reason_list)
        how_to_read("reasons")
    st.caption(ANONYMISED_NOTE)

    with st.expander("Detailed SHAP chart (all features, for technical review)"):
        plot_waterfall(shap_row, base_value, values, feature_names)
        st.markdown(HOW_TO_READ["waterfall"])
    st.caption("Suggested actions are decision support only. The analyst makes the final decision.")

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
    """Verdict, plain-language reasons and SHAP/LIME detail for one test transaction."""
    raw = unscale(X_test[idx:idx + 1], get_scorer().preprocessor, feature_names)[0]
    shap_values, base_value = test_shap_values()
    _, _, _, y_test, _ = load_data()
    show_explanation(float(proba[idx]), threshold, shap_values[idx], raw, feature_names,
                     base_value, actual=int(y_test[idx]))

    if st.toggle("Compare with LIME (alternative explanation method)"):
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_test, feature_names=feature_names, class_names=["Normal", "Fraud"],
            mode="classification", random_state=42)
        lime_exp = lime_explainer.explain_instance(
            X_test[idx], load_model()["model"].predict_proba, num_features=10)
        fig = lime_exp.as_pyplot_figure()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(HOW_TO_READ["lime"])
        st.caption("LIME works on the scaled values the model sees, so Time and Amount appear "
                   "standardised here.")

    with st.expander("All feature values for this transaction"):
        st.dataframe(pd.DataFrame({"Feature": feature_names, "Value": raw}), hide_index=True)


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
            names = list(batch.raw_features.columns)
            ref = get_reference()
            batch.results["risk_level"] = [risk_label(p, batch.threshold)
                                           for p in batch.results["fraud_probability"]]
            batch.results["main_reasons"] = [
                short_reasons(batch.shap_values[i], raw[i], names, ref)
                for i in range(len(batch.results))
            ]
        cache[file_key] = (batch, batch_id)
    batch, batch_id = cache[file_key]
    results = batch.results
    keys = np.array([risk_key(p, batch.threshold) for p in results["fraud_probability"]])

    st.success(f"Scored and saved to the audit log as batch #{batch_id}.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions scored", f"{len(results):,}")
    c2.metric("🔴 High risk", f"{int((keys == 'high').sum()):,}")
    c3.metric("🟡 Needs review", f"{int((keys == 'review').sum()):,}")
    c4.metric("🟢 Low risk", f"{int((keys == 'low').sum()):,}")
    c5.metric("Not scored", f"{len(prepared.rejected):,}")
    with st.expander("ℹ️ What do the risk levels mean?"):
        st.markdown(risk_level_table(batch.threshold))

    if len(prepared.rejected):
        with st.expander(f"{len(prepared.rejected)} rows not scored"):
            st.dataframe(prepared.rejected, hide_index=True)

    if "actual" in results.columns:
        actual = results["actual"].to_numpy()
        n_fraud = int(actual.sum())
        caught = int(((keys == "high") & (actual == 1)).sum())
        false_alarms = int(((keys == "high") & (actual == 0)).sum())
        in_review = int(((keys == "review") & (actual == 1)).sum())
        st.info(
            f"This file has labels, so the decisions can be checked: **{caught} of {n_fraud}** "
            f"frauds were marked High risk, with **{false_alarms}** false alarm"
            f"{'' if false_alarms == 1 else 's'}."
            + (f" Another **{in_review}** fraud{' was' if in_review == 1 else 's were'} marked "
               "Needs review." if in_review else "")
        )

    st.subheader("Results")
    show = st.radio("Show", ["High risk and Needs review", "All transactions"], horizontal=True,
                    index=0 if (keys != "low").any() else 1)
    shown = results[keys != "low"] if show.startswith("High") else results
    shown = shown.sort_values("fraud_probability", ascending=False)
    columns = ["row", "risk_level", "fraud_probability", "main_reasons"] + \
        (["actual"] if "actual" in results.columns else [])
    st.dataframe(
        shown[columns], hide_index=True, width="stretch",
        column_config={
            "row": st.column_config.NumberColumn("Row in file"),
            "risk_level": st.column_config.TextColumn("Risk level"),
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud score", min_value=0.0, max_value=1.0, format="percent"),
            "main_reasons": st.column_config.TextColumn(
                "Main reasons (↑ raises risk, ↓ lowers risk)", width="large"),
            "actual": st.column_config.NumberColumn("Actual (1 = fraud)"),
        },
    )
    st.download_button("Download results (CSV)", results.to_csv(index=False).encode(),
                       file_name=f"scored_batch_{batch_id}.csv", mime="text/csv")

    st.subheader("Explain a transaction")
    if shown.empty:
        st.info("No transactions to explain with the current filter.")
        return
    labels = dict(zip(shown["row"], shown["risk_level"] + "  ·  score " +
                      shown["fraud_probability"].map(fmt_prob)))
    row = st.selectbox("Transaction", shown["row"].tolist(),
                       format_func=lambda r: f"Row {r}  ·  {labels[r]}")
    i = int(np.where(results["row"].to_numpy() == row)[0][0])
    show_explanation(float(results["fraud_probability"].iat[i]), batch.threshold,
                     batch.shap_values[i], batch.raw_features.iloc[i].to_numpy(dtype=float),
                     batch.raw_features.columns, batch.base_value,
                     actual=int(results["actual"].iat[i]) if "actual" in results.columns else None)


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
    show = c2.selectbox("Show", ["High risk and Needs review", "High risk only", "All decisions"])
    batch_id = None if batch_choice == "All batches" else int(batch_choice.split()[0][1:])
    decisions = log.get_decisions(batch_id=batch_id, limit=20_000)
    keys = np.array([risk_key(p, t) for p, t in zip(decisions["fraud_probability"],
                                                    decisions["threshold"])])
    decisions["risk_level"] = [risk_label(p, t) for p, t in zip(decisions["fraud_probability"],
                                                                decisions["threshold"])]
    if show == "High risk only":
        decisions = decisions[keys == "high"]
    elif show.startswith("High risk and"):
        decisions = decisions[keys != "low"]
    if decisions.empty:
        st.info("No decisions match this filter.")
        return
    st.dataframe(
        decisions[["id", "batch_id", "created_at", "source_file", "row_number", "risk_level",
                   "fraud_probability", "actual", "top_factors"]],
        hide_index=True, width="stretch",
        column_config={
            "id": "Decision", "batch_id": "Batch", "created_at": "Logged at (UTC)",
            "source_file": "File", "row_number": "Row in file", "risk_level": "Risk level",
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud score", min_value=0.0, max_value=1.0, format="percent"),
            "actual": st.column_config.NumberColumn("Actual (1 = fraud)"),
            "top_factors": st.column_config.TextColumn("Main factors (as logged)", width="large"),
        },
    )
    st.download_button("Download decision log (CSV)", decisions.to_csv(index=False).encode(),
                       file_name="decision_log.csv", mime="text/csv")

    st.subheader("Review a decision")
    labels = dict(zip(decisions["id"], decisions["risk_level"] + "  ·  score " +
                      decisions["fraud_probability"].map(fmt_prob)))
    decision_id = st.selectbox("Decision", decisions["id"].tolist(),
                               format_func=lambda i: f"#{i}  ·  {labels[i]}")
    d = log.get_decision(decision_id)
    st.markdown(
        f"- **Logged:** {d['created_at']} UTC, batch #{d['batch_id']}, row {d['row_number']} "
        f"of `{d['source_file']}`\n"
        f"- **Model:** {d['model_name']} (trained {d['model_version']}), alert threshold "
        f"{d['threshold']:.4f}"
    )
    if d["hash_ok"]:
        st.caption(f"✅ Integrity check passed: the stored inputs match their SHA-256 fingerprint "
                   f"({d['input_sha256'][:16]}…), so this record has not been altered.")
    else:
        st.error("Integrity check FAILED: the stored inputs do not match their fingerprint. "
                 "This record may have been altered.")
    show_explanation(d["fraud_probability"], d["threshold"], d["shap_values"], d["features"],
                     d["feature_names"], d["base_value"], actual=d["actual"])


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

        st.subheader("Risk levels")
        st.markdown(risk_level_table(float(load_model()["metadata"]["threshold"])))

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

        st.subheader("In plain terms")
        y = y_test.astype(int)
        tp = int(((y_pred == 1) & (y == 1)).sum())
        fp = int(((y_pred == 1) & (y == 0)).sum())
        n_fraud, n_legit = int(y.sum()), int((y == 0).sum())
        review = (y_pred_proba >= REVIEW_FLOOR) & (y_pred == 0)
        review_fraud = int((review & (y == 1)).sum())
        st.markdown(
            f"- ✅ **{tp / (tp + fp) * 100:.0f} out of every 100 alerts are real fraud** "
            f"({tp} of {tp + fp} alerts).\n"
            f"- 🎯 **The model catches about {round(tp / n_fraud * 4)} in 4 frauds** "
            f"({tp} of {n_fraud}, {tp / n_fraud:.0%}).\n"
            f"- 🔕 **False alarms are rare: about 1 in {round(n_legit / max(fp, 1), -3):,.0f} legitimate "
            f"transactions** is wrongly flagged ({fp} of {n_legit:,}).\n"
            f"- 🟡 A further {int(review.sum())} transactions fall in the *Needs review* band; "
            f"{review_fraud} of them {'is' if review_fraud == 1 else 'are'} fraud.\n"
            f"- ⚠️ **Weak point:** about {round((n_fraud - tp) / n_fraud * 4)} in 4 frauds look "
            "like normal transactions to the model and are missed."
        )

        st.subheader("Performance Metrics")
        show_metrics(y_test, y_pred, y_pred_proba)
        with st.expander("ℹ️ What do these metrics mean?"):
            st.markdown(
                "- **Precision**: of the transactions flagged, the share that really were fraud. "
                "High precision means few false alarms.\n"
                "- **Recall**: of all the real frauds, the share the model caught.\n"
                "- **F1 score**: one number balancing precision and recall (1.0 is perfect).\n"
                "- **ROC-AUC**: how well the model ranks frauds above legitimate transactions "
                "overall (1.0 is perfect, 0.5 is guessing).\n"
                "- **Accuracy** is shown for completeness but is misleading here: only 0.17% of "
                "transactions are fraud, so a model that never flags anything would still score "
                "99.8%."
            )
        st.caption(f"Alert threshold: {threshold:.4f}, tuned on the training data.")

        st.subheader("Confusion Matrix")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(create_interactive_confusion_matrix(y_test, y_pred, "XGBoost"))
        with col2:
            st.markdown(HOW_TO_READ["confusion"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ROC Curve")
            st.plotly_chart(create_interactive_roc_curve(y_test, y_pred_proba, "XGBoost"))
            how_to_read("roc")
        with col2:
            st.subheader("Precision-Recall Curve")
            st.plotly_chart(create_interactive_precision_recall_curve(y_test, y_pred_proba, "XGBoost"))
            how_to_read("pr")

        st.subheader("Global Explanation (SHAP)")
        st.info("**Key takeaways**\n\n" + "\n".join(f"- {line}" for line in
                                                     global_takeaways(get_reference())))
        st.caption(ANONYMISED_NOTE)
        shap_values, base_value = test_shap_values()
        raw = unscale(X_test, get_scorer().preprocessor, feature_names)
        exp = shap.Explanation(values=shap_values, base_values=np.full(len(shap_values), base_value),
                               data=raw, feature_names=feature_names)
        col1, col2 = st.columns(2)
        with col1:
            shap.plots.bar(exp, max_display=15, show=False)
            st.pyplot(plt.gcf())
            plt.close("all")
            how_to_read("shap_bar")
        with col2:
            shap.plots.beeswarm(exp, max_display=15, show=False)
            st.pyplot(plt.gcf())
            plt.close("all")
            how_to_read("beeswarm")

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
