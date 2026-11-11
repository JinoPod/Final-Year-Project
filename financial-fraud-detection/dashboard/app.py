"""
Streamlit dashboard for financial fraud detection with explainable AI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
from src.models.explainability import explain_with_shap, explain_with_lime, explain_prediction
from src.inference import FraudScorer, prepare_upload, top_factors, MAX_ROWS
from src.audit_log import AuditLog
from src.visualization.visualization import (
    create_interactive_roc_curve, create_interactive_precision_recall_curve,
    create_interactive_confusion_matrix, create_interactive_feature_importance
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

# Function to load models
@st.cache_resource
def load_models():
    """Load trained models."""
    models = {}
    # XGBoost is the primary model (report Section 3.4) so it is listed first;
    # the rest are comparison baselines. Missing files are skipped.
    model_files = ["xgboost", "random_forest", "lightgbm", "logistic_regression", "ensemble"]
    
    for model_file in model_files:
        try:
            model_path = os.path.join(MODEL_DIR, "models", f"{model_file}.pkl")
            if os.path.exists(model_path):
                model_dict = joblib.load(model_path)
                # Ensure model has the expected structure
                if 'model' in model_dict:
                    models[model_file] = model_dict
                    print(f"Successfully loaded {model_file} model")
                else:
                    print(f"Warning: {model_file} model has unexpected structure")
        except Exception as e:
            print(f"Error loading {model_file} model: {str(e)}")
    
    return models

# Function to make predictions
def make_predictions(model, X, model_dict=None):
    """Make predictions using the selected model."""
    # Get the threshold from model metadata if available
    threshold = 0.5  # Default threshold
    if model_dict and 'metadata' in model_dict and 'threshold' in model_dict['metadata']:
        threshold = model_dict['metadata']['threshold']
        st.info(f"Using optimized threshold: {threshold:.4f}")
    else:
        # For imbalanced datasets, a lower threshold is often better
        threshold = 0.01
        st.warning(f"No threshold found in model metadata. Using default threshold for imbalanced data: {threshold:.4f}")
    
    if isinstance(model, dict) and "models" in model:
        # Ensemble model
        models = model["models"]
        weights = model["weights"]
        
        # Get predictions from each model
        y_pred_probas = []
        for m in models:
            y_pred_proba = m.predict_proba(X)[:, 1]
            y_pred_probas.append(y_pred_proba)
        
        # Weighted average of probabilities
        y_pred_proba = np.zeros_like(y_pred_probas[0])
        for i, weight in enumerate(weights):
            y_pred_proba += weight * y_pred_probas[i]
        
        # Convert to binary predictions using the threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        # Regular model
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    return y_pred, y_pred_proba

# Function to create metrics
def create_metrics(y_true, y_pred, y_pred_proba):
    """Calculate and display metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")
    col5.metric("AUC-ROC", f"{auc_roc:.4f}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc
    }

# Function to explain a transaction
def explain_transaction(model, X, feature_names, transaction_idx, threshold=0.5):
    """Explain a specific transaction, classifying it with the model's tuned threshold."""
    # Get the transaction
    transaction = X[transaction_idx:transaction_idx+1]
    
    # Make prediction
    if isinstance(model, dict) and "models" in model:
        # Ensemble model
        models = model["models"]
        weights = model["weights"]
        
        # Get predictions from each model
        y_pred_probas = []
        for m in models:
            y_pred_proba = m.predict_proba(transaction)[:, 1]
            y_pred_probas.append(y_pred_proba)
        
        # Weighted average of probabilities
        y_pred_proba = np.zeros_like(y_pred_probas[0])
        for i, weight in enumerate(weights):
            y_pred_proba += weight * y_pred_probas[i]
        
        # Convert to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Use first model for explanation
        model_for_explanation = models[0]
    else:
        # Regular model
        y_pred_proba = model.predict_proba(transaction)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        model_for_explanation = model

    # Display prediction
    st.subheader("Prediction")
    if y_pred[0] == 1:
        st.error(f"Fraudulent Transaction (fraud probability {y_pred_proba[0]:.4f}, threshold {threshold:.4f})")
    else:
        st.success(f"Normal Transaction (fraud probability {y_pred_proba[0]:.4f}, threshold {threshold:.4f})")
    
    # SHAP explanation
    st.subheader("SHAP Explanation")
    
    try:
        # Create SHAP explainer with proper handling for different model types
        if hasattr(model_for_explanation, 'predict_proba'):
            # For scikit-learn models
            explainer = shap.Explainer(model_for_explanation.predict_proba, X[:100])
            shap_values = explainer(transaction)
            
            # Plot SHAP values
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0, :, 1], show=False)
            st.pyplot(fig)
        elif str(type(model_for_explanation)).endswith("xgboost.sklearn.XGBClassifier'>"):
            # For XGBoost models
            explainer = shap.TreeExplainer(model_for_explanation)
            shap_values = explainer(transaction)
            
            # Plot SHAP values
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
        else:
            # Fallback approach
            st.warning("Using fallback SHAP explanation method")
            explainer = shap.Explainer(model_for_explanation, X[:100])
            shap_values = explainer(transaction)
            
            # Plot SHAP values
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {str(e)}")
        st.info("Displaying feature importance instead")
        
        # Display feature importance as fallback
        if hasattr(model_for_explanation, 'feature_importances_'):
            importances = model_for_explanation.feature_importances_
            feature_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(feature_imp.set_index('Feature'))
    
    # LIME explanation
    st.subheader("LIME Explanation")
    
    # Create LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X,
        feature_names=feature_names,
        class_names=["Normal", "Fraud"],
        mode="classification"
    )
    
    # Generate explanation
    lime_exp = lime_explainer.explain_instance(
        transaction[0], 
        model_for_explanation.predict_proba,
        num_features=10
    )
    
    # Plot LIME values
    fig = lime_exp.as_pyplot_figure()
    st.pyplot(fig)
    
    # Feature values
    st.subheader("Feature Values")
    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": transaction[0]
    })
    st.dataframe(feature_df)

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
    # Sidebar
    st.sidebar.title("Financial Fraud Detection")
    
    # Load data
    with st.sidebar.expander("Data", expanded=False):
        if st.button("Load Data"):
            with st.spinner("Loading data..."):
                X_train, X_test, y_train, y_test, feature_names = load_data()
                st.session_state["data_loaded"] = True
                st.session_state["X_train"] = X_train
                st.session_state["X_test"] = X_test
                st.session_state["y_train"] = y_train
                st.session_state["y_test"] = y_test
                st.session_state["feature_names"] = feature_names
                st.success("Data loaded successfully!")
    
    # Load models
    with st.sidebar.expander("Models", expanded=False):
        if st.button("Load Models"):
            with st.spinner("Loading models..."):
                models = load_models()
                st.session_state["models_loaded"] = True
                st.session_state["models"] = models
                st.success(f"Loaded {len(models)} models successfully!")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Home", "Upload & Score", "Audit Log", "Model Performance",
         "Transaction Explanation", "Live Monitoring"]
    )
    
    # Home page
    if page == "Home":
        st.title("Financial Fraud Detection Dashboard")
        st.write("""
        This dashboard provides tools for detecting and explaining fraudulent financial transactions.
        
        ### Features:
        - **Upload & Score**: Upload a CSV of transactions, get fraud decisions and a SHAP explanation for each
        - **Audit Log**: Review past decisions and their stored explanations
        - **Model Performance**: Evaluate and compare the fraud detection models
        - **Transaction Explanation**: Understand why a test-set transaction was flagged
        - **Live Monitoring**: Simulated stream of test-set transactions

        ### Getting Started:
        - **Upload & Score** and **Audit Log** work straight away.
        - For the other pages, load the data and models using the buttons in the sidebar.
        """)

        # Display project info
        st.subheader("Project Information")
        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            **Dataset**: European Credit Card Fraud (Kaggle / ULB), 284,807 transactions,
            0.17% fraud; features Time, Amount and PCA components V1-V28

            **Models**:
            - XGBoost (primary, tuned with Optuna)
            - Random Forest, LightGBM, Logistic Regression (baselines)

            **Explainability**:
            - SHAP (TreeSHAP for XGBoost)
            - LIME (comparison)
            """)

        with col2:
            st.info("""
            **Imbalance handling**: SMOTE on the training data only

            **Decision threshold**: tuned for F1 with cross-validation on the training
            data; the test set is used only for the final evaluation

            **Audit**: every uploaded decision is logged with its SHAP explanation
            """)
        
        # Check if data is loaded
        if "data_loaded" in st.session_state and st.session_state["data_loaded"]:
            # Display data info
            X_train = st.session_state["X_train"]
            X_test = st.session_state["X_test"]
            y_train = st.session_state["y_train"]
            y_test = st.session_state["y_test"]
            
            st.subheader("Data Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Training Samples", f"{X_train.shape[0]:,}")
            col2.metric("Test Samples", f"{X_test.shape[0]:,}")
            col3.metric("Features", f"{X_train.shape[1]:,}")
            
            # Display class distribution
            st.subheader("Class Distribution")
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Training set
            train_counts = np.bincount(y_train)
            ax[0].bar(["Normal", "Fraud"], train_counts)
            ax[0].set_title("Training Set")
            for i, count in enumerate(train_counts):
                ax[0].text(i, count + 10, f"{count:,}", ha="center")
            
            # Test set
            test_counts = np.bincount(y_test)
            ax[1].bar(["Normal", "Fraud"], test_counts)
            ax[1].set_title("Test Set")
            for i, count in enumerate(test_counts):
                ax[1].text(i, count + 10, f"{count:,}", ha="center")
            
            st.pyplot(fig)
    
    elif page == "Upload & Score":
        page_upload()

    elif page == "Audit Log":
        page_audit()

    # Model Performance page
    elif page == "Model Performance":
        st.title("Model Performance")
        
        # Check if data and models are loaded
        if not ("data_loaded" in st.session_state and st.session_state["data_loaded"]):
            st.warning("Please load the data first.")
            return
        
        if not ("models_loaded" in st.session_state and st.session_state["models_loaded"]):
            st.warning("Please load the models first.")
            return
        
        # Get data and models
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        feature_names = st.session_state["feature_names"]
        models = st.session_state["models"]
        
        # Select model
        model_name = st.selectbox("Select Model", list(models.keys()))
        model_dict = models[model_name]
        model = model_dict["model"]
        
        # Make predictions
        y_pred, y_pred_proba = make_predictions(model, X_test, model_dict)
        
        # Display metrics
        st.subheader("Performance Metrics")
        metrics = create_metrics(y_test, y_pred, y_pred_proba)
        
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        confusion_matrix_fig = create_interactive_confusion_matrix(y_test, y_pred, model_name)
        st.plotly_chart(confusion_matrix_fig)
        
        # Display ROC curve and PR curve
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curve")
            roc_curve_fig = create_interactive_roc_curve(y_test, y_pred_proba, model_name)
            st.plotly_chart(roc_curve_fig)
        
        with col2:
            st.subheader("Precision-Recall Curve")
            pr_curve_fig = create_interactive_precision_recall_curve(y_test, y_pred_proba, model_name)
            st.plotly_chart(pr_curve_fig)
        
        # Display feature importance
        st.subheader("Feature Importance")
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importance_fig = create_interactive_feature_importance(
                feature_names, importances, model_name
            )
            st.plotly_chart(feature_importance_fig)
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            feature_importance_fig = create_interactive_feature_importance(
                feature_names, importances, model_name
            )
            st.plotly_chart(feature_importance_fig)
        elif isinstance(model, dict) and "models" in model:
            # Use the first model in the ensemble
            if hasattr(model["models"][0], "feature_importances_"):
                importances = model["models"][0].feature_importances_
                feature_importance_fig = create_interactive_feature_importance(
                    feature_names, importances, f"{model_name} (First Model)"
                )
                st.plotly_chart(feature_importance_fig)
        else:
            st.info("Feature importance not available for this model.")
        
        # SHAP explanation
        st.subheader("SHAP Summary")
        
        if st.button("Generate SHAP Summary"):
            with st.spinner("Generating SHAP summary..."):
                try:
                    # Use a subset of test data for SHAP
                    sample_indices = np.random.choice(X_test.shape[0], min(100, X_test.shape[0]), replace=False)
                    X_sample = X_test[sample_indices]
                    
                    # Create SHAP explainer based on model type
                    if isinstance(model, dict) and "models" in model:
                        # Use the first model in the ensemble
                        model_for_shap = model["models"][0]
                    else:
                        model_for_shap = model
                    
                    if hasattr(model_for_shap, 'predict_proba'):
                        # For scikit-learn models
                        explainer = shap.Explainer(model_for_shap.predict_proba, X_sample[:10])
                        shap_values = explainer(X_sample)
                        
                        # Plot SHAP summary
                        fig, ax = plt.subplots(figsize=(10, 8))
                        shap.summary_plot(shap_values[:, :, 1], X_sample, feature_names=feature_names, show=False)
                        st.pyplot(fig)
                    elif str(type(model_for_shap)).endswith("xgboost.sklearn.XGBClassifier'>"):
                        # For XGBoost models
                        explainer = shap.TreeExplainer(model_for_shap)
                        shap_values = explainer(X_sample)
                        
                        # Plot SHAP summary
                        fig, ax = plt.subplots(figsize=(10, 8))
                        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                        st.pyplot(fig)
                    else:
                        # Fallback approach
                        st.warning("Using fallback SHAP explanation method")
                        explainer = shap.Explainer(model_for_shap, X_sample[:10])
                        shap_values = explainer(X_sample)
                        
                        # Plot SHAP summary
                        fig, ax = plt.subplots(figsize=(10, 8))
                        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating SHAP summary: {str(e)}")
                    
                    # Display feature importance as fallback
                    if hasattr(model_for_shap, 'feature_importances_'):
                        st.info("Displaying feature importance instead")
                        importances = model_for_shap.feature_importances_
                        feature_imp = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        st.bar_chart(feature_imp.set_index('Feature'))
    
    # Transaction Explanation page
    elif page == "Transaction Explanation":
        st.title("Transaction Explanation")
        
        # Check if data and models are loaded
        if not ("data_loaded" in st.session_state and st.session_state["data_loaded"]):
            st.warning("Please load the data first.")
            return
        
        if not ("models_loaded" in st.session_state and st.session_state["models_loaded"]):
            st.warning("Please load the models first.")
            return
        
        # Get data and models
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        feature_names = st.session_state["feature_names"]
        models = st.session_state["models"]
        
        # Select model
        model_name = st.selectbox("Select Model", list(models.keys()))
        model_dict = models[model_name]
        model = model_dict["model"]
        
        # Make predictions
        y_pred, y_pred_proba = make_predictions(model, X_test, model_dict)
        
        # Find interesting transactions
        fraud_indices = np.where((y_test == 1) & (y_pred == 1))[0]  # True positives
        normal_indices = np.where((y_test == 0) & (y_pred == 0))[0]  # True negatives
        false_pos_indices = np.where((y_test == 0) & (y_pred == 1))[0]  # False positives
        false_neg_indices = np.where((y_test == 1) & (y_pred == 0))[0]  # False negatives
        
        # Select transaction type
        transaction_type = st.selectbox(
            "Select Transaction Type",
            ["Fraudulent (True Positive)", "Normal (True Negative)", 
             "False Alarm (False Positive)", "Missed Fraud (False Negative)"]
        )
        
        if transaction_type == "Fraudulent (True Positive)":
            if len(fraud_indices) == 0:
                st.warning("No fraudulent transactions correctly identified.")
                return
            transaction_idx = st.selectbox("Select Transaction", fraud_indices)
        elif transaction_type == "Normal (True Negative)":
            if len(normal_indices) == 0:
                st.warning("No normal transactions correctly identified.")
                return
            transaction_idx = st.selectbox("Select Transaction", normal_indices)
        elif transaction_type == "False Alarm (False Positive)":
            if len(false_pos_indices) == 0:
                st.warning("No false alarms found.")
                return
            transaction_idx = st.selectbox("Select Transaction", false_pos_indices)
        else:  # "Missed Fraud (False Negative)"
            if len(false_neg_indices) == 0:
                st.warning("No missed frauds found.")
                return
            transaction_idx = st.selectbox("Select Transaction", false_neg_indices)
        
        # Explain the transaction
        explain_transaction(model, X_test, feature_names, transaction_idx,
                            threshold=model_dict.get("metadata", {}).get("threshold", 0.5))
    
    # Live Monitoring page
    elif page == "Live Monitoring":
        st.title("Live Monitoring")
        
        # Check if data and models are loaded
        if not ("data_loaded" in st.session_state and st.session_state["data_loaded"]):
            st.warning("Please load the data first.")
            return
        
        if not ("models_loaded" in st.session_state and st.session_state["models_loaded"]):
            st.warning("Please load the models first.")
            return
        
        # Get data and models
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        feature_names = st.session_state["feature_names"]
        models = st.session_state["models"]
        
        # Select model
        model_name = st.selectbox("Select Model", list(models.keys()))
        model_dict = models[model_name]
        model = model_dict["model"]
        
        # Simulate live monitoring
        st.subheader("Simulated Transaction Stream")
        st.write("This section simulates a stream of incoming transactions.")
        
        # Number of transactions to simulate
        n_transactions = st.slider("Number of Transactions", 10, 100, 20)
        
        # Randomly select transactions from test set
        transaction_indices = np.random.choice(X_test.shape[0], n_transactions, replace=False)
        X_sample = X_test[transaction_indices]
        y_sample = y_test[transaction_indices]
        
        # Make predictions
        y_pred, y_pred_proba = make_predictions(model, X_sample, model_dict)
        
        # Create DataFrame for display
        transactions_df = pd.DataFrame({
            "Transaction ID": transaction_indices,
            "Actual": ["Fraud" if y == 1 else "Normal" for y in y_sample],
            "Predicted": ["Fraud" if y == 1 else "Normal" for y in y_pred],
            "Fraud Probability": y_pred_proba,
            "Correct": y_sample == y_pred
        })
        
        # Display transactions
        st.dataframe(transactions_df)
        
        # Display fraud alerts
        st.subheader("Fraud Alerts")
        
        # Filter for predicted fraud
        fraud_alerts = transactions_df[transactions_df["Predicted"] == "Fraud"]
        
        if len(fraud_alerts) == 0:
            st.info("No fraud alerts detected.")
        else:
            st.warning(f"{len(fraud_alerts)} potential fraud transactions detected!")
            
            # Display alerts
            for i, (_, alert) in enumerate(fraud_alerts.iterrows()):
                with st.expander(f"Alert {i+1}: Transaction {alert['Transaction ID']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Transaction ID:** {alert['Transaction ID']}")
                        st.write(f"**Fraud Probability:** {alert['Fraud Probability']:.4f}")
                        st.write(f"**Actual Class:** {alert['Actual']}")
                        
                        if alert["Correct"]:
                            st.success("Correct prediction!")
                        else:
                            st.error("False alarm!")
                    
                    with col2:
                        # Explain the transaction
                        transaction_idx = alert["Transaction ID"]
                        transaction = X_test[transaction_idx:transaction_idx+1]
                        
                        # Create SHAP explainer
                        if isinstance(model, dict) and "models" in model:
                            # Use the first model in the ensemble
                            explainer = shap.Explainer(model["models"][0], X_test[:10])
                        else:
                            explainer = shap.Explainer(model, X_test[:10])
                        
                        # Calculate SHAP values
                        shap_values = explainer(transaction)
                        
                        # Plot SHAP values
                        fig, ax = plt.subplots(figsize=(8, 4))
                        # Tree models like Random Forest return one explanation per class
                        # (shape: samples x features x classes); plot the fraud class.
                        sv = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
                        shap.plots.waterfall(sv, show=False)
                        st.pyplot(fig)
        
        # Display monitoring metrics
        st.subheader("Monitoring Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        total = len(transactions_df)
        fraud_count = len(fraud_alerts)
        fraud_rate = fraud_count / total if total > 0 else 0
        true_fraud = len(transactions_df[(transactions_df["Predicted"] == "Fraud") & 
                                        (transactions_df["Actual"] == "Fraud")])
        false_alarms = len(transactions_df[(transactions_df["Predicted"] == "Fraud") & 
                                          (transactions_df["Actual"] == "Normal")])
        
        col1.metric("Total Transactions", total)
        col2.metric("Fraud Alerts", fraud_count)
        col3.metric("Fraud Rate", f"{fraud_rate:.2%}")
        col4.metric("False Alarms", false_alarms)
        
        # Plot fraud rate over time (simulated)
        st.subheader("Fraud Rate Over Time")
        
        # Generate simulated data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq="D")
        fraud_rates = np.random.uniform(0.01, 0.05, size=30)
        
        # Create plot
        fig = px.line(
            x=dates, y=fraud_rates,
            labels={"x": "Date", "y": "Fraud Rate"},
            title="Fraud Rate Trend (Simulated)"
        )
        
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
