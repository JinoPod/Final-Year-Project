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
from src.audit_log import AuditLog, list_archives, start_fresh_log
from src.explanations import (
    load_reference, verdict, reasons, short_reasons, summary_sentence, risk_key, risk_label,
    global_takeaways, feature_guide, fmt_prob, REVIEW_FLOOR, RISK_LEVELS
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theme  # noqa: E402  (dashboard/theme.py)

# Set page config
st.set_page_config(
    page_title="CLARIX · XAI Fraud Detection",
    page_icon=str(theme.APP_ICON),
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
    "feature_guide": (
        "Each row is one feature, most influential at the top. The **blue band** is where the "
        "middle 90% of normal (legitimate) transactions fall. The **red dot** is the typical "
        "(median) value among frauds.\n\n"
        "A red dot far outside the band means frauds usually look very different on that "
        "feature, e.g. V14: normal transactions sit between about -1.4 and 1.4, while a typical "
        "fraud is around -6.8. A red dot inside the band (e.g. V8) means the feature still "
        "matters to the model, but not through a simple high/low rule."
    ),
}


def how_to_read(key):
    with st.expander("How to read this chart", icon=":material/info:"):
        st.markdown(HOW_TO_READ[key])


def feature_guide_section():
    """Model Performance: what the data shows about each (anonymised) feature."""
    st.subheader("Feature guide")
    st.markdown(
        "V1–V28 have no published meaning, so this guide describes **how each feature behaves in "
        "the training data** instead: how much the model relies on it, where normal transactions "
        "fall, and where frauds typically fall. Use it to recognise warning signs, e.g. "
        "*\"very low V14\"*, when reading an explanation.")
    rows = feature_guide(get_reference())
    st.plotly_chart(theme.feature_guide_chart(rows, top=12), width="stretch", theme=None)
    how_to_read("feature_guide")
    with st.expander("All 30 features", icon=":material/table_rows:"):
        table = pd.DataFrame(rows)[["rank", "feature", "influence", "normal_range",
                                    "typical_fraud", "pattern"]]
        st.dataframe(table, hide_index=True, width="stretch", height="content", column_config={
            "rank": st.column_config.NumberColumn("Rank", help="1 = the feature the model relies on most "
                                                  "(mean absolute SHAP value)"),
            "feature": "Feature",
            "influence": "Influence on the model",
            "normal_range": st.column_config.TextColumn(
                "Normal range", width=fit_width(table["normal_range"], minimum=140),
                help="Middle 90% of legitimate training transactions (5th to 95th percentile)"),
            "typical_fraud": st.column_config.TextColumn(
                "Typical fraud", help="Median value among fraudulent training transactions"),
            "pattern": st.column_config.TextColumn("What the data shows", width=fit_width(table["pattern"])),
        })
    ref = get_reference()
    st.caption(
        f"Ranges and typical values come from the training data only ({ref['n_normal']:,} legitimate "
        f"and {ref['n_fraud']:,} fraudulent transactions); influence comes from SHAP. These are "
        "tendencies, not rules: a fraud can still have normal values, which is how frauds get missed.")


def plot_waterfall(shap_values, base_value, feature_values, feature_names, max_display=12):
    """SHAP waterfall from stored values (no model call needed)."""
    explanation = shap.Explanation(values=np.asarray(shap_values), base_values=base_value,
                                   data=np.asarray(feature_values), feature_names=list(feature_names))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)


def plot_reasons(reason_list):
    """Simple horizontal bar chart of the strongest factors (crimson = towards fraud)."""
    st.plotly_chart(theme.reasons_chart(reason_list), width="stretch", theme=None)


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
    theme.verdict_card(v, probability, threshold, REVIEW_FLOOR, fmt_prob(probability))
    if v["borderline"]:
        st.info(v["borderline_note"], icon=":material/warning:")
    if actual is not None:
        st.caption(f"Actual label in the data: **{'fraud' if actual == 1 else 'legitimate'}**.")

    reason_list = reasons(shap_row, values, list(feature_names), get_reference())
    st.markdown("#### Why")
    st.markdown(summary_sentence(reason_list, probability, threshold))
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("\n".join(f"{RAISES if r['raises_risk'] else LOWERS} {r['sentence']}\n"
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


@st.cache_data(show_spinner="Drawing the global SHAP charts...")
def global_shap_images():
    """Global SHAP bar and beeswarm charts as PNG bytes (drawn once: the beeswarm has ~850k points)."""
    _, X_test, _, _, feature_names = load_data()
    shap_values, base_value = test_shap_values()
    raw = unscale(X_test, get_scorer().preprocessor, feature_names)
    exp = shap.Explanation(values=shap_values, base_values=np.full(len(shap_values), base_value),
                           data=raw, feature_names=feature_names)
    images = []
    for kind in ("bar", "beeswarm"):
        plt.figure()
        if kind == "bar":
            # Importance is a magnitude, not a direction, so it uses one neutral colour.
            with shap.plots._style.style_context(primary_color_positive=theme.BLUE):
                shap.plots.bar(exp, max_display=15, show=False)
        else:
            shap.plots.beeswarm(exp, max_display=15, show=False, color=theme.SHAP_CMAP)
        buf = io.BytesIO()
        plt.gcf().savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close("all")
        images.append(buf.getvalue())
    return images


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
        fig = theme.recolor_lime(lime_exp.as_pyplot_figure())
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
        earlier = get_audit_log().find_file(file_bytes)
        if len(earlier) and st.session_state.get("rescore_ok") != file_key:
            last = earlier.iloc[0]
            times = f" (and {len(earlier) - 1} time{'s' if len(earlier) > 2 else ''} before that)" \
                if len(earlier) > 1 else ""
            st.warning(f"This file was already scored as **batch #{last['id']}** on "
                       f"{fmt_time(last['created_at'])}{times}. Scoring it again would add a "
                       "duplicate batch to the audit log.")
            c1, c2, _ = st.columns([1, 1, 1])
            c1.button(f"View batch #{last['id']} in the Audit Log", type="primary",
                      on_click=view_batch, args=(int(last["id"]),), width="stretch")
            c2.button("Score it again as a new batch", width="stretch",
                      on_click=st.session_state.__setitem__, args=("rescore_ok", file_key))
            return
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
        st.session_state.audit_batch = batch_id  # the Audit Log opens on this upload
    batch, batch_id = cache[file_key]
    results = batch.results
    keys = np.array([risk_key(p, batch.threshold) for p in results["fraud_probability"]])

    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.success(f"Scored and saved to the audit log as batch #{batch_id}.")
    c2.button("View in Audit Log", on_click=view_batch, args=(batch_id,), width="stretch")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions scored", f"{len(results):,}")
    c2.metric("🔴 High risk", f"{int((keys == 'high').sum()):,}")
    c3.metric("🟡 Needs review", f"{int((keys == 'review').sum()):,}")
    c4.metric("🟢 Low risk", f"{int((keys == 'low').sum()):,}")
    c5.metric("Not scored", f"{len(prepared.rejected):,}")
    with st.expander("What do the risk levels mean?", icon=":material/info:"):
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
                "Main reasons (↑ raises risk, ↓ lowers risk)", width=fit_width(shown["main_reasons"])),
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


def fit_width(texts, px_per_char=8, minimum=400, maximum=2400):
    """Column width in pixels that shows the longest text in full.

    Table cells cannot wrap, so a wide column plus the table's horizontal scrollbar is
    the way to read long text without hovering over each cell.
    """
    longest = int(texts.astype(str).str.len().max()) if len(texts) else 0
    return int(min(max(longest * px_per_char + 32, minimum), maximum))


def fmt_time(iso):
    """'2026-11-11T13:50:53+00:00' -> '11 Nov 2026, 13:50 UTC'."""
    try:
        return pd.Timestamp(iso).strftime("%d %b %Y, %H:%M UTC")
    except (ValueError, TypeError):
        return str(iso)


def view_batch(batch_id):
    """Button callback: open the Audit Log on one batch."""
    st.session_state.audit_batch = batch_id
    go_to("Audit Log")


def fresh_log():
    """Button callback: archive the current log and start an empty one."""
    archived = start_fresh_log(AUDIT_DB_PATH)
    get_audit_log.clear()  # the next call creates a new, empty database
    for key in ("scored_uploads", "audit_batch", "rescore_ok", "fresh_ok", "older_pick"):
        st.session_state.pop(key, None)
    st.session_state.fresh_msg = archived


ALL_BATCHES = "all"
CARDS_SHOWN = 6  # newest uploads shown as cards; older ones are in a dropdown


def batch_cards(summaries, selected):
    """One card per upload, newest first; the batch being viewed is highlighted."""
    recent = summaries.head(CARDS_SHOWN)
    for start in range(0, len(recent), 3):
        cols = st.columns(3)
        for col, (_, b) in zip(cols, recent.iloc[start:start + 3].iterrows()):
            bid = int(b["id"])
            viewing = selected == bid
            with col.container(border=True, key="batch_card_sel" if viewing else f"batch_card_{bid}"):
                st.markdown(f"**Batch #{bid}**" + ("  ·  *viewing*" if viewing else ""))
                st.markdown(f":material/description: `{b['source_file']}`")
                st.caption(f":material/schedule: {fmt_time(b['created_at'])}")
                not_scored = f" ({int(b['rejected_rows'])} not scored)" if b["rejected_rows"] else ""
                st.markdown(
                    f"🔴 **{int(b['high'])}** high &nbsp; 🟡 **{int(b['review'])}** review &nbsp; "
                    f"of {int(b['scored_rows']):,} scored{not_scored}")
                st.button("Viewing" if viewing else "View", key=f"view_{bid}", disabled=viewing,
                          type="primary" if viewing else "secondary", width="stretch",
                          on_click=st.session_state.__setitem__, args=("audit_batch", bid))

    older = summaries.iloc[CARDS_SHOWN:]
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    if len(older):
        labels = {int(i): f"#{int(i)}  ·  {f}  ·  {fmt_time(t)}"
                  for i, f, t in zip(older["id"], older["source_file"], older["created_at"])}
        c1.selectbox(f"Older uploads ({len(older)})", list(labels), index=None,
                     placeholder="Choose an older batch", format_func=labels.get, key="older_pick",
                     on_change=lambda: st.session_state.update(audit_batch=st.session_state.older_pick))
    if len(summaries) > 1:
        c2.button("Show all batches together", disabled=selected == ALL_BATCHES, width="stretch",
                  on_click=st.session_state.__setitem__, args=("audit_batch", ALL_BATCHES))


def fresh_log_section():
    st.divider()
    with st.expander("Start a fresh log", icon=":material/inventory_2:"):
        st.markdown(
            "Moves the whole current log to `logs/archive/` and starts a new, empty one. "
            "**Nothing is deleted**: the archived file keeps every decision and its explanation, "
            "and can be inspected at any time. Useful for clearing test uploads before a "
            "demonstration.")
        archives = list_archives(AUDIT_DB_PATH)
        if archives:
            st.caption("Already archived: " + ", ".join(f"`{a}`" for a in archives))
        ok = st.checkbox("I understand that the current log will be moved to the archive",
                         key="fresh_ok")
        st.button("Start a fresh log", disabled=not ok, on_click=fresh_log)


def page_audit():
    """Audit Log page (report Section 3.6.5)."""
    st.title("Audit Log")
    st.write(
        "Every decision made on the Upload & Score page, with the explanation stored at the time. "
        "Explanations here are read from the log; the model is not re-run. The log is append-only: "
        "records cannot be edited or deleted."
    )
    if st.session_state.get("fresh_msg"):
        archived = st.session_state.pop("fresh_msg")
        st.success("Started a fresh log. The previous one was archived as "
                   f"`{os.path.relpath(archived, PROJECT_DIR)}`.")
    log = get_audit_log()
    summaries = log.batch_summaries(REVIEW_FLOOR)
    if summaries.empty:
        st.info("No decisions logged yet. Score a file on the Upload & Score page first.")
        archives = list_archives(AUDIT_DB_PATH)
        if archives:
            st.caption("Archived logs in `logs/archive/`: " + ", ".join(f"`{a}`" for a in archives))
        return

    ids = summaries["id"].astype(int).tolist()
    selected = st.session_state.get("audit_batch")
    if selected != ALL_BATCHES and selected not in ids:
        selected = ids[0]  # default: the newest upload
    st.session_state.audit_batch = selected

    st.subheader("Uploads")
    st.caption("Each uploaded file is saved as its own batch. Choose one to see its decisions.")
    batch_cards(summaries, selected)

    with st.expander("Batch details (technical)"):
        st.dataframe(
            log.list_batches()[["id", "created_at", "source_file", "file_sha256", "model_name",
                                "model_version", "threshold", "total_rows", "scored_rows",
                                "rejected_rows", "flagged_rows"]],
            hide_index=True, width="stretch",
            column_config={"created_at": "Logged at (UTC)", "id": "Batch",
                           "file_sha256": "File fingerprint (SHA-256)"},
        )

    if selected == ALL_BATCHES:
        st.subheader("Decisions from all batches")
    else:
        b = summaries[summaries["id"] == selected].iloc[0]
        st.subheader(f"Decisions in batch #{selected}")
        st.caption(f":material/description: {b['source_file']}  ·  :material/schedule: "
                   f"{fmt_time(b['created_at'])}  ·  "
                   f"alert threshold {b['threshold']:.4f}")
    show = st.selectbox("Show", ["High risk and Needs review", "High risk only", "All decisions"])
    decisions = log.get_decisions(batch_id=None if selected == ALL_BATCHES else selected,
                                  limit=20_000)
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
        fresh_log_section()
        return
    columns = ["id", "row_number", "risk_level", "fraud_probability", "actual", "top_factors"]
    if selected == ALL_BATCHES:  # the batch columns only matter when batches are mixed
        columns[1:1] = ["batch_id", "created_at", "source_file"]
    st.dataframe(
        decisions[columns],
        hide_index=True, width="stretch",
        column_config={
            "id": "Decision", "batch_id": "Batch", "created_at": "Logged at (UTC)",
            "source_file": "File", "row_number": "Row in file", "risk_level": "Risk level",
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud score", min_value=0.0, max_value=1.0, format="percent"),
            "actual": st.column_config.NumberColumn("Actual (1 = fraud)"),
            "top_factors": st.column_config.TextColumn(
                "Main factors (as logged)", width=fit_width(decisions["top_factors"])),
        },
    )
    name = "all_batches" if selected == ALL_BATCHES else f"batch_{selected}"
    st.download_button("Download these decisions (CSV)", decisions.to_csv(index=False).encode(),
                       file_name=f"decision_log_{name}.csv", mime="text/csv")

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
        st.caption(f":green[:material/verified:] Integrity check passed: the stored inputs match their SHA-256 fingerprint "
                   f"({d['input_sha256'][:16]}…), so this record has not been altered.")
    else:
        st.error("Integrity check FAILED: the stored inputs do not match their fingerprint. "
                 "This record may have been altered.")
    show_explanation(d["fraud_probability"], d["threshold"], d["shap_values"], d["features"],
                     d["feature_names"], d["base_value"], actual=d["actual"])
    fresh_log_section()


PAGES = list(theme.PAGE_ICON_KEYS)  # Home, Upload & Score, Audit Log, ...

# Direction markers for the reasons list, in the chart colours (crimson / blue).
RAISES = ":red[:material/arrow_upward:]"
LOWERS = ":blue[:material/arrow_downward:]"


def go_to(page):
    """Button callback: switch page and keep the URL in sync so it can be bookmarked."""
    st.session_state.page = page
    st.query_params["page"] = page


# Main function
def main():
    """Main function for the dashboard."""
    theme.apply_theme()
    theme.sidebar_brand()
    if "page" not in st.session_state:  # ?page=Audit Log opens that page directly
        requested = st.query_params.get("page", "Home")
        st.session_state.page = requested if requested in PAGES else "Home"
    page = st.session_state.page

    st.sidebar.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    for p in PAGES:
        st.sidebar.button(p, key=theme.nav_key(p), on_click=go_to, args=(p,),
                          type="primary" if p == page else "secondary", width="stretch")
    threshold = float(load_model()["metadata"]["threshold"])
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Model: XGBoost, trained {str(load_model().get('timestamp', 'unknown'))[:10]}  \n"
                       f"Alert threshold: {threshold:.3f}  \n"
                       f"Review band starts at: {REVIEW_FLOOR:.2f}")

    if page == "Home":
        theme.hero(
            "Explainable Fraud Detection",
            "Scores credit card transactions with a tuned <span class='accent'>XGBoost</span> "
            "model and explains every decision with <span class='accent'>SHAP</span>, in plain "
            "language an analyst can act on. Every decision is kept in a tamper-evident audit log.",
            kicker="CLARIX",
            footnote="Final Year Project: XAI-Based Fraud Detection: An Interpretable Machine "
                     "Learning Approach for Transaction Monitoring")

        cards = [
            ("Upload & Score", "Upload a CSV of transactions. Each one gets a risk level, "
                                    "a suggested action and the reasons behind it."),
            ("Audit Log", "Review any past decision with the explanation stored at the time, "
                               "and confirm the record has not been altered."),
            ("Model Performance", "How well the model does on 56,746 transactions it never "
                                       "saw during training, in plain terms."),
            ("Transaction Explanation", "Explore caught frauds, false alarms and missed frauds "
                                             "from the test set and see why each happened."),
        ]
        cols = st.container(key="home_cards").columns(4)
        for col, (title, text) in zip(cols, cards):
            with col.container(border=True, height="stretch"):
                st.markdown(theme.card_title(title), unsafe_allow_html=True)
                st.caption(text)
                st.button("Open →", key=f"card_{title}", on_click=go_to, args=(title,))

        st.subheader("Risk levels")
        st.markdown(risk_level_table(threshold))

        X_train, X_test, y_train, y_test, _ = load_data()
        st.subheader("The data")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Training transactions", f"{X_train.shape[0]:,}",
                    f"{int(y_train.sum())} fraud", delta_color="off", delta_arrow="off")
        col2.metric("Test transactions", f"{X_test.shape[0]:,}",
                    f"{int(y_test.sum())} fraud", delta_color="off", delta_arrow="off")
        col3.metric("Share that is fraud", f"{(y_train.sum() + y_test.sum()) / (len(y_train) + len(y_test)):.2%}",
                    "about 1 in 600", delta_color="off", delta_arrow="off")
        col4.metric("Features per transaction", f"{X_train.shape[1]}",
                    "Time, Amount, V1–V28", delta_color="off", delta_arrow="off")

        with st.expander("How the model was built"):
            st.markdown(
                "- **Dataset:** European Credit Card Fraud (Kaggle / ULB), 284,807 transactions; "
                "1,081 duplicates removed. V1–V28 are PCA components anonymised by the provider.\n"
                "- **Model:** XGBoost, tuned with Optuna using 3-fold cross-validation.\n"
                "- **Imbalance:** SMOTE, applied only to training folds.\n"
                "- **Alert threshold:** tuned for F1 on training data; the test set was used once, "
                "for the final evaluation.\n"
                "- **Explanations:** exact TreeSHAP, with LIME available for comparison.\n"
                "- **Audit:** every uploaded decision is stored with its explanation and a SHA-256 "
                "fingerprint of its inputs.")

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
        fn = int(((y_pred == 0) & (y == 1)).sum())
        tn = int(((y_pred == 0) & (y == 0)).sum())
        n_fraud, n_legit = int(y.sum()), int((y == 0).sum())
        review = (y_pred_proba >= REVIEW_FLOOR) & (y_pred == 0)
        review_fraud = int((review & (y == 1)).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(":material/verified: Alerts that are real fraud", f"{tp / (tp + fp) * 100:.0f} in 100",
                  f"{tp} of {tp + fp} alerts", delta_color="off", delta_arrow="off")
        c2.metric(":material/target: Frauds caught", f"about {round(tp / n_fraud * 4)} in 4",
                  f"{tp} of {n_fraud} ({tp / n_fraud:.0%})", delta_color="off", delta_arrow="off")
        c3.metric(":material/notifications_off: False alarm rate", f"1 in {round(n_legit / max(fp, 1), -3):,.0f}",
                  f"{fp} of {n_legit:,} legitimate", delta_color="off", delta_arrow="off")
        c4.metric(":material/hourglass_top: Sent for review", f"{int(review.sum())}",
                  f"{review_fraud} of them fraud", delta_color="off", delta_arrow="off")
        st.caption(f":material/warning: Weak point: about {round(fn / n_fraud * 4)} in 4 frauds look like normal "
                   "transactions to the model and are missed.")

        st.subheader("Performance Metrics")
        show_metrics(y_test, y_pred, y_pred_proba)
        with st.expander("What do these metrics mean?", icon=":material/info:"):
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
            theme.confusion_grid(tn, fp, fn, tp)
        with col2:
            st.markdown(HOW_TO_READ["confusion"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ROC Curve")
            st.plotly_chart(theme.roc_chart(y, y_pred_proba, threshold), width="stretch", theme=None)
            how_to_read("roc")
        with col2:
            st.subheader("Precision-Recall Curve")
            st.plotly_chart(theme.pr_chart(y, y_pred_proba, threshold), width="stretch", theme=None)
            how_to_read("pr")

        st.subheader("Global Explanation (SHAP)")
        st.info("**Key takeaways**\n\n" + "\n".join(f"- {line}" for line in
                                                     global_takeaways(get_reference())))
        st.caption(ANONYMISED_NOTE)
        bar_png, beeswarm_png = global_shap_images()
        col1, col2 = st.columns(2)
        with col1:
            st.image(bar_png, width="stretch")
            how_to_read("shap_bar")
        with col2:
            st.image(beeswarm_png, width="stretch")
            how_to_read("beeswarm")
        feature_guide_section()

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
