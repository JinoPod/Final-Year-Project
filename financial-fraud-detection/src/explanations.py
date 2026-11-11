"""
Analyst-facing explanation layer: turns a fraud score and its SHAP values into a
risk level, a suggested action and plain-language reasons.

V1-V28 are PCA components anonymised by the dataset publisher, so a reason
cannot say what a feature means in business terms. Instead each reason says
how unusual the value is compared with normal (legitimate) training
transactions, what fraud typically looks like for that feature, and how
strongly it moved the score. The comparison data comes from the training set
only and is stored in models/models/feature_reference.json.
"""

import json
import os

import joblib
import numpy as np

# A transaction scoring at least the standard 0.5 cut-off but below the tuned
# alert threshold is sent for manual review. 0.5 is the default decision point
# of a probabilistic classifier, so the band is fixed a priori, not tuned on test data.
REVIEW_FLOOR = 0.5
# Scores this close to the alert threshold get a "close to the decision line" note.
BORDERLINE_MARGIN = 0.05

REFERENCE_FILE = os.path.join("models", "models", "feature_reference.json")
PERCENTILES = [0, 0.1, 0.5, 1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5, 99.9, 100]

RISK_LEVELS = {
    "high": {
        "label": "High risk", "icon": "🔴", "headline": "Likely fraud",
        "action": "Hold the transaction and contact the cardholder to confirm it before releasing it.",
    },
    "review": {
        "label": "Needs review", "icon": "🟡", "headline": "Uncertain",
        "action": "Review manually before approving: check the card's recent activity and the "
                  "reasons below.",
    },
    "low": {
        "label": "Low risk", "icon": "🟢", "headline": "Looks legitimate",
        "action": "No action needed.",
    },
}


# --------------------------------------------------------------------------
# Risk level
# --------------------------------------------------------------------------

def fmt_prob(probability):
    """Fraud score as a percentage, without rounding near-certain scores to 0% or 100%."""
    if probability > 0.999:
        return ">99.9%"
    if probability < 0.001:
        return "<0.1%"
    return f"{probability:.1%}"


def risk_key(probability, threshold):
    if probability >= threshold:
        return "high"
    if probability >= REVIEW_FLOOR:
        return "review"
    return "low"


def risk_label(probability, threshold):
    level = RISK_LEVELS[risk_key(probability, threshold)]
    return f"{level['icon']} {level['label']}"


def is_borderline(probability, threshold):
    return abs(probability - threshold) < BORDERLINE_MARGIN


def verdict(probability, threshold):
    """Headline, explanation of the score and suggested action for one transaction."""
    key = risk_key(probability, threshold)
    level = dict(RISK_LEVELS[key], key=key)
    if key == "high":
        detail = (f"The fraud score of {fmt_prob(probability)} is above the alert threshold of "
                  f"{threshold:.1%}.")
    elif key == "review":
        detail = (f"The fraud score of {fmt_prob(probability)} is below the alert threshold of "
                  f"{threshold:.1%}, so no alert is raised, but it is above "
                  f"{REVIEW_FLOOR:.0%}: the model leans towards fraud.")
    else:
        detail = (f"The fraud score of {fmt_prob(probability)} is well below the alert threshold "
                  f"of {threshold:.1%}.")
    level["detail"] = detail
    level["borderline"] = is_borderline(probability, threshold)
    if level["borderline"]:
        level["borderline_note"] = (
            "This score is close to the decision line, so a small change in the transaction "
            "could change the decision. Treat it with extra care.")
    return level


# --------------------------------------------------------------------------
# Reference data (training set only)
# --------------------------------------------------------------------------

def build_reference(base_dir, shap_sample=20_000, seed=42):
    """
    Summarise the training data for plain-language comparisons:
    percentiles of normal transactions, the typical fraud value and its direction,
    and each feature's global importance (mean |SHAP| on a training sample).
    """
    from src.inference import FraudScorer
    from src.models.report_figures import unscale

    processed = os.path.join(base_dir, "data", "processed")
    X = np.load(os.path.join(processed, "X_train.npy"))
    y = np.load(os.path.join(processed, "y_train.npy")).astype(int)
    with open(os.path.join(processed, "feature_names.txt")) as f:
        names = f.read().splitlines()
    preprocessor = joblib.load(os.path.join(processed, "preprocessor.pkl"))
    raw = unscale(X, preprocessor, names)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(shap_sample, len(X)), replace=False)
    shap_values = FraudScorer.from_disk(base_dir).explainer(X[idx]).values
    mean_abs = np.abs(shap_values).mean(axis=0)
    rank = {names[i]: r + 1 for r, i in enumerate(np.argsort(-mean_abs))}

    features = {}
    for j, name in enumerate(names):
        legit, fraud = raw[y == 0, j], raw[y == 1, j]
        p5, p95 = np.percentile(legit, [5, 95])
        fraud_median = float(np.median(fraud))
        # Direction only when typical fraud sits outside the normal 5-95% range:
        # +1 frauds tend to be high, -1 frauds tend to be low, 0 no simple pattern.
        direction = 1 if fraud_median > p95 else -1 if fraud_median < p5 else 0
        features[name] = {
            "normal_percentiles": np.percentile(legit, PERCENTILES).round(6).tolist(),
            "normal_median": float(np.median(legit)),
            "fraud_median": fraud_median,
            "importance_rank": rank[name],
            "mean_abs_shap": float(mean_abs[j]),
            "fraud_direction": direction,
        }
    reference = {
        "source": "training set only",
        "n_normal": int((y == 0).sum()), "n_fraud": int((y == 1).sum()),
        "percentile_levels": PERCENTILES,
        "features": features,
    }
    path = os.path.join(base_dir, REFERENCE_FILE)
    with open(path, "w") as f:
        json.dump(reference, f, indent=1)
    return reference


def load_reference(base_dir):
    path = os.path.join(base_dir, REFERENCE_FILE)
    if not os.path.exists(path):
        return build_reference(base_dir)
    with open(path) as f:
        return json.load(f)


def normal_percentile(reference, feature, value):
    """Share (0-100) of normal training transactions with a value below `value`."""
    ref = reference["features"][feature]
    return float(np.interp(value, ref["normal_percentiles"], reference["percentile_levels"]))


# --------------------------------------------------------------------------
# Reasons
# --------------------------------------------------------------------------

def _fmt(feature, value):
    if feature == "Amount":
        return f"€{value:,.2f}"
    if feature == "Time":
        return f"{value:,.0f} s"
    return f"{value:.2f}"


def _unusualness(pct):
    """Short description of how a value sits among normal transactions."""
    side = "high" if pct >= 50 else "low"
    tail = pct if side == "low" else 100 - pct
    if tail <= 0.1:
        return f"extremely {side}"
    if tail <= 1:
        return f"very {side}"
    if tail <= 10:
        return f"unusually {side}"
    return "typical"


def _strength(shap_value):
    s = abs(shap_value)
    if s >= 2:
        return "strongly"
    if s >= 0.5:
        return "moderately"
    return "slightly"


def reasons(shap_row, values, feature_names, reference, n=5):
    """The n strongest factors behind one decision, each with a plain-language sentence."""
    shap_row = np.asarray(shap_row, dtype=float)
    values = np.asarray(values, dtype=float)
    out = []
    for i in np.argsort(-np.abs(shap_row))[:n]:
        name, value, sv = feature_names[i], float(values[i]), float(shap_row[i])
        ref = reference["features"][name]
        effect = f"This {_strength(sv)} {'raises' if sv > 0 else 'lowers'} the fraud score."

        if name == "Time":
            pct, unusual = None, "n/a"
            sentence = (f"**Time = {_fmt(name, value)}** ({value / 3600:.1f} hours after the first "
                        f"transaction in the dataset). {effect}")
        else:
            pct = normal_percentile(reference, name, value)
            unusual = _unusualness(pct)
            p = reference["features"][name]["normal_percentiles"]
            lo, hi = p[PERCENTILES.index(5)], p[PERCENTILES.index(95)]
            if unusual == "typical":
                position = "within the normal range"
            else:
                share = max(pct, 100 - pct)
                share = "over 99.9%" if share > 99.9 else f"{share:.1f}%"
                position = f"{'lower' if pct < 50 else 'higher'} than {share} of normal transactions"
            sentence = (f"**{name} = {_fmt(name, value)}** is {unusual}: {position} "
                        f"(normal range {_fmt(name, lo)} to {_fmt(name, hi)}; frauds typically "
                        f"have about {_fmt(name, ref['fraud_median'])}). {effect}")
        if ref["importance_rank"] <= 3:
            sentence += f" {name} is one of the model's three most influential features."

        out.append({
            "feature": name, "value": value, "shap": sv,
            "raises_risk": sv > 0, "strength": _strength(sv),
            "normal_percentile": pct, "unusualness": unusual, "sentence": sentence,
        })
    return out


def short_reasons(shap_row, values, feature_names, reference, n=3):
    """One-line summary for result tables, e.g. 'V14 extremely low ↑, V4 very high ↑'."""
    parts = []
    for r in reasons(shap_row, values, feature_names, reference, n=n):
        desc = "" if r["feature"] == "Time" else f" {r['unusualness']}"
        parts.append(f"{r['feature']}{desc} {'↑' if r['raises_risk'] else '↓'}")
    return ", ".join(parts)


def summary_sentence(reason_list, probability, threshold):
    """One sentence tying the reasons to the decision."""
    up = [r for r in reason_list if r["raises_risk"]]
    down = [r for r in reason_list if not r["raises_risk"]]
    key = risk_key(probability, threshold)
    lead = (up or reason_list)[0] if key != "low" else (down or reason_list)[0]
    lead_desc = "" if lead["feature"] == "Time" else f" ({lead['unusualness']} value)"
    if key == "low":
        return (f"{len(down)} of the {len(reason_list)} strongest factors point towards a legitimate "
                f"transaction, led by {lead['feature']}{lead_desc}.")
    return (f"{len(up)} of the {len(reason_list)} strongest factors point towards fraud, "
            f"led by {lead['feature']}{lead_desc}.")


def global_takeaways(reference, n=5):
    """Plain-language summary of what drives the model overall."""
    top = sorted(reference["features"].items(), key=lambda kv: kv[1]["importance_rank"])[:n]
    low = [k for k, v in top if v["fraud_direction"] < 0]
    high = [k for k, v in top if v["fraud_direction"] > 0]
    mixed = [k for k, v in top if v["fraud_direction"] == 0]
    lines = [f"The {n} most influential features are {', '.join(k for k, _ in top)}."]
    if low:
        lines.append(f"Unusually **low** values of {', '.join(low)} push a transaction towards fraud.")
    if high:
        lines.append(f"Unusually **high** values of {', '.join(high)} push a transaction towards fraud.")
    if mixed:
        lines.append(f"{', '.join(mixed)} also matter{'s' if len(mixed) == 1 else ''}, but without a "
                     "simple high/low pattern: typical frauds fall inside the normal range.")
    return lines
