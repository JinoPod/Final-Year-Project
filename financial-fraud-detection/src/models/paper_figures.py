"""
Figures for the technical paper (IEEE two-column conference format).

Column-width versions (3.4 in wide, 8 pt Times New Roman, 300 dpi) of the dissertation figures,
built from the same saved models and test set, plus a pipeline diagram and a cropped dashboard view.

Outputs (models/visualizations/paper/):
  paper_pipeline.png        method overview
  paper_pr_curves.png       precision-recall curves of the four models, operating point marked
  paper_beeswarm.png        SHAP summary plot (top 10 features)
  paper_waterfall_fp.png    local explanation of the most confident false alarm
  paper_dashboard.png       CLARIX verdict card (cropped from the dissertation screenshot)

Usage:
  python src/models/paper_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from src.models import dissertation_figures as df  # noqa: E402
from src.models.report_figures import pick_cases, unscale  # noqa: E402

OUT_DIR = os.path.join(PROJECT_DIR, "models", "visualizations", "paper")
DPI = 600  # stays above 300 dpi when enlarged in the one-column paper
COL_W = 3.4  # inches, one IEEE column

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7})


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", name)


def fig_pipeline():
    steps = [("European credit\ncard data\n(284,807 records)", "#F2F2F2"),
             ("Duplicate removal,\nstratified\n80/20 split", "#F2F2F2"),
             ("SMOTE inside\ncross-validation,\nOptuna tuning", "#F2F2F2"),
             ("XGBoost with\nthreshold chosen\non OOF predictions", "#C9D8EA"),
             ("TreeSHAP global\nand local\nexplanations", "#C9D8EA"),
             ("CLARIX dashboard\nand append-only\naudit log", "#C9D8EA")]
    fig, ax = plt.subplots(figsize=(COL_W, 1.55))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.axis("off")
    w, h = 0.8, 0.74
    pos = [(0.5, 1.5), (1.5, 1.5), (2.5, 1.5), (2.5, 0.5), (1.5, 0.5), (0.5, 0.5)]
    for (text, color), (x, y) in zip(steps, pos):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                                    fc=color, ec="#444444", lw=0.7))
        ax.text(x, y, text, ha="center", va="center", fontsize=6.3, linespacing=1.15, color=df.INK)
    for (x0, y0), (x1, y1) in zip(pos, pos[1:]):
        if y0 == y1:
            s = np.sign(x1 - x0)
            a, b = (x0 + s * w / 2, y0), (x1 - s * w / 2, y1)
        else:
            a, b = (x0, y0 - h / 2), (x1, y1 + h / 2)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, color="#444444", lw=0.8))
    save(fig, "paper_pipeline.png")


def fig_pr(y, probas, thr):
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    for label, _ in df.MODELS:
        p, st = probas[label], df.STYLE[label]
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=st["color"], ls=st["ls"], lw=st["lw"] * 0.75,
                label=f"{label} (AP {average_precision_score(y, p):.3f})")
    flagged = probas["XGBoost"] >= thr
    r, pr = recall_score(y, flagged), precision_score(y, flagged)
    ax.plot([r], [pr], "o", ms=5, color=df.BLUE, mec="white", mew=1.0, zorder=5)
    ax.annotate("XGBoost operating point", (r, pr), xytext=(0.08, 0.62), fontsize=7,
                arrowprops=dict(arrowstyle="->", color="#444444", lw=0.6))
    ax.set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02), xlim=(0, 1.0))
    ax.legend(loc="lower left", frameon=True)
    save(fig, "paper_pr_curves.png")


def shap_figures(model, X_test, y, names, preprocessor, proba, thr):
    sv = shap.TreeExplainer(model)(X_test)
    raw = unscale(X_test, preprocessor, names)
    exp = shap.Explanation(values=sv.values, base_values=sv.base_values, data=raw, feature_names=names)

    shap.plots.beeswarm(exp, max_display=10, show=False, plot_size=(COL_W, 2.9))
    fig = plt.gcf()
    for ax in fig.axes:
        ax.tick_params(labelsize=7.5)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    fig.axes[0].set_xlabel("SHAP value (impact on log-odds of fraud)")
    save(fig, "paper_beeswarm.png")

    idx = pick_cases(y, proba, thr)["false_positive"]
    shap.plots.waterfall(exp[idx], max_display=8, show=False)
    fig = plt.gcf()
    fig.set_size_inches(COL_W, 2.6)
    for ax in fig.axes:
        ax.tick_params(labelsize=8)
        for t in ax.texts:
            t.set_fontsize(7.5)
    save(fig, "paper_waterfall_fp.png")


def fig_dashboard():
    src = Image.open(os.path.join(df.OUT_DIR, "shot_explanation.png"))
    s = src.width / 2000  # crop box measured on the 2000-px-wide preview
    box = [int(v * s) for v in (585, 585, 1885, 1065)]
    crop = src.crop(box).convert("RGB")
    framed = Image.new("RGB", (crop.width + 6, crop.height + 6), (150, 150, 150))
    framed.paste(crop, (3, 3))
    framed.save(os.path.join(OUT_DIR, "paper_dashboard.png"), dpi=(DPI, DPI))
    print("saved paper_dashboard.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _, X_test, y, names, preprocessor, models = df.load()
    probas = {label: m.predict_proba(X_test)[:, 1] for label, (m, _) in models.items()}
    xgb, thr = models["XGBoost"]
    fig_pipeline()
    fig_pr(y, probas, thr)
    shap_figures(xgb, X_test, y, names, preprocessor, probas["XGBoost"], thr)
    fig_dashboard()


if __name__ == "__main__":
    main()
