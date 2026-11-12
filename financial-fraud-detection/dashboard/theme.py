"""
Visual theme for the dashboard: navy, gold and creme.

Base colours (theme, sidebar, fonts) live in .streamlit/config.toml; this module
adds the finishing CSS, styles matplotlib/SHAP charts, and builds the themed
Plotly charts. Data colours were checked with a colour-vision-deficiency
validator against the creme background:

    BLUE    "towards legitimate" / the model's curve   (#2E64A0)
    CRIMSON "towards fraud"                             (#BF4B3C)

Gold is an accent for the interface only; it never encodes data.

Branding: the dashboard is called CLARIX. Logo files are in dashboard/assets/clarix/.
"""

import base64
import functools
import html
import re
from pathlib import Path

import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go
import shap
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import auc, precision_recall_curve, roc_curve

NAVY = "#1F3A5F"
NAVY_DARK = "#172C49"
GOLD = "#B8892B"
GOLD_LIGHT = "#D9B45A"
CREME = "#FBF8F1"
CREME_DARK = "#F2EBDD"
CARD = "#FFFFFF"
BORDER = "#E4D9C3"
INK = "#1E2A3A"
MUTED = "#5F6B7A"
GRID = "#ECE4D4"

BLUE = "#2E64A0"
CRIMSON = "#BF4B3C"

STATUS = {  # risk-level colours; always shown with an icon and a label
    "high": {"color": "#B23A30", "tint": "#F8E7E3"},
    "review": {"color": "#A97A12", "tint": "#F8EFD9"},
    "low": {"color": "#3E7D5A", "tint": "#E5F1E9"},
}

ASSETS = Path(__file__).resolve().parent / "assets" / "clarix"
APP_ICON = ASSETS / "clarix-app-icon.png"  # browser-tab icon, rendered from clarix-app-icon.svg

SHAP_CMAP = LinearSegmentedColormap.from_list("blue_crimson", [BLUE, "#D9D4CB", CRIMSON])

CSS = f"""
<style>
/* Headings */
h1 {{ color: {NAVY}; font-weight: 700; letter-spacing: -0.01em;
     border-bottom: 2px solid {GOLD}; padding-bottom: 0.35rem; margin-bottom: 0.8rem; }}
h2, h3, h4 {{ color: {NAVY}; }}

/* Metric cards */
[data-testid="stMetric"] {{
    background: {CARD}; border: 1px solid {BORDER}; border-left: 4px solid {GOLD};
    border-radius: 10px; padding: 0.8rem 1rem;
    box-shadow: 0 1px 3px rgba(31, 58, 95, 0.06);
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.85rem; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; font-size: 1.75rem; }}
[data-testid="stMetricValue"] *, [data-testid="stMetricLabel"] * {{
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }}

/* Expanders and dataframes as soft cards */
[data-testid="stExpander"] details {{ background: {CARD}; border-color: {BORDER}; border-radius: 10px; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}

/* Sidebar */
[data-testid="stSidebar"] {{ border-right: 3px solid {GOLD}; }}
.sidebar-brand {{ padding: 0.2rem 0 0.9rem 0; border-bottom: 1px solid rgba(217, 180, 90, 0.45);
                  margin-bottom: 0.8rem; }}
.sidebar-brand img {{ width: 100%; display: block; }}
.nav-label {{ color: #C9D3E0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
              margin: 0.2rem 0 0.1rem 0.2rem; }}

/* Sidebar navigation: flat buttons, the current page highlighted in gold */
[data-testid="stSidebar"] .stButton {{ margin-bottom: -0.55rem; }}
[data-testid="stSidebar"] .stButton button {{
    justify-content: flex-start; text-align: left; border: none; border-radius: 8px;
    background: transparent; color: #F5EFE3; padding: 0.5rem 0.8rem;
    border-left: 4px solid transparent; transition: background 0.15s;
}}
[data-testid="stSidebar"] .stButton button div {{ justify-content: flex-start; }}
[data-testid="stSidebar"] .stButton button:hover {{ background: rgba(255, 255, 255, 0.08); color: #FFFFFF; }}
[data-testid="stSidebar"] .stButton button[kind="primary"] {{
    background: rgba(217, 180, 90, 0.18); border-left: 4px solid {GOLD_LIGHT};
    color: {GOLD_LIGHT}; font-weight: 600;
}}
[data-testid="stSidebar"] .stButton button:focus:not(:active) {{ box-shadow: none; }}

/* Home page cards: "Open" button pinned to the bottom of each card */
.st-key-home_cards [data-testid="stVerticalBlock"] {{ height: 100%; }}
.st-key-home_cards [data-testid="stElementContainer"]:has(.stButton) {{ margin-top: auto; }}
.st-key-home_cards .stButton button {{ background: {NAVY}; color: {CREME}; border: none; }}
.st-key-home_cards .stButton button:hover {{ background: {NAVY_DARK}; color: {GOLD_LIGHT}; }}

/* Audit Log: the batch card being viewed */
.st-key-batch_card_sel {{ border: 2px solid {GOLD} !important; background: {CARD};
                          box-shadow: 0 2px 6px rgba(184, 137, 43, 0.18); }}

/* Home hero banner */
.hero {{ background: linear-gradient(135deg, {NAVY} 0%, {NAVY_DARK} 100%); color: {CREME};
         border-radius: 14px; padding: 1.8rem 2rem; margin-bottom: 1.4rem;
         border-bottom: 4px solid {GOLD}; }}
.hero h1 {{ color: {CREME}; border: none; margin: 0 0 0.4rem 0; padding: 0; font-family: serif; }}
.hero p {{ color: #D8DEE8; margin: 0; font-size: 1.02rem; max-width: 52rem; }}
.hero .kicker {{ color: {GOLD_LIGHT}; letter-spacing: 0.32em; font-size: 0.8rem; font-weight: 600;
                 margin-bottom: 0.35rem; }}
.hero .fyp {{ color: #AEB9C9; font-size: 0.82rem; margin-top: 0.9rem; }}
.hero .accent {{ color: {GOLD_LIGHT}; font-weight: 600; }}

/* Verdict card */
.verdict {{ background: {CARD}; border: 1px solid {BORDER}; border-left: 6px solid var(--c);
            border-radius: 12px; padding: 1rem 1.3rem; margin: 0.4rem 0 1rem 0;
            box-shadow: 0 1px 3px rgba(31, 58, 95, 0.06); }}
.verdict .title {{ font-size: 1.25rem; font-weight: 700; color: var(--c); }}
.verdict .score {{ float: right; text-align: right; color: {MUTED}; font-size: 0.8rem; }}
.verdict .score b {{ display: block; font-size: 1.6rem; color: {INK}; }}
.verdict .detail {{ color: {INK}; margin: 0.35rem 0 0.6rem 0; }}
.verdict .action {{ background: var(--t); border-radius: 8px; padding: 0.55rem 0.8rem; color: {INK}; }}
.verdict .gauge {{ position: relative; height: 8px; border-radius: 4px; margin: 0.2rem 0 0.9rem 0;
                   background: linear-gradient(90deg, {STATUS['low']['tint']} 0%,
                   {STATUS['low']['tint']} var(--floor), {STATUS['review']['tint']} var(--floor),
                   {STATUS['review']['tint']} var(--thr), {STATUS['high']['tint']} var(--thr)); }}
.verdict .gauge .dot {{ position: absolute; top: -4px; width: 16px; height: 16px; border-radius: 50%;
                        background: var(--c); border: 2px solid {CARD}; left: calc(var(--p) - 8px); }}
.verdict .gauge .tick {{ position: absolute; top: -3px; width: 2px; height: 14px; background: {MUTED};
                         left: var(--thr); }}

/* Confusion grid */
.cm {{ display: grid; grid-template-columns: auto 1fr 1fr; gap: 6px; max-width: 34rem; }}
.cm div {{ border-radius: 10px; padding: 0.7rem 0.9rem; }}
.cm .h {{ color: {MUTED}; font-size: 0.8rem; text-align: center; align-self: end; padding: 0.2rem; }}
.cm .r {{ color: {MUTED}; font-size: 0.8rem; text-align: right; align-self: center; padding: 0.2rem; }}
.cm .cell {{ background: {CARD}; border: 1px solid {BORDER}; }}
.cm .cell b {{ display: block; font-size: 1.5rem; color: {INK}; }}
.cm .cell span {{ color: {MUTED}; font-size: 0.82rem; }}
.cm .good {{ border-left: 4px solid {STATUS['low']['color']}; }}
.cm .bad {{ border-left: 4px solid {STATUS['high']['color']}; background: {STATUS['high']['tint']}; }}
</style>
"""


def apply_theme():
    """Inject CSS and style matplotlib/SHAP charts to match the theme."""
    st.markdown(CSS, unsafe_allow_html=True)
    mpl.rcParams.update({
        "figure.facecolor": CREME, "axes.facecolor": CREME, "savefig.facecolor": CREME,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": BORDER,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": NAVY,
        "font.family": "sans-serif", "axes.titleweight": "bold",
    })
    shap.plots._style.set_style(
        primary_color_positive=CRIMSON, primary_color_negative=BLUE,
        secondary_color_positive="#E2A9A0", secondary_color_negative="#A9C0DD",
        hlines_color=GRID, vlines_color=BORDER, tick_labels_color=MUTED,
    )


@functools.lru_cache(maxsize=None)
def logo_uri(name="clarix-logo-navy.svg", crop=True):
    """A logo file as a data URI for an <img> tag.

    Drops the full-size background rectangle (so the logo sits on the sidebar's own navy),
    trims the empty margin around the artwork, and leaves out the embedded C2PA metadata
    block, which is most of the file size. The original files are not changed.
    """
    svg = (ASSETS / name).read_text(encoding="utf-8")
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)
    svg = re.sub(r'<rect width="880" height="264" fill="[^"]+"/>', "", svg, count=1)
    if crop:
        svg = svg.replace('viewBox="0 0 880 264" width="880" height="264"',
                          'viewBox="62 46 790 176" width="790" height="176"', 1)
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def sidebar_brand():
    st.sidebar.markdown(
        f'<div class="sidebar-brand"><img src="{logo_uri()}" '
        'alt="CLARIX: XAI fraud detection and transaction monitoring"></div>',
        unsafe_allow_html=True)


def hero(title, subtitle_html, kicker=None, footnote=None):
    st.markdown(
        '<div class="hero">'
        + (f'<div class="kicker">{html.escape(kicker)}</div>' if kicker else "")
        + f'<h1>{html.escape(title)}</h1><p>{subtitle_html}</p>'
        + (f'<div class="fyp">{html.escape(footnote)}</div>' if footnote else "")
        + '</div>', unsafe_allow_html=True)


def verdict_card(v, probability, threshold, review_floor, score_text):
    """Status card: icon + label + score gauge + suggested action."""
    s = STATUS[v["key"]]
    style = (f"--c:{s['color']};--t:{s['tint']};--p:{probability * 100:.2f}%;"
             f"--thr:{threshold * 100:.2f}%;--floor:{review_floor * 100:.2f}%")
    st.markdown(
        f'<div class="verdict" style="{style}">'
        f'<div class="score">fraud score<b>{html.escape(score_text)}</b></div>'
        f'<div class="title">{v["icon"]} {html.escape(v["label"])}: {html.escape(v["headline"])}</div>'
        f'<div class="detail">{html.escape(v["detail"])}</div>'
        f'<div class="gauge" title="Low risk | Needs review | High risk"><div class="tick"></div>'
        f'<div class="dot"></div></div>'
        f'<div class="action"><b>Suggested action:</b> {html.escape(v["action"])}</div>'
        f'</div>', unsafe_allow_html=True)


def confusion_grid(tn, fp, fn, tp):
    """2x2 confusion matrix as labelled cards (a colour scale is useless at 56,648 vs 3)."""
    def cell(n, label, kind):
        return f'<div class="cell {kind}"><b>{n:,}</b><span>{label}</span></div>'
    st.markdown(
        '<div class="cm">'
        '<div></div><div class="h">Model: legitimate</div><div class="h">Model: fraud</div>'
        '<div class="r">Actually<br>legitimate</div>'
        + cell(tn, "correctly let through", "good") + cell(fp, "false alarms", "bad") +
        '<div class="r">Actually<br>fraud</div>'
        + cell(fn, "missed frauds", "bad") + cell(tp, "frauds caught", "good") +
        '</div>', unsafe_allow_html=True)


def _layout(fig, x_title, y_title, height=400):
    fig.update_layout(
        height=height, margin=dict(l=70, r=20, t=20, b=60), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="sans-serif", size=15),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font_color=INK),
    )
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickformat=".0%",
                tickfont=dict(color=MUTED), automargin=True)
    fig.update_xaxes(title=dict(text=x_title, standoff=12), range=[-0.01, 1.01], **axis)
    fig.update_yaxes(title=dict(text=y_title, standoff=12), range=[-0.01, 1.03], **axis)
    return fig


def _rates_at(y_true, proba, threshold):
    """Precision, recall and false-alarm rate at the actual alert threshold."""
    y_true = np.asarray(y_true).astype(int)
    flagged = proba >= threshold
    tp = int((flagged & (y_true == 1)).sum())
    fp = int((flagged & (y_true == 0)).sum())
    return tp / max(tp + fp, 1), tp / max(int(y_true.sum()), 1), fp / max(int((y_true == 0).sum()), 1)


def _operating_point(fig, x, y, text, position="bottom left"):
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text", text=[text], textposition=position,
        textfont=dict(color=INK, size=14),
        marker=dict(size=11, color=NAVY, line=dict(color=CREME, width=2)),
        hovertemplate=f"{text}<extra></extra>"))


def roc_chart(y_true, proba, threshold):
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", hoverinfo="skip",
                             line=dict(color=MUTED, width=1, dash="dot")))
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", line=dict(color=BLUE, width=2),
        hovertemplate="False alarms: %{x:.2%}<br>Frauds caught: %{y:.1%}<extra></extra>"))
    _, recall, fa_rate = _rates_at(y_true, proba, threshold)
    _operating_point(fig, fa_rate, recall, f"  alert threshold: {recall:.0%} caught", "bottom right")
    fig.add_annotation(x=0.98, y=0.04, xanchor="right", showarrow=False,
                       text=f"AUC {auc(fpr, tpr):.3f}", font=dict(color=MUTED))
    fig.add_annotation(x=0.62, y=0.55, showarrow=False, text="random guessing",
                       font=dict(color=MUTED, size=13), textangle=-38)
    return _layout(fig, "False alarm rate", "Share of frauds caught")


def pr_chart(y_true, proba, threshold):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode="lines", line=dict(color=BLUE, width=2),
        hovertemplate="Frauds caught: %{x:.1%}<br>Alerts that are fraud: %{y:.1%}<extra></extra>"))
    prec_t, rec_t, _ = _rates_at(y_true, proba, threshold)
    _operating_point(fig, rec_t, prec_t,
                     f"alert threshold: {prec_t:.0%} of alerts are fraud, {rec_t:.0%} caught  ")
    fig.add_annotation(x=0.02, y=0.04, xanchor="left", showarrow=False,
                       text=f"Area {auc(recall, precision):.3f}", font=dict(color=MUTED))
    return _layout(fig, "Recall: share of frauds caught", "Precision: share of alerts that are fraud")


def reasons_chart(reason_list):
    """Horizontal bars of the strongest factors: crimson = towards fraud, blue = towards legitimate."""
    rs = list(reversed(reason_list))  # strongest at the top
    labels = [f"{r['feature']} = {r['value']:,.2f}" for r in rs]
    fig = go.Figure(go.Bar(
        x=[r["shap"] for r in rs], y=labels, orientation="h",
        marker=dict(color=[CRIMSON if r["raises_risk"] else BLUE for r in rs],
                    line=dict(color=CREME, width=2), cornerradius=4),
        hovertext=[("raises" if r["raises_risk"] else "lowers") + f" risk ({r['strength']})"
                   for r in rs],
        hoverinfo="text",
    ))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_layout(
        height=70 + 44 * len(rs), margin=dict(l=120, r=10, t=10, b=40), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="sans-serif", size=15),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font_color=INK),
        xaxis=dict(title="← legitimate   |   fraud →", showticklabels=False,
                   zeroline=False, showgrid=False),
        yaxis=dict(showgrid=False, automargin=True, tickfont=dict(color=INK)),
    )
    return fig


def recolor_lime(fig):
    """LIME draws green/red bars; recolour to the dashboard's fraud/legitimate colours."""
    for ax in fig.axes:
        for patch in ax.patches:
            patch.set_color(CRIMSON if patch.get_width() > 0 else BLUE)
        ax.set_facecolor(CREME)
    return fig
