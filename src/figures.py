"""Figures and paper tables for the report (notebook 04).

Reads only `results/*.json`, so the whole stage runs on a laptop: no Drive, no
GPU, no model. Every figure has a table beside it in the notebook, which is also
what satisfies the palette's contrast caveat -- identity is never carried by
colour alone.

Palette: slots 1-3 of the validated categorical theme (blue / orange / aqua),
checked with the dataviz validator against the light surface -- lightness band,
chroma floor, CVD separation (worst adjacent dE 9.2 deutan), normal-vision floor
(27.6) all pass. Hues are assigned to a fixed meaning (English / Hausa /
back-translated) and never cycled, so a figure that drops a series keeps the
colours of the ones that remain.
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["llama", "qwen", "medgemma"]
LABEL = {"llama": "Llama-3.2-3B", "qwen": "Qwen2.5-3B", "medgemma": "MedGemma-4B"}

# Fixed hue -> meaning. Never reassigned between figures.
EN, HA, BT = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE = "#e5e4e0", "#fcfcfb"
CHANCE = 0.25


def style():
    """Recessive axes, thin marks, text in ink rather than series colour."""
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 9.5,
        "figure.dpi": 110,
    })


def load(results_dir="results"):
    r = Path(results_dir)
    out = {"baseline": json.loads((r / "baseline_results.json").read_text("utf-8")),
           "e1": json.loads((r / "e1_summary.json").read_text("utf-8")),
           "validation": json.loads((r / "validation_summary.json").read_text("utf-8")),
           "manifest": json.loads((r / "data_manifest.json").read_text("utf-8"))}
    gate = r / "h4_pilot_gate.json"
    out["h4_gate"] = json.loads(gate.read_text("utf-8")) if gate.exists() else None
    h4 = r / "h4_results.json"
    out["h4"] = json.loads(h4.read_text("utf-8")) if h4.exists() else None
    return out


def save(fig, path, also_pdf=True):
    """PNG for the repo, PDF for the manuscript."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return str(path)


def _bar_labels(ax, bars, fmt="{:.3f}", dy=0.008):
    """Direct labels on every bar: there are few enough that each one earns its
    label, and the palette's contrast warning requires visible text."""
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=8.5, color=INK2)


# ---------------------------------------------------------------- figure 1

def fig_accuracy(res, path=None):
    """Grouped bars: the headline. English vs Hausa, against the two reference
    lines that decide whether any of it is above trivial."""
    acc = res["baseline"]["accuracy"]
    base = res["baseline"]["majority_letter_baseline"]["p0"]
    x = np.arange(len(MODELS))
    w = 0.34          # paired bars keep a visible surface gap between fills

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    b1 = ax.bar(x - w / 2 - 0.025, [acc[m]["en_p0"] for m in MODELS], w,
                label="English", color=EN)
    b2 = ax.bar(x + w / 2 + 0.025, [acc[m]["ha_p0"] for m in MODELS], w,
                label="Hausa", color=HA)
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)

    # The reference lines are named in the legend, not annotated in the plot:
    # an in-plot label collides with whichever bar happens to be tallest.
    ax.axhline(base, color=INK2, lw=1.4, ls="--", zorder=1)
    ax.axhline(CHANCE, color=MUTED, lw=1.0, ls=":", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in MODELS])
    ax.set_ylabel("accuracy (canonical option order)")
    ax.set_ylim(0, 0.80)
    ax.set_title("Every model loses most of its medical accuracy in Hausa")
    handles = [b1, b2,
               plt.Line2D([], [], color=INK2, lw=1.4, ls="--",
                          label=f"majority-letter baseline ({base:.3f})"),
               plt.Line2D([], [], color=MUTED, lw=1.0, ls=":",
                          label=f"chance ({CHANCE:.2f})")]
    ax.legend(handles=handles, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.6)
    fig.tight_layout()
    if path:
        save(fig, path)
    return fig


# ---------------------------------------------------------------- figure 2

def fig_effects(res, path=None):
    """Three panels because there are three quantities on three scales. A
    shared axis here would be a lie, and a second y-axis worse."""
    p = res["baseline"]["primary"]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5))
    panels = [
        (axes[0], "H1  English − Hausa", "accuracy gap",
         [(m, p["H1"][m]["ci"], p["H1"][m]["supported"]) for m in MODELS], 0.0),
        (axes[1], "H2  gap recovered by\nback-translating to English", "fraction of the gap",
         [(m, p["H2"][m]["recovery"], p["H2"][m]["supported"]) for m in MODELS], 0.0),
        (axes[2], "H3  consistency under\noption reordering", "English − Hausa",
         [(m, p["H3"][m]["ci"], p["H3"][m]["supported"]) for m in MODELS], 0.0),
    ]
    for ax, title, xlabel, rows, zero in panels:
        ax.set_axisbelow(True)
        ax.grid(axis="x")
        ys = np.arange(len(rows))[::-1]
        for y, (m, ci, ok) in zip(ys, rows):
            lo, hi, pt = ci["ci_low"], ci["ci_high"], ci["point"]
            colour = EN if ok else MUTED
            ax.plot([lo, hi], [y, y], color=colour, lw=2.0,
                    solid_capstyle="round", zorder=3)
            ax.plot([pt], [y], "o", ms=8, color=colour,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
            ax.text(hi + 0.02, y, f"{pt:.2f}", va="center", fontsize=8.5, color=INK2)
        ax.axvline(zero, color=INK2, lw=1.0, ls="--", zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([LABEL[m] for m, _, _ in rows])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.5, len(rows) - 0.5)
        # Room on the right for the value labels, which sit past the whisker.
        lo_x, hi_x = ax.get_xlim()
        ax.set_xlim(lo_x, hi_x + 0.18 * (hi_x - lo_x))
    fig.suptitle("Pre-registered effects, with 95% paired-bootstrap intervals",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.text(0.5, -0.03, "filled = decision rule met,   grey = not met",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    if path:
        save(fig, path)
    return fig


# ---------------------------------------------------------------- figure 3

def fig_parse_rate(res, path=None):
    """Qwen's Hausa collapse is partly a format-following failure, not only a
    knowledge one. That is invisible in an accuracy chart, so it gets its own."""
    pr = res["baseline"]["parse_rate"]
    x = np.arange(len(MODELS))
    w = 0.34          # paired bars keep a visible surface gap between fills

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    b1 = ax.bar(x - w / 2 - 0.025, [pr[m]["en_p0"] for m in MODELS], w,
                label="English", color=EN)
    b2 = ax.bar(x + w / 2 + 0.025, [pr[m]["ha_p0"] for m in MODELS], w,
                label="Hausa", color=HA)
    _bar_labels(ax, b1, fmt="{:.2f}")
    _bar_labels(ax, b2, fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in MODELS])
    ax.set_ylabel("answers parsed as a letter")
    # Five of the six bars sit at 1.00, so the legend needs its own band above
    # them rather than a corner inside the axes.
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Only Qwen fails to follow the answer format in Hausa")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    if path:
        save(fig, path)
    return fig


# ---------------------------------------------------------------- figure 4

def fig_scoring(res, path=None):
    """Dumbbell: how much of each model's Hausa score depends on the scoring
    rule. Qwen roughly doubles once the letter is read from probabilities
    rather than from text, which is the same finding as figure 3 priced in
    accuracy."""
    parsed = res["baseline"]["accuracy"]
    prob = res["baseline"]["accuracy_letter_prob"]
    base = res["baseline"]["majority_letter_baseline"]["p0"]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    ys = np.arange(len(MODELS))[::-1]
    for y, m in zip(ys, MODELS):
        a, b = parsed[m]["ha_p0"], prob[m]["ha_p0"]
        ax.plot([a, b], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=2)
        ax.plot([a], [y], "o", ms=9, color=HA, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=4)
        ax.plot([b], [y], "o", ms=9, color=BT, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=4)
        ax.text(min(a, b) - 0.012, y, f"{a:.3f}", va="center", ha="right",
                fontsize=8.5, color=INK2)
        ax.text(max(a, b) + 0.012, y, f"{b:.3f}", va="center", ha="left",
                fontsize=8.5, color=INK2)
    ax.axvline(base, color=INK2, lw=1.2, ls="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[m] for m in MODELS])
    ax.set_xlim(0.05, 0.45)
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.set_xlabel("Hausa accuracy")
    ax.set_title("Hausa accuracy under the two scoring rules")
    # Legend below the axes: three rows of dumbbells leave no interior space
    # that does not belong to some model's row.
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=HA,
                          label="parsed from text (primary)"),
               plt.Line2D([], [], marker="o", ls="", ms=8, color=BT,
                          label="letter probability (secondary)"),
               plt.Line2D([], [], color=INK2, lw=1.2, ls="--",
                          label=f"majority-letter baseline ({base:.3f})")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout()
    if path:
        save(fig, path)
    return fig


# ---------------------------------------------------------------- figure 5

def fig_e1(res, path=None):
    """E1: what goes wrong, and where it sits in the item."""
    pooled = res["e1"]["primary_pooled"]
    order = ["term", "translation", "format", "reasoning"]
    names = {"term": "medical-term loss", "translation": "translation error",
             "format": "option-format failure", "reasoning": "reasoning failure"}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4),
                             gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    ys = np.arange(len(order))[::-1]
    for y, c in zip(ys, order):
        v = pooled["codes"][c]
        lo, hi = v["ci95"]
        colour = MUTED if c == "reasoning" else HA
        ax.plot([lo, hi], [y, y], color=colour, lw=2.0,
                solid_capstyle="round", zorder=3)
        ax.plot([v["prop"]], [y], "o", ms=8, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.text(hi + 0.015, y, f"{v['prop']:.2f}  (n={v['n']})", va="center",
                fontsize=8.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([names[c] for c in order])
    ax.set_xlim(0, 0.95)
    ax.set_xlabel("proportion of 50 discordant items (95% Wilson)")
    ax.set_title("Why an item is right in English and wrong in Hausa")
    ax.text(0.0, -0.42, "orange = the Hausa text caused it,   "
                        "grey = residual, the item was answerable as presented",
            transform=ax.get_yaxis_transform(), fontsize=8.5, color=MUTED,
            va="top")

    ax = axes[1]
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    loc = pooled["text_error_location"]
    keys = ["stem", "chosen", "gold"]
    pretty = {"stem": "question\nstem", "chosen": "option the\nmodel chose",
              "gold": "gold\noption"}
    bars = ax.bar([pretty[k] for k in keys], [loc.get(k, 0) for k in keys],
                  0.6, color=HA)
    _bar_labels(ax, bars, fmt="{:.0f}", dy=0.3)
    ax.set_ylabel("text errors")
    ax.set_ylim(0, max(loc.values()) * 1.25)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("where the damage sits")
    fig.tight_layout()
    if path:
        save(fig, path)
    return fig


# ----------------------------------------------------------------- tables

def table_headline(res) -> pd.DataFrame:
    acc = res["baseline"]["accuracy"]
    pr = res["baseline"]["parse_rate"]
    p = res["baseline"]["primary"]
    base = res["baseline"]["majority_letter_baseline"]["p0"]
    rows = []
    for m in MODELS:
        ci = p["H1"][m]["ci"]
        rows.append({
            "model": LABEL[m],
            "EN": acc[m]["en_p0"],
            "HA": acc[m]["ha_p0"],
            "gap (pp)": round(100 * (acc[m]["en_p0"] - acc[m]["ha_p0"]), 1),
            "gap 95% CI": f"{100 * ci['ci_low']:.1f} to {100 * ci['ci_high']:.1f}",
            "HA vs baseline": round(acc[m]["ha_p0"] - base, 3),
            "HA parse rate": pr[m]["ha_p0"],
        })
    return pd.DataFrame(rows)


def table_decisions(res) -> pd.DataFrame:
    p = res["baseline"]["primary"]
    rows = []
    for h, label in [("H1", "H1  Hausa below English"),
                     ("H2", "H2  back-translation recovers the gap"),
                     ("H3", "H3  consistency lower in Hausa")]:
        for m in MODELS:
            d = p[h][m]
            ci = d.get("ci") or d.get("recovery")
            rows.append({
                "hypothesis": label, "model": LABEL[m],
                "estimate": round(ci["point"], 3),
                "95% CI": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
                "p (Holm)": (f"{d['p_holm']:.2e}" if "p_holm" in d else ""),
                "supported": "yes" if d["supported"] else "NO",
            })
    return pd.DataFrame(rows)


def table_sensitivity(res) -> pd.DataFrame:
    b = res["baseline"]
    rows = []
    for key, label in [("primary", "primary (all 500)"),
                       ("sens_a_no_rater_flagged", "a: rater-flagged items dropped"),
                       ("sens_b_no_english_options", "b: English-option items dropped"),
                       ("secondary_letter_prob", "letter-probability scoring")]:
        blk = b.get(key)
        if not blk:
            continue
        row = {"analysis": label, "n": blk.get("n_items")}
        for m in MODELS:
            row[LABEL[m]] = round(blk["H1"][m]["diff"], 3)
        rows.append(row)
    return pd.DataFrame(rows)


def all_figures(res, outdir="results/figures"):
    style()
    spec = [("fig1_accuracy", fig_accuracy), ("fig2_effects", fig_effects),
            ("fig3_parse_rate", fig_parse_rate), ("fig4_scoring", fig_scoring),
            ("fig5_e1_errors", fig_e1)]
    out = {}
    for name, fn in spec:
        target = f"{outdir}/{name}.png"
        fn(res, target)
        out[name] = target
    plt.close("all")
    return out
