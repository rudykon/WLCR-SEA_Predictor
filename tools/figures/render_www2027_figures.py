#!/usr/bin/env python3
"""Render the figures used by the WWW 2027 manuscript.

The script reads frozen analysis exports under artifacts/ and writes the data
figures paper/figures/fig2_comparison.pdf, fig3_robustness.pdf,
fig4_planning.pdf, and fig5_audit.pdf. The externally authored framework,
fig1_framework.pdf, is preserved and recorded in the manifest.
The data figures run no model, training, or
evaluation. Every plotted value used in the text is checked against the
values printed in the manuscript before drawing.

Run from the repository root with matplotlib on the path, for example:
    PYTHONPATH=.runtime/lightgbm python3 tools/figures/render_www2027_figures.py

Text uses pdfLaTeX with the manuscript's Libertine text and math fonts.
The framework PNG preview requires pdftocairo. Data-figure PNG previews
additionally require pdftocairo or Ghostscript. docs/images/ holds only the
five current manuscript previews.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "paper" / "figures"
MISSINGNESS_CSV = (
    ROOT / "artifacts/reproduction/analysis/missingness/comparative_missingness.csv"
)
PLANNING_CSV = (
    ROOT / "artifacts/reproduction/analysis/downstream_prb_20261019/scores.csv"
)
DELETION_CSV = (
    ROOT
    / "artifacts/reproduction/figures/readability_20261019/"
    / "wlcr_sea_auditability_panel_a_data.csv"
)
EXPERT_CSV = (
    ROOT
    / "artifacts/reproduction/figures/readability_20261019/"
    / "wlcr_sea_auditability_panel_c_data.csv"
)
PREVIEW_DIR = ROOT / "docs" / "images"
MANIFEST = FIGURE_DIR / "www2027_figures_manifest.json"

# Match Figure 1's blue, teal, pale-green training lane, and blue-gray details.
# The blue and slate are sampled from the framework; sage and teal are darker
# companion shades so curves remain readable on the white data-figure canvas.
# Model colors are shared across every panel; marker shapes remain redundant.
COLORS = {
    "wlcr": "#508DCF",
    "dlinear": "#8EB79B",
    "patchtst": "#75AEC3",
    "grud": "#617383",
    "lightgbm": "#7F7F7F",
    "top": "#C0504D",
    "random": "#7F7F7F",
    "text": "#222222",
    "grid": "#E3E3E3",
    "placeholder": "#B03A2E",
    "frame": "#8C8C8C",
}

METHODS = (
    ("A6_mixed_aug", "WLCR-SEA", "wlcr", "o", "-"),
    ("dlinear_aug", "DLinear-Aug", "dlinear", "s", "-"),
    ("patchtst_aug", "PatchTST-Aug", "patchtst", "^", "-"),
    ("grud_direct_aug", "GRU-D-Direct-Aug", "grud", "D", "-"),
    ("original_wlcr", "73D LightGBM (not augmentation-matched)", "lightgbm", "x", "--"),
)
MATCHED = ("A6_mixed_aug", "dlinear_aug", "patchtst_aug", "grud_direct_aug")
MATCHED_LABELS = {
    "A6_mixed_aug": "WLCR-SEA",
    "dlinear_aug": "DLinear-Aug",
    "patchtst_aug": "PatchTST-Aug",
    "grud_direct_aug": "GRU-D-Direct-Aug",
}
MECHANISMS = (
    ("mcar", "(a) MCAR"),
    ("block", "(b) Block"),
    ("recent_tail", "(c) Timeline tail"),
    ("asynchronous", "(d) Asynchronous"),
)
RATES_PERCENT = (0, 10, 20, 30, 50)
SEVERE_MECHANISMS = (
    ("block", "block_0.50", "Block"),
    ("recent_tail", "timeline_tail_0.50", "Timeline\ntail"),
    ("asynchronous", "asynchronous_0.50", "Asynchronous"),
)
COST_RATIOS = (1.0, 3.0, 9.0)

# Four-decimal values printed in paper/results.tex (Tables 2 and 3 of the manuscript).
EXPECTED_WAPE = {
    ("A6_mixed_aug", "mcar", 0.0): "0.1955",
    ("dlinear_aug", "mcar", 0.0): "0.1907",
    ("patchtst_aug", "mcar", 0.0): "0.1960",
    ("grud_direct_aug", "mcar", 0.0): "0.2235",
    ("original_wlcr", "mcar", 0.0): "0.1951",
    ("A6_mixed_aug", "block", 0.2): "0.2017",
    ("A6_mixed_aug", "recent_tail", 0.2): "0.2048",
    ("A6_mixed_aug", "asynchronous", 0.3): "0.2045",
    ("A6_mixed_aug", "block", 0.5): "0.2196",
    ("A6_mixed_aug", "recent_tail", 0.5): "0.2460",
    ("A6_mixed_aug", "asynchronous", 0.5): "0.2172",
    ("original_wlcr", "block", 0.5): "0.2636",
    ("original_wlcr", "recent_tail", 0.5): "0.3234",
    ("original_wlcr", "asynchronous", 0.5): "0.2261",
}
# Paired WLCR-SEA minus DLinear-Aug differences printed in paper/results.tex (Table robust).
EXPECTED_WAPE_DELTA_DLINEAR = {
    "block": "-0.04022",
    "recent_tail": "-0.03451",
    "asynchronous": "-0.02044",
}
# Four-decimal planning costs printed in paper/appendix_results.tex (Table prb-full).
EXPECTED_COST = {
    ("A6_mixed_aug", "clean", 3.0): "0.5087",
    ("A6_mixed_aug", "block_0.20", 3.0): "0.5114",
    ("A6_mixed_aug", "timeline_tail_0.20", 3.0): "0.5188",
    ("A6_mixed_aug", "asynchronous_0.30", 3.0): "0.5110",
    ("A6_mixed_aug", "block_0.50", 3.0): "0.5215",
    ("A6_mixed_aug", "timeline_tail_0.50", 3.0): "0.5615",
    ("A6_mixed_aug", "asynchronous_0.50", 3.0): "0.5180",
    ("A6_mixed_aug", "block_0.50", 1.0): "0.2963",
    ("A6_mixed_aug", "block_0.50", 9.0): "0.8699",
    ("A6_mixed_aug", "timeline_tail_0.50", 9.0): "0.9961",
    ("A6_mixed_aug", "asynchronous_0.50", 9.0): "0.8493",
    ("dlinear_aug", "block_0.50", 3.0): "0.5595",
}
# Printed paired differences for gamma = 3 (WLCR-SEA minus DLinear-Aug, Table prb-full).
EXPECTED_COST_DELTA_DLINEAR_G3 = {
    "clean": "+0.01963",
    "block_0.50": "-0.03804",
    "timeline_tail_0.50": "-0.03248",
    "asynchronous_0.50": "-0.02349",
}
EXPECTED_DELETION = {
    "Top-weight": ("0.00595", "0.00441", "0.00757"),
    "Matched-random": ("0.00104", "0.00079", "0.00131"),
}

EXPERT_LABEL_OFFSET = {
    "last_day": (-0.012, 0.014, "right"),
    "last_week": (0.014, -0.030, "left"),
    "last_biweek": (0.014, -0.030, "left"),
    "same_hour_median_7d": (0.014, -0.030, "left"),
    "same_hour_median_14d": (0.0, 0.038, "center"),
    "bounded_week_trend": (0.022, 0.024, "left"),
    "window_local_median": (0.014, 0.014, "left"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            # Match acmart's pdfLaTeX text and math fonts exactly. PGF embeds
            # the TeX Type 1 fonts instead of substituting a system font.
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": (
                r"\usepackage[T1]{fontenc}"
                r"\usepackage{libertine}"
                r"\usepackage[libertine]{newtxmath}"
            ),
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7.0,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "lines.linewidth": 1.3,
            "lines.markersize": 3.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=COLORS["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=2.5, width=0.6, pad=2)


def save(fig: plt.Figure, pdf_path: Path, preview_name: str) -> None:
    fig.savefig(pdf_path, backend="pgf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        PREVIEW_DIR / preview_name,
        backend="pgf", dpi=300, bbox_inches="tight", pad_inches=0.02,
    )
    plt.close(fig)



def render_framework_preview() -> None:
    """Export the current Figure 1 without modifying its authored PDF."""
    subprocess.run(
        [
            "pdftocairo", "-singlefile", "-png", "-r", "300",
            str(FIGURE_DIR / "fig1_framework.pdf"),
            str(PREVIEW_DIR / "paper_figure_framework"),
        ],
        check=True,
    )


def load_missingness() -> dict[tuple[str, str, float], float]:
    values: dict[tuple[str, str, float], float] = {}
    for row in read_rows(MISSINGNESS_CSV):
        key = (row["method"], row["mechanism"], float(row["requested_rate"]))
        values[key] = float(row["macro_wape"])
    for key, printed in EXPECTED_WAPE.items():
        actual = f"{values[key]:.4f}"
        if actual != printed:
            raise RuntimeError(f"robustness value {key} is {actual}, paper prints {printed}")
    for mechanism, printed in EXPECTED_WAPE_DELTA_DLINEAR.items():
        delta = values[("A6_mixed_aug", mechanism, 0.5)] - values[("dlinear_aug", mechanism, 0.5)]
        if f"{delta:.5f}" != printed:
            raise RuntimeError(f"50% {mechanism} delta is {delta:.5f}, paper prints {printed}")
    return values


def load_planning() -> dict[tuple[str, str, float], float]:
    costs: dict[tuple[str, str, float], float] = {}
    for row in read_rows(PLANNING_CSV):
        if row["resource"] != "macro_prb":
            continue
        key = (row["method"], row["scenario"], float(row["shortage_cost"]))
        costs[key] = float(row["cost"])
    for key, printed in EXPECTED_COST.items():
        actual = f"{costs[key]:.4f}"
        if actual != printed:
            raise RuntimeError(f"planning cost {key} is {actual}, paper prints {printed}")
    for scenario, printed in EXPECTED_COST_DELTA_DLINEAR_G3.items():
        delta = costs[("A6_mixed_aug", scenario, 3.0)] - costs[("dlinear_aug", scenario, 3.0)]
        if f"{delta:+.5f}" != printed:
            raise RuntimeError(f"gamma=3 {scenario} delta is {delta:+.5f}, paper prints {printed}")
    return costs


def render_robustness(output: Path, values: dict[tuple[str, str, float], float]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.85), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.74, bottom=0.2, wspace=0.16)
    handles = []
    for ax, (mechanism, title) in zip(axes, MECHANISMS):
        style_axis(ax)
        ax.set_title(title, loc="left", pad=3)
        for method, label, color_key, marker, linestyle in METHODS:
            xs = [0.0] + [rate for rate in (0.1, 0.2, 0.3, 0.5)]
            ys = [values[(method, "mcar", 0.0)]] + [
                values[(method, mechanism, rate)] for rate in (0.1, 0.2, 0.3, 0.5)
            ]
            line, = ax.plot(
                [100 * x for x in xs],
                ys,
                color=COLORS[color_key],
                marker=marker,
                linestyle=linestyle,
                zorder=3 if method == "A6_mixed_aug" else 2,
                linewidth=1.8 if method == "A6_mixed_aug" else 1.1,
                markeredgewidth=0.6,
                markerfacecolor="white" if marker not in {"x"} else COLORS[color_key],
                label=label,
            )
            if mechanism == "mcar":
                handles.append(line)
        ax.set_xticks(RATES_PERCENT)
        ax.set_xlim(-2.5, 53)
        ax.set_xlabel("Requested loss (%)")
    axes[0].set_ylabel("Macro WAPE")
    axes[0].set_ylim(0.185, 0.335)
    axes[0].set_yticks([0.20, 0.23, 0.26, 0.29, 0.32])
    fig.legend(
        handles=handles,
        labels=[method[1] for method in METHODS],
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.1,
        handlelength=2.2,
    )
    save(fig, output, "paper_figure_robustness.png")


def load_deletion() -> dict[str, tuple[float, float, float]]:
    values: dict[str, tuple[float, float, float]] = {}
    for row in read_rows(DELETION_CSV):
        name = row["condition"].replace("\n", " ").replace(" deletion", "")
        values[name] = (float(row["delta"]), float(row["low"]), float(row["high"]))
    for name, printed in EXPECTED_DELETION.items():
        delta, low, high = values[name]
        actual = (f"{delta:.5f}", f"{low:.5f}", f"{high:.5f}")
        if actual != printed:
            raise RuntimeError(f"deletion {name} is {actual}, paper prints {printed}")
    return values


def load_experts() -> list[dict[str, float | str]]:
    experts = []
    for row in read_rows(EXPERT_CSV):
        experts.append(
            {
                "key": row["expert"],
                "label": row["display"],
                "availability": float(row["availability_rate"]),
                "attention": float(row["mean_attention_when_available"]),
                "attention_low": float(row["mean_attention_when_available_ci_low"]),
                "attention_high": float(row["mean_attention_when_available_ci_high"]),
                "change": float(row["mean_absolute_prediction_change"]),
                "change_low": float(row["mean_absolute_prediction_change_ci_low"]),
                "change_high": float(row["mean_absolute_prediction_change_ci_high"]),
            }
        )
    if len(experts) != 7:
        raise RuntimeError("expert panel must contain the seven non-prior experts")
    return experts


def render_audit(output: Path, deletion, experts) -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(3.33, 1.95), gridspec_kw={"width_ratios": [0.8, 1.3], "wspace": 0.62}
    )
    fig.subplots_adjust(left=0.19, right=0.99, top=0.86, bottom=0.2)

    style_axis(ax_a, axis="x")
    rows = (("Top-weight", 1.0, COLORS["top"]), ("Matched-random", 0.0, COLORS["random"]))
    for name, y, color in rows:
        delta, low, high = deletion[name]
        ax_a.errorbar(
            delta,
            y,
            xerr=[[delta - low], [high - delta]],
            fmt="o",
            color=color,
            markersize=4.2,
            capsize=2.0,
            elinewidth=1.0,
            capthick=1.0,
        )
        ax_a.annotate(
            f"{delta:.5f}",
            (high, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=5.8,
        )
    ax_a.set_yticks([1.0, 0.0])
    ax_a.set_yticklabels(["Top-weight", "Matched\nrandom"])
    ax_a.set_ylim(-0.6, 1.5)
    ax_a.set_xlim(0.0, 0.0105)
    ax_a.set_xticks([0.0, 0.004, 0.008])
    ax_a.set_xlabel("Increase in macro WAPE")
    ax_a.set_title("(a) Rerouted deletion", loc="left", pad=3)

    style_axis(ax_b, axis="both")
    for expert in experts:
        size = 50 * float(expert["availability"])
        ax_b.errorbar(
            expert["attention"],
            expert["change"],
            xerr=[[expert["attention"] - expert["attention_low"]], [expert["attention_high"] - expert["attention"]]],
            yerr=[[expert["change"] - expert["change_low"]], [expert["change_high"] - expert["change"]]],
            fmt="none",
            ecolor="#7FA6D6",
            elinewidth=0.6,
            capsize=1.0,
            alpha=0.7,
            zorder=2,
        )
        ax_b.scatter(
            expert["attention"],
            expert["change"],
            s=size,
            color=COLORS["wlcr"],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        dx, dy, ha = EXPERT_LABEL_OFFSET[str(expert["key"])]
        ax_b.text(
            float(expert["attention"]) + dx,
            float(expert["change"]) + dy,
            str(expert["label"]),
            ha=ha,
            va="center",
            fontsize=5.6,
        )
    ax_b.set_xlim(-0.02, 0.52)
    ax_b.set_ylim(-0.01, 0.48)
    ax_b.set_xticks([0.0, 0.2, 0.4])
    ax_b.set_yticks([0.0, 0.2, 0.4])
    ax_b.set_xlabel("Mean allocation weight")
    ax_b.set_ylabel(r"Mean |$\Delta$ prediction|")
    ax_b.set_title("(b) Expert evidence", loc="left", pad=3)

    save(fig, output, "paper_figure_audit.png")


def severe_mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values)


def render_comparison(output: Path, missingness, planning) -> None:
    """Figure 3: three views of the 50% comparison, in the style of RegimeGuard Fig. 3."""
    rates = {method[0]: method for method in METHODS}
    group_labels = [label for _, _, label in SEVERE_MECHANISMS]
    series = [method for method in METHODS]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.78, bottom=0.2, wspace=0.42)
    ax_a, ax_b, ax_c = axes

    style_axis(ax_a)
    width = 0.15
    for index, (method, label, color_key, _, _) in enumerate(series):
        offsets = [group + (index - 2) * width for group in range(len(SEVERE_MECHANISMS))]
        heights = [missingness[(method, mechanism, 0.5)] for mechanism, _, _ in SEVERE_MECHANISMS]
        hatched = method == "original_wlcr"
        ax_a.bar(
            offsets,
            heights,
            width=width,
            facecolor="white" if hatched else COLORS[color_key],
            edgecolor=COLORS[color_key] if hatched else "white",
            hatch="////" if hatched else None,
            linewidth=0.5,
            zorder=3,
        )
    ax_a.set_xticks(range(len(SEVERE_MECHANISMS)))
    ax_a.set_xticklabels(group_labels)
    ax_a.set_ylim(0, 0.36)
    ax_a.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax_a.set_ylabel("Macro WAPE")
    ax_a.set_title("(a) Macro WAPE, 50% loss", loc="left", pad=3)

    style_axis(ax_b)
    for index, method in enumerate(MATCHED):
        offsets = [group + (index - 1.5) * 0.2 for group in range(len(SEVERE_MECHANISMS))]
        heights = [planning[(method, scenario, 3.0)] for _, scenario, _ in SEVERE_MECHANISMS]
        ax_b.bar(
            offsets,
            heights,
            width=0.2,
            color=COLORS[rates[method][2]],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    ax_b.set_xticks(range(len(SEVERE_MECHANISMS)))
    ax_b.set_xticklabels(group_labels)
    ax_b.set_ylim(0, 0.7)
    ax_b.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax_b.set_ylabel(r"Planning cost $C_3$")
    ax_b.set_title(r"(b) Planning cost $C_3$, 50% loss", loc="left", pad=3)

    style_axis(ax_c, axis="both")
    clean = {method: missingness[(method, "mcar", 0.0)] for method, _, _, _, _ in METHODS}
    severe = {
        method: severe_mean({mechanism: missingness[(method, mechanism, 0.5)]
                             for mechanism, _, _ in SEVERE_MECHANISMS})
        for method, _, _, _, _ in METHODS
    }
    label_offsets = {
        "A6_mixed_aug": (0.0008, -0.0012, "left"),
        "dlinear_aug": (0.0008, 0.0002, "left"),
        "patchtst_aug": (0.0008, -0.0040, "left"),
        "grud_direct_aug": (-0.0014, 0.0002, "right"),
        "original_wlcr": (0.0008, 0.0014, "left"),
    }
    for method, label, color_key, marker, _ in METHODS:
        hollow = method == "original_wlcr"
        ax_c.scatter(
            clean[method],
            severe[method],
            s=22,
            marker="o" if hollow else marker,
            facecolors="white" if hollow else COLORS[color_key],
            edgecolors=COLORS[color_key],
            linewidths=0.9,
            zorder=3,
        )
        dx, dy, ha = label_offsets[method]
        short = {"A6_mixed_aug": "WLCR-SEA", "dlinear_aug": "DLinear-Aug",
                 "patchtst_aug": "PatchTST-Aug", "grud_direct_aug": "GRU-D-Direct-Aug",
                 "original_wlcr": "73D LightGBM"}[method]
        ax_c.text(clean[method] + dx, severe[method] + dy, short, ha=ha, va="center", fontsize=5.6)
    ax_c.set_xlim(0.1870, 0.2305)
    ax_c.set_ylim(0.2225, 0.2865)
    ax_c.set_xticks([0.19, 0.20, 0.21, 0.22])
    ax_c.set_yticks([0.23, 0.25, 0.27])
    ax_c.set_xlabel("Clean macro WAPE")
    ax_c.set_ylabel("Mean 50% macro WAPE")
    ax_c.set_title("(c) Clean vs. severe-loss error", loc="left", pad=3)

    handles = [Patch(facecolor="white" if method == "original_wlcr" else COLORS[color_key],
                     edgecolor=COLORS[color_key],
                     hatch="////" if method == "original_wlcr" else None,
                     linewidth=0.5, label=label)
               for method, label, color_key, _, _ in METHODS]
    fig.legend(handles=handles, labels=[m[1] for m in METHODS], loc="upper center",
               ncol=5, bbox_to_anchor=(0.5, 1.0), columnspacing=1.1, handlelength=1.6)
    save(fig, output, "paper_figure_comparison.png")


def render_planning(output: Path, planning) -> None:
    """Figure 5: planning-cost proxy across conditions and cost ratios."""
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.0, 2.0), gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.3}
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.78, bottom=0.2)

    conditions = (
        ("clean", "Clean"),
        ("block_0.50", "Block"),
        ("timeline_tail_0.50", "Timeline\ntail"),
        ("asynchronous_0.50", "Asynchronous"),
    )
    style_axis(ax_a)
    width = 0.2
    for index, method in enumerate(MATCHED):
        offsets = [group + (index - 1.5) * width for group in range(len(conditions))]
        heights = [planning[(method, scenario, 3.0)] for scenario, _ in conditions]
        ax_a.bar(
            offsets,
            heights,
            width=width,
            color=COLORS[dict((m[0], m[2]) for m in METHODS)[method]],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    ax_a.set_xticks(range(len(conditions)))
    ax_a.set_xticklabels([label for _, label in conditions])
    ax_a.set_ylim(0, 0.7)
    ax_a.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax_a.set_ylabel(r"Planning cost $C_3$")
    ax_a.set_title(r"(a) Cost at $\gamma=3$ by condition", loc="left", pad=3)

    style_axis(ax_b)
    xs = list(range(len(COST_RATIOS)))
    for method, label, color_key, marker, _ in METHODS[:4]:
        ys = [
            severe_mean({scenario: planning[(method, scenario, gamma)]
                         for _, scenario, _ in SEVERE_MECHANISMS})
            for gamma in COST_RATIOS
        ]
        ax_b.plot(
            xs,
            ys,
            color=COLORS[color_key],
            marker=marker,
            linestyle="-",
            linewidth=1.8 if method == "A6_mixed_aug" else 1.1,
            markeredgewidth=0.6,
            markerfacecolor="white",
            zorder=3 if method == "A6_mixed_aug" else 2,
        )
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(["1", "3", "9"])
    ax_b.set_xlim(-0.25, 2.25)
    ax_b.set_ylim(0.2, 1.4)
    ax_b.set_yticks([0.4, 0.8, 1.2])
    ax_b.set_xlabel(r"Cost ratio $\gamma$")
    ax_b.set_ylabel(r"Mean 50% cost $C_\gamma$")
    ax_b.set_title(r"(b) Cost across $\gamma$, 50% loss", loc="left", pad=3)

    handles = [Patch(facecolor=COLORS[color_key], edgecolor=COLORS[color_key], label=label)
               for method, label, color_key, _, _ in METHODS[:4]]
    fig.legend(handles=handles, labels=[m[1] for m in METHODS[:4]], loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.6)
    save(fig, output, "paper_figure_planning.png")


def placeholder_box(ax, xy, width, height, text, dashed=False, fontsize=5.8, color="#4D4D4D"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0,rounding_size=0.012",
        linewidth=0.7,
        edgecolor=COLORS["frame"] if not dashed else "#7F7F7F",
        facecolor="#F4F4F4" if not dashed else "none",
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, color=color, linespacing=1.15)


def arrow(ax, start, end, dashed=False):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", color="#7F7F7F", lw=0.8,
                        linestyle="--" if dashed else "-", mutation_scale=6),
    )


def render_placeholders() -> None:
    """Legacy placeholders, not included in the current manuscript or main()."""
    fig, ax = plt.subplots(figsize=(3.33, 1.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.012, 0.012), 0.976, 0.976, boxstyle="round,pad=0,rounding_size=0.02",
                                fill=False, linestyle="--", linewidth=0.8, edgecolor=COLORS["frame"]))
    ax.text(0.5, 0.92, "PLACEHOLDER: Overview", ha="center", va="center",
            fontsize=6.4, weight="bold", color=COLORS["placeholder"])
    placeholder_box(ax, (0.05, 0.60), 0.26, 0.18, "Masked\n14-day history")
    placeholder_box(ax, (0.37, 0.60), 0.26, 0.18, "Request-local\nscorer")
    placeholder_box(ax, (0.69, 0.60), 0.26, 0.18, "24×4 forecast\n+ audit record")
    arrow(ax, (0.31, 0.69), (0.37, 0.69))
    arrow(ax, (0.63, 0.69), (0.69, 0.69))
    placeholder_box(ax, (0.33, 0.27), 0.34, 0.18, "Evidence loss\n(clean, 20–30%, 50%)", dashed=True)
    arrow(ax, (0.50, 0.45), (0.50, 0.60), dashed=True)
    ax.text(0.5, 0.13, "Replace with the final drawing (same file name).\n"
            "Prompt: paper/figures/PLACEHOLDER_PROMPTS.md, section 1.",
            ha="center", va="center", fontsize=4.9, color="#595959", linespacing=1.3)
    fig.savefig(FIGURE_DIR / "unused_overview_placeholder.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 1.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, "PLACEHOLDER: Framework", ha="center", va="center",
            fontsize=6.8, weight="bold", color=COLORS["placeholder"])
    ax.add_patch(FancyBboxPatch((0.005, 0.10), 0.64, 0.74, boxstyle="round,pad=0,rounding_size=0.015",
                                fill=False, linestyle="--", linewidth=0.8, edgecolor="#7F7F7F"))
    ax.text(0.325, 0.80, "Serving path (per request)", ha="center", va="center", fontsize=6.0, color="#404040")
    placeholder_box(ax, (0.03, 0.30), 0.13, 0.40, "Ingress\n(authorization,\nwindow build)\nout of scope")
    placeholder_box(ax, (0.20, 0.30), 0.15, 0.40, "Request-local\nscorer\n(5-member\nensemble)")
    placeholder_box(ax, (0.41, 0.30), 0.12, 0.40, "Audit writer\n(JSON/UTF-8)")
    placeholder_box(ax, (0.57, 0.30), 0.06, 0.40, "Caller", fontsize=5.4)
    arrow(ax, (0.16, 0.50), (0.20, 0.50))
    arrow(ax, (0.35, 0.50), (0.41, 0.50))
    arrow(ax, (0.53, 0.50), (0.57, 0.50))

    ax.add_patch(FancyBboxPatch((0.675, 0.10), 0.32, 0.74, boxstyle="round,pad=0,rounding_size=0.015",
                                fill=False, linestyle="--", linewidth=0.8, edgecolor="#7F7F7F"))
    ax.text(0.835, 0.80, "Control plane (offline)", ha="center", va="center", fontsize=6.0, color="#404040")
    placeholder_box(ax, (0.70, 0.56), 0.27, 0.13, "Fit / inner / evaluation splits")
    placeholder_box(ax, (0.70, 0.38), 0.27, 0.13, "Training and seed-wise selection")
    placeholder_box(ax, (0.70, 0.20), 0.27, 0.13, "Frozen checkpoint, prior")
    arrow(ax, (0.835, 0.56), (0.835, 0.51))
    arrow(ax, (0.835, 0.38), (0.835, 0.33))
    ax.text(0.5, 0.02, "Replace with the final drawing (same file name). "
            "Prompt: paper/figures/PLACEHOLDER_PROMPTS.md, section 2.",
            ha="center", va="center", fontsize=5.2, color="#595959")
    arrow(ax, (0.70, 0.265), (0.275, 0.302), dashed=True)
    ax.text(0.47, 0.225, "frozen assets only", ha="center", va="center", fontsize=4.9,
            color="#595959")
    fig.savefig(FIGURE_DIR / "unused_framework_placeholder.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    render_framework_preview()
    comparison_out = FIGURE_DIR / "fig2_comparison.pdf"
    robustness_out = FIGURE_DIR / "fig3_robustness.pdf"
    planning_out = FIGURE_DIR / "fig4_planning.pdf"
    audit_out = FIGURE_DIR / "fig5_audit.pdf"
    missingness = load_missingness()
    planning = load_planning()
    deletion = load_deletion()
    experts = load_experts()
    render_comparison(comparison_out, missingness, planning)
    render_robustness(robustness_out, missingness)
    render_planning(planning_out, planning)
    render_audit(audit_out, deletion, experts)
    inputs = [MISSINGNESS_CSV, PLANNING_CSV, DELETION_CSV, EXPERT_CSV, Path(__file__).resolve()]
    outputs = [comparison_out, robustness_out, planning_out, audit_out]
    schematics = [FIGURE_DIR / "fig1_framework.pdf"]
    manifest = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in inputs
        ],
        "outputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "figure": number}
            for number, path in enumerate(outputs, start=2)
        ],
        "schematics": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "figure": 1,
                "note": "Externally authored framework; preserved by this script.",
            }
            for path in schematics
        ],
        "unused_figures": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in sorted(FIGURE_DIR.glob("unused_*.pdf"))
        ],
        "palette": {
            "reference": "paper/figures/fig1_framework.pdf",
            "model_colors": {key: COLORS[key] for key in
                             ("wlcr", "dlinear", "patchtst", "grud", "lightgbm")},
            "note": "Blue, sage, teal and slate coordinated with Figure 1.",
        },
        "previews": [
            "docs/images/paper_figure_framework.png",
            "docs/images/paper_figure_comparison.png",
            "docs/images/paper_figure_robustness.png",
            "docs/images/paper_figure_planning.png",
            "docs/images/paper_figure_audit.png",
        ],
        "model_training_or_evaluation": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote the data figures, docs/images previews and manifest; preserved the schematics")


if __name__ == "__main__":
    main()
