"""Shared figure style so every chart in the project reads as one system.

Colors are role-based: SERIES for categorical identity (fixed order, validated
for color-vision deficiency), a blue<->red diverging pair with a gray midpoint
for signed error, and recessive ink/grid colors for chrome.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # first 3 slots: safe even all-pairs
HIGHLIGHT, CONTEXT = SERIES[0], "#b5b3ad"     # "the method we care about" vs the rest

# Diverging: red (overstates Republican share) <-> gray <-> blue (overstates Democratic)
DIVERGING = LinearSegmentedColormap.from_list(
    "wawv_div", ["#b8282a", "#e34948", "#f0efec", "#3987e5", "#184f95"])

METHOD_LABELS = {
    "raw": "Unweighted",
    "ces_wt": "CES weight",
    "raked_lv": "Raking + likely voter",
    "xgb_lv": "XGBoost poststrat",
    "mrp_lv": "MRP + likely voter",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "figure.dpi": 150, "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9.5, "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.labelcolor": INK_2, "axes.labelsize": 9,
        "text.color": INK, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 8.5,
    })


def subtitle(ax, text: str) -> None:
    """Secondary line under the title (matplotlib has no native subtitle)."""
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=8.5,
            color=INK_2, va="bottom")


def source_note(fig, text: str, y: float = 0.01) -> None:
    fig.text(0.01, y, text, fontsize=7, color=MUTED, ha="left", va="bottom")


def save(fig, path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")
