"""Shared matplotlib style."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
MODEL_COLORS = {"Exact-match rules": MUTED, "Logistic Regression": BLUE, "Random Forest": ORANGE,
                "XGBoost": AQUA, "LightGBM": YELLOW}
GROUP_COLORS = {"Identifier": BLUE, "Amount": ORANGE, "Dates": AQUA, "Rate": YELLOW,
                "Counterparty": MAGENTA, "Direction/product": GREEN}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
    "font.size": 10, "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 130,
    "axes.axisbelow": True,
})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
