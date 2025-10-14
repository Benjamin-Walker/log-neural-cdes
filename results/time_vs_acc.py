import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


matplotlib.rcParams.update(
    {
        "font.size": 14,
    }
)  # Update default font size

# ─────────────────────────── 1. DATA ──────────────────────────────
# Average test accuracy (%)
accuracy = {
    "LRU": 61.7,
    "S5": 61.8,
    "S6": 62.0,
    "MAMBA": 58.6,
    "NCDE": 60.2,
    "NRDE": 60.6,
    "Log‑NCDE": 64.3,
    "D‑SLiCE": 61.6,
    "BD-SLiCE": 64.5,
    "DE‑LNCDE": 62.2,
    "DPLR-SLiCE": 60.4,
    "D-DE-SLiCE": 61.9,
    "S-SLiCE": 61.5,
    "WH-SLiCE": 62.1,
}

# Time per 1 000 training steps (s)
time_1k = {
    "LRU": 30.971666666666664,
    "S5": 15.42,
    "S6": 19.85,
    "MAMBA": 80.82166666666669,
    "NCDE": 5665.208333333333,
    "NRDE": 4535.818333333334,
    "Log‑NCDE": 1131.1683333333335,
    "D‑LNCDE": 6.96,
    "BD-LNCDE": 55.29,
    "DE‑LNCDE": 114.23,
}

# GPU memory (MB)
gpu_mem = {
    "LRU": 4121.666666666667,
    "S5": 2815.0,
    "S6": 2608.0,
    "MAMBA": 4450.333333333333,
    "NCDE": 1759.6666666666667,
    "NRDE": 2676.3333333333335,
    "Log‑NCDE": 1999.6666666666667,
    "D‑LNCDE": 2159.0,  # diagonal_linear_ncde
    "BD-LNCDE": 2938.0,  # bd_linear_ncde
    "DE‑LNCDE": 8820.333333333334,  # dense_linear_ncde
}

text_pos = {
    "LRU": (-12.5, -30),
    "S5": (-10, -25),
    "S6": (-10, 15),
    "MAMBA": (-25, 18),
    "NCDE": (-20, -23),
    "NRDE": (-20, 15),
    "Log‑NCDE": (-35, -25),
    "D‑LNCDE": (-25, -25),
    "BD-LNCDE": (-35, 15),
    "DE‑LNCDE": (-32, -37),
}

# ─────────────────────── 2. GROUP / COLOUR MAPPING ─────────────────
groups = {
    "Non‑linear CDE": ["NCDE", "NRDE", "Log‑NCDE"],
    "Linear CDE": ["D‑LNCDE", "BD-LNCDE", "DE‑LNCDE"],
    "SSM": ["S5", "S6", "MAMBA"],
    "RNN": ["LRU"],
}

colours = {
    "Non‑linear CDE": "#0072B2",  # Blue
    "Linear CDE": "#E69F00",  # Orange
    "SSM": "#009E73",  # Green
    "RNN": "#D55E00",  # Vermilion (orange-red, not pure red)
}

# ─────────────────────────── 3. PLOT ───────────────────────────────
plt.figure(figsize=(10, 7))

handles = []
for family, models in groups.items():
    x = [time_1k[m] for m in models]
    y = [accuracy[m] for m in models]
    sizes = [gpu_mem[m] / 4 for m in models]  # scale so bubbles look good
    h = plt.scatter(
        x,
        y,
        s=sizes,
        c=colours[family],
        label=family,
        alpha=0.75,
        edgecolors="w",
        linewidths=0.5,
    )
    handles.append(h)
    # annotate each point
    for m in models:
        pos = text_pos[m]
        plt.annotate(
            m,
            (time_1k[m], accuracy[m]),
            textcoords="offset points",
            xytext=pos,
            ha="left",
            fontsize=14,
        )

plt.xscale("log")
plt.xlabel("Time per 1000 training steps (s)")
plt.ylabel("Average test accuracy (%)")
plt.title("Accuracy, Speed, and Memory Footprint on the UEA-MTSCA")
# plt.grid(alpha=0.3, which='both', linestyle='--')
plt.ylim(58, 66)
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=family,
        markerfacecolor=colours[family],
        markersize=14,
        alpha=0.75,
        markeredgecolor="w",
        markeredgewidth=0.5,
    )
    for family in groups
]

plt.legend(handles=legend_elements, title="Model family")
plt.tight_layout()
plt.savefig("results/images/time_vs_acc.png", dpi=300)
plt.show()
