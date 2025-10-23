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
    "D‑SLiCE": 61.7,
    "BD-SLiCE": 64.0,
    "DE‑LNCDE": 61.6,
}

# Time per 1 000 training steps (s)
time_1k = {
    "LRU": 26.91,
    "S5": 21.92,
    "S6": 20.09,
    "MAMBA": 59.97,
    "NCDE": 6923.06,
    "NRDE": 3431.09,
    "Log‑NCDE": 1321.69,
    "D‑SLiCE": 11.0,
    "BD-SLiCE": 68.1,
    "DE‑LNCDE": 77.2,
}

# GPU memory (MB)
gpu_mem = {
    "LRU": 4308.0,
    "S5": 3327.33,
    "S6": 2938.33,
    "MAMBA": 4434.67,
    "NCDE": 1961.67,
    "NRDE": 2858.33,
    "Log‑NCDE": 2177.33,
    "D‑SLiCE": 1875.33,
    "BD-SLiCE": 2344.00,
    "DE‑LNCDE": 12756,
}

text_pos = {
    "LRU": (-12.5, -30),
    "S5": (-13, -26),
    "S6": (-10, 15),
    "MAMBA": (-25, 18),
    "NCDE": (-20, -23),
    "NRDE": (-20, 15),
    "Log‑NCDE": (-35, -25),
    "D‑SLiCE": (-25, -25),
    "BD-SLiCE": (-35, 15),
    "DE‑LNCDE": (-35, 32),
}

# ─────────────────────── 2. GROUP / COLOUR MAPPING ─────────────────
groups = {
    "Non‑linear CDE": ["NCDE", "NRDE", "Log‑NCDE"],
    "Linear CDE": [
        "D‑SLiCE",
        "BD-SLiCE",
        "DE‑LNCDE",
    ],
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
plt.ylim(58, 65)
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

plt.legend(handles=legend_elements, title="Model family", loc=[0.5, 0.2])
plt.tight_layout()
plt.savefig("results/images/time_vs_acc.pdf", dpi=300)
plt.show()
