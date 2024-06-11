"""
This script plots the results from the four classifications considered on the toy dataset.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


font = {"size": 17}
matplotlib.rc("font", **font)
plt.rcParams["text.usetex"] = True

plot_names = {
    "lru": "LRU",
    "S5": "S5",
    "S6": "S6",
    "mamba": "MAMBA",
    "ncde": "NCDE",
    "nrde": "NRDE",
    "log_ncde": "Log-NCDE",
}
colors = {"lru": 0, "S5": 1, "S6": 2, "mamba": 3, "ncde": 4, "nrde": 5, "log_ncde": 6}
CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]
markers = ["o", "s", "D", "^", "p", "v", "P", "*", "X"]
titles = {
    "signature1": "$\int_0^1\mathrm{d}X^3_s>0?$",
    "signature2": "$\int_0^1\int_0^u\mathrm{d}X^3_s\mathrm{d}X^6_u>0?$",
    "signature3": "$\int_0^1\int_0^v\int_0^u\mathrm{d}X^3_s\mathrm{d}X^6_u\mathrm{d}X^1_v>0?$",
    "signature4": "$\int_0^1\int_0^w\int_0^v\int_0^u\mathrm{d}X^3_s\mathrm{d}X^6_u\mathrm{d}X^1_v\mathrm{d}X^4_w>0?$",
}

markerpoints = {
    "signature1": {
        "lru": [21, 80],
        "S5": [18, 70],
        "S6": [40, -1],
        "mamba": [40, -1],
        "ncde": [20, -1],
        "nrde": [30, -1],
        "log_ncde": [40, -1],
    },
    "signature2": {
        "lru": [19, 90],
        "S5": [13, -1],
        "S6": [40, -1],
        "mamba": [15, -1],
        "ncde": [-1],
        "nrde": [-1],
        "log_ncde": [-1],
    },
    "signature3": {
        "lru": [30, -1],
        "S5": [20, 90],
        "S6": [40, -1],
        "mamba": [40, -1],
        "ncde": [-1],
        "nrde": [-1],
        "log_ncde": [-1],
    },
    "signature4": {
        "lru": [20, 90],
        "S5": [10, 80],
        "S6": [30, -1],
        "mamba": [40, -1],
        "ncde": [-1],
        "nrde": [-1],
        "log_ncde": [-1],
    },
}


fig, axes = plt.subplots(2, 2, figsize=(14, 12))
datasets = ["signature1", "signature2", "signature3", "signature4"]
axes = axes.flatten()

for i, dataset in enumerate(datasets):
    ax = axes[i]
    for model in plot_names.keys():
        exp = os.listdir(f"results/paper_outputs/toy_outputs/{model}/toy/{dataset}")[0]
        with open(
            f"results/paper_outputs/toy_outputs/{model}/toy/{dataset}/{exp}/all_val_acc.npy",
            "rb",
        ) as f:
            val_acc = np.load(f)
            val_acc[0] = 0.5
        with open(
            f"results/paper_outputs/toy_outputs/{model}/toy/{dataset}/{exp}/steps.npy",
            "rb",
        ) as f:
            steps = np.load(f)
        if dataset == "signature1":
            if model in ["ncde", "nrde", "lru", "S5"]:
                markevery = [20, -1]
            elif model in ["S6", "mamba", "log_ncde"]:
                markevery = [40, -1]
        else:
            if model in ["ncde", "nrde", "log_ncde"]:
                markevery = [-1]
            else:
                markevery = [20, -1]
        ax.semilogy(
            steps,
            1 - val_acc,
            linewidth=2.5,
            label=plot_names[model],
            color=CB_color_cycle[colors[model]],
            marker=markers[colors[model]],
            markersize=14,
            markevery=[],
            markeredgecolor="black",
        )
        for idx in markerpoints[dataset][model]:
            ax.scatter(
                steps[idx],
                1 - val_acc[idx],
                color=CB_color_cycle[colors[model]],
                marker=markers[colors[model]],
                s=200,
                edgecolors="black",
                zorder=10,
            )
    if dataset == "signature1":
        start = -500
        end = 10500
    else:
        start = -5000
        end = 105000
    ax.set_xlim([start, end])
    ax.hlines(0.5, start, end, linestyle="--", color="black", linewidth=1.5, zorder=0)
    ax.hlines(0.1, start, end, linestyle="--", color="black", linewidth=1.5, zorder=0)
    ax.hlines(0.025, start, end, linestyle="--", color="black", linewidth=1.5, zorder=0)
    if i in [0, 2]:
        ax.set_yticks([0.025, 0.1, 0.5], ["V=97.5\%", "V=90\%", "V=50\%"])
        ax.set_ylabel("$\log(1-V)$, $V$ is Validation Accuracy")
    else:
        ax.set_yticks([0.025, 0.1, 0.5], ["", "", ""])
    ax.set_xlabel("Steps")
    ax.set_title(titles[dataset], pad=10)
    if dataset == "signature1":
        ax.legend(ncols=2, loc="upper right")
    ax.set_ylim([0.019, 0.6])

plt.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs("results/images", exist_ok=True)
plt.savefig("results/images/combined_plot.png", dpi=300, bbox_inches="tight")
plt.show()
