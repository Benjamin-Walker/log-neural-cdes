import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


font = {"size": 14}
matplotlib.rc("font", **font)

plot_names = {
    "lru": "LRU",
    "S5": "S5",
    "S6": "S6",
    "ncde": "NCDE",
    "nrde": "NRDE",
    "log_ncde": "Log-NCDE",
}
colors = {"lru": 0, "S5": 1, "S6": 2, "ncde": 3, "nrde": 4, "log_ncde": 5}
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

for dataset in ["signature1", "signature2", "signature3", "signature4"]:
    plt.figure(figsize=(7, 6))
    for model in ["lru", "S5", "S6", "ncde", "nrde", "log_ncde"]:
        exp = os.listdir(f"results/toy_outputs/{model}/toy/{dataset}")[0]
        with open(
            f"results/toy_outputs/{model}/toy/{dataset}/{exp}/all_val_acc.npy", "rb"
        ) as f:
            val_acc = np.load(f)
            val_acc[0] = 0.5
        with open(
            f"results/toy_outputs/{model}/toy/{dataset}/{exp}/steps.npy", "rb"
        ) as f:
            steps = np.load(f)
        plt.plot(
            steps,
            val_acc,
            linewidth=3,
            label=plot_names[model],
            color=CB_color_cycle[colors[model]],
        )
    plt.xlabel("Steps")
    plt.ylabel("Validation Accuracy")
    plt.title(f"{dataset}")
    plt.legend()
    plt.ylim([0.45, 1])
    plt.savefig(f"results/{dataset}.png", dpi=300, bbox_inches="tight")
    plt.close()
