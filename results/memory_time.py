"""
This script reads the results from the memory_time_results.json file and plots the results in a matrix format.
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, colors


font = {"size": 13}

matplotlib.rc("font", **font)

with open("results/memory_time_results.json", "r") as file:
    data = json.load(file)

memory = data["memory"]
time = data["time"]
num_steps = data["num_steps"]
short_name = data["short_name"]
model_name = data["model_name"]

model_order = ["lru", "S5", "S6", "mamba", "ncde", "nrde", "log_ncde"]
memory = {model: memory[model] for model in model_order}
time = {model: time[model] for model in model_order}
num_steps = {model: num_steps[model] for model in model_order}

memory_matrix = [
    [memory[model][dataset] for dataset in memory[model]] for model in memory
]
time_matrix = [[time[model][dataset] for dataset in time[model]] for model in time]
num_steps_matrix = [
    [num_steps[model][dataset] for dataset in num_steps[model]] for model in num_steps
]

av_time = np.mean(time_matrix, axis=1)
av_memory = np.mean(memory_matrix, axis=1)
av_num_steps = np.mean(num_steps_matrix, axis=1)
av_total_time = np.mean(
    np.array(time_matrix) * np.array(num_steps_matrix) / 1000, axis=1
)

for i, model in enumerate(model_order):
    print(
        f"{model_name[model]} & {av_time[i]:.2f} & {av_memory[i]:.2f} & {av_num_steps[i]:.2f} "
        f"& {av_total_time[i]:.2f} \\\\"
    )


def plot_matrix(data, matrix, fignum, cmp, title, filename):
    fig_size = (7, 6)
    plt.figure(figsize=fig_size)
    plt.matshow(matrix, norm=colors.LogNorm(), fignum=fignum, cmap=cmp)
    for i in range(len(data)):
        for j in range(len(data["S5"])):
            plt.text(
                j,
                i,
                str(round(data[list(data.keys())[i]][list(data["S5"].keys())[j]])),
                ha="center",
                va="center",
                color="white",
            )
    plt.colorbar()
    plt.xticks(range(len(data["S5"])), [short_name[x] for x in list(data["S5"].keys())])
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(len(data)), [model_name[x] for x in list(data.keys())])
    plt.xlabel("Dataset")
    plt.ylabel("Model")
    plt.title(title)
    os.makedirs("results/images", exist_ok=True)
    plt.savefig(f"results/images/{filename}.png", dpi=300, bbox_inches="tight")


viridis = colormaps["viridis"]
newcolors = viridis(np.linspace(0, 0.95, 256))
newcmp = colors.LinearSegmentedColormap.from_list("Viridis", newcolors)
plot_matrix(memory, memory_matrix, 1, newcmp, "Memory Usage (MB)", "memory")
plot_matrix(time, time_matrix, 2, newcmp, "Time for 1000 Steps (s)", "time")
plot_matrix(num_steps, num_steps_matrix, 3, newcmp, "Number of Steps", "num_steps")
plot_matrix(
    time,
    np.array(time_matrix) * np.array(num_steps_matrix) / 1000,
    4,
    newcmp,
    "Total Time Usage (s)",
    "total_time",
)
plt.show()
