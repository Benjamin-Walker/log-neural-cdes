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

viridis = colormaps["viridis"]
newcolors = viridis(np.linspace(0, 0.95, 256))
newcmp = colors.LinearSegmentedColormap.from_list("Viridis", newcolors)

fig_size = (7, 6)

plt.figure(figsize=fig_size)
plt.matshow(memory_matrix, norm=colors.LogNorm(), fignum=1, cmap=newcmp)
for i in range(len(memory)):
    for j in range(len(memory["S5"])):
        plt.text(
            j,
            i,
            round(memory[list(memory.keys())[i]][list(memory["S5"].keys())[j]]),
            ha="center",
            va="center",
            color="white",
        )
plt.colorbar()
plt.xticks(range(len(memory["S5"])), [short_name[x] for x in list(time["S5"].keys())])
plt.gca().xaxis.tick_bottom()
plt.yticks(range(len(memory)), [model_name[x] for x in list(memory.keys())])
plt.xlabel("Dataset")
plt.ylabel("Model")
plt.title("Memory Usage (MB)")
os.makedirs("results/images", exist_ok=True)
plt.savefig("results/images/memory.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=fig_size)
plt.matshow(time_matrix, norm=colors.LogNorm(), fignum=2, cmap=newcmp)
for i in range(len(time)):
    for j in range(len(time["S5"])):
        plt.text(
            j,
            i,
            round(time[list(time.keys())[i]][list(time["S5"].keys())[j]]),
            ha="center",
            va="center",
            color="white",
        )
plt.colorbar()
plt.xticks(range(len(time["S5"])), [short_name[x] for x in list(time["S5"].keys())])
plt.gca().xaxis.tick_bottom()
plt.yticks(range(len(memory)), [model_name[x] for x in list(memory.keys())])
plt.xlabel("Dataset")
plt.ylabel("Model")
plt.title("Time for 1000 Steps (s)")
plt.savefig("results/images/time.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=fig_size)
plt.matshow(num_steps_matrix, norm=colors.LogNorm(), fignum=3, cmap=newcmp)
for i in range(len(num_steps)):
    for j in range(len(num_steps["S5"])):
        plt.text(
            j,
            i,
            round(
                num_steps[list(num_steps.keys())[i]][list(num_steps["S5"].keys())[j]]
            ),
            ha="center",
            va="center",
            color="white",
        )
plt.colorbar()
plt.xticks(
    range(len(num_steps["S5"])), [short_name[x] for x in list(num_steps["S5"].keys())]
)
plt.gca().xaxis.tick_bottom()
plt.yticks(range(len(memory)), [model_name[x] for x in list(memory.keys())])
plt.xlabel("Dataset")
plt.ylabel("Model")
plt.title("Number of Steps")
plt.savefig("results/images/num_steps.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=fig_size)
plt.matshow(
    np.array(time_matrix) * np.array(num_steps_matrix) / 1000,
    norm=colors.LogNorm(),
    fignum=4,
    cmap=newcmp,
)
for i in range(len(time)):
    for j in range(len(time["S5"])):
        time_ij = time[list(time.keys())[i]][list(time["S5"].keys())[j]]
        num_steps_ij = num_steps[list(num_steps.keys())[i]][
            list(num_steps["S5"].keys())[j]
        ]
        plt.text(
            j,
            i,
            f"{round(time_ij * num_steps_ij / 1000)}",
            ha="center",
            va="center",
            color="white",
        )
plt.colorbar()
plt.xticks(range(len(time["S5"])), [short_name[x] for x in list(time["S5"].keys())])
plt.gca().xaxis.tick_bottom()
plt.yticks(range(len(memory)), [model_name[x] for x in list(memory.keys())])
plt.xlabel("Dataset")
plt.ylabel("Model")
plt.title("Total Time Usage (s)")
plt.savefig("results/images/total_time.png", dpi=300, bbox_inches="tight")

plt.show()
