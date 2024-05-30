import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, colors


font = {"size": 13}

matplotlib.rc("font", **font)

memory = {}
time = {}
num_steps = {}


memory["ncde"] = {
    "EigenWorms": 2488,
    "EthanolConcentration": 692,
    "Heartbeat": 948,
    "MotorImagery": 4534,
    "SelfRegulationSCP1": 692,
    "SelfRegulationSCP2": 692,
}
time["ncde"] = {
    "EigenWorms": 23346.67,
    "EthanolConcentration": 1751.48,
    "Heartbeat": 408.09,
    "MotorImagery": 7393.39,
    "SelfRegulationSCP1": 846.34,
    "SelfRegulationSCP2": 1233.28,
}
num_steps["ncde"] = {
    "EigenWorms": 1060,
    "EthanolConcentration": 1100,
    "Heartbeat": 1500,
    "MotorImagery": 2000,
    "SelfRegulationSCP1": 2400,
    "SelfRegulationSCP2": 1800,
}

memory["nrde"] = {
    "EigenWorms": 2486,
    "EthanolConcentration": 692,
    "Heartbeat": 10934,
    "MotorImagery": 4532,
    "SelfRegulationSCP1": 694,
    "SelfRegulationSCP2": 692,
}
time["nrde"] = {
    "EigenWorms": 1160.47,
    "EthanolConcentration": 1884.41,
    "Heartbeat": 20384.05,
    "MotorImagery": 7402.05,
    "SelfRegulationSCP1": 376.32,
    "SelfRegulationSCP2": 1147.78,
}
num_steps["nrde"] = {
    "EigenWorms": 1900,
    "EthanolConcentration": 1880,
    "Heartbeat": 1420,
    "MotorImagery": 1660,
    "SelfRegulationSCP1": 1760,
    "SelfRegulationSCP2": 2080,
}

memory["log_ncde"] = {
    "EigenWorms": 2490,
    "EthanolConcentration": 698,
    "Heartbeat": 2746,
    "MotorImagery": 8636,
    "SelfRegulationSCP1": 700,
    "SelfRegulationSCP2": 698,
}
time["log_ncde"] = {
    "EigenWorms": 1431.00,
    "EthanolConcentration": 1080.32,
    "Heartbeat": 600.02,
    "MotorImagery": 2761.72,
    "SelfRegulationSCP1": 535.55,
    "SelfRegulationSCP2": 597.74,
}
num_steps["log_ncde"] = {
    "EigenWorms": 1740,
    "EthanolConcentration": 2080,
    "Heartbeat": 1480,
    "MotorImagery": 1920,
    "SelfRegulationSCP1": 2280,
    "SelfRegulationSCP2": 1760,
}

memory["lru"] = {
    "EigenWorms": 10690,
    "EthanolConcentration": 1456,
    "Heartbeat": 1988,
    "MotorImagery": 12742,
    "SelfRegulationSCP1": 3014,
    "SelfRegulationSCP2": 3016,
}
time["lru"] = {
    "EigenWorms": 70.21,
    "EthanolConcentration": 6.58,
    "Heartbeat": 11.72,
    "MotorImagery": 51.41,
    "SelfRegulationSCP1": 17.38,
    "SelfRegulationSCP2": 13.26,
}
num_steps["lru"] = {
    "EigenWorms": 12800,
    "EthanolConcentration": 11000,
    "Heartbeat": 11000,
    "MotorImagery": 12800,
    "SelfRegulationSCP1": 16600,
    "SelfRegulationSCP2": 17400,
}

memory["S5"] = {
    "EigenWorms": 18686,
    "EthanolConcentration": 1028,
    "Heartbeat": 1010,
    "MotorImagery": 8700,
    "SelfRegulationSCP1": 1020,
    "SelfRegulationSCP2": 1522,
}
time["S5"] = {
    "EigenWorms": 111.72,
    "EthanolConcentration": 9.12,
    "Heartbeat": 4.74,
    "MotorImagery": 32.00,
    "SelfRegulationSCP1": 6.96,
    "SelfRegulationSCP2": 5.20,
}
num_steps["S5"] = {
    "EigenWorms": 14600,
    "EthanolConcentration": 11000,
    "Heartbeat": 21400,
    "MotorImagery": 16400,
    "SelfRegulationSCP1": 28600,
    "SelfRegulationSCP2": 17600,
}

memory["mamba"] = {
    "EigenWorms": 13486,
    "EthanolConcentration": 4876,
    "Heartbeat": 1650,
    "MotorImagery": 3120,
    "SelfRegulationSCP1": 1110,
    "SelfRegulationSCP2": 2460,
}
time["mamba"] = {
    "EigenWorms": 122.16,
    "EthanolConcentration": 255.18,
    "Heartbeat": 33.97,
    "MotorImagery": 34.92,
    "SelfRegulationSCP1": 6.66,
    "SelfRegulationSCP2": 32.04,
}
num_steps["mamba"] = {
    "EigenWorms": 23000,
    "EthanolConcentration": 16200,
    "Heartbeat": 17800,
    "MotorImagery": 20800,
    "SelfRegulationSCP1": 18600,
    "SelfRegulationSCP2": 39200,
}

memory["S6"] = {
    "EigenWorms": 7922,
    "EthanolConcentration": 938,
    "Heartbeat": 606,
    "MotorImagery": 4056,
    "SelfRegulationSCP1": 904,
    "SelfRegulationSCP2": 1222,
}
time["S6"] = {
    "EigenWorms": 67.72,
    "EthanolConcentration": 4.15,
    "Heartbeat": 4.04,
    "MotorImagery": 33.72,
    "SelfRegulationSCP1": 2.62,
    "SelfRegulationSCP2": 6.84,
}

num_steps["S6"] = {
    "EigenWorms": 36400,
    "EthanolConcentration": 24600,
    "Heartbeat": 19200,
    "MotorImagery": 17200,
    "SelfRegulationSCP1": 29000,
    "SelfRegulationSCP2": 24600,
}

short_name = {
    "EigenWorms": "EW",
    "EthanolConcentration": "EC",
    "Heartbeat": "HB",
    "MotorImagery": "MI",
    "SelfRegulationSCP1": "SCP1",
    "SelfRegulationSCP2": "SCP2",
}

model_name = {
    "mamba": "MAMBA",
    "ncde": "NCDE",
    "nrde": "NRDE",
    "log_ncde": "Log-NCDE",
    "lru": "LRU",
    "S5": "S5",
    "S6": "S6",
}

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

print(av_time)
print(av_memory)
print(av_num_steps)
print(av_total_time)

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
plt.savefig("memory.png", dpi=300, bbox_inches="tight")

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
plt.savefig("time.png", dpi=300, bbox_inches="tight")

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
plt.savefig("num_steps.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=fig_size)
plt.matshow(
    np.array(time_matrix) * np.array(num_steps_matrix) / 1000,
    norm=colors.LogNorm(),
    fignum=4,
    cmap=newcmp,
)
for i in range(len(time)):
    for j in range(len(time["S5"])):
        time = time[list(time.keys())[i]][list(time["S5"].keys())[j]]
        num_steps = num_steps[list(num_steps.keys())[i]][
            list(num_steps["S5"].keys())[j]
        ]
        plt.text(
            j,
            i,
            f"{round(time * num_steps / 1000)}",
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
plt.savefig("total_time.png", dpi=300, bbox_inches="tight")
