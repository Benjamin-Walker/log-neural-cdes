"""
This script is used to produce a table of the hyperparameters selected from the grid search optimisation for the UEA
and PPG experiments.
"""

import json
import os


with open("results/hyperparam_info_for_table.json", "r") as file:
    data = json.load(file)

short_names = data["short_names"]
hypparams_srm = data["hypparams_srm"]
hypparams_ncde = data["hypparams_ncde"]
dir_spot = data["dir_spot"]

for model_list, hypparams in zip(
    [["lru", "S5", "S6", "mamba"], ["ncde", "nrde", "log_ncde"]],
    [hypparams_srm, hypparams_ncde],
):
    for key in hypparams.keys():
        count_key = 0
        values = hypparams[key]
        for val in values:
            if isinstance(val, list):
                name = f"({val[0]}, {int(float(val[1]))})"
            else:
                name = val
            if count_key == 0:
                print_ = f"{key} & {name}"
            else:
                print_ = f" & {name}"
            count_key += 1
            for model in model_list:
                if key in dir_spot[model].keys():
                    print_ += " &"
                    count = 0
                    for dataset in [
                        "EigenWorms",
                        "EthanolConcentration",
                        "Heartbeat",
                        "MotorImagery",
                        "SelfRegulationSCP1",
                        "SelfRegulationSCP2",
                        "ppg",
                    ]:
                        for parent_dir in [
                            "UEA_outputs_repeats",
                            "PPG_outputs_repeats",
                        ]:
                            if os.path.exists(
                                f"results/paper_outputs/{parent_dir}/{model}/{dataset}/"
                            ):
                                dir = os.listdir(
                                    f"results/paper_outputs/{parent_dir}/{model}/{dataset}/"
                                )[0].split("_")
                                idxs = dir_spot[model][key]
                                if isinstance(idxs, list):
                                    if (
                                        dir[idxs[0]] == val[0]
                                        and dir[idxs[1]] == val[1]
                                    ):
                                        if count == 0:
                                            print_ += f" {short_names[dataset]}"
                                        else:
                                            print_ += f", {short_names[dataset]}"
                                        count += 1
                                else:
                                    if dir[idxs] == val:
                                        if count == 0:
                                            print_ += f" {short_names[dataset]}"
                                        else:
                                            print_ += f", {short_names[dataset]}"
                                        count += 1
                else:
                    print_ += " & \ding{55}"
            print_ += " \\\\ \\cline{2-6}"
            if count_key == len(values):
                print_ += " \Xhline{2\\arrayrulewidth}"
            print(print_)
    print()
