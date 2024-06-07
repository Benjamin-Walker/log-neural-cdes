import json
import os


with open("results/hyperparam_info_for_table.json", "r") as file:
    data = json.load(file)

dir_spot = data["dir_spot"]

hypparams_short_names = {
    "Learning Rate": "lr",
    "Include Time": "time",
    "Hidden Dimension": "hidden_dim",
    "Number of Layers": "num_blocks",
    "State Dimension": "ssm_dim",
    "S5 Initialisation Blocks": "ssm_blocks",
    "Convolution Dimension": "convdim",
    "Expansion Factor": "expansion",
    "Vector Field (Depth, Width)": ["vf_depth", "vf_width"],
    "Log-ODE (Depth, Step)": ["depth", "stepsize"],
    "Regularisation λ": "lambd",
}


model_list = ["lru", "S5", "S6", "mamba", "ncde", "nrde", "log_ncde"]
dataset_list = [
    "EigenWorms",
    "EthanolConcentration",
    "Heartbeat",
    "MotorImagery",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
]

for model in model_list:
    dir_spot_model = dir_spot[model]
    for dataset in dataset_list:
        dir = os.listdir(f"results/UEA_PPG_toy/UEA_outputs_repeats/{model}/{dataset}/")[
            0
        ].split("_")
        json_dict = {}
        json_dict["seeds"] = [2345, 3456, 4567, 5678, 6789]
        json_dict["data_dir"] = "data_dir"
        json_dict["output_parent_dir"] = ""
        json_dict["lr_scheduler"] = "lambda lr: lr"
        json_dict["num_steps"] = 100000
        if model in ["lru", "S5", "S6", "mamba"]:
            json_dict["print_steps"] = 1000
        else:
            json_dict["print_steps"] = 100
        if model in ["lru", "S5", "S6", "mamba"] and dataset == "ppg":
            json_dict["batch_size"] = 4
        else:
            json_dict["batch_size"] = 32
        json_dict["model_name"] = model
        if dataset == "ppg":
            json_dict["metric"] = "mse"
            json_dict["classification"] = False
        else:
            json_dict["metric"] = "accuracy"
            json_dict["classification"] = True
        json_dict["dataset_name"] = dataset
        json_dict["use_presplit"] = False
        json_dict["T"] = 1
        if model in ["lru", "S5", "S6", "mamba"]:
            json_dict["dt0"] = None
        else:
            if model == "ncde":
                json_dict["dt0"] = dir[27]
            else:
                json_dict["dt0"] = dir[31]
        if model == "log_ncde":
            json_dict["scale"] = 1000
        else:
            json_dict["scale"] = 1

        for key in dir_spot_model.keys():
            label = hypparams_short_names[key]
            if isinstance(label, list):
                json_dict[label[0]] = dir[dir_spot_model[key][0]]
                json_dict[label[1]] = dir[dir_spot_model[key][1]]
            else:
                json_dict[label] = dir[dir_spot_model[key]]

        os.makedirs(f"experiments/repeats/{model}", exist_ok=True)
        with open(f"experiments/repeats/{model}/{dataset}.json", "w") as file:
            json.dump(json_dict, file, indent=4)
#
#
# {
#     "seed": [2345, 3456, 4567, 5678, 6789],
#     "data_dir": "data_dir",
#     "output_parent_dir": "",
#     "lr_scheduler": "lambda lr: lr",
#     "num_steps": 101,
#     "print_steps": 100,
#     "batch_size": 32,
#     "model_name": "nrde",
#     "metric": "accuracy",
#     "dataset_name": "Heartbeat",
#     "learning_rate": 0.0001,
#     "use_presplit": true,
#     "classification": true,
#     "include_time": false,
#     "T": 1,
#     "stepsize": 2,
#     "logsig_depth": 2,
#     "hidden_dim": 128,
#     "vf_depth": 4,
#     "vf_width": 128,
#     "dt0": 0.002,
#     "scale": 1,
#     "lambd": 0.0,
#     "solver": "diffrax.Heun()",
#     "stepsize_controller": "diffrax.ConstantStepSize()",
#     "model_args": {
#       "hidden_dim": 128,
#       "vf_depth": 4,
#       "vf_width": 128,
#       "dt0": 0.002,
#       "solver": "diffrax.Heun()",
#       "stepsize_controller": "diffrax.ConstantStepSize()",
#       "scale": 1,
#       "lambd": 0.0
#     }
# }
