"""
This script loads a JSON file containing the hyperparameters for each model and dataset, and uses
create_dataset_model_and_train from train.py to train the models on the datasets using the hyperparameters. The results
are saved in the output directory specified in the JSON file.
"""

import json

import diffrax

from train import create_dataset_model_and_train


model_names = ["ncde", "log_ncde", "nrde", "lru", "S5"]
dataset_names = [
    "EigenWorms",
    "EthanolConcentration",
    "Heartbeat",
    "MotorImagery",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
]
experiment_folder = "experiment_configs/repeats"

for model_name in model_names:
    for dataset_name in dataset_names:
        with open(
            experiment_folder + f"/{model_name}/{dataset_name}.json", "r"
        ) as file:
            data = json.load(file)

        seeds = data["seeds"]
        data_dir = data["data_dir"]
        output_parent_dir = data["output_parent_dir"]
        lr_scheduler = eval(data["lr_scheduler"])
        num_steps = data["num_steps"]
        print_steps = data["print_steps"]
        batch_size = data["batch_size"]
        metric = data["metric"]
        classification = data["classification"]
        use_presplit = data["use_presplit"]
        T = data["T"]
        if model_name in ["lru", "S5", "S6", "mamba"]:
            dt0 = None
        else:
            dt0 = float(data["dt0"])
        scale = data["scale"]
        lr = float(data["lr"])
        include_time = data["time"].lower() == "true"
        hidden_dim = int(data["hidden_dim"])
        if model_name in ["log_ncde", "nrde", "ncde"]:
            vf_depth = int(data["vf_depth"])
            vf_width = int(data["vf_width"])
            if model_name in ["log_ncde", "nrde"]:
                logsig_depth = int(data["depth"])
                stepsize = int(float(data["stepsize"]))
            else:
                logsig_depth = 1
                stepsize = 1
            if model_name == "log_ncde":
                lambd = float(data["lambd"])
            else:
                lambd = None
            ssm_dim = None
            num_blocks = None
        else:
            vf_depth = None
            vf_width = None
            logsig_depth = 1
            stepsize = 1
            lambd = None
            ssm_dim = int(data["ssm_dim"])
            num_blocks = int(data["num_blocks"])
        if model_name == "S5":
            ssm_blocks = int(data["ssm_blocks"])
        else:
            ssm_blocks = None
        if dataset_name == "ppg":
            output_step = int(data["output_step"])
        else:
            output_step = 1

        model_args = {
            "num_blocks": num_blocks,
            "hidden_dim": hidden_dim,
            "vf_depth": vf_depth,
            "vf_width": vf_width,
            "ssm_dim": ssm_dim,
            "ssm_blocks": ssm_blocks,
            "dt0": dt0,
            "solver": diffrax.Heun(),
            "stepsize_controller": diffrax.ConstantStepSize(),
            "scale": scale,
            "lambd": lambd,
        }

        for seed in seeds:
            print(f"Running experiment with seed: {seed}")
            create_dataset_model_and_train(
                seed=seed,
                data_dir=data_dir,
                use_presplit=use_presplit,
                dataset_name=dataset_name,
                output_step=output_step,
                metric=metric,
                include_time=include_time,
                T=T,
                model_name=model_name,
                stepsize=stepsize,
                logsig_depth=logsig_depth,
                model_args=model_args,
                num_steps=num_steps,
                print_steps=print_steps,
                lr=lr,
                lr_scheduler=lr_scheduler,
                batch_size=batch_size,
                output_parent_dir=output_parent_dir,
            )
