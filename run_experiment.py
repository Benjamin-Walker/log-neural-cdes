"""
This script loads a JSON file containing the hyperparameters for each model and dataset, and uses
create_dataset_model_and_train from train.py to train the models on the datasets using the hyperparameters. The results
are saved in the output directory specified in the JSON file.
"""

import argparse
import json


def run_experiments(model_names, dataset_names, experiment_folder, pytorch_experiments):

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
            if model_name == "mamba":
                conv_dim = int(data["convdim"])
                expansion = int(data["expansion"])
            if model_name == "S6":
                conv_dim = None
                expansion = None

            if pytorch_experiments:
                from torch_experiments.train import (
                    create_dataset_model_and_train as torch_create_dataset_model_and_train,
                )

                exps_n_samples = {
                    "EigenWorms": 236,
                    "EthanolConcentration": 524,
                    "Heartbeat": 409,
                    "MotorImagery": 378,
                    "SelfRegulationSCP1": 561,
                    "SelfRegulationSCP2": 380,
                    "ppg": 1232,
                    "signature1": 100000,
                    "signature2": 100000,
                    "signature3": 100000,
                    "signature4": 100000,
                }
                n_samples = exps_n_samples[dataset_name]

                model_args = {
                    "num_blocks": num_blocks,
                    "hidden_dim": hidden_dim,
                    "state_dim": ssm_dim,
                    "conv_dim": conv_dim,
                    "expansion": expansion,
                }
                run_args = {
                    "data_dir": data_dir,
                    "output_parent_dir": output_parent_dir,
                    "model_name": model_name,
                    "metric": metric,
                    "batch_size": batch_size,
                    "dataset_name": dataset_name,
                    "n_samples": n_samples,
                    "output_step": output_step,
                    "use_presplit": use_presplit,
                    "include_time": include_time,
                    "num_steps": num_steps,
                    "print_steps": print_steps,
                    "lr": lr,
                    "model_args": model_args,
                }
                run_fn = torch_create_dataset_model_and_train
            else:
                import diffrax

                from train import create_dataset_model_and_train

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
                run_args = {
                    "data_dir": data_dir,
                    "use_presplit": use_presplit,
                    "dataset_name": dataset_name,
                    "output_step": output_step,
                    "metric": metric,
                    "include_time": include_time,
                    "T": T,
                    "model_name": model_name,
                    "stepsize": stepsize,
                    "logsig_depth": logsig_depth,
                    "model_args": model_args,
                    "num_steps": num_steps,
                    "print_steps": print_steps,
                    "lr": lr,
                    "lr_scheduler": lr_scheduler,
                    "batch_size": batch_size,
                    "output_parent_dir": output_parent_dir,
                }
                run_fn = create_dataset_model_and_train

            for seed in seeds:
                print(f"Running experiment with seed: {seed}")
                run_fn(seed=seed, **run_args)


if __name__ == "__main__":

    args = argparse.ArgumentParser()

    args.add_argument("--pytorch_experiments", action="store_true")
    args = args.parse_args()
    pytorch_experiments = args.pytorch_experiments

    if pytorch_experiments:
        model_names = ["mamba", "S6"]
    else:
        model_names = ["ncde", "log_ncde", "nrde", "S5", "lru"]
    dataset_names = [
        "EigenWorms",
        "EthanolConcentration",
        "Heartbeat",
        "MotorImagery",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
    ]
    experiment_folder = "experiment_configs/repeats"

    run_experiments(model_names, dataset_names, experiment_folder, pytorch_experiments)
