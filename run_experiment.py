"""
This script loads hyperparameters from JSON files and trains models on specified datasets using
the `create_dataset_model_and_train` function from `train.py` or its PyTorch equivalent. The results
are saved in the output directories defined in the JSON files.

The `run_experiments` function iterates over model names and dataset names, loading configuration
files from a specified folder, and then calls the appropriate training function based on the
framework (PyTorch or JAX).

Arguments for `run_experiments`:
- `model_names`: List of model architectures to use.
- `dataset_names`: List of datasets to train on.
- `experiment_folder`: Directory containing JSON configuration files.
- `pytorch_experiments`: Boolean indicating whether to use PyTorch (True) or JAX (False).

The script also provides a command-line interface (CLI) for specifying whether to run PyTorch experiments.

Usage:
- Use the `--pytorch_experiments` flag to run experiments with PyTorch; otherwise, JAX is used by default.
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
            early_stopping_steps = data["early_stopping_steps"]
            batch_size = data["batch_size"]
            metric = data["metric"]
            use_presplit = data["use_presplit"]
            T = data["T"]
            if model_name in [
                "lru",
                "S5",
                "S6",
                "mamba",
                "rnn_linear",
                "rnn_lstm",
                "rnn_gru",
                "bd_linear_ncde",
                "diagonal_linear_ncde",
                "dense_linear_ncde",
            ]:
                dt0 = None
            else:
                dt0 = float(data["dt0"])
            scale = data["scale"]
            lr = float(data["lr"])
            include_time = data["time"].lower() == "true"
            hidden_dim = int(data["hidden_dim"])
            if model_name in ["log_ncde", "nrde", "ncde"]:
                block_size = None
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
                if (
                    model_name == "bd_linear_ncde"
                    or model_name == "diagonal_linear_ncde"
                    or model_name == "dense_linear_ncde"
                ):
                    block_size = int(data["block_size"])
                    ssm_dim = None
                    stepsize = int(float(data["stepsize"]))
                    logsig_depth = int(data["depth"])
                    lambd = float(data["lambd"])
                    num_blocks = None
                else:
                    block_size = None
                    ssm_dim = int(data["ssm_dim"])
                    stepsize = 1
                    logsig_depth = 1
                    lambd = None
                    num_blocks = int(data["num_blocks"])
                vf_depth = None
                vf_width = None
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
                    "early_stopping_steps": early_stopping_steps,
                    "lr": lr,
                    "model_args": model_args,
                }
                run_fn = torch_create_dataset_model_and_train
            else:
                import diffrax

                from train import create_dataset_model_and_train

                model_args = {
                    "num_blocks": num_blocks,
                    "block_size": block_size,
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
                    "early_stopping_steps": early_stopping_steps,
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
        model_names = [
            "ncde",
            "log_ncde",
            "nrde",
            "S5",
            "lru",
            "bd_linear_ncde",
            "dense_linear_ncde",
            "diagonal_linear_ncde",
        ]
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
