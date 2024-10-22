"""
This script analyses the results of the UEA and PPG experiments. It is designed to determine the
optimal hyperparameters for each model and dataset following a grid search optimisation, and to
compare the performance of models across different random seeds using fixed hyperparameters.

The script performs the following tasks:
- Identifies the best hyperparameters based on validation metric for each model and dataset.
- Compares model performance across different random seeds, calculating the mean and standard
  deviation of test metric.

The script can handle two types of experiments:
1. `hypopt`: Hyperparameter optimisation, where the best configuration is selected based on
   validation metric.
2. `repeats`: Repeated experiments with fixed hyperparameters, where the results across multiple
   runs are aggregated.
"""

import os

import numpy as np


benchmark = "UEA"
experiment = "repeats"
results_dir = "results/paper_outputs/" + benchmark + "_outputs_" + experiment + "/"

if benchmark == "UEA":
    best_idx = np.argmax
    best_val = max
    operator = lambda x, y: x >= y
elif benchmark == "PPG":
    best_idx = np.argmin
    best_val = min
    operator = lambda x, y: x <= y
else:
    raise ValueError(f"Unknown benchmark: {benchmark}")

for model in sorted(os.listdir(results_dir)):
    if os.path.isdir(results_dir + model):
        model += "/"
        for dataset in sorted(os.listdir(results_dir + model)):
            if os.path.isdir(results_dir + model + dataset):
                train_metrics = []
                val_metrics = []
                steps = []
                test_metrics = []
                exps = []
                dataset += "/"
                for exp in os.listdir(results_dir + model + dataset):
                    if os.path.isdir(results_dir + model + dataset + exp):
                        if len(os.listdir(results_dir + model + dataset + exp)) != 0:
                            all_val_metric = np.load(
                                results_dir
                                + model
                                + dataset
                                + exp
                                + "/all_val_metric.npy"
                            )
                            all_train_metric = np.load(
                                results_dir
                                + model
                                + dataset
                                + exp
                                + "/all_train_metric.npy"
                            )
                            if benchmark == "PPG":
                                all_val_metric = all_val_metric[1:]
                                all_train_metric = all_train_metric[1:]
                            if experiment == "hypopt":
                                if operator(
                                    all_train_metric[best_idx(all_val_metric)],
                                    best_val(all_val_metric),
                                ):
                                    exps.append(exp)
                                    val_metrics.append(best_val(all_val_metric))
                                    train_metrics.append(
                                        all_train_metric[best_idx(all_val_metric)]
                                    )
                            elif experiment == "repeats":
                                val_metrics.append(
                                    np.load(
                                        results_dir
                                        + model
                                        + dataset
                                        + exp
                                        + "/all_val_metric.npy"
                                    )
                                )
                                test_metrics.append(
                                    np.load(
                                        results_dir
                                        + model
                                        + dataset
                                        + exp
                                        + "/test_metric.npy"
                                    )
                                )
                            else:
                                raise ValueError(f"Unknown experiment: {experiment}")
                if experiment == "hypopt":
                    val_metrics = np.array(val_metrics)
                    train_metrics = np.array(train_metrics)
                    idxs = np.where(val_metrics == best_val(val_metrics))[0]
                    train_idxs = np.where(
                        train_metrics[idxs] == best_val(train_metrics[idxs])
                    )[0]
                    for tr_idx in train_idxs:
                        idx = idxs[tr_idx]
                        print(
                            f"{model[:-1]} {dataset[:-1]} {exps[idx]} {100*val_metrics[idx]}"
                        )

                elif experiment == "repeats":
                    test_metrics = np.array(test_metrics)
                    print(
                        f"{model[:-1]} {dataset[:-1]} {np.mean([len(x) for x in val_metrics])} "
                        f"{100*np.mean(test_metrics)} {100*np.std(test_metrics)}"
                    )
