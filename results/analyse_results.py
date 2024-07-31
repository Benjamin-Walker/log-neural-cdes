"""
This script analyses the results of the UEA and PPG experiments. It is designed to determine the
optimal hyperparameters for each model and dataset following a grid search optimisation, and to
compare the performance of models across different random seeds using fixed hyperparameters.

The script performs the following tasks:
- Identifies the best hyperparameters based on validation accuracy for each model and dataset.
- Compares model performance across different random seeds, calculating the mean and standard
  deviation of test accuracy.

The script can handle two types of experiments:
1. `hypopt`: Hyperparameter optimisation, where the best configuration is selected based on
   validation accuracy.
2. `repeats`: Repeated experiments with fixed hyperparameters, where the results across multiple
   runs are aggregated.
"""

import os

import numpy as np


benchmark = "PPG"
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
                train_accs = []
                val_accs = []
                steps = []
                test_accs = []
                exps = []
                dataset += "/"
                for exp in os.listdir(results_dir + model + dataset):
                    if os.path.isdir(results_dir + model + dataset + exp):
                        if len(os.listdir(results_dir + model + dataset + exp)) != 0:
                            if model == "mamba/" and benchmark != "PPG":
                                all_val_acc = np.load(
                                    results_dir + model + dataset + exp + "/val_acc.npy"
                                )
                            else:
                                all_val_acc = np.load(
                                    results_dir
                                    + model
                                    + dataset
                                    + exp
                                    + "/all_val_acc.npy"
                                )
                            if model == "S6/" or model == "mamba/":
                                all_train_acc = np.load(
                                    results_dir
                                    + model
                                    + dataset
                                    + exp
                                    + "/train_acc.npy"
                                )
                            else:
                                all_train_acc = np.load(
                                    results_dir
                                    + model
                                    + dataset
                                    + exp
                                    + "/all_train_acc.npy"
                                )
                            if benchmark == "PPG":
                                all_val_acc = all_val_acc[1:]
                                all_train_acc = all_train_acc[1:]
                            if experiment == "hypopt":
                                if operator(
                                    all_train_acc[best_idx(all_val_acc)],
                                    best_val(all_val_acc),
                                ):
                                    exps.append(exp)
                                    val_accs.append(best_val(all_val_acc))
                                    train_accs.append(
                                        all_train_acc[best_idx(all_val_acc)]
                                    )
                            elif experiment == "repeats":
                                if model == "mamba/" and benchmark != "PPG":
                                    val_accs.append(
                                        np.load(
                                            results_dir
                                            + model
                                            + dataset
                                            + exp
                                            + "/val_acc.npy"
                                        )
                                    )
                                else:
                                    val_accs.append(
                                        np.load(
                                            results_dir
                                            + model
                                            + dataset
                                            + exp
                                            + "/all_val_acc.npy"
                                        )
                                    )
                                test_accs.append(
                                    np.load(
                                        results_dir
                                        + model
                                        + dataset
                                        + exp
                                        + "/test_acc.npy"
                                    )
                                )
                            else:
                                raise ValueError(f"Unknown experiment: {experiment}")
                if experiment == "hypopt":
                    val_accs = np.array(val_accs)
                    train_accs = np.array(train_accs)
                    idxs = np.where(val_accs == best_val(val_accs))[0]
                    train_idxs = np.where(
                        train_accs[idxs] == best_val(train_accs[idxs])
                    )[0]
                    for tr_idx in train_idxs:
                        idx = idxs[tr_idx]
                        print(
                            f"{model[:-1]} {dataset[:-1]} {exps[idx]} {100*val_accs[idx]}"
                        )
                elif experiment == "repeats":
                    test_accs = np.array(test_accs)
                    print(
                        f"{model[:-1]} {dataset[:-1]} {np.mean([len(x) for x in val_accs])} "
                        f"{100*np.mean(test_accs)} {100*np.std(test_accs)}"
                    )
