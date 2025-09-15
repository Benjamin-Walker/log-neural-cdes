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
from collections import defaultdict
from typing import Dict, List

import numpy as np


def rank_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
    """Return the rank (1 = best) for each key in *score_dict*.

    If two models obtain exactly the same score on a dataset they receive the
    same rank; the next rank is offset accordingly (i.e. *dense* ranking /
    competition ranking scheme).
    """

    # Sort by *descending* score because higher accuracy is better.
    sorted_items = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)

    ranks: Dict[str, float] = {}
    current_rank = 1
    i = 0
    while i < len(sorted_items):
        # Find block of items with identical score.
        j = i + 1
        while j < len(sorted_items) and np.isclose(
            sorted_items[j][1], sorted_items[i][1]
        ):
            j += 1

        # All models in [i, j) share the same rank (dense/competition ranking).
        for k in range(i, j):
            model_name = sorted_items[k][0]
            ranks[model_name] = current_rank

        current_rank += j - i  # Dense ranking.
        i = j

    return ranks


# -----------------------------------------------------------------------------
# User‑configurable settings
# -----------------------------------------------------------------------------

benchmark = "UEA"  # Either "UEA" or "PPG".
experiment = "repeats"  # Either "hypopt" or "repeats".
results_dir = f"results/paper_outputs/{benchmark}_outputs_{experiment}/"

# Determine optimisation direction.
if benchmark == "UEA":
    best_idx = np.argmax
    best_val = max
    operator = lambda x, y: x >= y  # noqa: E731  (keep as simple lambda)
elif benchmark == "PPG":
    best_idx = np.argmin
    best_val = min
    operator = lambda x, y: x <= y  # noqa: E731
else:
    raise ValueError(f"Unknown benchmark: {benchmark}")

# -----------------------------------------------------------------------------
# Containers for summary statistics (used for *average* accuracy and *rank*)
# -----------------------------------------------------------------------------

model_to_dataset_scores: Dict[str, List[float]] = defaultdict(list)
# e.g. {"ncde": [74.0, 53.1, ...]}

dataset_to_model_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
# e.g. {"Heartbeat": {"ncde": 74.0, "mamba": 76.3, ...}}

# -----------------------------------------------------------------------------
# Main loop over all saved experiment results
# -----------------------------------------------------------------------------

for model in sorted(os.listdir(results_dir)):
    if not os.path.isdir(os.path.join(results_dir, model)):
        continue

    model_dir = os.path.join(results_dir, model)
    for dataset in sorted(os.listdir(model_dir)):
        dataset_dir = os.path.join(model_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        train_metrics = []  # Only used for *hypopt*.
        val_metrics = []
        test_metrics = []  # Only used for *repeats*.

        for exp in os.listdir(dataset_dir):
            exp_dir = os.path.join(dataset_dir, exp)
            if not os.path.isdir(exp_dir):
                continue
            if not os.listdir(exp_dir):
                continue  # Empty directory – skip.

            all_val_metric = np.load(os.path.join(exp_dir, "all_val_metric.npy"))
            all_train_metric = np.load(os.path.join(exp_dir, "all_train_metric.npy"))

            if benchmark == "PPG":  # First element is burn‑in.
                all_val_metric = all_val_metric[1:]
                all_train_metric = all_train_metric[1:]

            if experiment == "hypopt":
                if operator(
                    all_train_metric[best_idx(all_val_metric)], best_val(all_val_metric)
                ):
                    val_metrics.append(best_val(all_val_metric))
                    train_metrics.append(all_train_metric[best_idx(all_val_metric)])

            elif experiment == "repeats":
                val_metrics.append(all_val_metric)
                test_metrics.append(np.load(os.path.join(exp_dir, "test_metric.npy")))
            else:
                raise ValueError(f"Unknown experiment: {experiment}")

        # ---------------------------------------------------------------------
        # Per‑dataset output
        # ---------------------------------------------------------------------
        if experiment == "hypopt":
            if not val_metrics:  # No valid experiments found.
                continue
            val_metrics = np.array(val_metrics)
            train_metrics = np.array(train_metrics)
            idxs = np.where(val_metrics == best_val(val_metrics))[0]
            train_idxs = np.where(train_metrics[idxs] == best_val(train_metrics[idxs]))[
                0
            ]
            for tr_idx in train_idxs:
                idx = idxs[tr_idx]
                print(f"{model} {dataset} {exp} {100 * val_metrics[idx]:.4f}")

        elif experiment == "repeats":
            if not test_metrics:  # No runs – skip.
                continue
            test_metrics = np.array(test_metrics)
            mean_test = 100 * np.mean(test_metrics)
            std_test = 100 * np.std(test_metrics)
            num_seeds = np.mean([len(x) for x in val_metrics])  # For completeness.

            print(f"{model} {dataset} {num_seeds:.1f} {mean_test:.8f} {std_test:.8f}")

            # Store for summary.
            model_to_dataset_scores[model].append(mean_test)
            dataset_to_model_scores[dataset][model] = mean_test

# -----------------------------------------------------------------------------
# Summary across datasets (only relevant for *repeats*)
# -----------------------------------------------------------------------------

if experiment == "repeats" and model_to_dataset_scores:
    print("\n=== Summary across all datasets ===")

    # 1) Average test accuracy per model.
    avg_test_accuracy = {
        model: float(np.mean(scores))
        for model, scores in model_to_dataset_scores.items()
    }

    # 2) Average rank per model.
    model_ranks: Dict[str, List[int]] = defaultdict(list)
    for dataset, scores in dataset_to_model_scores.items():
        ranks = rank_scores(scores)  # dict {model: rank}
        for mdl, rk in ranks.items():
            model_ranks[mdl].append(rk)

    avg_rank = {model: float(np.mean(ranks)) for model, ranks in model_ranks.items()}

    # Pretty print – sorted by average rank (ascending = better).
    header = f"{'Model':<30s} {'Avg Test Acc (%)':>17s} {'Avg Rank':>10s}"
    print(header)
    print("-" * len(header))
    for mdl in sorted(avg_rank, key=lambda m: avg_rank[m]):
        print(f"{mdl:<30s} {avg_test_accuracy[mdl]:>17.4f} {avg_rank[mdl]:>10.2f}")
