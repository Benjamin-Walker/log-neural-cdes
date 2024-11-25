"""
This module defines classes and functions for creating and training Mamba and S6 using PyTorch.
The main function, `create_dataset_model_and_train`, is designed to initialise the dataset, construct the model, and
execute the training process.

The function `create_dataset_model_and_train` takes the following arguments:

- `seed`: An integer representing the random seed for reproducibility.
- `data_dir`: The directory where the dataset is stored.
- `output_parent_dir`: The parent directory where the training outputs will be saved.
- `model_name`: A string specifying the model architecture to use ('mamba' or 'S6').
- `metric`: The evaluation metric to use during training, either 'accuracy' for classification or 'mse' for regression.
- `batch_size`: The number of samples per batch during training.
- `dataset_name`: The name of the dataset to load and use for training.
- `n_samples`: The total number of samples in the dataset.
- `output_step`: For regression tasks, defines the interval for outputting predictions.
- `use_presplit`: A boolean indicating whether to use a pre-split dataset.
- `include_time`: A boolean that determines whether to include time as a feature in the dataset.
- `num_steps`: The total number of steps for training the model.
- `print_steps`: The interval of steps after which to print training progress and metrics.
- `lr`: The learning rate for the optimiser.
- `model_args`: A dictionary containing additional arguments and hyperparameters for model customisation.

Classes defined in this module:

- `GLU`: Implements a Gated Linear Unit (GLU) layer, which applies a linear transformation followed by a gated
         activation.
- `MambaBlock`: A block that consists of normalisation, a Mamba or S6 layer, a GLU layer, and dropout. It serves as a
                basic building block for the Mamba model.
- `Mamba`: A sequence model that stacks multiple MambaBlock layers and includes an encoder and decoder for input/output
           transformation.
"""

import os
import shutil
import time

import numpy as np
import torch
from mamba_ssm import Mamba as MambaLayer

from torch_experiments.jax_dataset import Dataset
from torch_experiments.s6_recurrence import S6Layer


class GLU(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, input_dim * 2)

    def forward(self, x):
        out = self.linear(x)
        return out[:, :, : x.shape[2]] * torch.sigmoid(out[:, :, x.shape[2] :])


class MambaBlock(torch.nn.Module):
    def __init__(self, hidden_dim, state_dim, S6, conv_dim, expansion):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_dim)
        if S6:
            self.mamba = S6Layer(d_model=hidden_dim, d_state=state_dim)
        else:
            self.mamba = MambaLayer(
                d_model=hidden_dim, d_state=state_dim, d_conv=conv_dim, expand=expansion
            )
        self.glu = GLU(hidden_dim)
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(self.activation(x))
        x = self.glu(x)
        x = self.dropout(x)
        x = x + skip
        return x


class Mamba(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        input_dim,
        output_dim,
        hidden_dim,
        state_dim,
        S6,
        conv_dim,
        expansion,
        classification,
        output_step=1,
    ):
        super().__init__()
        self.linear_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.blocks = torch.nn.Sequential(
            *[
                MambaBlock(hidden_dim, state_dim, S6, conv_dim, expansion)
                for _ in range(num_blocks)
            ]
        )
        self.linear_decoder = torch.nn.Linear(hidden_dim, output_dim)
        self.classification = classification
        self.output_step = output_step

    def forward(self, x):
        x = self.linear_encoder(x)
        x = self.blocks(x)
        if self.classification:
            x = torch.mean(x, dim=1)
            x = torch.softmax(self.linear_decoder(x), dim=1)
        else:
            x = x[:, self.output_step - 1 :: self.output_step]
            x = torch.tanh(self.linear_decoder(x))
        return x


def create_dataset_model_and_train(
    seed,
    data_dir,
    output_parent_dir,
    model_name,
    metric,
    batch_size,
    dataset_name,
    n_samples,
    output_step,
    use_presplit,
    include_time,
    num_steps,
    print_steps,
    early_stopping_steps,
    lr,
    model_args,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classification = metric == "accuracy"

    if metric == "accuracy":
        best_val = max
        operator_improv = lambda x, y: x >= y
        operator_no_improv = lambda x, y: x <= y
    elif metric == "mse":
        best_val = min
        operator_improv = lambda x, y: x <= y
        operator_no_improv = lambda x, y: x >= y
    else:
        raise ValueError(f"Unknown metric: {metric}")

    output_dir = output_parent_dir + f"outputs/{model_name}" + f"/{dataset_name}/"
    output_dir += f"lr_{lr}_time_{include_time}"
    for k, v in model_args.items():
        output_dir += f"_{k}_{v}"
    output_dir += f"_seed_{seed}"

    if os.path.isdir(output_dir):
        user_input = input(
            f"Warning: Output directory {output_dir} already exists. Do you want to "
            f" delete it? (yes/no): "
        )
        if user_input.lower() == "yes":
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            print(f"Directory {output_dir} has been deleted and recreated.")
        else:
            raise ValueError(f"Directory {output_dir} already exists. Exiting.")
    else:
        os.makedirs(output_dir)
        print(f"Directory {output_dir} has been created.")

    indexes = torch.randperm(n_samples)

    train_dataset = Dataset(
        data_dir,
        dataset_name,
        True,
        False,
        False,
        indexes,
        presplit=use_presplit,
        include_time=include_time,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataset = Dataset(
        data_dir,
        dataset_name,
        False,
        True,
        False,
        indexes,
        presplit=use_presplit,
        include_time=include_time,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = Dataset(
        data_dir,
        dataset_name,
        False,
        False,
        True,
        indexes,
        presplit=use_presplit,
        include_time=include_time,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    input_dim = train_dataset.input_dim
    output_dim = train_dataset.output_dim

    if model_name == "mamba":
        S6 = False
    elif model_name == "S6":
        S6 = True
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = Mamba(
        input_dim=input_dim,
        output_dim=output_dim,
        classification=classification,
        S6=S6,
        output_step=output_step,
        **model_args,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
    all_train_metrics = []
    all_val_metrics = []
    val_metric_for_best_model = []
    no_val_improvement = 0.0
    steps = []
    step = 0
    start = time.time()
    while step <= num_steps:
        for X, y in train_dataloader:
            optimizer.zero_grad()

            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            if classification:
                loss = torch.nn.functional.cross_entropy(y_hat, y.argmax(dim=1))
            else:
                loss = torch.nn.functional.mse_loss(y_hat[:, :, 0], y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % print_steps == 0:

                model.eval()

                train_metric = 0.0
                for X, y in train_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    y_hat = model(X)
                    if classification:
                        metric = (
                            y_hat.argmax(dim=1) == y.argmax(dim=1)
                        ).float().cpu().sum() / len(y)
                    else:
                        metric = torch.nn.functional.mse_loss(y_hat[:, :, 0], y).item()
                    train_metric += metric
                all_train_metrics.append((train_metric / len(train_dataloader)))

                val_metric = 0.0
                for X, y in val_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    y_hat = model(X)
                    if classification:
                        metric = (
                            y_hat.argmax(dim=1) == y.argmax(dim=1)
                        ).float().cpu().sum() / len(y)
                    else:
                        metric = torch.nn.functional.mse_loss(y_hat[:, :, 0], y).item()
                    val_metric += metric
                end = time.time()
                print(
                    f"Step: {step}, "
                    f"Train Metric: {train_metric / len(train_dataloader)},"
                    f"Val Metric: {val_metric / len(val_dataloader)}, "
                    f"Time: {end - start}"
                )
                start = time.time()
                all_val_metrics.append((val_metric / len(val_dataloader)))
                steps.append(step)
                running_loss = 0.0

                if len(val_metric_for_best_model) == 0 or operator_improv(
                    all_val_metrics[-1], best_val(val_metric_for_best_model)
                ):
                    no_val_improvement = 0.0
                    val_metric_for_best_model.append(all_val_metrics[-1])
                    test_metric = 0.0
                    for X, y in test_dataloader:
                        X = X.to(device)
                        y = y.to(device)
                        y_hat = model(X)
                        if classification:
                            metric = (
                                y_hat.argmax(dim=1) == y.argmax(dim=1)
                            ).float().cpu().sum() / len(y)
                        else:
                            metric = torch.nn.functional.mse_loss(
                                y_hat[:, :, 0], y
                            ).item()
                        test_metric += metric
                    test_metric = test_metric / len(test_dataloader)

                    print(f"Test Metric: {test_metric}")
                if operator_no_improv(
                    all_val_metrics[-1], best_val(val_metric_for_best_model)
                ):
                    no_val_improvement += 1
                    if no_val_improvement > early_stopping_steps:
                        steps_save = np.array(steps)
                        all_train_metrics_save = np.array(all_train_metrics)
                        all_val_metrics_save = np.array(all_val_metrics)
                        test_metric = np.array(test_metric)
                        np.save(output_dir + "/steps.npy", steps_save)
                        np.save(
                            output_dir + "/all_train_metric.npy", all_train_metrics_save
                        )
                        np.save(
                            output_dir + "/all_val_metric.npy", all_val_metrics_save
                        )
                        np.save(output_dir + "/test_metric.npy", test_metric)
                        return

                steps_save = np.array(steps)
                all_train_metrics_save = np.array(all_train_metrics)
                all_val_metrics_save = np.array(all_val_metrics)
                test_metric = np.array(test_metric)
                np.save(output_dir + "/steps.npy", steps_save)
                np.save(output_dir + "/all_train_metric.npy", all_train_metrics_save)
                np.save(output_dir + "/all_val_metric.npy", all_val_metrics_save)
                np.save(output_dir + "/test_metric.npy", test_metric)
            model.train()
            step += 1
