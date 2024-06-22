import os
import shutil
import time

import numpy as np
from jax_dataset import Dataset
from mamba_recurrence import S6Layer
from mamba_ssm import Mamba as MambaLayer

import torch


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
    output_step,
    use_presplit,
    include_time,
    num_steps,
    print_steps,
    lr,
    model_args,
):
    torch.manual_seed(seed)

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

    output_dir = output_parent_dir + f"/{model_name}" + f"/{dataset_name}/"
    output_dir += (
        f"lr_{lr}_time_{include_time}_numblocks_{num_blocks}_"
        f"hiddendim_{hidden_dim}_statedim_{state_dim}_convdim_{conv_dim}_"
        f"expansion_{expansion}_seed_{seed}"
    )

    if os.path.isdir(output_dir):
        user_input = input(
            f"Warning: Output directory {output_dir} already exists. Do you want to delete it? (yes/no): "
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

    n_samples = {
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
    indexes = torch.randperm(n_samples[dataset_name])

    device = "cuda"
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
    all_train_accuracies = []
    all_val_accuracies = []
    val_acc_for_best_model = []
    no_val_improvement = 0.0
    steps = []
    step = 0
    num_epochs = 10 * int(num_steps / len(train_dataloader))
    start = time.time()
    for _ in range(num_epochs):
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

                train_accuracy = 0.0
                for X, y in train_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    y_hat = model(X)
                    if classification:
                        accuracy = (
                            y_hat.argmax(dim=1) == y.argmax(dim=1)
                        ).float().cpu().sum() / len(y)
                    else:
                        accuracy = torch.nn.functional.mse_loss(
                            y_hat[:, :, 0], y
                        ).item()
                    train_accuracy += accuracy
                all_train_accuracies.append((train_accuracy / len(train_dataloader)))

                val_accuracy = 0.0
                for X, y in val_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    y_hat = model(X)
                    if classification:
                        accuracy = (
                            y_hat.argmax(dim=1) == y.argmax(dim=1)
                        ).float().cpu().sum() / len(y)
                    else:
                        accuracy = torch.nn.functional.mse_loss(
                            y_hat[:, :, 0], y
                        ).item()
                    val_accuracy += accuracy
                end = time.time()
                print(
                    f"Step: {step}, Train Accuracy: {train_accuracy / len(train_dataloader)},"
                    f"Val Accuracy: {val_accuracy / len(val_dataloader)}, Time: {end - start}"
                )
                start = time.time()
                all_val_accuracies.append((val_accuracy / len(val_dataloader)))
                steps.append(step)
                running_loss = 0.0

                if len(val_acc_for_best_model) == 0 or operator_improv(
                    all_val_accuracies[-1], best_val(val_acc_for_best_model)
                ):
                    no_val_improvement = 0.0
                    val_acc_for_best_model.append(all_val_accuracies[-1])
                    test_accuracy = 0.0
                    for X, y in test_dataloader:
                        X = X.to(device)
                        y = y.to(device)
                        y_hat = model(X)
                        if classification:
                            accuracy = (
                                y_hat.argmax(dim=1) == y.argmax(dim=1)
                            ).float().cpu().sum() / len(y)
                        else:
                            accuracy = torch.nn.functional.mse_loss(
                                y_hat[:, :, 0], y
                            ).item()
                        test_accuracy += accuracy
                    test_accuracy = test_accuracy / len(test_dataloader)

                    print(f"Test Accuracy: {test_accuracy}")
                if operator_no_improv(
                    all_val_accuracies[-1], best_val(val_acc_for_best_model)
                ):
                    no_val_improvement += 1
                    if no_val_improvement > 10:
                        steps_save = np.array(steps)
                        all_train_accuracies_save = np.array(all_train_accuracies)
                        all_val_accuracies_save = np.array(all_val_accuracies)
                        test_accuracy = np.array(test_accuracy)
                        np.save(output_dir + "/steps.npy", steps_save)
                        np.save(
                            output_dir + "/train_acc.npy", all_train_accuracies_save
                        )
                        np.save(
                            output_dir + "/all_val_acc.npy", all_val_accuracies_save
                        )
                        np.save(output_dir + "/test_acc.npy", test_accuracy)
                        return

                steps_save = np.array(steps)
                all_train_accuracies_save = np.array(all_train_accuracies)
                all_val_accuracies_save = np.array(all_val_accuracies)
                test_accuracy = np.array(test_accuracy)
                np.save(output_dir + "/steps.npy", steps_save)
                np.save(output_dir + "/train_acc.npy", all_train_accuracies_save)
                np.save(output_dir + "/all_val_acc.npy", all_val_accuracies_save)
                np.save(output_dir + "/test_acc.npy", test_accuracy)
            if step == num_steps:
                return
            model.train()
            step += 1


if __name__ == "__main__":

    lr = 1e-4
    model_name = "S6"
    metric = "mse"
    S6 = False

    for seed in [2345, 3456, 4567, 5678, 6789]:
        for dataset in ["ppg"]:
            for include_time in [True]:
                for num_blocks in [2]:
                    for hidden_dim in [64]:
                        for state_dim in [64]:
                            for conv_dim in [4]:
                                for expansion in [2]:
                                    model_args = {
                                        "num_blocks": num_blocks,
                                        "hidden_dim": hidden_dim,
                                        "state_dim": state_dim,
                                        "conv_dim": conv_dim,
                                        "expansion": expansion,
                                    }
                                    create_dataset_model_and_train(
                                        seed=seed,
                                        data_dir="data_dir",
                                        output_parent_dir="outputs_S6_ppg_repeats",
                                        model_name=model_name,
                                        metric=metric,
                                        batch_size=4,
                                        dataset_name=dataset,
                                        output_step=128,
                                        use_presplit=True,
                                        include_time=include_time,
                                        num_steps=100000,
                                        print_steps=1000,
                                        lr=lr,
                                        model_args=model_args,
                                    )
