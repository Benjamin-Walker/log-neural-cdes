"""
This module defines functions for creating datasets, building models, and training them using JAX
and Equinox. The main function, `create_dataset_model_and_train`, is designed to initialise the
dataset, construct the model, and execute the training process.

The function `create_dataset_model_and_train` takes the following arguments:

- `seed`: A random seed for reproducibility.
- `data_dir`: The directory where the dataset is stored.
- `use_presplit`: A boolean indicating whether to use a pre-split dataset.
- `dataset_name`: The name of the dataset to load and use for training.
- `output_step`: For regression tasks, the number of steps to skip before outputting a prediction.
- `metric`: The metric to use for evaluation. Supported values are `'mse'` for regression and `'accuracy'` for
            classification.
- `include_time`: A boolean indicating whether to include time as a channel in the time series data.
- `T`: The maximum time value to scale time data to [0, T].
- `model_name`: The name of the model architecture to use.
- `stepsize`: The size of the intervals for the Log-ODE method.
- `logsig_depth`: The depth of the Log-ODE method. Currently implemented for depths 1 and 2.
- `model_args`: A dictionary of additional arguments to customise the model.
- `num_steps`: The number of steps to train the model.
- `print_steps`: How often to print the loss during training.
- `lr`: The learning rate for the optimiser.
- `lr_scheduler`: The learning rate scheduler function.
- `batch_size`: The number of samples per batch during training.
- `output_parent_dir`: The parent directory where the training outputs will be saved.

The module also includes the following key functions:

- `calc_output`: Computes the model output, handling stateful and nondeterministic models with JAX's `vmap` for
                 batching.
- `classification_loss`: Computes the loss for classification tasks, including optional regularisation.
- `regression_loss`: Computes the loss for regression tasks, including optional regularisation.
- `make_step`: Performs a single optimisation step, updating model parameters based on the computed gradients.
- `train_model`: Handles the training loop, managing metrics, early stopping, and saving progress at regular intervals.
"""

import os
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from data_dir.datasets import create_dataset
from models.generate_model import create_model


def _compact_output_value(value):
    if isinstance(value, bool):
        return str(int(value))
    name = str(value)
    if "(" in name:
        name = name.split("(", 1)[0]
    return name


def _append_output_component(output_dir, key, value):
    if value is None:
        return output_dir
    key_aliases = {
        "block_size": "bs",
        "hidden_dim": "hd",
        "vf_depth": "vfd",
        "vf_width": "vfw",
        "ssm_dim": "sd",
        "ssm_blocks": "sb",
        "dt0": "dt0",
        "solver": "sol",
        "stepsize_controller": "sc",
        "scale": "scale",
        "lambd": "lam",
        "parallel_steps": "ps",
        "walsh_hadamard": "wh",
        "diagonal_dense": "dd",
        "sparsity": "sp",
        "piecewise_abelian": "pa",
        "rank": "r",
        "num_blocks": "nb",
        "buf_len": "buf",
    }
    key_name = key_aliases.get(key, key)
    value_name = _compact_output_value(value)
    return f"{output_dir}_{key_name}{value_name}"


def build_output_dir(
    output_parent_dir,
    model_name,
    dataset_name,
    T,
    include_time,
    num_steps,
    lr,
    drop_percentage,
    path_drop_window_mode,
    stepsize,
    logsig_depth,
    model_args,
    seed,
    drop_mode="same",
):
    if dataset_name.lower() in {"pm25", "pm10"}:
        output_parent_dir += (
            f"outputs_pm_drop_{drop_percentage}_{drop_mode}/"
            + model_name
            + "/"
            + dataset_name
        )
        output_dir = f"T_{T:.2f}_time_{include_time}_nsteps_{num_steps}_lr_{lr}"
        if model_name == "log_ncde" or model_name == "nrde":
            output_dir += f"_stepsize_{stepsize:.2f}_depth_{logsig_depth}"
        for k, v in model_args.items():
            if v is not None:
                if k == "dt0":
                    output_dir += f"_{k}_{v:.2f}"
                else:
                    output_dir += f"_{k}_{_compact_output_value(v)}"
                if _compact_output_value(v) == "PIDController":
                    output_dir += f"_rtol_{v.rtol}_atol_{v.atol}"
        output_dir += f"_seed_{seed}"
        return output_parent_dir + "/" + output_dir

    output_parent_dir += "outputs/" + model_name + "/" + dataset_name
    output_dir = f"T{T:.2f}_time{int(include_time)}_n{num_steps}_lr{lr}"
    if drop_percentage is not None:
        output_dir += f"_drop{drop_percentage:.2f}"
        if (
            model_name == "log_ncde"
            or model_name == "nrde"
            or model_name.endswith("linear_ncde")
        ):
            output_dir += f"_pwm{path_drop_window_mode}"
    if model_name == "log_ncde" or model_name == "nrde":
        output_dir += f"_step{stepsize:.2f}_depth{logsig_depth}"
    for k, v in model_args.items():
        if k == "dt0" and v is not None:
            output_dir = _append_output_component(output_dir, k, f"{v:.2f}")
        else:
            output_dir = _append_output_component(output_dir, k, v)
        if v is not None and _compact_output_value(v) == "PIDController":
            output_dir += f"_rtol{v.rtol}_atol{v.atol}"
    output_dir += f"_seed{seed}"
    return output_parent_dir + "/" + output_dir


def run_output_files_exist(output_dir):
    required_files = (
        "steps.npy",
        "all_train_metric.npy",
        "all_val_metric.npy",
        "all_time.npy",
        "test_metric.npy",
    )
    return all(
        os.path.isfile(os.path.join(output_dir, filename))
        for filename in required_files
    )


@eqx.filter_jit
def calc_output(model, X, state, key, stateful, nondeterministic):
    if stateful:
        if nondeterministic:
            output, state = jax.vmap(
                model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
            )(X, state, key)
        else:
            output, state = jax.vmap(
                model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
            )(X, state)
    elif nondeterministic:
        output = jax.vmap(model, in_axes=(0, None))(X, key)
    else:
        output = jax.vmap(model)(X)

    return output, state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def classification_loss(diff_model, static_model, X, y, state, key):
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(
        model, X, state, key, model.stateful, model.nondeterministic
    )
    norm = 0
    if model.lip2:
        if hasattr(model, "vf"):
            for layer in model.vf.mlp.layers:
                norm += jnp.mean(
                    jnp.linalg.norm(layer.weight, axis=-1)
                    + jnp.linalg.norm(layer.bias, axis=-1)
                )
        elif model.vf_A is not None:
            norm += jnp.mean(jnp.linalg.norm(model.vf_A, axis=-1))
        elif model.vf_A_sparse is not None:
            vf_A = model.vf_A_sparse.todense()
            norm += jnp.mean(jnp.linalg.norm(vf_A, axis=-1))
        else:
            norm = 0.0
        norm *= model.lambd
    return (
        jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1)) + norm,
        state,
    )


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def regression_loss(diff_model, static_model, X, y, state, key):
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(
        model, X, state, key, model.stateful, model.nondeterministic
    )
    pred_y = pred_y[:, :, 0]
    norm = 0
    if model.lip2:
        if hasattr(model, "vf"):
            for layer in model.vf.mlp.layers:
                norm += jnp.mean(
                    jnp.linalg.norm(layer.weight, axis=-1)
                    + jnp.linalg.norm(layer.bias, axis=-1)
                )
        elif model.vf_A is not None:
            norm += jnp.mean(jnp.linalg.norm(model.vf_A, axis=-1))
        elif model.vf_A_sparse is not None:
            vf_A = model.vf_A_sparse.todense()
            norm += jnp.mean(jnp.linalg.norm(vf_A, axis=-1))
        else:
            norm = 0.0
        norm *= model.lambd
    return (
        jnp.mean(jnp.mean((pred_y - y) ** 2, axis=1)) + norm,
        state,
    )


@eqx.filter_jit
def make_step(model, filter_spec, X, y, loss_fn, state, opt, opt_state, key):
    diff_model, static_model = eqx.partition(model, filter_spec)
    (value, state), grads = loss_fn(diff_model, static_model, X, y, state, key)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, value


def train_model(
    model_name,
    dataset_name,
    model,
    metric,
    filter_spec,
    state,
    dataloaders,
    num_steps,
    print_steps,
    early_stopping_steps,
    lr,
    lr_scheduler,
    batch_size,
    key,
    output_dir,
):

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

    if os.path.isdir(output_dir):
        if run_output_files_exist(output_dir):
            print(f"Skipping completed run in existing directory {output_dir}.")
            return None
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        print(f"Directory {output_dir} existed but was incomplete; recreated it.")
    else:
        os.makedirs(output_dir)
        print(f"Directory {output_dir} has been created.")

    batchkey, key = jr.split(key, 2)
    opt = optax.adam(learning_rate=lr_scheduler(lr))
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    if model.classification:
        loss_fn = classification_loss
    else:
        loss_fn = regression_loss

    running_loss = 0.0
    if metric == "accuracy":
        all_val_metric = [0.0]
        all_train_metric = [0.0]
        val_metric_for_best_model = [0.0]
    elif metric == "mse":
        all_val_metric = [100.0]
        all_train_metric = [100.0]
        val_metric_for_best_model = [100.0]
    no_val_improvement = 0
    all_time = []
    start = time.time()

    def _rescale_X_if_needed(X):
        if model_name == "dplr_linear_ncde":
            if dataset_name == "Heartbeat" or dataset_name == "MotorImagery":
                return (X[0], X[1] / 100, X[2])
        elif model_name.endswith("linear_ncde") and dataset_name == "Heartbeat":
            return (X[0], X[1] / 10, X[2])
        return X

    for step, data in zip(
        range(num_steps),
        dataloaders["train"].loop(batch_size, key=batchkey),
    ):
        stepkey, key = jr.split(key, 2)
        X, y = data
        X = _rescale_X_if_needed(X)
        model, state, opt_state, value = make_step(
            model, filter_spec, X, y, loss_fn, state, opt, opt_state, stepkey
        )
        running_loss += value
        if (step + 1) % print_steps == 0:
            end = time.time()

            def _compute_metric(loader, key):
                inference_model = eqx.tree_inference(model, value=True)

                if model.classification:
                    correct = 0.0
                    total = 0
                else:
                    mse_sum = 0.0  # sum over samples of (mean_t (sq err))
                    total = 0  # number of samples

                for data in loader.loop_epoch(batch_size):
                    stepkey, key = jr.split(key, 2)
                    X, y = data
                    X = _rescale_X_if_needed(X)

                    prediction, _ = calc_output(
                        inference_model,
                        X,
                        state,
                        stepkey,
                        model.stateful,
                        model.nondeterministic,
                    )

                    if model.classification:
                        correct += jnp.sum(
                            jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                        )
                        total += y.shape[0]
                    else:
                        pred = prediction[:, :, 0]
                        y_ = (
                            y[:, :, 0]
                            if (hasattr(y, "ndim") and y.ndim == 3 and y.shape[-1] == 1)
                            else y
                        )
                        per_sample_mse = jnp.mean((pred - y_) ** 2, axis=1)  # (batch,)
                        mse_sum += jnp.sum(per_sample_mse)
                        total += per_sample_mse.shape[0]

                metric = (
                    (correct / total) if model.classification else (mse_sum / total)
                )
                return metric, key

            train_metric, key = _compute_metric(dataloaders["train"], key)
            val_metric, key = _compute_metric(dataloaders["val"], key)

            total_time = end - start
            print(
                f"Step: {step + 1}, Loss: {running_loss / print_steps}, "
                f"Train metric: {train_metric}, "
                f"Validation metric: {val_metric}, Time: {total_time}"
            )
            start = time.time()

            if step > 0:
                if operator_no_improv(val_metric, best_val(val_metric_for_best_model)):
                    no_val_improvement += 1
                    if no_val_improvement > early_stopping_steps:
                        break
                else:
                    no_val_improvement = 0

                if operator_improv(val_metric, best_val(val_metric_for_best_model)):
                    val_metric_for_best_model.append(val_metric)

                    test_metric, key = _compute_metric(dataloaders["test"], key)
                    print(f"Test metric: {test_metric}")

                    if (not model.classification) and (
                        dataset_name.lower() in ["pm25", "pm10"]
                    ):

                        eps = 1e-8
                        stats_path = "data_dir/processed/PM/25/norm_stats_time1.npz"

                        try:
                            stats = np.load(stats_path)
                            y_min = float(stats["y_min"])
                            y_max = float(stats["y_max"])

                            rmse_scaled = jnp.sqrt(test_metric)
                            scale_factor = (y_max - y_min + eps) / 2.0
                            rmse_unscaled = scale_factor * rmse_scaled

                            print(f"Test RMSE (unscaled): {rmse_unscaled}")
                        except FileNotFoundError:
                            print(
                                f"(warn) Could not find stats file at {stats_path}, skipping RMSE unscale."
                            )

            running_loss = 0.0
            all_train_metric.append(train_metric)
            all_val_metric.append(val_metric)
            all_time.append(total_time)
            steps = jnp.arange(0, step + 1, print_steps)
            all_train_metric_save = jnp.array(all_train_metric)
            all_val_metric_save = jnp.array(all_val_metric)
            all_time_save = jnp.array(all_time)
            test_metric_save = jnp.array(test_metric)
            jnp.save(output_dir + "/steps.npy", steps)
            jnp.save(output_dir + "/all_train_metric.npy", all_train_metric_save)
            jnp.save(output_dir + "/all_val_metric.npy", all_val_metric_save)
            jnp.save(output_dir + "/all_time.npy", all_time_save)
            jnp.save(output_dir + "/test_metric.npy", test_metric_save)

    print(f"Test metric: {test_metric}")
    steps = jnp.arange(0, num_steps + 1, print_steps)
    all_train_metric = jnp.array(all_train_metric)
    all_val_metric = jnp.array(all_val_metric)
    all_time = jnp.array(all_time)
    test_metric = jnp.array(test_metric)
    jnp.save(output_dir + "/steps.npy", steps)
    jnp.save(output_dir + "/all_train_metric.npy", all_train_metric)
    jnp.save(output_dir + "/all_val_metric.npy", all_val_metric)
    jnp.save(output_dir + "/all_time.npy", all_time)
    jnp.save(output_dir + "/test_metric.npy", test_metric)

    return model


def create_dataset_model_and_train(
    seed,
    data_dir,
    use_presplit,
    dataset_name,
    output_step,
    metric,
    include_time,
    T,
    drop_percentage,
    drop_mode,
    path_drop_window_mode,
    model_name,
    stepsize,
    logsig_depth,
    model_args,
    num_steps,
    print_steps,
    early_stopping_steps,
    lr,
    lr_scheduler,
    batch_size,
    output_parent_dir="",
):
    output_dir = build_output_dir(
        output_parent_dir=output_parent_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        T=T,
        include_time=include_time,
        num_steps=num_steps,
        lr=lr,
        drop_percentage=drop_percentage,
        path_drop_window_mode=path_drop_window_mode,
        stepsize=stepsize,
        logsig_depth=logsig_depth,
        model_args=model_args,
        seed=seed,
        drop_mode=drop_mode,
    )

    key = jr.PRNGKey(seed)

    datasetkey, modelkey, trainkey, key = jr.split(key, 4)
    print(f"Creating dataset {dataset_name}")

    if model_name.endswith("linear_ncde"):
        scale = True
    else:
        scale = False

    dataset = create_dataset(
        data_dir,
        dataset_name,
        stepsize=stepsize,
        depth=logsig_depth,
        include_time=include_time,
        T=T,
        drop_percentage=drop_percentage,
        drop_mode=drop_mode,
        path_drop_window_mode=path_drop_window_mode,
        use_idxs=False,
        use_presplit=use_presplit,
        scale=scale,
        key=datasetkey,
    )
    buf_len = dataset.buf_len
    model_args["buf_len"] = buf_len

    print(f"Creating model {model_name}")
    classification = metric == "accuracy"
    model, state = create_model(
        model_name,
        dataset.data_dim,
        dataset.logsig_dim,
        logsig_depth,
        dataset.intervals,
        dataset.label_dim,
        classification=classification,
        output_step=output_step,
        **model_args,
        key=modelkey,
    )
    filter_spec = jax.tree_util.tree_map(lambda _: True, model)
    if (
        model_name == "nrde"
        or model_name == "log_ncde"
        or model_name.endswith("linear_ncde")
    ):
        dataloaders = dataset.path_dataloaders
        if model_name == "log_ncde":
            where = lambda model: (model.intervals, model.pairs)
            filter_spec = eqx.tree_at(
                where, filter_spec, replace=(False, False), is_leaf=lambda x: x is None
            )
        elif model_name == "nrde":
            where = lambda model: (model.intervals,)
            filter_spec = eqx.tree_at(where, filter_spec, replace=(False,))
        elif model_name == "wh_linear_ncde":
            where = lambda model: (model.hadamard_matrix,)
            filter_spec = eqx.tree_at(where, filter_spec, replace=(False,))
    elif model_name == "ncde":
        dataloaders = dataset.coeff_dataloaders
    else:
        dataloaders = dataset.raw_dataloaders

    return train_model(
        model_name,
        dataset_name,
        model,
        metric,
        filter_spec,
        state,
        dataloaders,
        num_steps,
        print_steps,
        early_stopping_steps,
        lr,
        lr_scheduler,
        batch_size,
        trainkey,
        output_parent_dir + "/" + output_dir,
    )
