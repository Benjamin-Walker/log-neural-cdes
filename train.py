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
import optax

from data_dir.datasets import create_dataset
from models.generate_model import create_model


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
    for step, data in zip(
        range(num_steps),
        dataloaders["train"].loop(batch_size, key=batchkey),
    ):
        stepkey, key = jr.split(key, 2)
        X, y = data

        if model_name.endswith("linear_ncde") and dataset_name == "Heartbeat":
            X = (X[0], X[1] / 10, X[2])
        model, state, opt_state, value = make_step(
            model, filter_spec, X, y, loss_fn, state, opt, opt_state, stepkey
        )
        running_loss += value
        if (step + 1) % print_steps == 0:
            predictions = []
            labels = []
            for data in dataloaders["train"].loop_epoch(batch_size):
                stepkey, key = jr.split(key, 2)
                inference_model = eqx.tree_inference(model, value=True)
                X, y = data
                if model_name.endswith("linear_ncde") and dataset_name == "Heartbeat":
                    X = (X[0], X[1] / 10, X[2])
                prediction, _ = calc_output(
                    inference_model,
                    X,
                    state,
                    stepkey,
                    model.stateful,
                    model.nondeterministic,
                )
                predictions.append(prediction)
                labels.append(y)
            prediction = jnp.vstack(predictions)
            y = jnp.vstack(labels)
            if model.classification:
                train_metric = jnp.mean(
                    jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                )
            else:
                prediction = prediction[:, :, 0]
                train_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
            predictions = []
            labels = []
            for data in dataloaders["val"].loop_epoch(batch_size):
                stepkey, key = jr.split(key, 2)
                inference_model = eqx.tree_inference(model, value=True)
                X, y = data
                if model_name.endswith("linear_ncde") and dataset_name == "Heartbeat":
                    X = (X[0], X[1] / 10, X[2])
                prediction, _ = calc_output(
                    inference_model,
                    X,
                    state,
                    stepkey,
                    model.stateful,
                    model.nondeterministic,
                )
                predictions.append(prediction)
                labels.append(y)
            prediction = jnp.vstack(predictions)
            y = jnp.vstack(labels)
            if model.classification:
                val_metric = jnp.mean(
                    jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                )
            else:
                prediction = prediction[:, :, 0]
                val_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
            end = time.time()
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
                    predictions = []
                    labels = []
                    for data in dataloaders["test"].loop_epoch(batch_size):
                        stepkey, key = jr.split(key, 2)
                        inference_model = eqx.tree_inference(model, value=True)
                        X, y = data
                        if (
                            model_name.endswith("linear_ncde")
                            and dataset_name == "Heartbeat"
                        ):
                            X = (X[0], X[1] / 10, X[2])
                        prediction, _ = calc_output(
                            inference_model,
                            X,
                            state,
                            stepkey,
                            model.stateful,
                            model.nondeterministic,
                        )
                        predictions.append(prediction)
                        labels.append(y)
                    prediction = jnp.vstack(predictions)
                    y = jnp.vstack(labels)
                    if model.classification:
                        test_metric = jnp.mean(
                            jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                        )
                    else:
                        prediction = prediction[:, :, 0]
                        test_metric = jnp.mean(
                            jnp.mean((prediction - y) ** 2, axis=1), axis=0
                        )
                    print(f"Test metric: {test_metric}")
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
    irregularly_sampled=1.0,
    output_parent_dir="",
):
    output_parent_dir += "outputs_Oct25/" + model_name + "/" + dataset_name
    output_dir = f"T_{T:.2f}_time_{include_time}_nsteps_{num_steps}_lr_{lr}"
    if model_name == "log_ncde" or model_name == "nrde":
        output_dir += f"_stepsize_{stepsize:.2f}_depth_{logsig_depth}"
    for k, v in model_args.items():
        name = str(v)
        if v is not None:
            if "(" in name:
                name = name.split("(", 1)[0]
            if name == "dt0":
                output_dir += f"_{k}_" + f"{v:.2f}"
            else:
                output_dir += f"_{k}_" + name
            if name == "PIDController":
                output_dir += f"_rtol_{v.rtol}_atol_{v.atol}"
    output_dir += f"_seed_{seed}"

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
        use_idxs=False,
        use_presplit=use_presplit,
        irregularly_sampled=irregularly_sampled,
        scale=scale,
        key=datasetkey,
    )

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
