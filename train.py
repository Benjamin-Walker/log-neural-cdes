import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from data.datasets import create_dataset
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
def classification_loss(model, X, y, state, key):
    pred_y, state = calc_output(
        model, X, state, key, model.stateful, model.nondeterministic
    )
    return jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1)), state


@eqx.filter_jit
def make_step(model, X, y, state, opt, opt_state, key):
    (value, state), grads = classification_loss(model, X, y, state, key)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, state, value


def train_model(
    model,
    state,
    dataloaders,
    num_steps,
    print_steps,
    lr,
    batch_size,
    key,
    output_dir,
    slurm=False,
):
    if not slurm:
        if os.path.isdir(output_dir):
            raise ValueError(f"Warning: Output directory {output_dir} already exists")
        else:
            os.makedirs(output_dir)
    model_file = output_dir + "/model.checkpoint.npz"

    batchkey, key = jr.split(key, 2)
    warmup_cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=lr,
        warmup_steps=int(num_steps * 0.1),
        decay_steps=num_steps,
        end_value=1e-6,
    )
    opt = optax.adam(learning_rate=warmup_cosine_decay_scheduler)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    running_loss = 0.0
    all_val_acc = [0.0]
    all_time = []
    start = time.time()
    for step, data in zip(
        range(num_steps),
        dataloaders["train"].loop(batch_size, key=batchkey),
    ):
        stepkey, key = jr.split(key, 2)
        X, y = data
        model, state, value = make_step(model, X, y, state, opt, opt_state, stepkey)
        running_loss += value
        if (step + 1) % print_steps == 0:

            for _, data in zip(
                range(1),
                dataloaders["val"].loop(dataloaders["val"].size, key=None),
            ):
                stepkey, key = jr.split(key, 2)
                inference_model = eqx.tree_inference(model, value=True)
                X, y = data
                prediction, _ = calc_output(
                    inference_model,
                    X,
                    state,
                    stepkey,
                    model.stateful,
                    model.nondeterministic,
                )
                val_accuracy = jnp.mean(
                    jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                )
                end = time.time()
                total_time = end - start
                print(
                    f"Step: {step + 1}, Loss: {running_loss / print_steps}, "
                    f"Validation accuracy: {val_accuracy}, Time: {total_time}"
                )
                start = time.time()
                if val_accuracy >= max(all_val_acc):
                    print("Saving model")
                    eqx.tree_serialise_leaves(model_file, model)
                running_loss = 0.0
                all_val_acc.append(val_accuracy)
                all_time.append(total_time)

    steps = jnp.arange(0, num_steps + 1, print_steps)
    all_val_acc = jnp.array(all_val_acc)
    all_time = jnp.array(all_time)
    jnp.save(output_dir + "/steps.npy", steps)
    jnp.save(output_dir + "/all_val_acc.npy", all_val_acc)
    jnp.save(output_dir + "/all_time.npy", all_time)

    best_model = eqx.tree_deserialise_leaves(model_file, model)
    inference_model = eqx.tree_inference(best_model, value=True)
    for _, data in zip(
        range(1),
        dataloaders["test"].loop(dataloaders["test"].size, key=None),
    ):
        X, y = data
        stepkey, key = jr.split(key, 2)
        prediction, _ = calc_output(
            inference_model,
            X,
            state,
            stepkey,
            best_model.stateful,
            best_model.nondeterministic,
        )
        test_accuracy = jnp.mean(
            jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
        )
        print(f"Test accuracy: {test_accuracy}")

    jnp.save(output_dir + "/test_acc.npy", test_accuracy)


def run_training(
    seed,
    dataset_name,
    model_name,
    output_parent_dir,
    output_dir,
    stepsize,
    logsig_depth,
    model_args,
    num_steps,
    print_steps,
    lr,
    batch_size,
    slurm=False,
):
    key = jr.PRNGKey(seed)

    datasetkey, modelkey, key = jr.split(key, 3)
    print(f"Creating dataset {dataset_name}")
    dataset = create_dataset(
        data_dir,
        dataset_name,
        stepsize=stepsize,
        depth=logsig_depth,
        use_idxs=False,
        key=datasetkey,
    )
    print(f"Creating model {model_name}")
    model = create_model(
        model_name,
        dataset.data_dim,
        dataset.logsig_dim,
        logsig_depth,
        dataset.intervals,
        dataset.label_dim,
        **model_args,
        key=modelkey,
    )

    if model_name == "nrde" or model_name == "log_ncde":
        dataloaders = dataset.path_dataloaders
    elif model_name == "ncde":
        dataloaders = dataset.coeff_dataloaders
    else:
        dataloaders = dataset.raw_dataloaders

    train_model(
        model,
        dataloaders,
        num_steps,
        print_steps,
        lr,
        batch_size,
        key,
        output_parent_dir + "/" + output_dir,
        slurm,
    )


def create_model_and_train(
    seed,
    dataset_name,
    model_name,
    stepsize,
    logsig_depth,
    model_args,
    num_steps,
    print_steps,
    lr,
    batch_size,
    *,
    key,
):
    output_parent_dir = "outputs/" + model_name + "/" + dataset_name
    output_dir = f"nsteps_{num_steps}_lr_{lr}"
    if model_name == "log_ncde" or model_name == "nrde":
        output_dir += f"_stepsize_{stepsize}_logsigdepth_{logsig_depth}"
    for k, v in model_args.items():
        output_dir += f"_{k}_{v}"
    output_dir += f"_seed_{seed}"

    modelkey, trainkey, key = jr.split(key, 3)

    print(f"Creating model {model_name}")
    model, state = create_model(
        model_name,
        dataset.data_dim,
        dataset.logsig_dim,
        logsig_depth,
        dataset.intervals,
        dataset.label_dim,
        **model_args,
        key=modelkey,
    )

    if model_name == "nrde" or model_name == "log_ncde":
        dataloaders = dataset.path_dataloaders
    elif model_name == "ncde":
        dataloaders = dataset.coeff_dataloaders
    else:
        dataloaders = dataset.raw_dataloaders

    train_model(
        model,
        state,
        dataloaders,
        num_steps,
        print_steps,
        lr,
        batch_size,
        trainkey,
        output_parent_dir + "/" + output_dir,
    )


if __name__ == "__main__":
    data_dir = "data"
    seed = 9012
    num_steps = 100000
    print_steps = 2000
    batch_size = 32
    lr = 3e-4
    # Spoken Arabic Digits has nan values in training data
    dataset_names = [
        "toy",
    ]
    stepsize = 4
    logsig_depth = 2
    model_names = [
        "log_ncde",
    ]

    model_args = {
        "num_blocks": 6,
        "hidden_dim": 16,
        "vf_depth": 3,
        "vf_width": 8,
        "ssm_dim": 32,
        "ssm_blocks": 2,
    }

    for dataset_name in dataset_names:

        key = jr.PRNGKey(seed)

        datasetkey, modelkey, key = jr.split(key, 3)
        print(f"Creating dataset {dataset_name}")
        dataset = create_dataset(
            data_dir,
            dataset_name,
            stepsize=stepsize,
            depth=logsig_depth,
            use_idxs=False,
            key=datasetkey,
        )
        for model_name in model_names:
            create_model_and_train(
                seed,
                dataset_name,
                model_name,
                stepsize,
                logsig_depth,
                model_args,
                num_steps,
                print_steps,
                lr,
                batch_size,
                key=modelkey,
            )
