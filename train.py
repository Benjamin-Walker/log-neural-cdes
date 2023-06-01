import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from data.datasets import create_uea_dataset
from models.generate_model import create_model


@eqx.filter_jit
def calc_output(model, X):
    return jax.vmap(model)(X)


@eqx.filter_jit
@eqx.filter_value_and_grad
def classification_loss(model, X, y):
    pred_y = calc_output(model, X)
    return jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1))


@eqx.filter_jit
def make_step(model, X, y, opt, opt_state):
    value, grads = classification_loss(model, X, y)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, value


def train_model(
    model,
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

    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    running_loss = 0.0
    all_val_acc = [0.0]
    all_time = []
    start = time.time()
    for step, data in zip(
        range(num_steps),
        dataloaders["train"].loop(batch_size, key=batchkey),
    ):
        X, y = data
        model, value = make_step(model, X, y, opt, opt_state)
        running_loss += value
        if (step + 1) % print_steps == 0:

            for _, data in zip(
                range(1),
                dataloaders["val"].loop(dataloaders["val"].size, key=None),
            ):
                X, y = data
                prediction = calc_output(model, X)
                val_accuracy = jnp.mean(
                    jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                )
                end = time.time()
                total_time = end - start
                print(
                    f"Step: {step + 1}, Loss: {running_loss / 100}, "
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
    for _, data in zip(
        range(1),
        dataloaders["test"].loop(dataloaders["test"].size, key=None),
    ):
        X, y = data
        prediction = calc_output(best_model, X)
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
    dataset = create_uea_dataset(
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


if __name__ == "__main__":
    seed = 1234
    num_steps = 100000
    print_steps = 200
    batch_size = 32
    lr = 3e-5
    # Spoken Arabic Digits has nan values in training data
    dataset_names = [
        # "EigenWorms",
        "EthanolConcentration",
        # "FaceDetection", # not enough memory
        "FingerMovements",
        "HandMovementDirection",
        "Handwriting",
        "Heartbeat",
        "Libras",
        "LSST",
        # "InsectWingbeat", # not enough memory to process
        # "MotorImagery", # not enough memory
        "NATOPS",
        "PhonemeSpectra",
        "RacketSports",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "UWaveGestureLibrary",
    ]
    stepsize = 4
    logsig_depth = 2
    model_names = [
        "rnn_linear",
        "rnn_gru",
        "rnn_lstm",
        "rnn_mlp",
        # "ncde",
        # "nrde",
        # "log_ncde"
    ]

    model_args = {"hidden_dim": 20, "vf_depth": 3, "vf_width": 8}

    for dataset_name in dataset_names:

        key = jr.PRNGKey(seed)

        datasetkey, modelkey, key = jr.split(key, 3)
        print(f"Creating dataset {dataset_name}")
        dataset = create_uea_dataset(
            dataset_name,
            stepsize=stepsize,
            depth=logsig_depth,
            use_idxs=False,
            key=datasetkey,
        )

        for model_name in model_names:
            output_parent_dir = "outputs/" + model_name + "/" + dataset_name
            output_dir = f"nsteps_{num_steps}_lr_{lr}"
            if model_name == "log_ncde" or model_name == "nrde":
                output_dir += f"_stepsize_{stepsize}_logsigdepth_{logsig_depth}"
            for k, v in model_args.items():
                output_dir += f"_{k}_{v}"
            output_dir += f"_seed_{seed}"

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
            )
