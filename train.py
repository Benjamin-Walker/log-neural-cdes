import os
import time

import diffrax
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
    lr_scheduler,
    batch_size,
    key,
    output_dir,
):
    if os.path.isdir(output_dir):
        raise ValueError(f"Warning: Output directory {output_dir} already exists")
    else:
        os.makedirs(output_dir)
    model_file = output_dir + "/model.checkpoint.npz"

    batchkey, key = jr.split(key, 2)
    opt = optax.adam(learning_rate=lr_scheduler(lr))
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


def create_dataset_model_and_train(
    seed,
    data_dir,
    dataset_name,
    T,
    model_name,
    stepsize,
    logsig_depth,
    model_args,
    num_steps,
    print_steps,
    lr,
    lr_scheduler,
    batch_size,
    output_parent_dir="",
):
    output_parent_dir += "outputs/" + model_name + "/" + dataset_name
    output_dir = f"T_{T}_nsteps_{num_steps}_lr_{lr}"
    if model_name == "log_ncde" or model_name == "nrde":
        output_dir += f"_stepsize_{stepsize}_logsigdepth_{logsig_depth}"
    for k, v in model_args.items():
        name = str(v)
        if "(" in name:
            name = name.split("(", 1)[0]
        output_dir += f"_{k}_" + name
        if name == "PIDController":
            output_dir += f"_rtol_{v.rtol}_atol_{v.atol}"
    output_dir += f"_seed_{seed}"

    key = jr.PRNGKey(seed)

    datasetkey, modelkey, trainkey, key = jr.split(key, 4)
    print(f"Creating dataset {dataset_name}")

    dataset = create_dataset(
        data_dir,
        dataset_name,
        stepsize=stepsize,
        depth=logsig_depth,
        include_time=model_args["include_time"],
        T=T,
        use_idxs=False,
        key=datasetkey,
    )

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
        lr_scheduler,
        batch_size,
        trainkey,
        output_parent_dir + "/" + output_dir,
    )


if __name__ == "__main__":
    data_dir = "data"
    seed = 1234
    num_steps = 1000
    print_steps = 200
    batch_size = 32
    lr = 1e-3
    T = 1
    include_time = False
    solver = diffrax.Heun()
    stepsize_controller = diffrax.ConstantStepSize()
    stepsize = 4
    logsig_depth = 2
    # Spoken Arabic Digits has nan values in training data
    dataset_names = [
        "EigenWorms",
        "EthanolConcentration",
        "FaceDetection",
        "FingerMovements",
        "HandMovementDirection",
        "Handwriting",
        "Heartbeat",
        "Libras",
        "LSST",
        "InsectWingbeat",
        "MotorImagery",
        "NATOPS",
        "PhonemeSpectra",
        "RacketSports",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "UWaveGestureLibrary",
    ]
    model_names = ["log_ncde"]

    model_args = {
        "num_blocks": 6,
        "hidden_dim": 64,
        "vf_depth": 2,
        "vf_width": 32,
        "ssm_dim": 32,
        "ssm_blocks": 2,
        "dt0": T / 2284,
        "include_time": include_time,
        "solver": solver,
        "stepsize_controller": stepsize_controller,
    }
    for dataset_name in dataset_names:
        for model_name in model_names:
            create_dataset_model_and_train(
                seed,
                dataset_name,
                T,
                model_name,
                stepsize,
                logsig_depth,
                model_args,
                num_steps,
                print_steps,
                lr,
                batch_size,
            )
