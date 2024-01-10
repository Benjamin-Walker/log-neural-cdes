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
def classification_loss(diff_model, static_model, X, y, state, key):
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(
        model, X, state, key, model.stateful, model.nondeterministic
    )
    norm = 0
    if model.lip2:
        for layer in model.vf.mlp.layers:
            norm += jnp.mean(
                jnp.linalg.norm(layer.weight, axis=-1)
                + jnp.linalg.norm(layer.bias, axis=-1)
            )
        norm *= model.lambd
    return (
        jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1)) + norm,
        state,
    )


@eqx.filter_jit
def make_step(model, filter_spec, X, y, state, opt, opt_state, key):
    diff_model, static_model = eqx.partition(model, filter_spec)
    (value, state), grads = classification_loss(
        diff_model, static_model, X, y, state, key
    )
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, state, value


def train_model(
    model,
    filter_spec,
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
    all_train_acc = [0.0]
    val_acc_for_best_model = [0.0]
    no_val_improvement = 0
    all_time = []
    start = time.time()
    for step, data in zip(
        range(num_steps),
        dataloaders["train"].loop(batch_size, key=batchkey),
    ):
        stepkey, key = jr.split(key, 2)
        X, y = data
        model, state, value = make_step(
            model, filter_spec, X, y, state, opt, opt_state, stepkey
        )
        running_loss += value
        if (step + 1) % print_steps == 0:
            predictions = []
            labels = []
            for data in dataloaders["train"].loop_epoch(batch_size):
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
                predictions.append(prediction)
                labels.append(y)
            prediction = jnp.vstack(predictions)
            y = jnp.vstack(labels)
            train_accuracy = jnp.mean(
                jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
            )
            predictions = []
            labels = []
            for data in dataloaders["val"].loop_epoch(batch_size):
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
                predictions.append(prediction)
                labels.append(y)
            prediction = jnp.vstack(predictions)
            y = jnp.vstack(labels)
            val_accuracy = jnp.mean(
                jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
            )
            end = time.time()
            total_time = end - start
            print(
                f"Step: {step + 1}, Loss: {running_loss / print_steps}, "
                f"Train accuracy: {train_accuracy}, "
                f"Validation accuracy: {val_accuracy}, Time: {total_time}"
            )
            start = time.time()
            if step > 0:
                if val_accuracy <= max(val_acc_for_best_model):
                    no_val_improvement += 1
                    if no_val_improvement > 10:
                        break
                else:
                    no_val_improvement = 0
                if val_accuracy >= max(val_acc_for_best_model):
                    print("Saving model")
                    eqx.tree_serialise_leaves(model_file, model)
                    val_acc_for_best_model.append(val_accuracy)
                    predictions = []
                    labels = []
                    for data in dataloaders["test"].loop_epoch(batch_size):
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
                        predictions.append(prediction)
                        labels.append(y)
                    prediction = jnp.vstack(predictions)
                    y = jnp.vstack(labels)
                    test_accuracy = jnp.mean(
                        jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                    )
                    print(f"Test accuracy: {test_accuracy}")
                running_loss = 0.0
                all_train_acc.append(train_accuracy)
                all_val_acc.append(val_accuracy)
                all_time.append(total_time)
                steps = jnp.arange(0, step + 1, print_steps)
                all_train_acc_save = jnp.array(all_train_acc)
                all_val_acc_save = jnp.array(all_val_acc)
                all_time_save = jnp.array(all_time)
                test_acc_save = jnp.array(test_accuracy)
                jnp.save(output_dir + "/steps.npy", steps)
                jnp.save(output_dir + "/all_train_acc.npy", all_train_acc_save)
                jnp.save(output_dir + "/all_val_acc.npy", all_val_acc_save)
                jnp.save(output_dir + "/all_time.npy", all_time_save)
                jnp.save(output_dir + "/test_acc.npy", test_acc_save)

    print(f"Test accuracy: {test_accuracy}")
    steps = jnp.arange(0, num_steps + 1, print_steps)
    all_train_acc = jnp.array(all_train_acc)
    all_val_acc = jnp.array(all_val_acc)
    all_time = jnp.array(all_time)
    test_acc = jnp.array(test_accuracy)
    jnp.save(output_dir + "/steps.npy", steps)
    jnp.save(output_dir + "/all_train_acc.npy", all_train_acc)
    jnp.save(output_dir + "/all_val_acc.npy", all_val_acc)
    jnp.save(output_dir + "/all_time.npy", all_time)
    jnp.save(output_dir + "/test_acc.npy", test_acc)


def create_dataset_model_and_train(
    seed,
    data_dir,
    use_presplit,
    dataset_name,
    include_time,
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
    output_parent_dir += "outputs_missing_channel/" + model_name + "/" + dataset_name
    output_dir = f"T_{T:.2f}_time_{include_time}_nsteps_{num_steps}_lr_{lr}"
    if model_name == "log_ncde" or model_name == "nrde":
        output_dir += f"_stepsize_{stepsize:.2f}_depth_{logsig_depth}"
    for k, v in model_args.items():
        name = str(v)
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

    dataset = create_dataset(
        data_dir,
        dataset_name,
        stepsize=stepsize,
        depth=logsig_depth,
        include_time=include_time,
        T=T,
        use_idxs=False,
        use_presplit=use_presplit,
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
    filter_spec = jax.tree_util.tree_map(lambda _: True, model)
    if model_name == "nrde" or model_name == "log_ncde":
        dataloaders = dataset.path_dataloaders
        if model_name == "log_ncde":
            where = lambda model: (model.intervals, model.pairs)
            filter_spec = eqx.tree_at(where, filter_spec, replace=(False, False))
        elif model_name == "nrde":
            where = lambda model: (model.intervals,)
            filter_spec = eqx.tree_at(where, filter_spec, replace=(False,))
    elif model_name == "ncde":
        dataloaders = dataset.coeff_dataloaders
    else:
        dataloaders = dataset.raw_dataloaders

    train_model(
        model,
        filter_spec,
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
    data_dir = "/data/math-datasig/shug6778/Log-Neural-CDEs/data"
    use_presplit = True
    output_parent_dir = ""
    seed = 1234
    num_steps = 10000
    print_steps = 100
    batch_size = 32
    lr = 1e-4
    lr_scheduler = lambda lr: lr
    T = 1
    dt0 = T / 10
    include_time = True
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)
    stepsize = 16
    logsig_depth = 2
    hidden_dim = 64
    scale = T * 1000
    lambd = 1e-6
    dataset_names = [
        "EigenWorms",
        # "EthanolConcentration",
        # "FaceDetection",
        # "FingerMovements",
        # "HandMovementDirection",
        # "Handwriting",
        # "Heartbeat",
        # "Libras",
        # "LSST",
        # "MotorImagery",
        # "NATOPS",
        # "PEMS-SF",
        # "PhonemeSpectra",
        # "SelfRegulationSCP1",
        # "SelfRegulationSCP2",
    ]
    model_names = ["rnn_lstm"]

    for dataset_name in dataset_names:
        for model_name in model_names:
            for include_time in [True, False]:
                for hidden_dim in [64]:
                    model_args = {
                        "num_blocks": 6,
                        "hidden_dim": hidden_dim,
                        "vf_depth": 3,
                        "vf_width": 64,
                        "ssm_dim": 32,
                        "ssm_blocks": 2,
                        "dt0": dt0,
                        "solver": solver,
                        "stepsize_controller": stepsize_controller,
                        "scale": scale,
                        "lambd": lambd,
                    }
                    create_dataset_model_and_train(
                        seed,
                        data_dir,
                        use_presplit,
                        dataset_name,
                        include_time,
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
                        output_parent_dir,
                    )
