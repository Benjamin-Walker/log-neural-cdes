import os
import pickle
from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from data.dataloaders import Dataloader
from data.generate_coeffs import calc_coeffs
from data.generate_paths import calc_paths


@dataclass
class Dataset:
    name: str
    raw_dataloaders: Dict[str, Dataloader]
    coeff_dataloaders: Dict[str, Dataloader]
    path_dataloaders: Dict[str, Dataloader]
    data_dim: int
    logsig_dim: int
    intervals: jnp.ndarray
    label_dim: int


def batch_calc_paths(data, stepsize, depth, inmemory=True):
    N = len(data)
    batchsize = 128
    num_batches = N // batchsize
    remainder = N % batchsize
    path_data = []
    if inmemory:
        out_func = lambda x: x
        in_func = lambda x: x
    else:
        out_func = lambda x: np.array(x)
        in_func = lambda x: jnp.array(x)
    for i in range(num_batches):
        path_data.append(
            out_func(
                calc_paths(
                    in_func(data[i * batchsize : (i + 1) * batchsize]), stepsize, depth
                )
            )
        )
    if remainder > 0:
        path_data.append(
            out_func(calc_paths(in_func(data[-remainder:]), stepsize, depth))
        )
    if inmemory:
        path_data = jnp.concatenate(path_data)
    else:
        path_data = np.concatenate(path_data)
    return path_data


def batch_calc_coeffs(data, include_time, T, inmemory=True):
    N = len(data)
    batchsize = 128
    num_batches = N // batchsize
    remainder = N % batchsize
    coeffs = []
    if inmemory:
        out_func = lambda x: x
        in_func = lambda x: x
    else:
        out_func = lambda x: np.array(x)
        in_func = lambda x: jnp.array(x)
    for i in range(num_batches):
        coeffs.append(
            out_func(
                calc_coeffs(
                    in_func(data[i * batchsize : (i + 1) * batchsize]), include_time, T
                )
            )
        )
    if remainder > 0:
        coeffs.append(
            out_func(calc_coeffs(in_func(data[-remainder:]), include_time, T))
        )
    if inmemory:
        coeffs = jnp.concatenate(coeffs)
    else:
        coeffs = np.concatenate(coeffs)
    return coeffs


def dataset_generator(
    name,
    data,
    labels,
    stepsize,
    depth,
    include_time,
    T,
    inmemory=True,
    coeffs_needed=True,
    idxs=None,
    use_presplit=False,
    *,
    key,
):
    N = len(data)
    if idxs is None:
        if use_presplit:
            train_data, val_data, test_data = data
            train_labels, val_labels, test_labels = labels
        else:
            permkey, key = jr.split(key)
            bound1 = int(N * 0.7)
            bound2 = int(N * 0.85)
            idxs_new = jr.permutation(permkey, N)
            train_data, train_labels = (
                data[idxs_new[:bound1]],
                labels[idxs_new[:bound1]],
            )
            val_data, val_labels = (
                data[idxs_new[bound1:bound2]],
                labels[idxs_new[bound1:bound2]],
            )
            test_data, test_labels = data[idxs_new[bound2:]], labels[idxs_new[bound2:]]
    else:
        train_data, train_labels = data[idxs[0]], labels[idxs[0]]
        val_data, val_labels = data[idxs[1]], labels[idxs[1]]
        test_data, test_labels = None, None

    train_paths = batch_calc_paths(train_data, stepsize, depth)
    val_paths = batch_calc_paths(val_data, stepsize, depth)
    test_paths = batch_calc_paths(test_data, stepsize, depth)
    intervals = jnp.arange(0, train_data.shape[1], stepsize)
    intervals = jnp.concatenate((intervals, jnp.array([train_data.shape[1]])))
    intervals = intervals * (T / train_data.shape[1])

    if coeffs_needed:
        train_coeffs = calc_coeffs(train_data, include_time, T)
        val_coeffs = calc_coeffs(val_data, include_time, T)
        test_coeffs = calc_coeffs(test_data, include_time, T)
        train_coeff_data = (
            (T / train_data.shape[1])
            * jnp.repeat(
                jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
            ),
            train_coeffs,
            train_data[:, 0, :],
        )
        val_coeff_data = (
            (T / val_data.shape[1])
            * jnp.repeat(
                jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0
            ),
            val_coeffs,
            val_data[:, 0, :],
        )
        if idxs is None:
            test_coeff_data = (
                (T / test_data.shape[1])
                * jnp.repeat(
                    jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
                ),
                test_coeffs,
                test_data[:, 0, :],
            )
    else:
        train_coeff_data = None
        val_coeff_data = None
        test_coeff_data = None

    train_path_data = (
        (T / train_data.shape[1])
        * jnp.repeat(
            jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
        ),
        train_paths,
        train_data[:, 0, :],
    )
    val_path_data = (
        (T / val_data.shape[1])
        * jnp.repeat(jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0),
        val_paths,
        val_data[:, 0, :],
    )
    if idxs is None:
        test_path_data = (
            (T / test_data.shape[1])
            * jnp.repeat(
                jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
            ),
            test_paths,
            test_data[:, 0, :],
        )

    data_dim = train_data.shape[-1]
    if len(train_labels.shape) == 1:
        label_dim = 1
    else:
        label_dim = train_labels.shape[-1]
    logsig_dim = train_paths.shape[-1]

    raw_dataloaders = {
        "train": Dataloader(train_data, train_labels, inmemory),
        "val": Dataloader(val_data, val_labels, inmemory),
        "test": Dataloader(test_data, test_labels, inmemory),
    }
    coeff_dataloaders = {
        "train": Dataloader(train_coeff_data, train_labels, inmemory),
        "val": Dataloader(val_coeff_data, val_labels, inmemory),
        "test": Dataloader(test_coeff_data, test_labels, inmemory),
    }

    path_dataloaders = {
        "train": Dataloader(train_path_data, train_labels, inmemory),
        "val": Dataloader(val_path_data, val_labels, inmemory),
        "test": Dataloader(test_path_data, test_labels, inmemory),
    }
    return Dataset(
        name,
        raw_dataloaders,
        coeff_dataloaders,
        path_dataloaders,
        data_dim,
        logsig_dim,
        intervals,
        label_dim,
    )


def create_uea_dataset(
    data_dir, name, use_idxs, use_presplit, stepsize, depth, include_time, T, *, key
):

    if use_presplit:
        idxs = None
        with open(data_dir + f"/processed/UEA/{name}/X_train.pkl", "rb") as f:
            train_data = pickle.load(f)
        with open(data_dir + f"/processed/UEA/{name}/y_train.pkl", "rb") as f:
            train_labels = pickle.load(f)
        with open(data_dir + f"/processed/UEA/{name}/X_val.pkl", "rb") as f:
            val_data = pickle.load(f)
        with open(data_dir + f"/processed/UEA/{name}/y_val.pkl", "rb") as f:
            val_labels = pickle.load(f)
        with open(data_dir + f"/processed/UEA/{name}/X_test.pkl", "rb") as f:
            test_data = pickle.load(f)
        with open(data_dir + f"/processed/UEA/{name}/y_test.pkl", "rb") as f:
            test_labels = pickle.load(f)
        if include_time:
            ts = (T / train_data.shape[1]) * jnp.repeat(
                jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
            )
            train_data = jnp.concatenate([ts[:, :, None], train_data[:, :, 1:]], axis=2)
            ts = (T / val_data.shape[1]) * jnp.repeat(
                jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0
            )
            val_data = jnp.concatenate([ts[:, :, None], val_data[:, :, 1:]], axis=2)
            ts = (T / test_data.shape[1]) * jnp.repeat(
                jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
            )
            test_data = jnp.concatenate([ts[:, :, None], test_data[:, :, 1:]], axis=2)
        else:
            train_data = train_data[:, :, 1:]
            val_data = val_data[:, :, 1:]
            test_data = test_data[:, :, 1:]
        data = (train_data, val_data, test_data)
        onehot_labels = (train_labels, val_labels, test_labels)
    else:
        with open(data_dir + f"/processed/UEA/{name}/data.pkl", "rb") as f:
            data = pickle.load(f)
        with open(data_dir + f"/processed/UEA/{name}/labels.pkl", "rb") as f:
            labels = pickle.load(f)
        onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
        onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)
        if use_idxs:
            with open(data_dir + f"/processed/UEA/{name}/original_idxs.pkl", "rb") as f:
                idxs = pickle.load(f)
        else:
            idxs = None

        if include_time:
            ts = (T / data.shape[1]) * jnp.repeat(
                jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
            )
            data = jnp.concatenate([ts[:, :, None], data], axis=2)

    return dataset_generator(
        name,
        data,
        onehot_labels,
        stepsize,
        depth,
        include_time,
        T,
        idxs=idxs,
        use_presplit=use_presplit,
        key=key,
    )


def create_fex_dataset(
    data_dir, name, use_presplit, stepsize, depth, include_time, T, *, key
):
    if use_presplit:
        raise ValueError("FEX datasets do not have presplit data")

    with open(data_dir + f"/processed/FEX/{name}/data.pkl", "rb") as f:
        data = jnp.array(pickle.load(f))
    with open(data_dir + f"/processed/FEX/{name}/labels.pkl", "rb") as f:
        labels = jnp.array(pickle.load(f))
    idxs = jnp.arange(len(data))
    key, subkey = jr.split(key)
    shuffle = jr.permutation(subkey, idxs, independent=True)
    data = data[shuffle]
    labels = labels[shuffle]
    onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
    onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)
    idxs = None

    return dataset_generator(
        name,
        data,
        onehot_labels,
        stepsize,
        depth,
        include_time,
        T,
        idxs=idxs,
        use_presplit=use_presplit,
        key=key,
    )


def create_lra_dataset(
    data_dir, name, use_idxs, use_presplit, stepsize, depth, include_time, T, *, key
):
    if use_presplit:
        idxs = None
        with open(data_dir + f"/processed/LRA/{name}/X_train.pkl", "rb") as f:
            train_data = pickle.load(f)
        with open(data_dir + f"/processed/LRA/{name}/y_train.pkl", "rb") as f:
            train_labels = pickle.load(f)
        with open(data_dir + f"/processed/LRA/{name}/X_val.pkl", "rb") as f:
            val_data = pickle.load(f)
        with open(data_dir + f"/processed/LRA/{name}/y_val.pkl", "rb") as f:
            val_labels = pickle.load(f)
        with open(data_dir + f"/processed/LRA/{name}/X_test.pkl", "rb") as f:
            test_data = pickle.load(f)
        with open(data_dir + f"/processed/LRA/{name}/y_test.pkl", "rb") as f:
            test_labels = pickle.load(f)
        ts = (T / train_data.shape[1]) * jnp.repeat(
            jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
        )
        train_data = jnp.concatenate([ts[:, :, None], train_data[:, :, 1:]], axis=2)
        ts = (T / val_data.shape[1]) * jnp.repeat(
            jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0
        )
        val_data = jnp.concatenate([ts[:, :, None], val_data[:, :, 1:]], axis=2)
        ts = (T / test_data.shape[1]) * jnp.repeat(
            jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
        )
        test_data = jnp.concatenate([ts[:, :, None], test_data[:, :, 1:]], axis=2)
        data = (train_data, val_data, test_data)
        onehot_labels = (train_labels, val_labels, test_labels)
    else:
        with open(data_dir + f"/processed/LRA/{name}/data.pkl", "rb") as f:
            data = pickle.load(f)
        with open(data_dir + f"/processed/LRA/{name}/labels.pkl", "rb") as f:
            labels = pickle.load(f)
        onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
        onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)
        if use_idxs:
            with open(data_dir + f"/processed/LRA/{name}/original_idxs.pkl", "rb") as f:
                idxs = pickle.load(f)
        else:
            idxs = None

    return dataset_generator(
        name,
        data,
        onehot_labels,
        stepsize,
        depth,
        include_time,
        T,
        idxs=idxs,
        use_presplit=use_presplit,
        key=key,
    )


def create_toy_dataset(data_dir, stepsize, depth, include_time, T, *, key):
    with open(data_dir + "/processed/toy/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(data_dir + "/processed/toy/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    labels = ((jnp.sign(labels[3][:, 2, 5, 0, 3]) + 1) / 2).astype(int)  # 2,5,0,3
    onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
    onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)
    idxs = None

    if include_time:
        ts = (T / data.shape[1]) * jnp.repeat(
            jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
        )
        data = jnp.concatenate([ts[:, :, None], data], axis=2)

    return dataset_generator(
        "toy", data, onehot_labels, stepsize, depth, include_time, T, idxs, key=key
    )


def create_speech_dataset(data_dir, stepsize, depth, include_time, T, *, key):
    data = []
    labels = []
    for i in range(10):
        data.append(np.load(data_dir + f"/processed/speech/data_{i}.npy"))
        labels.append(np.load(data_dir + f"/processed/speech/labels_{i}.npy"))
    data = np.concatenate(data)
    labels = np.concatenate(labels)

    return dataset_generator(
        "speech",
        data,
        labels,
        stepsize,
        depth,
        include_time,
        T,
        inmemory=False,
        coeffs_needed=False,
        use_presplit=False,
        key=key,
    )


def create_ppg_dataset(data_dir, stepsize, depth, include_time, T, *, key):
    with open(data_dir + "/processed/PPG/X_train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(data_dir + "/processed/PPG/y_train.pkl", "rb") as f:
        train_labels = pickle.load(f)
    with open(data_dir + "/processed/PPG/X_val.pkl", "rb") as f:
        val_data = pickle.load(f)
    with open(data_dir + "/processed/PPG/y_val.pkl", "rb") as f:
        val_labels = pickle.load(f)
    with open(data_dir + "/processed/PPG/X_test.pkl", "rb") as f:
        test_data = pickle.load(f)
    with open(data_dir + "/processed/PPG/y_test.pkl", "rb") as f:
        test_labels = pickle.load(f)

    data = (train_data, val_data, test_data)
    labels = (train_labels, val_labels, test_labels)

    return dataset_generator(
        "speech",
        data,
        labels,
        stepsize,
        depth,
        include_time,
        T,
        inmemory=False,
        coeffs_needed=False,
        use_presplit=True,
        key=key,
    )


def create_dataset(
    data_dir, name, use_idxs, use_presplit, stepsize, depth, include_time, T, *, key
):
    uea_subfolders = [
        f.name for f in os.scandir(data_dir + "/processed/UEA") if f.is_dir()
    ]
    lra_subfolders = [
        f.name for f in os.scandir(data_dir + "/processed/LRA") if f.is_dir()
    ]
    fex_subfolders = [
        f.name for f in os.scandir(data_dir + "/processed/FEX") if f.is_dir()
    ]
    if name in uea_subfolders:
        return create_uea_dataset(
            data_dir,
            name,
            use_idxs,
            use_presplit,
            stepsize,
            depth,
            include_time,
            T,
            key=key,
        )
    elif name in lra_subfolders:
        return create_lra_dataset(
            data_dir,
            name,
            use_idxs,
            use_presplit,
            stepsize,
            depth,
            include_time,
            T,
            key=key,
        )
    elif name in fex_subfolders:
        return create_fex_dataset(
            data_dir,
            name,
            use_presplit,
            stepsize,
            depth,
            include_time,
            T,
            key=key,
        )
    elif name == "toy":
        return create_toy_dataset(data_dir, stepsize, depth, include_time, T, key=key)
    elif name == "speech":
        return create_speech_dataset(
            data_dir, stepsize, depth, include_time, T, key=key
        )
    elif name == "ppg":
        return create_ppg_dataset(data_dir, stepsize, depth, include_time, T, key=key)
    else:
        raise ValueError(f"Dataset {name} not found in UEA folder and not toy dataset")
