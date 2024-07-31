"""
This module defines the `Dataset` class and functions for generating datasets tailored to different model types.
A `Dataset` object in this module contains three different dataloaders, each providing a specific version of the data
required by different models:

- `raw_dataloaders`: Returns the raw time series data, suitable for recurrent neural networks (RNNs) and structured
  state space models (SSMs).
- `coeff_dataloaders`: Provides the coefficients of an interpolation of the data, used by Neural Controlled Differential
  Equations (NCDEs).
- `path_dataloaders`: Provides the log-signature of the data over intervals, used by Neural Rough Differential Equations
  (NRDEs) and Log-NCDEs.

The module also includes utility functions for processing and generating these datasets, ensuring compatibility with
different model requirements.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from data_dir.dataloaders import Dataloader
from data_dir.generate_coeffs import calc_coeffs
from data_dir.generate_paths import calc_paths


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
        * jnp.repeat(jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0),
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
    if len(train_labels.shape) == 1 or name == "ppg":
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
    data_dir,
    name,
    use_idxs,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    *,
    key,
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
            train_data = jnp.concatenate([ts[:, :, None], train_data], axis=2)
            ts = (T / val_data.shape[1]) * jnp.repeat(
                jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0
            )
            val_data = jnp.concatenate([ts[:, :, None], val_data], axis=2)
            ts = (T / test_data.shape[1]) * jnp.repeat(
                jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
            )
            test_data = jnp.concatenate([ts[:, :, None], test_data], axis=2)
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


def create_toy_dataset(data_dir, name, stepsize, depth, include_time, T, *, key):
    with open(data_dir + "/processed/toy/signature/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(data_dir + "/processed/toy/signature/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    if name == "signature1":
        labels = ((jnp.sign(labels[0][:, 2]) + 1) / 2).astype(int)
    elif name == "signature2":
        labels = ((jnp.sign(labels[1][:, 2, 5]) + 1) / 2).astype(int)
    elif name == "signature3":
        labels = ((jnp.sign(labels[2][:, 2, 5, 0]) + 1) / 2).astype(int)
    elif name == "signature4":
        labels = ((jnp.sign(labels[3][:, 2, 5, 0, 3]) + 1) / 2).astype(int)
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


def create_ppg_dataset(
    data_dir, use_presplit, stepsize, depth, include_time, T, *, key
):
    with open(data_dir + "/processed/PPG/ppg/X_train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/y_train.pkl", "rb") as f:
        train_labels = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/X_val.pkl", "rb") as f:
        val_data = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/y_val.pkl", "rb") as f:
        val_labels = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/X_test.pkl", "rb") as f:
        test_data = pickle.load(f)
    with open(data_dir + "/processed/PPG/ppg/y_test.pkl", "rb") as f:
        test_labels = pickle.load(f)

    if include_time:
        ts = (T / train_data.shape[1]) * jnp.repeat(
            jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
        )
        train_data = jnp.concatenate([ts[:, :, None], train_data], axis=2)
        ts = (T / val_data.shape[1]) * jnp.repeat(
            jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0
        )
        val_data = jnp.concatenate([ts[:, :, None], val_data], axis=2)
        ts = (T / test_data.shape[1]) * jnp.repeat(
            jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
        )
        test_data = jnp.concatenate([ts[:, :, None], test_data], axis=2)

    if use_presplit:
        data = (train_data, val_data, test_data)
        labels = (train_labels, val_labels, test_labels)
    else:
        data = jnp.concatenate((train_data, val_data, test_data), axis=0)
        labels = jnp.concatenate((train_labels, val_labels, test_labels), axis=0)

    return dataset_generator(
        "ppg",
        data,
        labels,
        stepsize,
        depth,
        include_time,
        T,
        inmemory=False,
        use_presplit=use_presplit,
        key=key,
    )


def create_dataset(
    data_dir,
    name,
    use_idxs,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    *,
    key,
):
    uea_subfolders = [
        f.name for f in os.scandir(data_dir + "/processed/UEA") if f.is_dir()
    ]
    toy_subfolders = [
        f.name for f in os.scandir(data_dir + "/processed/toy") if f.is_dir()
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
    elif name[:-1] in toy_subfolders:
        return create_toy_dataset(
            data_dir, name, stepsize, depth, include_time, T, key=key
        )
    elif name == "ppg":
        return create_ppg_dataset(
            data_dir, use_presplit, stepsize, depth, include_time, T, key=key
        )
    else:
        raise ValueError(f"Dataset {name} not found in UEA folder and not toy dataset")
