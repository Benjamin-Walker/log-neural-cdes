import os
import pickle
from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp
import jax.random as jr

from data.dataloaders import InMemoryDataloader
from data.generate_coeffs import calc_coeffs
from data.generate_paths import calc_paths


@dataclass
class Dataset:
    name: str
    raw_dataloaders: Dict[str, InMemoryDataloader]
    coeff_dataloaders: Dict[str, InMemoryDataloader]
    path_dataloaders: Dict[str, InMemoryDataloader]
    data_dim: int
    logsig_dim: int
    intervals: jnp.ndarray
    label_dim: int


def dataset_generator(
    name, data, labels, stepsize, depth, include_time, idxs=None, *, key
):
    N = len(data)
    batchsize = 128
    num_batches = N // batchsize
    remainder = N % batchsize
    path_data = []
    for i in range(num_batches):
        path_data.append(
            calc_paths(
                data[i * batchsize : (i + 1) * batchsize], stepsize, depth, include_time
            )
        )
    if remainder > 0:
        path_data.append(calc_paths(data[-remainder:], stepsize, depth, include_time))
    path_data = jnp.concatenate(path_data)
    intervals = jnp.arange(0, data.shape[1], stepsize)
    intervals = jnp.concatenate((intervals, jnp.array([data.shape[1]])))

    coeff_data = calc_coeffs(data)

    if idxs is None:
        permkey, key = jr.split(key)
        bound1 = int(N * 0.7)
        bound2 = int(N * 0.85)
        idxs = jr.permutation(permkey, N)
        train_data, train_labels = data[idxs[:bound1]], labels[idxs[:bound1]]
        train_path_data = (
            data[idxs[:bound1], :, 0],
            path_data[idxs[:bound1]],
            data[idxs[:bound1], 0, :],
        )
        train_coeff_data = (
            data[idxs[:bound1], :, 0],
            tuple(data[idxs[:bound1]] for data in coeff_data),
            data[idxs[:bound1], 0, :],
        )
        val_data, val_labels = data[idxs[bound1:bound2]], labels[idxs[bound1:bound2]]
        val_path_data = (
            data[idxs[bound1:bound2], :, 0],
            path_data[idxs[bound1:bound2]],
            data[idxs[bound1:bound2], 0, :],
        )
        val_coeff_data = (
            data[idxs[bound1:bound2], :, 0],
            tuple(data[idxs[bound1:bound2]] for data in coeff_data),
            data[idxs[bound1:bound2], 0, :],
        )
        test_data, test_labels = data[idxs[bound2:]], labels[idxs[bound2:]]
        test_path_data = (
            data[idxs[bound2:], :, 0],
            path_data[idxs[bound2:]],
            data[idxs[bound2:], 0, :],
        )
        test_coeff_data = (
            data[idxs[bound2:], :, 0],
            tuple(data[idxs[bound2:]] for data in coeff_data),
            data[idxs[bound2:], 0, :],
        )
    else:
        train_data, train_labels = data[idxs[0]], labels[idxs[0]]
        train_path_data = (
            data[idxs[0], :, 0],
            path_data[idxs[0]],
            data[idxs[0], 0, :],
        )
        train_coeff_data = (
            data[idxs[0], :, 0],
            tuple(data[idxs[0]] for data in coeff_data),
            data[idxs[0], 0, :],
        )
        val_data, val_labels = data[idxs[1]], labels[idxs[1]]
        val_path_data = (
            data[idxs[1], :, 0],
            path_data[idxs[1]],
            data[idxs[1], 0, :],
        )
        val_coeff_data = (
            data[idxs[1], :, 0],
            tuple(data[idxs[1]] for data in coeff_data),
            data[idxs[1], 0, :],
        )
        test_data, test_labels = None, None
        test_path_data = None
        test_coeff_data = None

    data_dim = train_data.shape[-1]
    if len(labels.shape) == 1:
        label_dim = 1
    else:
        label_dim = labels.shape[-1]
    logsig_dim = path_data.shape[-1]

    raw_dataloaders = {
        "train": InMemoryDataloader(train_data, train_labels),
        "val": InMemoryDataloader(val_data, val_labels),
        "test": InMemoryDataloader(test_data, test_labels),
    }

    coeff_dataloaders = {
        "train": InMemoryDataloader(train_coeff_data, train_labels),
        "val": InMemoryDataloader(val_coeff_data, val_labels),
        "test": InMemoryDataloader(test_coeff_data, test_labels),
    }

    path_dataloaders = {
        "train": InMemoryDataloader(train_path_data, train_labels),
        "val": InMemoryDataloader(val_path_data, val_labels),
        "test": InMemoryDataloader(test_path_data, test_labels),
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
    data_dir, name, use_idxs, stepsize, depth, include_time, T, *, key
):
    subfolders = [f.name for f in os.scandir(data_dir + "/processed/UEA") if f.is_dir()]
    if name not in subfolders:
        raise ValueError(f"Dataset {name} not found in UEA folder")

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

    ts = (
        T
        * jnp.repeat(jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0)
        / (data.shape[1])
    )
    data = jnp.concatenate([ts[:, :, None], data], axis=2)

    return dataset_generator(
        name, data, onehot_labels, stepsize, depth, include_time, idxs, key=key
    )
