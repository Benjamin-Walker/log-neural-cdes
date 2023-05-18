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
    label_dim: int


def dataset_generator(name, data, labels, idxs=None, *, key):
    path_data = calc_paths(data)
    coeff_data = calc_coeffs(data)

    if idxs is None:
        permkey, key = jr.split(key)
        N = len(data)
        bound1 = int(N * 0.7)
        bound2 = int(N * 0.85)
        idxs = jr.permutation(permkey, N)
        train_data, train_labels = data[idxs[:bound1]], labels[idxs[:bound1]]
        train_path_data = path_data[idxs[:bound1]]
        train_coeff_data = tuple(data[idxs[:bound1]] for data in coeff_data)
        val_data, val_labels = data[idxs[bound1:bound2]], labels[idxs[bound1:bound2]]
        val_path_data = path_data[idxs[bound1:bound2]]
        val_coeff_data = tuple(data[idxs[bound1:bound2]] for data in coeff_data)
        test_data, test_labels = data[idxs[bound2:]], labels[idxs[bound2:]]
        test_path_data = path_data[idxs[bound2:]]
        test_coeff_data = tuple(data[idxs[bound2:]] for data in coeff_data)
    else:
        train_data, train_labels = data[idxs[0]], labels[idxs[0]]
        train_path_data = path_data[idxs[0]]
        train_coeff_data = tuple(data[idxs[0]] for data in coeff_data)
        val_data, val_labels = data[idxs[1]], labels[idxs[1]]
        val_path_data = path_data[idxs[1]]
        val_coeff_data = tuple(data[idxs[1]] for data in coeff_data)
        test_data, test_labels = None, None
        test_path_data = None
        test_coeff_data = None

    data_dim = train_data.shape[-1]
    if len(labels.shape) == 1:
        label_dim = 1
    else:
        label_dim = labels.shape[-1]

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
        name, raw_dataloaders, coeff_dataloaders, path_dataloaders, data_dim, label_dim
    )


def create_uea_dataset(name, use_idxs, *, key):
    subfolders = [f.name for f in os.scandir("data/processed/UEA") if f.is_dir()]
    if name not in subfolders:
        raise ValueError(f"Dataset {name} not found in UEA folder")

    with open(f"data/processed/UEA/{name}/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(f"data/processed/UEA/{name}/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
    onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)
    if use_idxs:
        with open(f"data/processed/UEA/{name}/original_idxs.pkl", "rb") as f:
            idxs = pickle.load(f)
    else:
        idxs = None

    ts = jnp.repeat(jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0)
    data = jnp.concatenate([ts[:, :, None], data], axis=2)

    return dataset_generator(name, data, onehot_labels, idxs, key=key)
