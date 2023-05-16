import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict

import jax.random as jr

from data.dataloaders import InMemoryDataloader
from data.generate_paths import calc_paths


@dataclass
class Dataset:
    name: str
    raw_dataloaders: Dict[str, Any]
    path_dataloaders: Dict[str, Any]
    data_dim: int
    label_dim: int


def dataset_generator(name, data, labels, idxs=None, *, key):

    path_data = calc_paths(data)

    if idxs is None:
        permkey, key = jr.split(key)
        N = len(data)
        bound1 = int(N * 0.7)
        bound2 = int(N * 0.85)
        idxs = jr.permutation(permkey, N)
        train_data, train_labels = data[idxs[:bound1]], labels[idxs[:bound1]]
        train_path_data = path_data[idxs[:bound1]]
        val_data, val_labels = data[idxs[bound1:bound2]], labels[idxs[bound1:bound2]]
        val_path_data = path_data[idxs[bound1:bound2]]
        test_data, test_labels = data[idxs[bound2:]], labels[idxs[bound2:]]
        test_path_data = path_data[idxs[bound2:]]
    else:
        train_data, train_labels = data[idxs[0]], labels[idxs[0]]
        train_path_data = path_data[idxs[0]]
        val_data, val_labels = data[idxs[1]], labels[idxs[1]]
        val_path_data = path_data[idxs[1]]
        test_data, test_labels = None, None
        test_path_data = None

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

    path_dataloaders = {
        "train": InMemoryDataloader(train_path_data, train_labels),
        "val": InMemoryDataloader(val_path_data, val_labels),
        "test": InMemoryDataloader(test_path_data, test_labels),
    }

    return Dataset(name, raw_dataloaders, path_dataloaders, data_dim, label_dim)


def create_uea_dataset(name, use_idxs, *, key):
    subfolders = [f.name for f in os.scandir("processed/UEA") if f.is_dir()]
    if name not in subfolders:
        raise ValueError(f"Dataset {name} not found in UEA folder")

    with open(f"processed/UEA/{name}/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(f"processed/UEA/{name}/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    if use_idxs:
        with open(f"processed/UEA/{name}/original_idxs.pkl", "rb") as f:
            idxs = pickle.load(f)
    else:
        idxs = None

    return dataset_generator(name, data, labels, idxs, key=key)
