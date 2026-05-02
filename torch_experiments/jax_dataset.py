"""
This module defines a custom PyTorch `Dataset` class for loading and processing time series data 
from different benchmarks (UEA, toy, PPG) which have been preprocessed and saved as Jax numpy arrays.
The dataset can be pre-split into training, validation, and test sets, or dynamically split based on provided indexes.

Classes:
- `Dataset`: A PyTorch dataset class that handles loading data and labels from pickle files of jax numpy arrays,
  optional inclusion of time as a feature, and splitting of data into train/val/test sets.

Methods:
- `__len__`: Returns the length of the dataset.
- `__getitem__`: Retrieves a data-label pair at the specified index.
"""

import os
import pickle

import numpy as np
import torch


def _uniform_times(length):
    return (1.0 / length) * np.arange(length, dtype=np.float32)


def _drop_observations(data, observation_times, drop_percentage, *, seed):
    if drop_percentage is None:
        return data, observation_times

    if not 0.0 <= drop_percentage < 1.0:
        raise ValueError("drop_percentage must satisfy 0.0 <= drop_percentage < 1.0")

    length = data.shape[1]
    keep = int(round(length * (1.0 - drop_percentage)))
    keep = min(length, max(2, keep))

    if keep == length:
        return data, observation_times

    rng = np.random.default_rng(seed)
    indices = np.empty((data.shape[0], keep), dtype=np.int64)
    for i in range(data.shape[0]):
        if keep == 2:
            indices[i] = np.array([0, length - 1], dtype=np.int64)
            continue
        middle = rng.choice(length - 2, size=keep - 2, replace=False) + 1
        indices[i] = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.sort(middle.astype(np.int64)),
                np.array([length - 1], dtype=np.int64),
            )
        )

    batch_indices = np.arange(data.shape[0])[:, None]
    dropped_data = data[batch_indices, indices]
    if observation_times is None:
        return dropped_data, None
    dropped_times = observation_times[batch_indices, indices]
    return dropped_data, dropped_times


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        name,
        train,
        val,
        test,
        indexes,
        presplit,
        include_time,
        drop_percentage=None,
        drop_seed=0,
    ):
        super().__init__()
        indexes = np.asarray(indexes)

        uea_subfolders = [
            f.name for f in os.scandir(data_dir + "/processed/UEA") if f.is_dir()
        ]
        toy_subfolders = [
            f.name for f in os.scandir(data_dir + "/processed/toy") if f.is_dir()
        ]
        ppg_subfolders = [
            f.name for f in os.scandir(data_dir + "/processed/PPG") if f.is_dir()
        ]
        if name in uea_subfolders:
            benchmark = "UEA"
        elif name[:-1] in toy_subfolders:
            benchmark = "toy"
        elif name in ppg_subfolders:
            benchmark = "PPG"
        else:
            raise ValueError("Benchmark not found")

        if presplit:
            if train:
                with open(
                    data_dir + f"/processed/{benchmark}/{name}/X_train.pkl", "rb"
                ) as f:
                    data = np.array(pickle.load(f))
                with open(
                    data_dir + f"/processed/{benchmark}/{name}/y_train.pkl", "rb"
                ) as f:
                    labels = np.array(pickle.load(f))
            elif val:
                with open(
                    data_dir + f"/processed/{benchmark}/{name}/X_val.pkl", "rb"
                ) as f:
                    data = np.array(pickle.load(f))
                with open(
                    data_dir + f"/processed/{benchmark}/{name}/y_val.pkl", "rb"
                ) as f:
                    labels = np.array(pickle.load(f))
            elif test:
                with open(
                    data_dir + f"/processed/{benchmark}/{name}/X_test.pkl", "rb"
                ) as f:
                    data = np.array(pickle.load(f))
                with open(
                    data_dir + f"/processed/{benchmark}/{name}/y_test.pkl", "rb"
                ) as f:
                    labels = np.array(pickle.load(f))
            ts = None
            if include_time or drop_percentage is not None:
                ts = np.repeat(
                    _uniform_times(data.shape[1])[None, :], data.shape[0], axis=0
                )
            if drop_percentage is not None:
                data, ts = _drop_observations(data, ts, drop_percentage, seed=drop_seed)
            if include_time:
                data = np.concatenate([ts[:, :, None], data], axis=2)

            self.data = torch.from_numpy(data).to(torch.float32)
            self.labels = torch.from_numpy(labels).to(torch.float32)
        else:
            if name[:-1] == "signature":
                name_dir = name[:-1]
            else:
                name_dir = name
            with open(
                data_dir + f"/processed/{benchmark}/{name_dir}/data.pkl",
                "rb",
            ) as f:
                data = np.array(pickle.load(f))
            with open(
                data_dir + f"/processed/{benchmark}/{name_dir}/labels.pkl",
                "rb",
            ) as f:
                if benchmark == "toy":
                    labels = pickle.load(f)
                    if name == "signature1":
                        labels = ((np.sign(labels[0][:, 2]) + 1) / 2).astype(int)
                    elif name == "signature2":
                        labels = ((np.sign(labels[1][:, 2, 5]) + 1) / 2).astype(int)
                    elif name == "signature3":
                        labels = ((np.sign(labels[2][:, 2, 5, 0]) + 1) / 2).astype(int)
                    elif name == "signature4":
                        labels = ((np.sign(labels[3][:, 2, 5, 0, 3]) + 1) / 2).astype(
                            int
                        )
                    labels = np.array(labels)
                else:
                    labels = np.array(pickle.load(f))
            ts = None
            if include_time or drop_percentage is not None:
                ts = np.repeat(
                    _uniform_times(data.shape[1])[None, :], data.shape[0], axis=0
                )
            assert len(indexes) == len(data)
            data = data[indexes]
            labels = labels[indexes]
            if ts is not None:
                ts = ts[indexes]
            num_classes = len(np.unique(labels))
            if train:
                data = data[: int(0.7 * len(data))]
                labels = labels[: int(0.7 * len(labels))]
                if ts is not None:
                    ts = ts[: int(0.7 * len(ts))]
            elif val:
                data = data[int(0.7 * len(data)) : int(0.85 * len(data))]
                labels = labels[int(0.7 * len(labels)) : int(0.85 * len(labels))]
                if ts is not None:
                    ts = ts[int(0.7 * len(ts)) : int(0.85 * len(ts))]
            elif test:
                data = data[int(0.85 * len(data)) :]
                labels = labels[int(0.85 * len(labels)) :]
                if ts is not None:
                    ts = ts[int(0.85 * len(ts)) :]
            if drop_percentage is not None:
                data, ts = _drop_observations(data, ts, drop_percentage, seed=drop_seed)
            if include_time:
                data = np.concatenate([ts[:, :, None], data], axis=2)
            self.data = torch.from_numpy(data).to(torch.float32)
            labels = torch.from_numpy(labels).to(torch.float32)
            self.labels = torch.nn.functional.one_hot(
                labels.to(torch.int64), num_classes
            ).to(torch.float32)
        self.length = len(self.data)
        self.input_dim = self.data.shape[2]
        self.output_dim = self.labels.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
