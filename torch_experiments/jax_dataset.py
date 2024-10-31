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


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dir, name, train, val, test, indexes, presplit, include_time
    ):
        super().__init__()
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
            if include_time:
                ts = (1 / data.shape[1]) * np.repeat(
                    np.arange(data.shape[1])[None, :], data.shape[0], axis=0
                )
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
            if include_time:
                ts = (1 / data.shape[1]) * np.repeat(
                    np.arange(data.shape[1])[None, :], data.shape[0], axis=0
                )
                data = np.concatenate([ts[:, :, None], data], axis=2)
            data = torch.from_numpy(data).to(torch.float32)
            labels = torch.from_numpy(labels).to(torch.float32)
            assert len(indexes) == len(data)
            data = data[indexes]
            labels = labels[indexes]
            num_classes = len(torch.unique(labels))
            if train:
                data = data[: int(0.7 * len(data))]
                labels = labels[: int(0.7 * len(labels))]
            elif val:
                data = data[int(0.7 * len(data)) : int(0.85 * len(data))]
                labels = labels[int(0.7 * len(labels)) : int(0.85 * len(labels))]
            elif test:
                data = data[int(0.85 * len(data)) :]
                labels = labels[int(0.85 * len(labels)) :]
            self.data = data
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
