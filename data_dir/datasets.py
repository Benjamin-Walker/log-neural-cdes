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

import jax
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
    buf_len: int = None


def batch_calc_paths(
    data,
    stepsize,
    depth,
    inmemory=True,
    include_time=False,
    interval_times=None,
):
    N = len(data)
    batchsize = 128
    num_batches = N // batchsize
    remainder = N % batchsize
    path_data = []
    obs_masks = []
    if inmemory:
        out_func = lambda x: x
        in_func = lambda x: x
    else:
        out_func = lambda x: np.array(x)
        in_func = lambda x: jnp.array(x)
    for i in range(num_batches):
        logsigs, obs_mask = out_func(
            calc_paths(
                in_func(data[i * batchsize : (i + 1) * batchsize]),
                stepsize,
                depth,
                include_time,
                0,
                (
                    interval_times[i * batchsize : (i + 1) * batchsize]
                    if interval_times is not None
                    else None
                ),
            )
        )
        path_data.append(logsigs)
        obs_masks.append(obs_mask)
    if remainder > 0:
        logsig, obs_mask = out_func(
            calc_paths(
                in_func(data[-remainder:]),
                stepsize,
                depth,
                include_time,
                0,
                interval_times[-remainder:] if interval_times is not None else None,
            )
        )
        path_data.append(logsig)
        obs_masks.append(obs_mask)
    if inmemory:
        path_data = jnp.concatenate(path_data)
        obs_masks = jnp.concatenate(obs_masks)
    else:
        path_data = np.concatenate(path_data)
        obs_masks = np.concatenate(obs_masks)
    return path_data, obs_masks


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


def max_false_run_length_1d(obs_mask) -> int:
    """
    Max length of consecutive False values in a 1D obs_mask.
    obs_mask: shape (T,) array-like, True = observation interval.
    """
    m = np.asarray(obs_mask, dtype=bool).reshape(-1)  # force 1D
    f = ~m
    if f.size == 0 or not f.any():
        return 0

    padded = np.concatenate(([False], f, [False]))
    changes = np.diff(padded.astype(np.int8))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    return int((ends - starts).max())


def max_false_run_length(obs_masks) -> int:
    """
    Max length of consecutive False values across:
      - a single 1D mask (T,)
      - a batch of masks (N, T)
      - a list/tuple of 1D masks
    """
    arr = np.asarray(obs_masks, dtype=bool)

    if arr.ndim == 1:
        return max_false_run_length_1d(arr)

    if arr.ndim == 2:
        # max over samples
        return max(max_false_run_length_1d(arr[i]) for i in range(arr.shape[0]))

    # If you accidentally pass something like (N, T, 1), squeeze it.
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        return max_false_run_length_1d(arr)
    if arr.ndim == 2:
        return max(max_false_run_length_1d(arr[i]) for i in range(arr.shape[0]))

    raise ValueError(
        f"obs_masks must be 1D or 2D, got shape {np.asarray(obs_masks).shape}"
    )


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
    interval_times=None,
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

    if include_time:
        ts_train = train_data[:, :, 0]
        ts_val = val_data[:, :, 0]
        ts_test = test_data[:, :, 0]
    else:
        ts_train = (T / train_data.shape[1]) * jnp.repeat(
            jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
        )
        ts_val = (T / val_data.shape[1]) * jnp.repeat(
            jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0
        )
        ts_test = (T / test_data.shape[1]) * jnp.repeat(
            jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
        )

    if not stepsize:
        interval_times_train, interval_times_val, interval_times_test = interval_times
    else:
        interval_times_train = None
        interval_times_val = None
        interval_times_test = None

    train_paths, train_obs_masks = batch_calc_paths(
        train_data,
        stepsize,
        depth,
        inmemory,
        include_time,
        interval_times_train,
    )
    val_paths, val_obs_masks = batch_calc_paths(
        val_data,
        stepsize,
        depth,
        inmemory,
        include_time,
        interval_times_val,
    )
    test_paths, test_obs_masks = batch_calc_paths(
        test_data,
        stepsize,
        depth,
        inmemory,
        include_time,
        interval_times_test,
    )
    train_buf_len = max_false_run_length(train_obs_masks)
    val_buf_len = max_false_run_length(val_obs_masks)
    test_buf_len = max_false_run_length(test_obs_masks)
    buf_len = max(train_buf_len, val_buf_len, test_buf_len) + 1
    indexes = np.unique(np.r_[0 : train_data.shape[1] : stepsize])
    intervals = ts_train[0, indexes]
    intervals = jnp.concatenate((intervals, jnp.array([T])))

    train_coeffs = calc_coeffs(train_data, include_time, T)
    val_coeffs = calc_coeffs(val_data, include_time, T)
    test_coeffs = calc_coeffs(test_data, include_time, T)
    train_coeff_data = (
        ts_train,
        train_coeffs,
        train_data[:, 0, :],
    )
    val_coeff_data = (
        ts_val,
        val_coeffs,
        val_data[:, 0, :],
    )
    if idxs is None:
        test_coeff_data = (
            ts_test,
            test_coeffs,
            test_data[:, 0, :],
        )

    train_path_data = (
        interval_times_train if interval_times_train is not None else ts_train,
        (train_paths, train_obs_masks),
        train_data[:, 0, :],
    )
    val_path_data = (
        interval_times_val if interval_times_val is not None else ts_val,
        (val_paths, val_obs_masks),
        val_data[:, 0, :],
    )
    if idxs is None:
        test_path_data = (
            interval_times_test if interval_times_test is not None else ts_test,
            (test_paths, test_obs_masks),
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
        buf_len,
    )


def _scale_to_minus_one_one(x, data_min, data_max, eps=1e-8):
    """Affine-maps x from [data_min,data_max] → [-1,1] with broadcasting."""
    return 2.0 * (x - data_min) / (data_max - data_min + eps) - 1.0


def create_uea_dataset(
    data_dir,
    name,
    use_idxs,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    scale=False,
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
        t = (T / train_data.shape[1]) * jnp.arange(train_data.shape[1])[None, :]
        if include_time:
            ts = jnp.repeat(t, train_data.shape[0], axis=0)
            train_data = jnp.concatenate([ts[:, :, None], train_data], axis=2)
            ts = jnp.repeat(t, val_data.shape[0], axis=0)
            val_data = jnp.concatenate([ts[:, :, None], val_data], axis=2)
            ts = jnp.repeat(t, test_data.shape[0], axis=0)
            test_data = jnp.concatenate([ts[:, :, None], test_data], axis=2)
        data = (train_data, val_data, test_data)
        onehot_labels = (train_labels, val_labels, test_labels)
    else:
        with open(data_dir + f"/processed/UEA/{name}/data.pkl", "rb") as f:
            data = pickle.load(f)
        with open(data_dir + f"/processed/UEA/{name}/labels.pkl", "rb") as f:
            labels = pickle.load(f)
        t = (T / data.shape[1]) * jnp.arange(data.shape[1])[None, :]
        onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
        onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)
        if use_idxs:
            with open(data_dir + f"/processed/UEA/{name}/original_idxs.pkl", "rb") as f:
                idxs = pickle.load(f)
        else:
            idxs = None

        if include_time:
            ts = jnp.repeat(t, data.shape[0], axis=0)
            data = jnp.concatenate([ts[:, :, None], data], axis=2)

    if scale:
        if use_presplit:
            # stack (N,L,C) arrays along N to get all samples
            all_data = jnp.concatenate([train_data, val_data, test_data], axis=0)
            data_min = all_data.min(axis=(0, 1), keepdims=True)
            data_max = all_data.max(axis=(0, 1), keepdims=True)

            train_data = _scale_to_minus_one_one(train_data, data_min, data_max)
            val_data = _scale_to_minus_one_one(val_data, data_min, data_max)
            test_data = _scale_to_minus_one_one(test_data, data_min, data_max)
        else:
            data_min = data.min(axis=(0, 1), keepdims=True)
            data_max = data.max(axis=(0, 1), keepdims=True)
            data = _scale_to_minus_one_one(data, data_min, data_max)

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


def create_PM_dataset(
    data_dir,
    name,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    drop_percentage,
    drop_mode,
    *,
    key,
):
    """
    PM sequence regression dataset.

    Dataset names:
      - "pm2.5" -> processed/PM/25/
      - "pm10"  -> processed/PM/10/

    X saved format (from preprocessing):
      X_*: (N, L, 11) where X[..., 0] is sample_number
      y_*: (N, L) or (N, L, 1)

    Behaviour:
      - include_time=False: drop sample_number channel -> X becomes (N, L, 10)
      - include_time=True: replace sample_number with time in [0,T] -> X becomes (N, L, 11)
          where X[..., 0] is time in [0,T] and remaining 10 are features

    """
    if not use_presplit:
        raise ValueError(
            "PM dataset requires use_presplit=True (already saved as train/val/test)."
        )

    name_lower = name.lower()
    if name_lower == "pm25":
        task_dir = "25"
    elif name_lower == "pm10":
        task_dir = "10"
    else:
        raise ValueError("PM dataset name must be exactly 'pm25' or 'pm10'.")

    base = os.path.join(data_dir, f"processed/PM/{task_dir}")

    # Load arrays
    X_train = jnp.asarray(np.load(os.path.join(base, "X_train.npy")))
    X_val = jnp.asarray(np.load(os.path.join(base, "X_val.npy")))
    X_test = jnp.asarray(np.load(os.path.join(base, "X_test.npy")))

    y_train = jnp.asarray(np.load(os.path.join(base, "y_train.npy")))
    y_val = jnp.asarray(np.load(os.path.join(base, "y_val.npy")))
    y_test = jnp.asarray(np.load(os.path.join(base, "y_test.npy")))

    # Ensure y is (N, L, 1)
    if y_train.ndim == 2:
        y_train = y_train[:, :, None]
    if y_val.ndim == 2:
        y_val = y_val[:, :, None]
    if y_test.ndim == 2:
        y_test = y_test[:, :, None]

    if X_train.ndim != 3:
        raise ValueError(f"Expected X_train ndim=3, got {X_train.shape}")
    if y_train.ndim != 3:
        raise ValueError(f"Expected y_train ndim=3, got {y_train.shape}")
    if X_train.shape[1] != y_train.shape[1]:
        raise ValueError(
            f"X and y length mismatch: {X_train.shape[1]} vs {y_train.shape[1]}"
        )

    eps = 1e-8

    # ---------------------------------------------------------------------
    # Drop sample_number if include_time=False
    # Replace sample_number by time in [0,T] if include_time=True
    # ---------------------------------------------------------------------
    def _transform_X(X):
        # X: (N, L, 11), channel 0 is sample_number
        sample = X[:, :, 0]  # (N, L)
        feats = X[:, :, 1:]  # (N, L, 10)

        if not include_time:
            return feats

        # time := affine map of sample_number to [0,T] per-window
        s0 = sample[:, :1]  # (N, 1)
        sT = sample[:, -1:]  # (N, 1)
        denom = sT - s0  # (N, 1)

        # fallback to uniform grid if denom==0 for some reason
        L = X.shape[1]
        uniform = (T / (L - 1)) * jnp.arange(L, dtype=X.dtype)[None, :]  # (1, L)
        time = jnp.where(
            denom > 0,
            ((sample - s0) / (denom + eps)) * T,
            uniform,
        )  # (N, L)

        return jnp.concatenate([time[:, :, None], feats], axis=2)  # (N, L, 11)

    X_train = _transform_X(X_train)
    X_val = _transform_X(X_val)
    X_test = _transform_X(X_test)

    # ---------------------------------------------------------------------
    # Normalisation (train stats only)
    # ---------------------------------------------------------------------
    stats_path = os.path.join(base, f"norm_stats_time{int(include_time)}.npz")

    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        X_max = stats["X_max"]
        X_min = stats["X_min"]
        y_max = stats["y_max"]
        y_min = stats["y_min"]
    else:
        # Determine which channels to normalise
        # include_time=True: do NOT normalise channel 0 (time), normalise channels 1: (10 features)
        # include_time=False: normalise all channels (10 features)
        norm_start = 1 if include_time else 0

        X_max = X_train[:, :, norm_start:].max(axis=(0, 1), keepdims=True)
        X_min = X_train[:, :, norm_start:].min(axis=(0, 1), keepdims=True)
        y_max = y_train.max()
        y_min = y_train.min()

        np.savez(
            stats_path,
            X_max=X_max,
            X_min=X_min,
            y_max=y_max,
            y_min=y_min,
        )

    def _normalise_X(X):
        if include_time:
            time = X[:, :, :1]  # (N, L, 1)
            feats = X[:, :, 1:]  # (N, L, 10)
            feats_norm = _scale_to_minus_one_one(feats, X_min, X_max, eps)
            return jnp.concatenate([time, feats_norm], axis=2)
        else:
            return _scale_to_minus_one_one(X, X_min, X_max, eps)

    X_train = _normalise_X(X_train)
    X_val = _normalise_X(X_val)
    X_test = _normalise_X(X_test)
    y_train = _scale_to_minus_one_one(y_train, y_min, y_max, eps)
    y_val = _scale_to_minus_one_one(y_val, y_min, y_max, eps)
    y_test = _scale_to_minus_one_one(y_test, y_min, y_max, eps)

    def _drop_percent(X, y, drop_percentage, *, key, mode="same"):
        if drop_percentage is None:
            # y_times: full time grid
            if include_time:
                y_times = X[:, :, 0]
            else:
                N, L, _ = X.shape
                y_times = (T / (L - 1)) * jnp.arange(L, dtype=X.dtype)[None, :]
                y_times = jnp.broadcast_to(y_times, (N, L))
            return X, y, y_times

        N, L, _ = X.shape
        keep = max(2, int(round((1.0 - drop_percentage) * L)))
        keep = min(keep, L)

        # full y time-grid (before any dropping)
        if include_time:
            t_full = X[:, :, 0]  # (N, L)
        else:
            t_full = (T / (L - 1)) * jnp.arange(L, dtype=X.dtype)[None, :]
            t_full = jnp.broadcast_to(t_full, (N, L))

        def _sample_idx(k, *, keep_first):
            if keep == L:
                return jnp.arange(L, dtype=jnp.int32)
            if keep_first:
                rest = jax.random.choice(k, L - 1, shape=(keep - 1,), replace=False) + 1
                idx = jnp.concatenate(
                    [jnp.array([0], dtype=jnp.int32), rest.astype(jnp.int32)]
                )
                return jnp.sort(idx)
            idx = jax.random.choice(k, L, shape=(keep,), replace=False)
            return jnp.sort(idx.astype(jnp.int32))

        def _gather_time(t, idx):
            return jax.vmap(lambda tt, ii: tt[ii], in_axes=(0, 0))(t, idx)

        if mode == "same":
            keys = jax.random.split(key, N)
            idx = jax.vmap(lambda k: _sample_idx(k, keep_first=True))(keys)

            X2 = jax.vmap(lambda x, ii: x[ii, :], in_axes=(0, 0))(X, idx)
            y2 = jax.vmap(lambda yy, ii: yy[ii, :], in_axes=(0, 0))(y, idx)
            y_times = _gather_time(t_full, idx)
            return X2, y2, y_times

        if mode == "input_only":
            keys = jax.random.split(key, N)
            idx_x = jax.vmap(lambda k: _sample_idx(k, keep_first=True))(keys)

            X2 = jax.vmap(lambda x, ii: x[ii, :], in_axes=(0, 0))(X, idx_x)
            y_times = t_full
            return X2, y, y_times

        if mode == "independent":
            kx, ky = jax.random.split(key, 2)
            keys_x = jax.random.split(kx, N)
            keys_y = jax.random.split(ky, N)

            idx_x = jax.vmap(lambda k: _sample_idx(k, keep_first=True))(keys_x)
            idx_y = jax.vmap(lambda k: _sample_idx(k, keep_first=True))(keys_y)

            X2 = jax.vmap(lambda x, ii: x[ii, :], in_axes=(0, 0))(X, idx_x)
            y2 = jax.vmap(lambda yy, ii: yy[ii, :], in_axes=(0, 0))(y, idx_y)
            y_times = _gather_time(t_full, idx_y)
            return X2, y2, y_times

        raise ValueError(f"Unknown drop mode: {mode}")

    key, k_tr, k_va, k_te = jax.random.split(key, 4)

    X_train, y_train, ytime_train = _drop_percent(
        X_train, y_train, drop_percentage, key=k_tr, mode=drop_mode
    )
    X_val, y_val, ytime_val = _drop_percent(
        X_val, y_val, drop_percentage, key=k_va, mode=drop_mode
    )
    X_test, y_test, ytime_test = _drop_percent(
        X_test, y_test, drop_percentage, key=k_te, mode=drop_mode
    )

    data = (X_train, X_val, X_test)
    labels = (y_train[:, :, 0], y_val[:, :, 0], y_test[:, :, 0])
    times = (ytime_train, ytime_val, ytime_test)

    return dataset_generator(
        name,
        data,
        labels,
        None,
        depth,
        include_time,
        T,
        use_presplit=True,
        interval_times=times,
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
    scale=False,
    drop_percentage=None,
    drop_mode="same",
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
            scale=scale,
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
    elif name.lower() in ["pm25", "pm10"]:
        return create_PM_dataset(
            data_dir,
            name,
            use_presplit,
            stepsize,
            depth,
            include_time,
            T,
            drop_percentage,
            drop_mode,
            key=key,
        )
    else:
        raise ValueError(f"Dataset {name} not found in UEA folder and not toy dataset")
