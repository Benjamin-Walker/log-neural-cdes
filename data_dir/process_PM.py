from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view


RAW_PATH = Path("data_dir/raw/PM/prsa_all_stations.pt")
OUT_BASE = Path("data_dir/processed/PM")

WINDOW_LEN = 1000
WINDOW_STEP = 100

SPLIT_FRACS = (0.70, 0.15, 0.15)  # train, val, test

CHANNELS: Dict[str, int] = {
    "sample_number": 0,
    "year": 1,
    "month": 2,
    "day": 3,
    "hour": 4,
    "PM2.5": 5,
    "PM10": 6,
    "SO2": 7,
    "NO2": 8,
    "CO": 9,
    "O3": 10,
    "TEMP": 11,
    "PRES": 12,
    "DEWP": 13,
    "RAIN": 14,
    "WSPM": 15,
    "Wd_cos": 16,
    "Wd_sin": 17,
}

INPUT_CHANNEL_NAMES = [
    "sample_number",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
    "Wd_cos",
    "Wd_sin",
]
INPUT_IDXS = [CHANNELS[name] for name in INPUT_CHANNEL_NAMES]


def forward_fill_nan_1d(x: np.ndarray) -> np.ndarray:
    """
    Forward-fill NaNs in a 1D array.
    Leading NaNs are filled with the first valid value (or 0 if all NaN).
    """
    assert x.ndim == 1
    out = x.copy()
    n = out.shape[0]

    if np.isnan(out[0]):
        first_valid = np.where(~np.isnan(out))[0]
        if first_valid.size == 0:
            return np.zeros_like(out)
        out[: first_valid[0]] = out[first_valid[0]]

    for i in range(1, n):
        if np.isnan(out[i]):
            out[i] = out[i - 1]

    return np.nan_to_num(out, nan=0.0)


def impute_forward_fill(data: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along the length axis, per station and channel."""
    data = data.copy()
    S, L, C = data.shape
    if not np.isnan(data).any():
        return data

    for s in range(S):
        for c in range(C):
            data[s, :, c] = forward_fill_nan_1d(data[s, :, c])
    return data


def split_indices(length: int) -> Tuple[slice, slice, slice]:
    frac_train, frac_val, _ = SPLIT_FRACS
    n_train = int(frac_train * length)
    n_val = int(frac_val * length)
    n_test = length - n_train - n_val

    train_sl = slice(0, n_train)
    val_sl = slice(n_train, n_train + n_val)
    test_sl = slice(n_train + n_val, n_train + n_val + n_test)
    return train_sl, val_sl, test_sl


def make_windows_X_y(
    data_split: np.ndarray,
    input_idxs: List[int],
    target_idx: int,
    window_len: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    data_split: [stations, split_len, channels]

    Returns sequence targets:
      X: [num_windows_total, window_len, num_inputs]
      y: [num_windows_total, window_len]
    """
    S, L, C = data_split.shape
    if L < window_len:
        raise ValueError(f"Split length {L} is smaller than window_len {window_len}.")

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for s in range(S):
        # IMPORTANT: avoid numpy advanced-indexing reordering
        # data_split[s] has shape [L, C]
        x_s = np.take(data_split[s], input_idxs, axis=-1).astype(
            np.float32
        )  # [L, num_inputs]
        y_s = data_split[s, :, target_idx].astype(np.float32)  # [L]

        Xw = sliding_window_view(x_s, window_shape=window_len, axis=0)[::step]
        yw = sliding_window_view(y_s, window_shape=window_len, axis=0)[::step]

        # In numpy, Xw often comes out as [nwin, num_inputs, window_len]
        # Convert to [nwin, window_len, num_inputs]
        if Xw.shape[1] == x_s.shape[1] and Xw.shape[2] == window_len:
            Xw = Xw.transpose(0, 2, 1)
        elif Xw.shape[1] == window_len and Xw.shape[2] == x_s.shape[1]:
            pass
        else:
            raise RuntimeError(
                f"Unexpected window shape from sliding_window_view: {Xw.shape}"
            )

        # Safety alignment
        min_n = min(Xw.shape[0], yw.shape[0])
        Xw = Xw[:min_n]
        yw = yw[:min_n]

        X_list.append(Xw)
        y_list.append(yw)

    X = np.concatenate(X_list, axis=0)  # [S*nwin, window_len, num_inputs]
    y = np.concatenate(y_list, axis=0)  # [S*nwin, window_len]
    return X, y


def save_split_arrays(
    out_dir: Path, X_train, X_val, X_test, y_train, y_val, y_test
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", np.asarray(X_train))
    np.save(out_dir / "X_val.npy", np.asarray(X_val))
    np.save(out_dir / "X_test.npy", np.asarray(X_test))

    np.save(out_dir / "y_train.npy", np.asarray(y_train))
    np.save(out_dir / "y_val.npy", np.asarray(y_val))
    np.save(out_dir / "y_test.npy", np.asarray(y_test))


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Could not find: {RAW_PATH}")

    obj = torch.load(RAW_PATH, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        data_t = obj
    elif isinstance(obj, dict):
        tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if not tensors:
            raise ValueError(
                f"{RAW_PATH} loaded as dict but no torch.Tensor values found."
            )
        data_t = tensors[0]
    else:
        raise ValueError(f"Unsupported data type loaded from {RAW_PATH}: {type(obj)}")

    if data_t.ndim != 3:
        raise ValueError(
            f"Expected [stations, length, channels], got {tuple(data_t.shape)}"
        )

    data = data_t.detach().cpu().numpy().astype(np.float32)
    S, L, C = data.shape
    print(f"Loaded tensor: shape={data.shape}")

    if np.isnan(data).any():
        print("NaNs detected: applying forward-fill imputation per station/channel...")
        data = impute_forward_fill(data)

    train_sl, val_sl, test_sl = split_indices(L)
    train_data = data[:, train_sl, :]
    val_data = data[:, val_sl, :]
    test_data = data[:, test_sl, :]

    print(
        f"Split lengths: train={train_data.shape[1]}, val={val_data.shape[1]}, test={test_data.shape[1]}"
    )
    print(f"Window config: window_len={WINDOW_LEN}, step={WINDOW_STEP}")

    tasks = [
        ("25", "PM2.5", CHANNELS["PM2.5"]),
        ("10", "PM10", CHANNELS["PM10"]),
    ]

    for task_dir, target_name, target_idx in tasks:
        print(
            f"\n=== Building task {task_dir}: predict {target_name} (sequence target) ==="
        )

        X_train, y_train = make_windows_X_y(
            train_data, INPUT_IDXS, target_idx, WINDOW_LEN, WINDOW_STEP
        )
        X_val, y_val = make_windows_X_y(
            val_data, INPUT_IDXS, target_idx, WINDOW_LEN, WINDOW_STEP
        )
        X_test, y_test = make_windows_X_y(
            test_data, INPUT_IDXS, target_idx, WINDOW_LEN, WINDOW_STEP
        )

        # Convert to JAX arrays (requested)
        X_train_j = jnp.asarray(X_train)
        X_val_j = jnp.asarray(X_val)
        X_test_j = jnp.asarray(X_test)

        y_train_j = jnp.asarray(y_train)
        y_val_j = jnp.asarray(y_val)
        y_test_j = jnp.asarray(y_test)

        out_dir = OUT_BASE / task_dir
        save_split_arrays(
            out_dir, X_train_j, X_val_j, X_test_j, y_train_j, y_val_j, y_test_j
        )

        print(f"Saved to: {out_dir}")
        print(f"X_train: {X_train_j.shape}, y_train: {y_train_j.shape}")
        print(f"X_val:   {X_val_j.shape}, y_val:   {y_val_j.shape}")
        print(f"X_test:  {X_test_j.shape}, y_test: {y_test_j.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
