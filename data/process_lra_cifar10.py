import argparse
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from flax.training import common_utils

from data import pathfinder


AUTOTUNE = tf.data.experimental.AUTOTUNE


def extract_data_from_tf_dataset(dataset):
    data_iter = iter(dataset)
    data_list, targets_list = [], []
    for idx, batch in enumerate(data_iter):
        print("Idx:{}".format(idx))
        batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))
        data_list.append(batch["inputs"])
        targets_list.append(batch["targets"])
    return data_list, targets_list


def convert_data(data):
    # Single array, then to tensor
    data_numpy = np.vstack(data)
    # flatten for images
    data_numpy = data_numpy.reshape(data_numpy.shape[0], 1, -1)
    data_numpy = np.swapaxes(data_numpy, 1, 2)
    data_jnumpy = jnp.array(data_numpy)
    return data_jnumpy


def save_pickle(obj, filename):
    """Saves a pickle object."""
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_cifar10_datasets(n_devices, batch_size=256, normalize=False):
    """Get CIFAR-10 dataset splits."""
    if batch_size % n_devices:
        raise ValueError(
            "Batch size %d isn't divided evenly by n_devices %d"
            % (batch_size, n_devices)
        )

    train_dataset = tfds.load("cifar10", split="train[:90%]")
    val_dataset = tfds.load("cifar10", split="train[90%:]")
    test_dataset = tfds.load("cifar10", split="test")

    def decode(x):
        decoded = {
            "inputs": tf.cast(tf.image.rgb_to_grayscale(x["image"]), dtype=tf.int32),
            "targets": x["label"],
        }
        if normalize:
            decoded["inputs"] = decoded["inputs"] / 255
        return decoded

    train_dataset = train_dataset.map(decode, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(decode, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(decode, num_parallel_calls=AUTOTUNE)

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, 10, 256, (batch_size, 32, 32, 1)


def get_pathfinder_base_datasets(
    n_devices, batch_size=256, resolution=32, normalize=False, split="easy"
):
    """Get Pathfinder dataset splits."""
    if batch_size % n_devices:
        raise ValueError(
            "Batch size %d isn't divided evenly by n_devices %d"
            % (batch_size, n_devices)
        )

    if split not in ["easy", "intermediate", "hard"]:
        raise ValueError("split must be in ['easy', 'intermediate', 'hard'].")

    if resolution == 32:
        builder = pathfinder.Pathfinder32(data_dir=_PATHFINER_TFDS_PATH)
        inputs_shape = (batch_size, 32, 32, 1)
    elif resolution == 64:
        builder = pathfinder.Pathfinder64(data_dir=_PATHFINER_TFDS_PATH)
        inputs_shape = (batch_size, 64, 64, 1)
    elif resolution == 128:
        builder = pathfinder.Pathfinder128(data_dir=_PATHFINER_TFDS_PATH)
        inputs_shape = (batch_size, 128, 128, 1)
    elif resolution == 256:
        builder = pathfinder.Pathfinder256(data_dir=_PATHFINER_TFDS_PATH)
        inputs_shape = (batch_size, 256, 256, 1)
    else:
        raise ValueError("Resolution must be in [32, 64, 128, 256].")

    # builder.download_and_prepare()
    def get_split(split):
        ds = builder.as_dataset(
            split=split, decoders={"image": tfds.decode.SkipDecoding()}
        )
        # Filter out examples with empty images:
        ds = ds.filter(lambda x: tf.strings.length((x["image"])) > 0)
        return ds

    train_dataset = get_split(f"{split}[:80%]")
    val_dataset = get_split(f"{split}[80%:90%]")
    test_dataset = get_split(f"{split}[90%:]")

    def decode(x):
        decoded = {
            "inputs": tf.cast(tf.image.decode_png(x["image"]), dtype=tf.int32),
            "targets": x["label"],
        }
        if normalize:
            decoded["inputs"] = decoded["inputs"] / 255
        return decoded

    train_dataset = train_dataset.map(decode, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(decode, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(decode, num_parallel_calls=AUTOTUNE)

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, 2, 256, inputs_shape


def parse_args():
    parser = argparse.ArgumentParser(description="Params for processing data from LRA")
    parser.add_argument("--task-name", type=str, default="Pathfinder-X")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="processed/LRA/")
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    args = parse_args()
    if args.task_name.lower() == "cifar10":
        ds = tfds.load(args.task_name.lower(), split=["train", "test"])
        _PATHFINER_TFDS_PATH = "/Users/tiexin/tensorflow_datasets/"
        (
            train_dataset,
            val_dataset,
            test_dataset,
            num_classes,
            vocab_size,
            input_shape,
        ) = get_cifar10_datasets(
            n_devices=jax.local_device_count(), batch_size=args.batch_size
        )
    elif args.task_name.lower() == "pathfinder":
        _PATHFINER_TFDS_PATH = "/Users/tiexin/Research/DataSets/LRA/TFDS/"
        (
            train_dataset,
            val_dataset,
            test_dataset,
            num_classes,
            vocab_size,
            input_shape,
        ) = get_pathfinder_base_datasets(
            n_devices=jax.local_device_count(),
            batch_size=args.batch_size,
            resolution=32,
            split="hard",
        )
    elif args.task_name.lower() == "pathfinder-x":
        _PATHFINER_TFDS_PATH = "/Users/tiexin/Research/DataSets/LRA/TFDS/"
        (
            train_dataset,
            val_dataset,
            test_dataset,
            num_classes,
            vocab_size,
            input_shape,
        ) = get_pathfinder_base_datasets(
            n_devices=jax.local_device_count(),
            batch_size=args.batch_size,
            resolution=128,
            split="hard",
        )
    else:
        raise ValueError("Task name: {} is not supported.")

    args.save_dir = os.path.join(args.save_dir, args.task_name)

    print("Convert to list")
    train_data_list, train_targets_list = extract_data_from_tf_dataset(train_dataset)
    val_data_list, val_targets_list = extract_data_from_tf_dataset(val_dataset)
    test_data_list, test_targets_list = extract_data_from_tf_dataset(test_dataset)
    n_train, n_val, n_test = (
        len(train_targets_list),
        len(val_targets_list),
        len(test_targets_list),
    )

    all_data = train_data_list + val_data_list + test_data_list
    all_targets = train_targets_list + val_targets_list + test_targets_list

    original_idxs = (
        np.arange(0, n_train),
        np.arange(n_train, n_train + n_val),
        np.arange(n_train + n_val, n_train + n_val + n_test),
    )
    print("Convert to jax array")
    all_data = convert_data(all_data)
    all_targets = jnp.array(np.squeeze(np.vstack(all_targets)))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Save data
    save_pickle(all_data, os.path.join(args.save_dir, "data.pkl"))
    save_pickle(all_targets, os.path.join(args.save_dir, "labels.pkl"))
    save_pickle(original_idxs, os.path.join(args.save_dir, "original_idxs.pkl"))
