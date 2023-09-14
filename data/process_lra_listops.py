# Following code is adapted to Jax from https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/data/listops.py

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
from flax.training import common_utils


AUTOTUNE = tf.data.experimental.AUTOTUNE


def rename_close_brackets(x):
    source = x["Source"]
    source = tf.strings.regex_replace(source, "]", "X")
    source = tf.strings.regex_replace(source, r"\(", "")
    source = tf.strings.regex_replace(source, r"\)", "")
    return {"Source": source, "Target": x["Target"]}


def preprocess_dataset(file_path, batch_size):
    """Preprocess dataset."""
    tf.logging.info(file_path)
    sel_cols = ["Source", "Target"]
    col_defaults = [tf.string, tf.int32]
    ds = tf.data.experimental.make_csv_dataset(
        [file_path],
        batch_size,
        column_defaults=col_defaults,
        select_columns=sel_cols,
        field_delim="\t",
        header=True,
        num_epochs=1,
    )
    ds = ds.unbatch()
    # we rename close brackets to X for this particular task because
    # tokenizer removes non alphanumeric.
    # since there is no trivial way to change this behaviour
    # we opt for an equivalent fix since the vocab in listops is fixed.
    ds = ds.map(rename_close_brackets, num_parallel_calls=AUTOTUNE)
    return ds


def read_csv_file(file_path):
    return pd.read_csv(file_path)


def save_pickle(obj, filename):
    """Saves a pickle object."""
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_data(data):
    # Single array, then to tensor
    data_numpy = np.vstack(data)
    data_numpy = np.swapaxes(data_numpy, 1, 2)
    data_jnumpy = jnp.array(data_numpy)
    return data_jnumpy


def extract_data_from_tf_dataset(dataset):
    data_iter = iter(dataset)
    data_list, targets_list = [], []
    for idx, batch in enumerate(data_iter):
        batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))
        data_list.append(batch["inputs"])
        targets_list.append(batch["targets"])
    return data_list, targets_list


def convert_listops_file(file_path, data_name):
    """Preprocess dataset."""
    data_dir = os.path.join(file_path, data_name)
    task_name = "basic"
    train_path = os.path.join(data_dir, task_name + "_train.tsv")
    val_path = os.path.join(data_dir, task_name + "_val.tsv")
    test_path = os.path.join(data_dir, task_name + "_test.tsv")

    train_dataset = preprocess_dataset(train_path, batch_size)
    val_dataset = preprocess_dataset(val_path, batch_size)
    test_dataset = preprocess_dataset(test_path, batch_size)

    tf.logging.info("Finished preprocessing")
    tf.logging.info("Building vocab")
    # build vocab
    vocab_set = set()
    tokenizer = text.WhitespaceTokenizer()

    lengths = []
    for i, data in enumerate(val_dataset):
        examples = data["Source"]
        examples = tokenizer.tokenize(examples.numpy())
        examples = np.reshape(examples, (-1)).tolist()
        lengths.append(len(examples))
        vocab_set.update(examples)
        if i % 1000 == 0:
            tf.logging.info("Processed {}".format(i))
        if i > 1000:
            break
    vocab_set = list(set(vocab_set))
    tf.logging.info("Finished processing vocab size={}".format(len(vocab_set)))

    encoder = tfds.deprecated.text.TokenTextEncoder(vocab_set)

    def tf_encode(x):
        result = tf.py_function(
            lambda s: tf.constant(encoder.encode(s.numpy())),
            [
                x,
            ],
            tf.int32,
        )
        result.set_shape([None])
        return result

    def tokenize(d):
        return {"inputs": tf_encode(d["Source"])[:max_length], "targets": d["Target"]}

    train_dataset = train_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)

    max_shape = {"inputs": [max_length], "targets": []}
    # Padding with fixed length
    train_dataset = train_dataset.padded_batch(1, padded_shapes=max_shape)
    val_dataset = val_dataset.padded_batch(1, padded_shapes=max_shape)
    test_dataset = test_dataset.padded_batch(1, padded_shapes=max_shape)

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    train_data_list, train_targets_list = extract_data_from_tf_dataset(train_dataset)
    val_data_list, val_targets_list = extract_data_from_tf_dataset(val_dataset)
    test_data_list, test_targets_list = extract_data_from_tf_dataset(test_dataset)

    return (
        train_data_list,
        val_data_list,
        test_data_list,
        train_targets_list,
        val_targets_list,
        test_targets_list,
    )


if __name__ == "__main__":
    data_dir = "data/raw/LRA/lra_release"
    save_dir = "data/processed/LRA/Listops"
    data_name = "listops-1000"
    batch_size = 256
    max_length = 2000
    (
        train_data,
        val_data,
        test_data,
        train_labels,
        val_labels,
        test_labels,
    ) = convert_listops_file(data_dir, data_name)

    all_data = train_data + val_data + test_data
    all_targets = train_labels + val_labels + test_labels

    original_idxs = (
        np.arange(0, len(train_data)),
        np.arange(len(train_data), len(train_data) + len(val_data)),
        np.arange(len(train_data) + len(val_data), len(all_data)),
    )
    all_data = convert_data(all_data)
    all_targets = jnp.array(np.squeeze(np.vstack(all_targets)))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save data
    save_pickle(all_data, save_dir + "/data.pkl")
    save_pickle(all_targets, save_dir + "/labels.pkl")
    save_pickle(original_idxs, save_dir + "/original_idxs.pkl")
