"""Input pipeline for the imdb dataset."""
import pdb
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from absl import logging
from flax.training import common_utils


AUTOTUNE = tf.data.experimental.AUTOTUNE


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
        field_delim=",",
        header=True,
        shuffle=False,
        num_epochs=1,
    )
    ds = ds.unbatch()
    return ds


def get_imdb_dataset():
    """Get dataset from  imdb tfds. converts into src/tgt pairs."""
    data = tfds.load("imdb_reviews")
    train_raw = data["train"]
    valid_raw = data["test"]
    test_raw = data["test"]
    # use test set for validation because IMDb doesn't have val set.
    # Print an example.
    logging.info("Data sample: %s", next(iter(tfds.as_numpy(train_raw.skip(4)))))

    def adapt_example(example):
        return {"Source": example["text"], "Target": example["label"]}

    train = train_raw.map(adapt_example)
    valid = valid_raw.map(adapt_example)
    test = test_raw.map(adapt_example)

    return train, valid, test


def get_yelp_dataset():
    """Get dataset from yelp tfds. converts into src/tgt pairs."""
    data = tfds.load("yelp_polarity_reviews")
    train_raw = data["train"]
    valid_raw = data["test"]
    test_raw = data["test"]
    # use test set for validation because yelp doesn't have val set.
    # Print an example.
    logging.info("Data sample: %s", next(iter(tfds.as_numpy(train_raw.skip(4)))))

    def adapt_example(example):
        return {"Source": example["text"], "Target": example["label"]}

    train = train_raw.map(adapt_example)
    valid = valid_raw.map(adapt_example)
    test = test_raw.map(adapt_example)

    return train, valid, test


def get_agnews_dataset():
    """Get dataset from  agnews tfds. converts into src/tgt pairs."""
    data = tfds.load("ag_news_subset")
    train_raw = data["train"]
    valid_raw = data["test"]
    test_raw = data["test"]
    # use test set for validation because agnews doesn't have val set.
    # Print an example.
    logging.info("Data sample: %s", next(iter(tfds.as_numpy(train_raw.skip(4)))))

    def adapt_example(example):
        return {"Source": example["description"], "Target": example["label"]}

    train = train_raw.map(adapt_example)
    valid = valid_raw.map(adapt_example)
    test = test_raw.map(adapt_example)

    return train, valid, test


def convert_tc_datasets(
    n_devices,
    task_name,
    data_dir=None,
    batch_size=256,
    fixed_vocab=None,
    max_length=512,
    tokenizer="char",
):
    """Get text classification datasets."""
    if batch_size % n_devices:
        raise ValueError(
            "Batch size %d isn't divided evenly by n_devices %d"
            % (batch_size, n_devices)
        )

    if task_name == "imdb_reviews":
        train_dataset, val_dataset, test_dataset = get_imdb_dataset()
    elif task_name == "yelp_reviews":
        train_dataset, val_dataset, test_dataset = get_yelp_dataset()
    elif task_name == "agnews":
        train_dataset, val_dataset, test_dataset = get_agnews_dataset()
    else:
        train_path = data_dir + task_name + "_train.tsv"
        val_path = data_dir + task_name + "_val.tsv"
        test_path = data_dir + task_name + "_test.tsv"

        train_dataset = preprocess_dataset(train_path, batch_size)
        val_dataset = preprocess_dataset(val_path, batch_size)
        test_dataset = preprocess_dataset(test_path, batch_size)

    tf.logging.info("Finished preprocessing")

    tf.logging.info(val_dataset)

    if tokenizer == "char":
        logging.info("Using char/byte level vocab")
        encoder = tfds.deprecated.text.ByteTextEncoder()
    else:
        if fixed_vocab is None:
            tf.logging.info("Building vocab")
            # build vocab
            vocab_set = set()
            tokenizer = tfds.deprecated.text.Tokenizer()
            for i, data in enumerate(train_dataset):
                examples = data["Source"]
                examples = tokenizer.tokenize(examples.numpy())
                examples = np.reshape(examples, (-1)).tolist()
                vocab_set.update(examples)
                if i % 1000 == 0:
                    tf.logging.info("Processed {}".format(i))
            tf.logging.info(len(vocab_set))
            vocab_set = list(set(vocab_set))
            tf.logging.info("Finished processing vocab size={}".format(len(vocab_set)))
        else:
            vocab_set = list(set(fixed_vocab))
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
    train_dataset = train_dataset.padded_batch(batch_size, padded_shapes=max_shape)
    val_dataset = val_dataset.padded_batch(batch_size, padded_shapes=max_shape)
    test_dataset = test_dataset.padded_batch(batch_size, padded_shapes=max_shape)

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
    data_dir = "/Users/tiexin/tensorflow_datasets/imdb_reviews/plain_text/1.0.0/"
    save_dir = "processed/LRA/IMDb"

    task_name = "imdb_reviews"
    batch_size = 1
    max_length = 1000

    # Run the following command for downloading data from tensorflow_datasets
    # ds = tfds.load(task_name, split='train', shuffle_files=True)

    (
        train_data,
        val_data,
        test_data,
        train_labels,
        val_labels,
        test_labels,
    ) = convert_tc_datasets(
        n_devices=jax.local_device_count(),
        task_name=task_name,
        data_dir=data_dir,
        batch_size=batch_size,
        fixed_vocab=None,
        max_length=max_length,
    )

    pdb.set_trace()
    # all_data = train_data + val_data + test_data
    # all_targets = train_labels + val_labels + test_labels
    #
    # original_idxs = (
    #     np.arange(0, len(train_data)),
    #     np.arange(len(train_data), len(train_data)+len(val_data)),
    #     np.arange(len(train_data) + len(val_data), len(all_data))
    # )
    # all_data = convert_data(all_data)
    # all_targets = jnp.array(np.squeeze(np.vstack(all_targets)))
    #
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # # Save data
    # save_pickle(all_data, save_dir + "/data.pkl")
    # save_pickle(all_targets, save_dir + "/labels.pkl")
    # save_pickle(original_idxs, save_dir + "/original_idxs.pkl")
