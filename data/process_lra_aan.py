import os

import jax
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from absl import logging


AUTOTUNE = tf.data.experimental.AUTOTUNE

DATASET_PATHS = "/tmp/dataset"

SHUFFLE_BUFFER_SIZE = 2048


def get_tsv_dataset(file_path, batch_size):
    """Preprocess dataset."""
    tf.logging.info(file_path)
    # sel_cols = ['label', 'id1', 'id2']
    col_defaults = [tf.float32, tf.string, tf.string, tf.string, tf.string]
    col_names = ["label", "id1", "id2", "text1", "text2"]
    ds = tf.data.experimental.make_csv_dataset(
        [file_path],
        batch_size,
        column_names=col_names,
        column_defaults=col_defaults,
        use_quote_delim=False,
        field_delim="\t",
        header=False,
        shuffle=True,
        shuffle_buffer_size=10000,
        num_epochs=1,
    )
    ds = ds.unbatch()
    return ds


def get_dataset(batch_size, data_path):
    """Get dataset from matching datasets converts into src/tgt pairs."""
    train_fps = data_path + ".train.tsv"
    valid_fps = data_path + ".eval.tsv"
    test_fps = data_path + ".test.tsv"
    train = get_tsv_dataset(train_fps, batch_size)
    valid = get_tsv_dataset(valid_fps, batch_size)
    test = get_tsv_dataset(test_fps, batch_size)

    # Print an example.
    logging.info("Data sample: %s", next(iter(tfds.as_numpy(test.skip(4)))))

    def adapt_example(example):
        return {
            "Source1": example["text1"],
            "Source2": example["text2"],
            "Target": example["label"],
        }

    train = train.map(adapt_example)
    valid = valid.map(adapt_example)
    test = test.map(adapt_example)

    return train, valid, test


def get_matching_datasets(
    n_devices,
    task_name,
    data_dir=None,
    batch_size=256,
    fixed_vocab=None,
    max_length=512,
    tokenizer="subword",
    vocab_file_path=None,
):
    """Get text matching classification datasets."""
    if batch_size % n_devices:
        raise ValueError(
            "Batch size %d isn't divided evenly by n_devices %d"
            % (batch_size, n_devices)
        )

    del task_name  # not used but may be used in the future.

    if data_dir is None:
        data_path = DATASET_PATHS
    else:
        data_path = os.path.join(data_dir, "new_aan_pairs")

    train_dataset, val_dataset, test_dataset = get_dataset(batch_size, data_path)

    tf.logging.info("Finished getting dataset.")

    if tokenizer == "char":
        logging.info("Using char-level/byte dataset..")
        encoder = tfds.deprecated.text.ByteTextEncoder()
    elif tokenizer == "subword":
        logging.info("Building/loading subword tokenizer")
        if vocab_file_path is None:
            raise ValueError("tokenizer=subword requires vocab_file_path")
        if tf.io.gfile.exists(vocab_file_path + ".subwords"):
            logging.info("Found vocab..already exists. loading..")
            encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
                vocab_file_path
            )
            logging.info("Loaded encoder")
        else:
            encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (en["Source1"].numpy() for en in train_dataset),
                target_vocab_size=2**13,
            )
            encoder.save_to_file(vocab_file_path)
            logging.info("Saved!")
    else:
        if fixed_vocab is None:
            tf.logging.info("Building vocab")
            # build vocab
            vocab_set = set()
            tokenizer = tfds.deprecated.text.Tokenizer()
            i = 0
            for example in tfds.as_numpy(train_dataset):
                # examples = data['Source1']
                examples = tokenizer.tokenize(example["Source1"])
                # examples = np.reshape(examples, (-1)).tolist()
                vocab_set.update(examples)
                if i % 1000 == 0:
                    tf.logging.info("Processed {}".format(i))
                i += 1
            tf.logging.info(len(vocab_set))
            vocab_set = list(set(vocab_set))
            tf.logging.info("Finished processing vocab size={}".format(len(vocab_set)))
        else:
            vocab_set = list(set(fixed_vocab))

        vocab_set = ["<pad>"] + vocab_set

        encoder = tfds.deprecated.text.TokenTextEncoder(vocab_set)

    def tf_encode(x):
        result = tf.py_function(
            lambda s: tf.constant(encoder.encode(s.numpy()[:10000])),
            [
                x,
            ],
            tf.int32,
        )
        result.set_shape([None])
        return result

    def tokenize(d):
        return {
            "inputs1": tf_encode(d["Source1"])[:max_length],
            "inputs2": tf_encode(d["Source2"])[:max_length],
            "targets": tf.cast(d["Target"], tf.int32),
        }

    train_dataset = train_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)

    max_shape = {"inputs1": [max_length], "inputs2": [max_length], "targets": []}
    train_dataset = train_dataset.shuffle(
        buffer_size=SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True
    ).padded_batch(batch_size, padded_shapes=max_shape, drop_remainder=True)
    val_dataset = val_dataset.padded_batch(
        batch_size, padded_shapes=max_shape, drop_remainder=True
    )
    test_dataset = test_dataset.padded_batch(
        batch_size, padded_shapes=max_shape, drop_remainder=True
    )

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, encoder


if __name__ == "__main__":
    data_dir = "/Users/tiexin/tensorflow_datasets/imdb_reviews/plain_text/1.0.0/"
    save_dir = "processed/LRA/IMDb"

    task_name = "imdb_reviews"
    batch_size = 1
    max_length = 1000
    tokenizer = None
    vocab_file_path = ""

    train_ds, eval_ds, test_ds, encoder = get_matching_datasets(
        n_devices=jax.local_device_count(),
        task_name=task_name,
        data_dir=data_dir,
        batch_size=batch_size,
        fixed_vocab=None,
        max_length=max_length,
        tokenizer=tokenizer,
        vocab_file_path=vocab_file_path,
    )
