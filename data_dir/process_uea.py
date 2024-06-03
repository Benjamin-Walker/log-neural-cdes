# Following code is adapted to Jax from https://github.com/jambo6/neuralRDEs

import os
import pickle

import jax.numpy as jnp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sktime.datasets import load_from_arff_to_dataframe
from tqdm import tqdm


def save_pickle(obj, filename):
    """Saves a pickle object."""
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_jax_data(train_file, test_file):
    """Creates jax tensors for test and training from the UCR arff format.

    Args:
        train_file (str): The location of the training data arff file.
        test_file (str): The location of the testing data arff file.

    Returns:
        data_train, data_test, labels_train, labels_test: All as jax tensors.
    """
    # Get arff format
    train_data, train_labels = load_from_arff_to_dataframe(train_file)
    test_data, test_labels = load_from_arff_to_dataframe(test_file)

    def convert_data(data):
        # Expand the series to numpy
        data_expand = data.applymap(lambda x: x.values).values
        # Single array, then to tensor
        data_numpy = np.stack([np.vstack(x).T for x in data_expand])
        data_jnumpy = jnp.array(data_numpy)
        return data_jnumpy

    train_data, test_data = convert_data(train_data), convert_data(test_data)

    # Encode labels as often given as strings
    encoder = LabelEncoder().fit(train_labels)
    train_labels, test_labels = encoder.transform(train_labels), encoder.transform(
        test_labels
    )
    train_labels, test_labels = jnp.array(train_labels), jnp.array(test_labels)

    return train_data, test_data, train_labels, test_labels


def convert_all_files(data_dir):
    """Convert UEA files into jax data to be stored in /interim."""
    arff_folder = data_dir + "/raw/UEA/Multivariate_arff"

    for ds_name in tqdm(
        [x for x in os.listdir(arff_folder) if os.path.isdir(arff_folder + "/" + x)]
    ):
        # File locations
        train_file = arff_folder + "/{}/{}_TRAIN.arff".format(ds_name, ds_name)
        test_file = arff_folder + "/{}/{}_TEST.arff".format(ds_name, ds_name)

        # Ready save dir
        save_dir = data_dir + "_repeat/processed/UEA/{}".format(ds_name)

        # If files don't exist, skip.
        if any(
            [
                x.split("/")[-1] not in os.listdir(arff_folder + "/{}".format(ds_name))
                for x in (train_file, test_file)
            ]
        ):
            if ds_name not in ["Images", "Descriptions"]:
                print("No files found for folder: {}".format(ds_name))
            continue
        elif os.path.isdir(save_dir):
            print("Files already exist for: {}".format(ds_name))
            continue
        else:
            os.makedirs(save_dir)
            train_data, test_data, train_labels, test_labels = create_jax_data(
                train_file, test_file
            )
            # Compile train and test data together
            data = jnp.concatenate([train_data, test_data])
            labels = jnp.concatenate([train_labels, test_labels])

            for i in range(len(data)):
                for j in range(i, len(data)):
                    if (data[i] == data[j]).all():
                        if i != j:
                            print(i, j)
                            breakpoint()

            # Save original train test indexes in case we wish to use original splits
            original_idxs = (
                np.arange(0, train_data.shape[0]),
                np.arange(train_data.shape[0], data.shape[0]),
            )

            # Save data
            save_pickle(data, save_dir + "/data.pkl")
            save_pickle(labels, save_dir + "/labels.pkl")
            save_pickle(original_idxs, save_dir + "/original_idxs.pkl")


if __name__ == "__main__":
    data_dir = "data"
    convert_all_files(data_dir)
