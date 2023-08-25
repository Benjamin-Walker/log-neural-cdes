import jax.numpy as jnp
import jax.random as jr
import numpy as np
from sktime.classification.kernel_based import RocketClassifier

from data.datasets import create_dataset


def create_data(
    data_dir, dataset_name, stepsize=18000, depth=2, include_time=True, T=1, *, key
):
    print(f"Creating dataset {dataset_name}")
    dataset = create_dataset(
        data_dir,
        dataset_name,
        stepsize=stepsize,
        depth=depth,
        include_time=include_time,
        T=T,
        use_idxs=False,
        use_presplit=True,
        key=key,
    )

    dataloaders = dataset.raw_dataloaders

    for _, data in zip(
        range(1),
        dataloaders["train"].loop(dataloaders["train"].size, key=None),
    ):
        X_train, y_train = data
    for _, data in zip(
        range(1),
        dataloaders["val"].loop(dataloaders["val"].size, key=None),
    ):
        X_val, y_val = data
    for _, data in zip(
        range(1),
        dataloaders["test"].loop(dataloaders["test"].size, key=None),
    ):
        X_test, y_test = data

    return X_train, y_train, X_val, y_val, X_test, y_test


def train(num_kernels, X_train, y_train, X_val, y_val, X_test, y_test):
    clf = RocketClassifier(
        num_kernels=num_kernels, n_jobs=-1, use_multivariate="yes", random_state=1
    )
    clf.fit(X_train, np.argmax(np.array(y_train), axis=1))
    y_pred_val = clf.predict(X_val)
    val_acc = (np.argmax(np.array(y_val), axis=1) == y_pred_val).sum() / len(y_pred_val)
    print(f"Val accuracy: {val_acc}")
    y_pred_test = clf.predict(X_test)
    test_acc = (np.argmax(np.array(y_test), axis=1) == y_pred_test).sum() / len(
        y_pred_test
    )
    print(f"Test accuracy: {test_acc}")


if __name__ == "__main__":
    dataset_names = [
        # "EigenWorms",
        # "EthanolConcentration",
        # "FaceDetection",
        # "FingerMovements",
        # "HandMovementDirection",
        # "Handwriting",
        # "Heartbeat",
        # "InsectWingbeat",
        "JapaneseVowels",
        "Libras",
        "LSST",
        "MotorImagery",
        "NATOPS",
        "PEMS-SF",
        "PhonemeSpectra",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
    ]
    for datasetname in dataset_names:
        data_dir = "data"
        key = jr.PRNGKey(1234)
        datasetkey, modelkey, key = jr.split(key, 3)
        X_train, y_train, X_val, y_val, X_test, y_test = create_data(
            data_dir, datasetname, key=datasetkey
        )

        X_train = jnp.moveaxis(X_train, 1, 2)
        X_val = jnp.moveaxis(X_val, 1, 2)
        X_test = jnp.moveaxis(X_test, 1, 2)

        X_train = np.array(X_train)
        X_val = np.array(X_val)
        X_test = np.array(X_test)

        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        for num_kernels in [500, 2000, 10000, 40000]:
            print(f"Num kernels: {num_kernels}")
            train(num_kernels, X_train, y_train, X_val, y_val, X_test, y_test)
