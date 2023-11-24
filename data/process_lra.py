import jax.numpy as jnp
import numpy as np
from dataloading_s5 import create_lra_image_classification_dataset
from process_uea import save_pickle


(
    trn_loader,
    val_loader,
    tst_loader,
    aux_loaders,
    N_CLASSES,
    SEQ_LENGTH,
    IN_DIM,
    TRAIN_SIZE,
) = create_lra_image_classification_dataset(bsz=1000)


def get_jax_data(loader):
    data = []
    labels = []
    for batch in loader:
        x, y, rate = batch
        data.append(x.numpy())
        labels.append(y.numpy())

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = jnp.array(data)
    labels = jnp.array(labels)
    return data, labels


trn_data, trn_labels = get_jax_data(trn_loader)
val_data, val_labels = get_jax_data(val_loader)
tst_data, tst_labels = get_jax_data(tst_loader)

data = jnp.concatenate([trn_data, val_data, tst_data], axis=0)
labels = jnp.concatenate([trn_labels, val_labels, tst_labels], axis=0)

save_pickle(data, "data/processed/LRA/Cifar10/data.pkl")
save_pickle(labels, "data/processed/LRA/Cifar10/labels.pkl")

breakpoint()
