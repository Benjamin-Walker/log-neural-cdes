import jax.numpy as jnp
import jax.random as jr


class InMemoryDataloader:

    data: jnp.ndarray
    labels: jnp.ndarray

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __iter__(self):
        RuntimeError("Use .loop(batch_size) instead of __iter__")

    def loop(self, batch_size, *, key):

        if self.data is None or jnp.isnan(self.data).all():
            raise ValueError("This dataloader is empty")

        if self.data.shape[0] != self.labels.shape[0]:
            raise ValueError("Data and labels must have same length")

        if not isinstance(batch_size, int) & (batch_size > 0):
            raise ValueError("Batch size must be a positive integer")

        dataset_size = len(self.data)

        if batch_size > dataset_size:
            raise ValueError("Batch size larger than dataset size")
        elif batch_size == dataset_size:
            while True:
                yield self.data, self.label
        else:
            indices = jnp.arange(dataset_size)
            while True:
                subkey, key = jr.split(key)
                perm = jr.permutation(subkey, indices)
                start = 0
                end = batch_size
                while end < dataset_size:
                    batch_perm = perm[start:end]
                    yield self.data[batch_perm], self.labels[batch_perm]
                    start = end
                    end = start + batch_size
