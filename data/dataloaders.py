import jax.numpy as jnp
import jax.random as jr


class InMemoryDataloader:

    data: jnp.ndarray
    labels: jnp.ndarray
    data_is_tuple: bool
    size: int

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.data_is_tuple = type(self.data) == tuple
        if not self.data_is_tuple:
            if self.data is None or jnp.isnan(self.data).all():
                self.size = 0
            else:
                self.size = len(data)
        else:
            self.size = len(data[1][0])

    def __iter__(self):
        RuntimeError("Use .loop(batch_size) instead of __iter__")

    def loop(self, batch_size, *, key):

        if not self.data_is_tuple:
            if self.data is None or jnp.isnan(self.data).all():
                raise ValueError("This dataloader is empty")

            if self.data.shape[0] != self.labels.shape[0]:
                raise ValueError("Data and labels must have same length")
        else:
            if self.data[1][0].shape[0] != self.labels.shape[0]:
                raise ValueError("Data and labels must have same length")

        if not isinstance(batch_size, int) & (batch_size > 0):
            raise ValueError("Batch size must be a positive integer")

        if batch_size > self.size:
            raise ValueError("Batch size larger than dataset size")
        elif batch_size == self.size:
            while True:
                yield self.data, self.labels
        else:
            indices = jnp.arange(self.size)
            while True:
                subkey, key = jr.split(key)
                perm = jr.permutation(subkey, indices)
                start = 0
                end = batch_size
                while end < self.size:
                    batch_perm = perm[start:end]
                    if self.data_is_tuple:
                        yield (
                            self.data[0][batch_perm],
                            tuple(data[batch_perm] for data in self.data[1]),
                            self.data[2][batch_perm],
                        ), self.labels[batch_perm]
                    else:
                        yield self.data[batch_perm], self.labels[batch_perm]
                    start = end
                    end = start + batch_size
