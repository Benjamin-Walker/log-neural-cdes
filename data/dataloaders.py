import jax.numpy as jnp
import jax.random as jr


class InMemoryDataloader:
    data: jnp.ndarray
    labels: jnp.ndarray
    size: int
    data_is_coeffs: bool = False
    data_is_logsig: bool = False

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        if type(self.data) == tuple:
            if len(data[1][0].shape) > 2:
                self.data_is_coeffs = True
            else:
                self.data_is_logsig = True

        if self.data_is_coeffs:
            self.size = len(data[1][0])
        elif self.data_is_logsig:
            self.size = len(data[1])
        elif self.data is None or jnp.isnan(self.data).all():
            self.size = 0
        else:
            self.size = len(data)

    def __iter__(self):
        RuntimeError("Use .loop(batch_size) instead of __iter__")

    def loop(self, batch_size, *, key):

        if self.size == 0:
            raise ValueError("This dataloader is empty")

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
                    if self.data_is_coeffs:
                        yield (
                            self.data[0][batch_perm],
                            tuple(data[batch_perm] for data in self.data[1]),
                            self.data[2][batch_perm],
                        ), self.labels[batch_perm]
                    elif self.data_is_logsig:
                        yield (
                            self.data[0][batch_perm],
                            self.data[1][batch_perm],
                            self.data[2][batch_perm],
                        ), self.labels[batch_perm]
                    else:
                        yield self.data[batch_perm], self.labels[batch_perm]
                    start = end
                    end = start + batch_size

    def loop_epoch(self, batch_size):

        if self.size == 0:
            raise ValueError("This dataloader is empty")

        if not isinstance(batch_size, int) & (batch_size > 0):
            raise ValueError("Batch size must be a positive integer")

        if batch_size > self.size:
            raise ValueError("Batch size larger than dataset size")
        elif batch_size == self.size:
            yield self.data, self.labels
        else:
            indices = jnp.arange(self.size)
            start = 0
            end = batch_size
            while end < self.size:
                batch_indices = indices[start:end]
                if self.data_is_coeffs:
                    yield (
                        self.data[0][batch_indices],
                        tuple(data[batch_indices] for data in self.data[1]),
                        self.data[2][batch_indices],
                    ), self.labels[batch_indices]
                elif self.data_is_logsig:
                    yield (
                        self.data[0][batch_indices],
                        self.data[1][batch_indices],
                        self.data[2][batch_indices],
                    ), self.labels[batch_indices]
                else:
                    yield self.data[batch_indices], self.labels[batch_indices]
                start = end
                end = start + batch_size
            batch_indices = indices[start:]
            if self.data_is_coeffs:
                yield (
                    self.data[0][batch_indices],
                    tuple(data[batch_indices] for data in self.data[1]),
                    self.data[2][batch_indices],
                ), self.labels[batch_indices]
            elif self.data_is_logsig:
                yield (
                    self.data[0][batch_indices],
                    self.data[1][batch_indices],
                    self.data[2][batch_indices],
                ), self.labels[batch_indices]
            else:
                yield self.data[batch_indices], self.labels[batch_indices]
