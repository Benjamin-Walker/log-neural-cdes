"""
A dataloader class for loading data in batches. Each model in this repository is designed to take a single argument as
input. Hence, the dataloader can handle three different cases.
- The first is simply the value of the time series data. This is used by stacked recurrent models, such as recurrent
neural networks and structured state space models. In this case data should be a jnp.ndarray of shape
(n_samples, n_timesteps, n_features)
- The second case is NCDEs, which requires the sampling time, the coefficients of an interpolation, and the initial
value of the data. In this case, data should be a tuple of length 3, where the first element is a jnp.ndarray of shape
(n_samples, n_timesteps) for the sampling times, the third element is a jnp.ndarray of shape (n_samples, n_features)
for the initial value, and the second element is a tuple of length n_coeffs, where each element is a jnp.ndarray of
shape (n_samples, n_timesteps-1, n_features) for the coefficients of the interpolation.
- The third case are NRDEs and Log-NCDEs, which require the sampling time, the log-signature of the data, and the
initial value of the data. In this case, data should be a tuple of length 3, where the first element is a jnp.ndarray
of shape (n_samples, n_timesteps) for the sampling times, the third element is a jnp.ndarray of shape
(n_samples, n_features) for the initial value, and the second element is a jnp.ndarray of shape
(n_samples, n_intervals, n_logsig_features) for the log-signature of the data over n_intervals.

Additionally, the data can be stored as a numpy array, and each batch converted to a jax numpy array, to save GPU
memory.
"""

import jax.numpy as jnp
import jax.random as jr


class Dataloader:
    data: jnp.ndarray
    labels: jnp.ndarray
    size: int
    data_is_coeffs: bool = False
    data_is_logsig: bool = False
    func: callable

    def __init__(self, data, labels, inmemory=True):
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
        elif self.data is None:
            self.size = 0
        else:
            self.size = len(data)
        if inmemory:
            self.func = lambda x: x
        else:
            self.func = lambda x: jnp.asarray(x)

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
                yield self.func(self.data), self.func(self.labels)
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
                            self.func(self.data[0][batch_perm]),
                            tuple(self.func(data[batch_perm]) for data in self.data[1]),
                            self.func(self.data[2][batch_perm]),
                        ), self.func(self.labels[batch_perm])
                    elif self.data_is_logsig:
                        yield (
                            self.func(self.data[0][batch_perm]),
                            self.func(self.data[1][batch_perm]),
                            self.func(self.data[2][batch_perm]),
                        ), self.func(self.labels[batch_perm])
                    else:
                        yield self.func(self.data[batch_perm]), self.func(
                            self.labels[batch_perm]
                        )
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
                        self.func(self.data[0][batch_indices]),
                        tuple(self.func(data[batch_indices]) for data in self.data[1]),
                        self.func(self.data[2][batch_indices]),
                    ), self.func(self.labels[batch_indices])
                elif self.data_is_logsig:
                    yield (
                        self.func(self.data[0][batch_indices]),
                        self.func(self.data[1][batch_indices]),
                        self.func(self.data[2][batch_indices]),
                    ), self.func(self.labels[batch_indices])
                else:
                    yield self.func(self.data[batch_indices]), self.func(
                        self.labels[batch_indices]
                    )
                start = end
                end = start + batch_size
            batch_indices = indices[start:]
            if self.data_is_coeffs:
                yield (
                    self.func(self.data[0][batch_indices]),
                    tuple(self.func(data[batch_indices]) for data in self.data[1]),
                    self.func(self.data[2][batch_indices]),
                ), self.func(self.labels[batch_indices])
            elif self.data_is_logsig:
                yield (
                    self.func(self.data[0][batch_indices]),
                    self.func(self.data[1][batch_indices]),
                    self.func(self.data[2][batch_indices]),
                ), self.func(self.labels[batch_indices])
            else:
                yield self.func(self.data[batch_indices]), self.func(
                    self.labels[batch_indices]
                )
