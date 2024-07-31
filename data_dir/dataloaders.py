"""
This module implements a `Dataloader` class for loading and batching data. It supports three different types of
data inputs, tailored for different types of models used in this repository.

1. **Time Series Data**: Used by models like recurrent neural networks and structured state space models.
   - Input data should be a `jnp.ndarray` of shape `(n_samples, n_timesteps, n_features)`.

2. **Neural Controlled Differential Equations (NCDEs)**: Requires sampling times, coefficients of an interpolation,
   and the initial value of the data.
   - Input data should be a tuple of length 3:
     - The first element: `jnp.ndarray` of shape `(n_samples, n_timesteps)` for sampling times.
     - The second element: a tuple of length `n_coeffs`, where each element is a `jnp.ndarray` of shape
       `(n_samples, n_timesteps-1, n_features)` for interpolation coefficients.
     - The third element: `jnp.ndarray` of shape `(n_samples, n_features)` for the initial value.

3. **Neural Rough Differential Equations (NRDEs) and Log-NCDEs**: Requires sampling times, log-signature of the data,
   and the initial value of the data.
   - Input data should be a tuple of length 3:
     - The first element: `jnp.ndarray` of shape `(n_samples, n_timesteps)` for sampling times.
     - The second element: `jnp.ndarray` of shape `(n_samples, n_intervals, n_logsig_features)` for log-signature data.
     - The third element: `jnp.ndarray` of shape `(n_samples, n_features)` for the initial value.

Additionally, data can be stored as a NumPy array to save GPU memory, with each batch converted to a JAX NumPy array.

Methods:
- `loop(batch_size, *, key)`: Generates data batches indefinitely. Randomly shuffles data for each batch.
- `loop_epoch(batch_size)`: Generates data batches for one epoch (i.e., a full pass through the dataset).
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
