from dataclasses import dataclass
from typing import Any, Dict

import jax.numpy as jnp
import jax.random as jr


@dataclass
class Dataset:
    name: str
    raw_dataloaders: Dict[str, Any]
    sig_dataloaders: Dict[str, Any]
    data_dim: int
    label_dim: int


class InMemoryDataloader:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __iter__(self):
        RuntimeError("Use .loop(batch_size) instead of __iter__")

    def loop(self, batch_size, *, key):

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
                    yield self.data[batch_perm], self.data[batch_perm]
                    start = end
                    end = start + batch_size
