"""
This script generates a toy dataset of shape (100000, 100, 6) and calculates the signature of depth 4 for the label.
"""

import os

import jax
import jax.numpy as jnp
import jax.random as jr
from process_uea import save_pickle
from signax.signature import signature


key = jr.PRNGKey(1234)
depth = 4

data = jr.normal(key, shape=(100000, 100, 6))
data = jnp.round(data)
data = jnp.cumsum(data, axis=1)

vmap_calc_logsig = jax.vmap(signature, in_axes=(0, None))
labels = vmap_calc_logsig(data, depth)
idxs = []

save_dir = "data_dir/processed/toy/signature"
os.makedirs(save_dir, exist_ok=True)
save_pickle(data, save_dir + "/data.pkl")
save_pickle(labels, save_dir + "/labels.pkl")
