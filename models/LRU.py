"""
Code modified from https://gist.github.com/Ryu1845/7e78da4baa8925b4de482969befa949d

This module implements the `LRU` class, a model architecture using JAX and Equinox.

Attributes of the `LRU` class:
- `linear_encoder`: The linear encoder applied to the input time series data.
- `blocks`: A list of `LRUBlock` instances, each containing the LRU layer, normalization, GLU, and dropout.
- `linear_layer`: The final linear layer that outputs the model predictions.
- `classification`: A boolean indicating whether the model is used for classification tasks.
- `output_step`: For regression tasks, specifies how many steps to skip before outputting a prediction.

The module also includes the following classes and functions:
- `GLU`: Implements a Gated Linear Unit for non-linear transformations within the model.
- `LRULayer`: A single LRU layer that applies complex-valued transformations and projections to the input.
- `LRUBlock`: A block consisting of normalization, LRU layer, GLU, and dropout, used as a building block for the `LRU`
              model.
- `binary_operator_diag`: A helper function used in the associative scan operation within `LRULayer` to process diagonal
                          elements.
"""

from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def binary_operator_diag(element_i, element_j):
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j


class GLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x):
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))


class LRULayer(eqx.Module):
    nu_log: jnp.ndarray
    theta_log: jnp.ndarray
    B_re: jnp.ndarray
    B_im: jnp.ndarray
    C_re: jnp.ndarray
    C_im: jnp.ndarray
    D: jnp.ndarray
    gamma_log: jnp.ndarray

    def __init__(self, N, H, r_min=0, r_max=1, max_phase=6.28, *, key):
        u1_key, u2_key, B_re_key, B_im_key, C_re_key, C_im_key, D_key = jr.split(key, 7)

        # N: state dimension, H: model dimension
        # Initialization of Lambda is complex valued distributed uniformly on ring
        # between r_min and r_max, with phase in [0, max_phase].
        u1 = jr.uniform(u1_key, shape=(N,))
        u2 = jr.uniform(u2_key, shape=(N,))
        self.nu_log = jnp.log(
            -0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2)
        )
        self.theta_log = jnp.log(max_phase * u2)

        # Glorot initialized Input/Output projection matrices
        self.B_re = jr.normal(B_re_key, shape=(N, H)) / jnp.sqrt(2 * H)
        self.B_im = jr.normal(B_im_key, shape=(N, H)) / jnp.sqrt(2 * H)
        self.C_re = jr.normal(C_re_key, shape=(H, N)) / jnp.sqrt(N)
        self.C_im = jr.normal(C_im_key, shape=(H, N)) / jnp.sqrt(N)
        self.D = jr.normal(D_key, shape=(H,))

        # Normalization factor
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        self.gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))

    def __call__(self, x):
        # Materializing the diagonal of Lambda and projections
        Lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )
        C = self.C_re + 1j * self.C_im
        # Running the LRU + output projection
        Lambda_elements = jnp.repeat(Lambda[None, ...], x.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(x)
        elements = (Lambda_elements, Bu_elements)
        _, inner_states = jax.lax.associative_scan(
            binary_operator_diag, elements
        )  # all x_k
        y = jax.vmap(lambda z, u: (C @ z).real + (self.D * u))(inner_states, x)

        return y


class LRUBlock(eqx.Module):

    norm: eqx.nn.BatchNorm
    lru: LRULayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(self, N, H, r_min=0, r_max=1, max_phase=6.28, drop_rate=0.1, *, key):
        lrukey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=H, axis_name="batch", channelwise_affine=False
        )
        self.lru = LRULayer(N, H, r_min, r_max, max_phase, key=lrukey)
        self.glu = GLU(H, H, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.lru(x)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x
        return x, state


class LRU(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: List[LRUBlock]
    linear_layer: eqx.nn.Linear
    classification: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        num_blocks,
        data_dim,
        N,
        H,
        output_dim,
        classification,
        output_step,
        r_min=0,
        r_max=1,
        max_phase=6.28,
        drop_rate=0.1,
        *,
        key
    ):
        linear_encoder_key, *block_keys, linear_layer_key = jr.split(
            key, num_blocks + 2
        )
        self.linear_encoder = eqx.nn.Linear(data_dim, H, key=linear_encoder_key)
        self.blocks = [
            LRUBlock(N, H, r_min, r_max, max_phase, drop_rate, key=key)
            for key in block_keys
        ]
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)
        self.classification = classification
        self.output_step = output_step

    def __call__(self, x, state, key):
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)
        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)
        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.nn.tanh(jax.vmap(self.linear_layer)(x))
        return x, state
