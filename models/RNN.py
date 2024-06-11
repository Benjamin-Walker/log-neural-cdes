"""
This module implements the RNN class and the RNN cell classes. The RNN class has the following attributes:
- cell: The RNN cell used in the RNN.
- output_layer: The linear layer used to obtain the output of the RNN.
- hidden_dim: The dimension of the hidden state $h_t$.
- classification: Whether the model is used for classification.
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp


class _AbstractRNNCell(eqx.Module):
    """Abstract RNN Cell class."""

    cell: eqx.Module
    hidden_size: int

    @abc.abstractmethod
    def __init__(self, data_dim, hidden_dim, *, key):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, state, input):
        raise NotImplementedError


class LinearCell(_AbstractRNNCell):
    cell: eqx.nn.Linear
    hidden_size: int

    def __init__(self, data_dim, hidden_dim, *, key):
        self.cell = eqx.nn.Linear(data_dim + hidden_dim, hidden_dim, key=key)
        self.hidden_size = hidden_dim

    def __call__(self, state, input):
        return self.cell(jnp.concatenate([state, input]))


class GRUCell(_AbstractRNNCell):
    cell: eqx.nn.GRUCell
    hidden_size: int

    def __init__(self, data_dim, hidden_dim, *, key):
        self.cell = eqx.nn.GRUCell(data_dim, hidden_dim, key=key)
        self.hidden_size = hidden_dim

    def __call__(self, state, input):
        return self.cell(input, state)


class LSTMCell(_AbstractRNNCell):
    cell: eqx.nn.LSTMCell
    hidden_size: int

    def __init__(self, data_dim, hidden_dim, *, key):
        self.cell = eqx.nn.LSTMCell(data_dim, hidden_dim, key=key)
        self.hidden_size = hidden_dim

    def __call__(self, state, input):
        return self.cell(input, state)


class MLPCell(_AbstractRNNCell):
    cell: eqx.nn.MLP
    hidden_size: int

    def __init__(self, data_dim, hidden_dim, depth, width, *, key):
        self.cell = eqx.nn.MLP(data_dim + hidden_dim, hidden_dim, width, depth, key=key)
        self.hidden_size = hidden_dim

    def __call__(self, state, input):
        return self.cell(jnp.concatenate([state, input]))


class RNN(eqx.Module):
    cell: _AbstractRNNCell
    output_layer: eqx.nn.Linear
    hidden_dim: int
    classification: bool
    stateful: bool = False
    nondeterministic: bool = False
    lip2: bool = False
    output_step: int

    def __init__(
        self, cell, hidden_dim, label_dim, classification=True, output_step=1, *, key
    ):
        self.cell = cell
        self.output_layer = eqx.nn.Linear(
            hidden_dim, label_dim, use_bias=False, key=key
        )
        self.hidden_dim = self.cell.hidden_size
        self.classification = classification
        self.output_step = output_step

    def __call__(self, x):
        hidden = jnp.zeros((self.hidden_dim,))
        hidden = (hidden,) * 2 if isinstance(self.cell, LSTMCell) else hidden

        scan_fn = lambda state, input: (
            self.cell(state, input),
            self.cell(state, input),
        )
        final_state, all_states = jax.lax.scan(scan_fn, hidden, x)

        final_state = final_state[0] if isinstance(self.cell, LSTMCell) else final_state
        all_states = all_states[0] if isinstance(self.cell, LSTMCell) else all_states

        if self.classification:
            return jax.nn.softmax(self.output_layer(final_state), axis=0)
        else:
            all_states = all_states[self.output_step - 1 :: self.output_step]
            return jax.nn.tanh(jax.vmap(self.output_layer)(all_states))
