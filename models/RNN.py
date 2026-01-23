"""
This module implements the `RNN` class and various RNN cell classes using JAX and Equinox. The `RNN`
class is designed to handle both classification and regression tasks, and can be configured with different
types of RNN cells.

Attributes of the `RNN` class:
- `cell`: The RNN cell used within the RNN, which can be one of several types (e.g., `LinearCell`, `GRUCell`,
          `LSTMCell`, `MLPCell`).
- `output_layer`: The linear layer applied to the hidden state to produce the model's output.
- `hidden_dim`: The dimension of the hidden state $h_t$.
- `classification`: A boolean indicating whether the model is used for classification tasks.
- `output_step`: For regression tasks, specifies how many steps to skip before outputting a prediction.

RNN Cell Classes:
- `_AbstractRNNCell`: An abstract base class for all RNN cells, defining the interface for custom RNN cells.
- `LinearCell`: A simple RNN cell that applies a linear transformation to the concatenated input and hidden state.
- `GRUCell`: An implementation of the Gated Recurrent Unit (GRU) cell.
- `LSTMCell`: An implementation of the Long Short-Term Memory (LSTM) cell.
- `MLPCell`: An RNN cell that applies a multi-layer perceptron (MLP) to the concatenated input and hidden state.

Each RNN cell class implements the following methods:
- `__init__`: Initialises the RNN cell with the specified input dimensions and hidden state size.
- `__call__`: Applies the RNN cell to the input and hidden state, returning the updated hidden state.

The `RNN` class also includes:
- A `__call__` method that processes a sequence of inputs, returning either the final output for classification or a
sequence of outputs for regression.
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
