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
        """Initialize RNN cell."""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, state, input):
        """Call method for RNN cell."""
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

    def __init__(self, cell, output_layer, classification=True):
        self.cell = cell
        self.output_layer = output_layer
        self.hidden_dim = self.cell.hidden_size
        self.classification = classification

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
            return jax.vmap(self.output_layer)(all_states)


def create_rnn_model(
    cell_name,
    data_dim,
    label_dim,
    hidden_dim,
    depth=None,
    width=None,
    classification=True,
    *,
    key,
):
    """Create RNN model."""

    cellkey, outputkey = jax.random.split(key, 2)

    if cell_name == "linear":
        cell = LinearCell(data_dim, hidden_dim, key=cellkey)
    elif cell_name == "gru":
        cell = GRUCell(data_dim, hidden_dim, key=cellkey)
    elif cell_name == "lstm":
        cell = LSTMCell(data_dim, hidden_dim, key=cellkey)
    elif cell_name == "mlp":
        if width is None or depth is None:
            raise ValueError("Must specify width and depth for MLP cell.")
        cell = MLPCell(data_dim, hidden_dim, depth, width, key=cellkey)
    else:
        raise ValueError(f"Unknown cell name: {cell_name}")

    output_layer = eqx.nn.Linear(hidden_dim, label_dim, use_bias=False, key=outputkey)

    return RNN(cell, output_layer, classification)
