import jax.random as jr

from models.NeuralCDEs import NeuralCDE, NeuralRDE
from models.RNN import GRUCell, LinearCell, LSTMCell, MLPCell, RNN


def create_model(
    model_name,
    data_dim,
    logsig_dim,
    intervals,
    label_dim,
    hidden_dim,
    depth=None,
    width=None,
    classification=True,
    *,
    key,
):
    """Create RNN model."""

    cellkey, outputkey = jr.split(key, 2)

    if model_name == "ncde":
        if width is None or depth is None:
            raise ValueError("Must specify vf width and depth for a NCDE.")
        return NeuralCDE(
            width, depth, hidden_dim, data_dim, label_dim, classification, key=key
        )
    elif model_name == "nrde":
        if width is None or depth is None:
            raise ValueError("Must specify vf width and depth for a NCDE.")
        return NeuralRDE(
            width,
            depth,
            hidden_dim,
            data_dim,
            logsig_dim,
            label_dim,
            classification,
            intervals,
            key=key,
        )
    elif model_name == "rnn_linear":
        cell = LinearCell(data_dim, hidden_dim, key=cellkey)
    elif model_name == "rnn_gru":
        cell = GRUCell(data_dim, hidden_dim, key=cellkey)
    elif model_name == "rnn_lstm":
        cell = LSTMCell(data_dim, hidden_dim, key=cellkey)
    elif model_name == "rnn_mlp":
        if width is None or depth is None:
            raise ValueError("Must specify width and depth for MLP cell.")
        cell = MLPCell(data_dim, hidden_dim, depth, width, key=cellkey)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return RNN(cell, hidden_dim, label_dim, classification, key=key)
