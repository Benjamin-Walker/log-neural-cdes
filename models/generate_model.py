import jax.random as jr

from models.LogNeuralCDEs import LogNeuralCDE
from models.NeuralCDEs import NeuralCDE, NeuralRDE
from models.RNN import GRUCell, LinearCell, LSTMCell, MLPCell, RNN


def create_model(
    model_name,
    data_dim,
    logsig_dim,
    logsig_depth,
    intervals,
    label_dim,
    hidden_dim,
    vf_depth=None,
    vf_width=None,
    classification=True,
    *,
    key,
):
    """Create RNN model."""

    cellkey, outputkey = jr.split(key, 2)

    if model_name == "log_ncde":
        return LogNeuralCDE(
            vf_width,
            vf_depth,
            hidden_dim,
            data_dim,
            logsig_depth,
            label_dim,
            classification,
            intervals,
            key=key,
        )
    if model_name == "ncde":
        if vf_width is None or vf_depth is None:
            raise ValueError("Must specify vf vf_width and vf_depth for a NCDE.")
        return NeuralCDE(
            vf_width, vf_depth, hidden_dim, data_dim, label_dim, classification, key=key
        )
    elif model_name == "nrde":
        if vf_width is None or vf_depth is None:
            raise ValueError("Must specify vf vf_width and vf_depth for a NCDE.")
        return NeuralRDE(
            vf_width,
            vf_depth,
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
        if vf_width is None or vf_depth is None:
            raise ValueError("Must specify vf_width and vf_depth for MLP cell.")
        cell = MLPCell(data_dim, hidden_dim, vf_depth, vf_width, key=cellkey)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return RNN(cell, hidden_dim, label_dim, classification, key=key)
