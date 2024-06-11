"""
This module contains a function to generate a model based on the model name and the hyperparameters.
"""

import diffrax
import equinox as eqx
import jax.random as jr

from models.LogNeuralCDEs import LogNeuralCDE
from models.LRU import LRU
from models.NeuralCDEs import NeuralCDE, NeuralRDE
from models.RNN import GRUCell, LinearCell, LSTMCell, MLPCell, RNN
from models.S5 import S5


def create_model(
    model_name,
    data_dim,
    logsig_dim,
    logsig_depth,
    intervals,
    label_dim,
    hidden_dim,
    num_blocks=None,
    vf_depth=None,
    vf_width=None,
    classification=True,
    output_step=1,
    ssm_dim=None,
    ssm_blocks=None,
    solver=diffrax.Heun(),
    stepsize_controller=diffrax.ConstantStepSize(),
    dt0=1,
    max_steps=16**4,
    scale=1.0,
    lambd=0.0,
    *,
    key,
):
    cellkey, outputkey = jr.split(key, 2)

    if model_name == "log_ncde":
        if vf_width is None or vf_depth is None:
            raise ValueError("Must specify vf_width and vf_depth for a Log-NCDE.")
        return (
            LogNeuralCDE(
                vf_width,
                vf_depth,
                hidden_dim,
                data_dim,
                logsig_depth,
                label_dim,
                classification,
                output_step,
                intervals,
                solver,
                stepsize_controller,
                dt0,
                max_steps,
                scale,
                lambd,
                key=key,
            ),
            None,
        )
    if model_name == "ncde":
        if vf_width is None or vf_depth is None:
            raise ValueError("Must specify vf_width and vf_depth for a NCDE.")
        return (
            NeuralCDE(
                vf_width,
                vf_depth,
                hidden_dim,
                data_dim,
                label_dim,
                classification,
                output_step,
                solver,
                stepsize_controller,
                dt0,
                max_steps,
                scale,
                key=key,
            ),
            None,
        )
    elif model_name == "nrde":
        if vf_width is None or vf_depth is None:
            raise ValueError("Must specify vf_width and vf_depth for a NRDE.")
        return (
            NeuralRDE(
                vf_width,
                vf_depth,
                hidden_dim,
                data_dim,
                logsig_dim,
                label_dim,
                classification,
                output_step,
                intervals,
                solver,
                stepsize_controller,
                dt0,
                max_steps,
                scale,
                key=key,
            ),
            None,
        )
    elif model_name == "lru":
        if num_blocks is None:
            raise ValueError("Must specify num_blocks for LRU.")
        lru = LRU(
            num_blocks,
            data_dim,
            ssm_dim,
            hidden_dim,
            label_dim,
            classification,
            output_step,
            key=key,
        )
        state = eqx.nn.State(lru)
        return lru, state
    elif model_name == "S5":
        if num_blocks is None:
            raise ValueError("Must specify num_blocks for S5.")
        if ssm_dim is None:
            raise ValueError("Must specify ssm_dim for S5.")
        if ssm_blocks is None:
            raise ValueError("Must specify ssm_blocks for S5.")
        ssm = S5(
            num_blocks,
            data_dim,
            ssm_dim,
            ssm_blocks,
            hidden_dim,
            label_dim,
            classification,
            output_step,
            "lecun_normal",
            True,
            True,
            "zoh",
            0.001,
            0.1,
            1.0,
            key=key,
        )
        state = eqx.nn.State(ssm)
        return ssm, state
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

    return RNN(cell, hidden_dim, label_dim, classification, output_step, key=key), None
