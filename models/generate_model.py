"""
This module provides a function to generate a model based on a model name and hyperparameters.
It supports various types of models, including Neural CDEs, RNNs, and the S5 model.

Function:
- `create_model`: Generates and returns a model instance along with its state (if applicable)
  based on the provided model name and hyperparameters.

Parameters for `create_model`:
- `model_name`: A string specifying the model architecture to create. Supported values include
  'log_ncde', 'ncde', 'nrde', 'lru', 'S5', 'rnn_linear', 'rnn_gru', 'rnn_lstm', and 'rnn_mlp'.
- `data_dim`: The input data dimension.
- `logsig_dim`: The dimension of the log-signature used in NRDE and Log-NCDE models.
- `logsig_depth`: The depth of the log-signature used in NRDE and Log-NCDE models.
- `intervals`: The intervals used in NRDE and Log-NCDE models.
- `label_dim`: The output label dimension.
- `hidden_dim`: The hidden state dimension for the model.
- `num_blocks`: The number of blocks (layers) in models like LRU or S5.
- `vf_depth`: The depth of the vector field network for CDE models.
- `vf_width`: The width of the vector field network for CDE models.
- `classification`: A boolean indicating whether the task is classification (True) or regression (False).
- `output_step`: The step interval for outputting predictions in sequence models.
- `ssm_dim`: The state-space model dimension for S5 models.
- `ssm_blocks`: The number of SSM blocks in S5 models.
- `solver`: The ODE solver used in CDE models, with a default of `diffrax.Heun()`.
- `stepsize_controller`: The step size controller used in CDE models, with a default of `diffrax.ConstantStepSize()`.
- `dt0`: The initial time step for the solver.
- `max_steps`: The maximum number of steps for the solver.
- `scale`: A scaling factor applied to the vf initialisation in CDE models.
- `lambd`: A regularisation parameter used in Log-NCDE models.
- `key`: A JAX PRNG key for random number generation.

Returns:
- A tuple containing the created model and its state (if applicable).

Raises:
- `ValueError`: If required hyperparameters for the specified model are not provided or if an
  unknown model name is passed.
"""

import diffrax
import equinox as eqx
import jax.random as jr

from models.LinearNeuralCDEs import LogLinearCDE
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
    block_size=None,
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
    w_init_std=0.25,
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
    elif (
        model_name == "bd_linear_ncde" or "diagonal_linear_ncde" or "dense_linear_ncde"
    ):
        return (
            LogLinearCDE(
                data_dim=data_dim,
                hidden_dim=hidden_dim,
                label_dim=label_dim,
                block_size=block_size,
                logsig_depth=logsig_depth,
                lambd=lambd,
                w_init_std=w_init_std,
                classification=classification,
                key=key,
            ),
            None,
        )
    elif model_name == "ncde":
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
