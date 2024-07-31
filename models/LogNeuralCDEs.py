"""
This module implements the `LogNeuralCDE` class using JAX and Equinox. The model is a
Neural Controlled Differential Equation (NCDE), where the output is approximated during
training using the Log-ODE method.

Attributes of the `LogNeuralCDE` model:
- `vf`: The vector field $f_{\theta}$ of the NCDE.
- `data_dim`: The number of channels in the input time series.
- `depth`: The depth of the Log-ODE method, currently implemented for depth 1 and 2.
- `hidden_dim`: The dimension of the hidden state $h_t$.
- `linear1`: The input linear layer used to initialise the hidden state $h_0$.
- `linear2`: The output linear layer used to obtain predictions from $h_t$.
- `pairs`: The pairs of basis elements for the terms in the depth-2 log-signature of the path.
- `classification`: Boolean indicating if the model is used for classification tasks.
- `output_step`: If the model is used for regression, the number of steps to skip before outputting a prediction.
- `intervals`: The intervals used in the Log-ODE method.
- `solver`: The solver applied to the ODE produced by the Log-ODE method.
- `stepsize_controller`: The stepsize controller for the solver.
- `dt0`: The initial step size for the solver.
- `max_steps`: The maximum number of steps allowed for the solver.
- `lambd`: The Lip(2) regularisation parameter, used to control the smoothness of the vector field.

The class also includes methods for initialising the model and for performing the forward pass, where the dynamics are
solved using the specified ODE solver.
"""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from data_dir.hall_set import HallSet
from models.NeuralCDEs import VectorField


class LogNeuralCDE(eqx.Module):
    vf: VectorField
    data_dim: int
    depth: int
    hidden_dim: int
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    pairs: jnp.ndarray
    classification: bool
    output_step: int
    intervals: jnp.ndarray
    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractStepSizeController
    dt0: float
    max_steps: int
    lambd: float
    stateful: bool = False
    nondeterministic: bool = False
    lip2: bool = True

    def __init__(
        self,
        vf_hidden_dim,
        vf_num_hidden,
        hidden_dim,
        data_dim,
        depth,
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
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        vf_key, l1key, l2key, weightkey = jr.split(key, 4)
        vf = VectorField(
            hidden_dim,
            hidden_dim * data_dim,
            vf_hidden_dim,
            vf_num_hidden,
            activation=jax.nn.silu,
            scale=scale,
            key=vf_key,
        )
        self.vf = vf
        self.data_dim = data_dim
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.linear1 = eqx.nn.Linear(data_dim, hidden_dim, key=l1key)
        self.linear2 = eqx.nn.Linear(hidden_dim, label_dim, key=l2key)
        hs = HallSet(self.data_dim, self.depth)
        if self.depth == 1:
            self.pairs = None
        else:
            self.pairs = jnp.asarray(hs.data[1:])
        self.classification = classification
        self.output_step = output_step
        self.intervals = intervals
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.dt0 = dt0
        self.max_steps = max_steps
        self.lambd = lambd

    def __call__(self, X):

        ts, logsig, x0 = X

        y0 = self.linear1(x0)

        def func(t, y, args):
            idx = jnp.searchsorted(self.intervals, t)
            logsig_t = logsig[idx - 1]
            vf_out = jnp.reshape(self.vf(y), (self.data_dim, self.hidden_dim))

            if self.pairs is None:
                return jnp.dot(logsig_t[1:], vf_out) / (
                    self.intervals[idx] - self.intervals[idx - 1]
                )

            jvps = jnp.reshape(
                jax.vmap(lambda x: jax.jvp(self.vf, (y,), (x,))[1])(vf_out),
                (self.data_dim, self.data_dim, self.hidden_dim),
            )

            def liebracket(jvps, pair):
                return jvps[pair[0] - 1, pair[1] - 1] - jvps[pair[1] - 1, pair[0] - 1]

            lieout = jax.vmap(liebracket, in_axes=(None, 0))(
                jvps, self.pairs[self.data_dim :]
            )

            return (
                jnp.dot(logsig_t[1 : self.data_dim + 1], vf_out)
                + jnp.dot(logsig_t[self.data_dim + 1 :], lieout)
            ) / (self.intervals[idx] - self.intervals[idx - 1])

        if self.classification:
            saveat = diffrax.SaveAt(t1=True)
        else:
            step = self.output_step / len(ts)
            times = jnp.arange(step, 1.0, step)
            saveat = diffrax.SaveAt(ts=times, t1=True)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt0,
            y0=y0,
            stepsize_controller=self.stepsize_controller,
            saveat=saveat,
            max_steps=self.max_steps,
        )

        if self.classification:
            return jax.nn.softmax(self.linear2(solution.ys[-1]))
        else:
            return jax.nn.tanh(jax.vmap(self.linear2)(solution.ys))
