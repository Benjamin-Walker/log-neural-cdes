"""
This module implements the `NeuralCDE` and `NeuralRDE` classes using JAX and Equinox.

Attributes of `NeuralCDE`:
- `vf`: The vector field $f_{\theta}$ of the NCDE.
- `data_dim`: Number of channels in the input time series.
- `hidden_dim`: Dimension of the hidden state $h_t$.
- `linear1`: Input linear layer for initializing $h_0$.
- `linear2`: Output linear layer for generating predictions from $h_t$.
- `classification`: Boolean indicating if the model is used for classification.
- `output_step`: For regression tasks, specifies the step interval for outputting predictions.
- `solver`: The solver used to integrate the NCDE.
- `stepsize_controller`: Controls the step size for the solver.
- `dt0`: Initial step size for the solver.
- `max_steps`: Maximum number of steps allowed for the solver.

Attributes of `NeuralRDE`:
- `vf`: The vector field $\bar{f}_{\theta}$ of the NRDE (excluding the final linear layer).
- `data_dim`: Number of channels in the input time series.
- `logsig_dim`: Dimension of the log-signature used as input to the NRDE.
- `hidden_dim`: Dimension of the hidden state $h_t$.
- `mlp_linear`: Final linear layer of the vector field.
- `linear1`: Input linear layer for initializing $h_0$.
- `linear2`: Output linear layer for generating predictions from $h_t$.
- `classification`: Boolean indicating if the model is used for classification.
- `output_step`: For regression tasks, specifies the step interval for outputting predictions.
- `solver`: The solver used to integrate the NRDE.
- `stepsize_controller`: Controls the step size for the solver.
- `dt0`: Initial step size for the solver.
- `max_steps`: Maximum number of steps allowed for the solver.

The module also includes the `VectorField` class, which defines the vector fields used by both
`NeuralCDE` and `NeuralRDE`.
"""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class VectorField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self, in_size, out_size, width, depth, *, key, activation=jax.nn.relu, scale=1
    ):
        mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width,
            depth=depth,
            activation=activation,
            final_activation=jax.nn.tanh,
            key=key,
        )

        def init_weight(model):
            is_linear = lambda x: isinstance(x, eqx.nn.Linear)
            get_weights = lambda m: [
                x.weight
                for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                if is_linear(x)
            ]
            weights = get_weights(model)
            new_weights = [weight / scale for weight in weights]
            new_model = eqx.tree_at(get_weights, model, new_weights)
            get_bias = lambda m: [
                x.bias
                for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                if is_linear(x)
            ]
            biases = get_bias(model)
            new_bias = [bias / scale for bias in biases]
            new_model = eqx.tree_at(get_bias, new_model, new_bias)
            return new_model

        self.mlp = init_weight(mlp)

    def __call__(self, y):
        return self.mlp(y)


class NeuralCDE(eqx.Module):
    vf: VectorField
    data_dim: int
    hidden_dim: int
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    classification: bool
    output_step: int
    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractStepSizeController
    dt0: float
    max_steps: int
    stateful: bool = False
    nondeterministic: bool = False
    lip2: bool = False

    def __init__(
        self,
        vf_hidden_dim,
        vf_num_hidden,
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
        *,
        key,
        **kwargs
    ):
        super().__init__(**kwargs)
        vf_key, l1key, l2key = jr.split(key, 3)
        self.vf = VectorField(
            hidden_dim,
            hidden_dim * data_dim,
            vf_hidden_dim,
            vf_num_hidden,
            scale=scale,
            key=vf_key,
        )
        self.linear1 = eqx.nn.Linear(data_dim, hidden_dim, key=l1key)
        self.linear2 = eqx.nn.Linear(hidden_dim, label_dim, key=l2key)
        self.classification = classification
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.dt0 = dt0
        self.max_steps = max_steps
        self.output_step = output_step

    def __call__(self, X):
        ts, coeffs, x0 = X
        func = lambda t, y, args: jnp.reshape(
            self.vf(y), (self.hidden_dim, self.data_dim)
        )
        control = diffrax.CubicInterpolation(ts, coeffs)
        y0 = self.linear1(x0)
        if self.classification:
            saveat = diffrax.SaveAt(t1=True)
        else:
            step = self.output_step / len(ts)
            times = jnp.arange(step, 1.0, step)
            saveat = diffrax.SaveAt(ts=times, t1=True)
        solution = diffrax.diffeqsolve(
            terms=diffrax.ControlTerm(func, control).to_ode(),
            solver=self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt0,
            y0=y0,
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
            max_steps=self.max_steps,
        )
        if self.classification:
            return jax.nn.softmax(self.linear2(solution.ys[-1]))
        else:
            return jax.nn.tanh(jax.vmap(self.linear2)(solution.ys))


class NeuralRDE(eqx.Module):
    vf: VectorField
    data_dim: int
    logsig_dim: int
    hidden_dim: int
    mlp_linear: eqx.nn.Linear
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    classification: bool
    output_step: int
    intervals: jnp.ndarray
    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractStepSizeController
    dt0: float
    max_steps: int
    stateful: bool = False
    nondeterministic: bool = False
    lip2: bool = False

    def __init__(
        self,
        vf_hidden_dim,
        vf_num_hidden,
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
        *,
        key,
        **kwargs
    ):
        vf_key, mlplkey, l1key, l2key = jr.split(key, 4)
        # Exclude first element as always zero
        self.logsig_dim = logsig_dim - 1
        self.vf = VectorField(
            hidden_dim,
            vf_hidden_dim,
            vf_hidden_dim,
            vf_num_hidden - 1,
            scale=scale,
            key=vf_key,
        )
        self.mlp_linear = eqx.nn.Linear(
            vf_hidden_dim, hidden_dim * self.logsig_dim, key=mlplkey
        )
        self.linear1 = eqx.nn.Linear(data_dim, hidden_dim, key=l1key)
        self.linear2 = eqx.nn.Linear(hidden_dim, label_dim, key=l2key)
        self.classification = classification
        self.output_step = output_step
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.intervals = intervals
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.dt0 = dt0
        self.max_steps = max_steps

    def __call__(self, X):
        ts, logsig, x0 = X

        def func(t, y, args):
            idx = jnp.searchsorted(self.intervals, t)
            return jnp.dot(
                jnp.reshape(
                    self.mlp_linear(self.vf(y)), (self.hidden_dim, self.logsig_dim)
                ),
                logsig[idx - 1][1:],
            ) / (self.intervals[idx] - self.intervals[idx - 1])

        y0 = self.linear1(x0)
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
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
            max_steps=self.max_steps,
        )
        if self.classification:
            return jax.nn.softmax(self.linear2(solution.ys[-1]))
        else:
            return jax.nn.tanh(jax.vmap(self.linear2)(solution.ys))
