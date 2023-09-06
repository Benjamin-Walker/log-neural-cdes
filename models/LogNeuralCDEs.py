import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from data.hall_set import HallSet
from models.NeuralCDEs import VectorField


class LogNeuralCDE(eqx.Module):
    vf: VectorField
    width: int
    depth: int
    hidden_dim: int
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    pairs: jnp.array
    classification: bool
    intervals: jnp.ndarray
    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractStepSizeController
    dt0: float
    max_steps: int
    stateful: bool = False
    nondeterministic: bool = False

    def __init__(
        self,
        vf_hidden_dim,
        vf_num_hidden,
        hidden_dim,
        data_dim,
        depth,
        label_dim,
        classification,
        intervals,
        solver,
        stepsize_controller,
        dt0,
        max_steps,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        vf_key, l1key, l2key, weightkey = jr.split(key, 4)
        vf = VectorField(
            hidden_dim, hidden_dim * data_dim, vf_hidden_dim, vf_num_hidden, key=vf_key
        )
        self.vf = vf
        self.width = data_dim
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.linear1 = eqx.nn.Linear(data_dim, hidden_dim, key=l1key)
        self.linear2 = eqx.nn.Linear(hidden_dim, label_dim, key=l2key)
        hs = HallSet(self.width, self.depth)
        if self.depth == 1:
            self.pairs = None
        else:
            self.pairs = jnp.asarray(hs.data[1:])
        self.classification = classification
        self.intervals = intervals
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.dt0 = dt0
        self.max_steps = max_steps

    def __call__(self, X):

        ts, logsig, x0 = X

        y0 = self.linear1(x0)

        def func(t, y, args):
            idx = jnp.searchsorted(self.intervals, t)
            logsig_t = logsig[idx - 1]
            vf_out = jnp.reshape(self.vf(y), (self.width, self.hidden_dim))

            if self.pairs is None:
                return jnp.dot(logsig_t[1:], vf_out) / (
                    self.intervals[idx] - self.intervals[idx - 1]
                )

            jvps = jnp.reshape(
                jax.vmap(lambda x: jax.jvp(self.vf, (y,), (x,))[1])(vf_out),
                (self.width, self.width, self.hidden_dim),
            )

            def liebracket(jvps, pair):
                return (
                    jvps[pair[1] - 1, (pair[0] - 1)] - jvps[pair[0] - 1, (pair[1] - 1)]
                )

            lieout = jax.vmap(liebracket, in_axes=(None, 0))(
                jvps, self.pairs[self.width :]
            )

            return (
                jnp.dot(logsig_t[1 : self.width + 1], vf_out)
                + jnp.dot(logsig_t[self.width + 1 :], lieout)
            ) / (self.intervals[idx] - self.intervals[idx - 1])

        if self.classification:
            saveat = diffrax.SaveAt(t1=True)
        else:
            saveat = diffrax.SaveAt(ts=ts)

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
            return jax.vmap(self.linear2)(solution.ys)
