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
        *,
        key,
        **kwargs
    ):
        super().__init__(**kwargs)
        vf_key, l1key, l2key, weightkey = jr.split(key, 4)
        data_dim = data_dim - 1
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
        self.pairs = jnp.asarray(hs.data[1:])
        self.classification = classification
        self.intervals = intervals

    def __call__(self, X):

        ts, logsig, x0 = X
        y0 = self.linear1(x0[1:])

        def func(t, y, args):
            # idx = jnp.searchsorted(self.intervals, t)
            idx = jnp.searchsorted(ts, t) // 8
            logsig_t = logsig[idx]
            vf_out = jnp.reshape(self.vf(y), (self.width, self.hidden_dim))
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

            return jnp.dot(logsig_t[1 : self.width + 1], vf_out) + jnp.dot(
                logsig_t[self.width + 1 :], lieout
            )

        if self.classification:
            saveat = diffrax.SaveAt(t1=True)
        else:
            saveat = diffrax.SaveAt(ts=ts)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            diffrax.Heun(),
            t0=ts[0],
            t1=ts[-1],
            dt0=(ts[-1] - ts[0]) / 17984,
            y0=y0,
            # stepsize_controller=diffrax.PIDController(
            #     rtol=1e-2,
            #     atol=1e-4,  # dtmin=(ts[-1] - ts[0]) / 4095
            # ),
            saveat=saveat,
            max_steps=2 * 17984,
        )

        if self.classification:
            return jax.nn.softmax(self.linear2(solution.ys[-1]))
        else:
            return jax.vmap(self.linear2)(solution.ys)
