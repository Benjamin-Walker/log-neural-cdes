import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class VectorField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_size, out_size, width, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            final_activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, y):
        return self.mlp(y)


class NeuralCDE(eqx.Module):
    vf: VectorField
    data_dim: int
    hidden_dim: int
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    classification: bool

    def __init__(
        self,
        vf_hidden_dim,
        vf_num_hidden,
        hidden_dim,
        data_dim,
        label_dim,
        classification,
        *,
        key,
        **kwargs
    ):
        super().__init__(**kwargs)
        vf_key, l1key, l2key = jr.split(key, 3)
        self.vf = VectorField(
            hidden_dim, hidden_dim * data_dim, vf_hidden_dim, vf_num_hidden, key=vf_key
        )
        self.linear1 = eqx.nn.Linear(data_dim, hidden_dim, key=l1key)
        self.linear2 = eqx.nn.Linear(hidden_dim, label_dim, key=l2key)
        self.classification = classification
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim

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
            saveat = diffrax.SaveAt(ts=ts)
        solution = diffrax.diffeqsolve(
            terms=diffrax.ControlTerm(func, control).to_ode(),
            solver=diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=y0,
            saveat=saveat,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        )
        if self.classification:
            return jax.nn.softmax(self.linear2(solution.ys[-1]))
        else:
            return jax.vmap(self.linear2)(solution.ys)


class NeuralRDE(eqx.Module):
    vf: VectorField
    data_dim: int
    logsig_dim: int
    hidden_dim: int
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    classification: bool
    intervals: jnp.ndarray

    def __init__(
        self,
        vf_hidden_dim,
        vf_num_hidden,
        hidden_dim,
        data_dim,
        logsig_dim,
        label_dim,
        classification,
        intervals,
        *,
        key,
        **kwargs
    ):
        vf_key, l1key, l2key = jr.split(key, 3)
        # Exclude first element as always zero
        self.logsig_dim = logsig_dim - 1
        self.vf = VectorField(
            hidden_dim,
            hidden_dim * self.logsig_dim,
            vf_hidden_dim,
            vf_num_hidden,
            key=vf_key,
        )
        self.linear1 = eqx.nn.Linear(data_dim, hidden_dim, key=l1key)
        self.linear2 = eqx.nn.Linear(hidden_dim, label_dim, key=l2key)
        self.classification = classification
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.intervals = intervals

    def __call__(self, X):
        ts, logsig, x0 = X

        def func(t, y, args):
            idx = jnp.searchsorted(self.intervals, t)
            return jnp.dot(
                jnp.reshape(self.vf(y), (self.hidden_dim, self.logsig_dim)),
                logsig[idx][1:],
            )

        y0 = self.linear1(x0)
        if self.classification:
            saveat = diffrax.SaveAt(t1=True)
        else:
            saveat = diffrax.SaveAt(ts=ts)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=y0,
            saveat=saveat,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        )
        if self.classification:
            return jax.nn.softmax(self.linear2(solution.ys[-1]))
        else:
            return jax.vmap(self.linear2)(solution.ys)
