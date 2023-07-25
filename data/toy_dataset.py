import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import os
from process_uea import save_pickle


def get_interaction_kernel(label):
    f1 = lambda x: 0.2 * x
    f2 = lambda x: 2 * x
    f3 = lambda x: 0 * x
    if label == 0:
        def interaction_kernel(x, y):
            diff = x-y
            norm = jnp.linalg.norm(diff, ord=2)
            cond1 = norm < 2 ** 0.5
            cond2 = jnp.logical_and(2 ** 0.5 <= norm, norm < 2)
            return jnp.where(cond1, f1(diff), jnp.where(cond2, f2(diff), f3(diff)))
    elif label == 1:
        def interaction_kernel(x, y):
            diff = x-y
            norm = jnp.linalg.norm(diff, ord=2)
            cond1 = norm < 2 ** 0.5
            cond2 = jnp.logical_and(2 ** 0.5 <= norm, norm < 2)
            return jnp.where(cond1, f2(diff), jnp.where(cond2, f1(diff), f3(diff)))
    else:
        raise ValueError("Label must be 0 or 1")

    return interaction_kernel


def get_drift(label):
    interaction_kernel = get_interaction_kernel(label)

    def drift(t, y, args):
        N = y.shape[0] // 2
        y = jnp.reshape(y, (N, 2))

        def drift_j(y_j, y):
            return jnp.sum(jax.vmap(interaction_kernel, in_axes=(None, 0))(y_j, y), axis=0) / N

        return jnp.reshape(jax.vmap(drift_j, in_axes=(0, None))(y, y), (2 * N,))

    return drift


if __name__ == "__main__":
    save_dir = "data/processed/toy"
    os.mkdir(save_dir)
    N = 3
    d = 2 * N
    t0, t1 = 0, 2
    key = jr.PRNGKey(0)
    data = []
    labels = []
    for label in [0, 1]:
        for _ in range(1000):
            initkey, noisekey, key = jr.split(key, 3)
            y0 = jr.normal(initkey, (d,))
            drift = get_drift(0)
            diffusion = lambda t, y, args: jnp.eye(d)
            brownian_motion = diffrax.VirtualBrownianTree(t0, t1, tol=1e-3, shape=(d,), key=noisekey)
            terms = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian_motion))
            solver = diffrax.Euler()

            sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0=0.01, y0=y0, saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, 21)))
            data.append(jnp.concatenate((sol.ts[:, None], sol.ys), axis=1))
            labels.append(label)

    data = jnp.stack(data)
    labels = jnp.stack(labels)

    save_pickle(data, save_dir + "/data.pkl")
    save_pickle(labels, save_dir + "/labels.pkl")
