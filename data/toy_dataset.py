import diffrax
import jax.numpy as jnp
import jax.random as jr


def get_interaction_kernel(label):
    if label == 0:
        def interaction_kernel(x, y):
            diff = x-y
            norm = jnp.linalg.norm(diff, ord=2)
            if 0 <= norm < 2 ** 0.5:
                return 0.2 * diff
            if 2 ** 0.5 <= norm < 2:
                return 2 * diff
            if 2 <= norm:
                return 0
    elif label == 1:
        def interaction_kernel(x, y):
            diff = x-y
            norm = jnp.linalg.norm(diff, ord=2)
            if 0 <= norm < 2 ** 0.5:
                return 2 * diff
            if 2 ** 0.5 <= norm < 2:
                return 0.2 * diff
            if 2 <= norm:
                return 0
    else:
        raise ValueError("Label must be 0 or 1")

    return interaction_kernel


def get_drift(label):
    interaction_kernel = get_interaction_kernel(label)


t0, t1 = 1, 3
drift = lambda t, y, args: -y
diffusion = lambda t, y, args: 0.1 * t
brownian_motion = diffrax.VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=jr.PRNGKey(0))
terms = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian_motion))
solver = diffrax.Euler()
saveat = diffrax.SaveAt(dense=True)

sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0=0.05, y0=1.0, saveat=saveat)
print(sol.evaluate(1.1))  # DeviceArray(0.89436394)
