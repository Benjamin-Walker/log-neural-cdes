import diffrax
import jax.numpy as jnp
import jax.random as jr


def interaction_kernel(x, y, label):
    diff = x - y
    if label == 0:
        if 0 <= diff < 2 ** 0.5:
            return 0.2 * diff
        if 2 ** 0.5 <= diff < 2:
            return 2 * diff
        if 2 <= diff:
            return 0
    if label == 1:
        if 0 <= diff < 2 ** 0.5:
            return 2 * diff
        if 2 ** 0.5 <= diff < 2:
            return 0.2 * diff
        if 2 <= diff:
            return 0


def get_drift(label):
    diff = x-y
    dist = jnp.linalg.norm(diff, ord=2)
    return jnp.exp(-jnp.abs(x - y) ** 2)


t0, t1 = 1, 3
drift = lambda t, y, args: -y
diffusion = lambda t, y, args: 0.1 * t
brownian_motion = diffrax.VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=jr.PRNGKey(0))
terms = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian_motion))
solver = diffrax.Euler()
saveat = diffrax.SaveAt(dense=True)

sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0=0.05, y0=1.0, saveat=saveat)
print(sol.evaluate(1.1))  # DeviceArray(0.89436394)
