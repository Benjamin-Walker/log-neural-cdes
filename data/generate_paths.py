import jax
import jax.numpy as jnp
from signax.signature_flattened import logsignature


def calc_paths(data, stepsize, depth=2):
    """
    Generate log-signature objects from data.

    In the future, this function will use RoughPy, and return path objects,
    which can be queried over any interval for the log-signature. Right now,
    it is necessary to specify the stepsize and depth ahead of time.
    """

    final_data = data[:, -(data.shape[1] % stepsize) - 1 :, ...]
    data = data[:, : -(data.shape[1] % stepsize), ...].reshape(
        data.shape[0], -1, stepsize, data.shape[-1]
    )

    def calc_logsig(x):
        return jnp.concatenate((jnp.array([0]), logsignature(x, depth)))

    vmap_calc_logsig = jax.vmap(calc_logsig)
    final_logsigs = vmap_calc_logsig(final_data)[:, None, :]
    logsigs = jnp.concatenate((jax.vmap(vmap_calc_logsig)(data), final_logsigs), axis=1)

    return logsigs
