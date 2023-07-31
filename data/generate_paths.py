import jax
import jax.numpy as jnp
from signax.signature import signature
from signax.signature_flattened import flatten
from signax.tensor_ops import log

from data.hall_set import HallSet


def hall_basis_logsig(x, depth, t2l):
    logsig = flatten(log(signature(x, depth)))
    return t2l[:, 1:] @ logsig


def calc_paths(data, stepsize, depth, include_time):
    """
    Generate log-signature objects from data.

    In the future, this function will use RoughPy, and return path objects,
    which can be queried over any interval for the log-signature. Right now,
    it is necessary to specify the stepsize and depth ahead of time.
    """

    if not include_time:
        data = data[:, :, 1:]

    hs = HallSet(data.shape[-1], depth)
    t2l = hs.t2l_matrix(depth)

    if data.shape[1] % stepsize != 0:
        final_data = data[:, -(data.shape[1] % stepsize) - 1 :, ...]
        data = data[:, : -(data.shape[1] % stepsize), ...].reshape(
            data.shape[0], -1, stepsize, data.shape[-1]
        )
    else:
        data = data.reshape(data.shape[0], -1, stepsize, data.shape[-1])
        final_data = None

    vmap_calc_logsig = jax.vmap(hall_basis_logsig, in_axes=(0, None, None))
    logsigs = jax.vmap(vmap_calc_logsig, in_axes=(0, None, None))(data, depth, t2l)

    if final_data is not None:
        final_logsigs = vmap_calc_logsig(final_data, depth, t2l)[:, None, :]
        logsigs = jnp.concatenate(
            (
                logsigs,
                final_logsigs,
            ),
            axis=1,
        )

    return logsigs
