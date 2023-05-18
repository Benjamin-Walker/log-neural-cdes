import diffrax
import jax


def calc_coeffs(data):
    ts = data[:, :, 0]
    ys = data[:, :, 1:]
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys)
    return coeffs
