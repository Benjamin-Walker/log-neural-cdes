import diffrax
import jax


def calc_coeffs(data):
    ts = data[:, :, 0]
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, data)
    return coeffs