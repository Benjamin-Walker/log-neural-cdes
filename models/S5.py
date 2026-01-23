"""
S5 implementation modified from: https://github.com/lindermanlab/S5/blob/main/s5/ssm_init.py

This module implements S5 using JAX and Equinox.

Attributes of the S5 model:
- `linear_encoder`: The linear encoder applied to the input time series.
- `blocks`: A list of S5 blocks, each consisting of an S5 layer, normalisation, GLU, and dropout.
- `linear_layer`: The final linear layer that outputs the predictions of the model.
- `classification`: A boolean indicating whether the model is used for classification tasks.
- `output_step`: For regression tasks, specifies how many steps to skip before outputting a prediction.

The module also includes:
- `S5Layer`: Implements the core S5 layer using structured state space models with options for
  different discretisation methods and eigenvalue clipping.
- `S5Block`: Combines the S5 layer with batch normalisation, a GLU activation, and dropout.
- Utility functions for initialising and discretising the state space model components,
  such as `make_HiPPO`, `make_NPLR_HiPPO`, and `make_DPLR_HiPPO`.
"""

from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn.initializers import lecun_normal, normal
from jax.scipy.linalg import block_diag

from models.LRU import GLU


def make_HiPPO(N):
    """Create a HiPPO-LegS matrix.
    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix
    """
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size

    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = jnp.sqrt(jnp.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:

    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]

    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = jnp.linalg.eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """Initialize the learnable timescale Delta by sampling
    uniformly between dt_min and dt_max.
    Args:
        dt_min (float32): minimum value
        dt_max (float32): maximum value
    Returns:
        init function
    """

    def init(key, shape):
        """Init function
        Args:
            key: jax jr key
            shape tuple: desired shape
        Returns:
            sampled log_step (float32)
        """
        return jr.uniform(key, shape) * (jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(
            dt_min
        )

    return init


def init_log_steps(key, input):
    """Initialize an array of learnable timescale parameters
    Args:
        key: jax jr key
        input: tuple containing the array shape H and
               dt_min and dt_max
    Returns:
        initialized array of timescales (float32): (H,)
    """
    H, dt_min, dt_max = input
    log_steps = []
    for i in range(H):
        key, skey = jr.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)

    return jnp.array(log_steps)


def init_VinvB(init_fun, rng, shape, Vinv):
    """Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
    Note we will parameterize this with two different matrices for complex
    numbers.
     Args:
         init_fun:  the initialization function to use, e.g. lecun_normal()
         rng:       jax jr key to be used with init function.
         shape (tuple): desired shape  (P,H)
         Vinv: (complex64)     the inverse eigenvectors used for initialization
     Returns:
         B_tilde (complex64) of shape (P,H,2)
    """
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return jnp.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def trunc_standard_normal(key, shape):
    """Sample C with a truncated normal distribution with standard deviation 1.
    Args:
        key: jax jr key
        shape (tuple): desired shape, of length 3, (H,P,_)
    Returns:
        sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
    """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = jr.split(key)
        C = lecun_normal()(skey, shape=(1, P, 2))
        Cs.append(C)
    return jnp.array(Cs)[:, 0]


def init_CV(init_fun, rng, shape, V):
    """Initialize C_tilde=CV. First sample C. Then compute CV.
    Note we will parameterize this with two different matrices for complex
    numbers.
     Args:
         init_fun:  the initialization function to use, e.g. lecun_normal()
         rng:       jax jr key to be used with init function.
         shape (tuple): desired shape  (H,P)
         V: (complex64)     the eigenvectors used for initialization
     Returns:
         C_tilde (complex64) of shape (H,P,2)
    """
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return jnp.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretisation step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = jnp.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretisation step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym):
    """Compute the LxH output of discretized SSM given an LxH input.
    Args:
        Lambda_bar (complex64): discretized diagonal state matrix    (P,)
        B_bar      (complex64): discretized input matrix             (P, H)
        C_tilde    (complex64): output matrix                        (H, P)
        input_sequence (float32): input sequence of features         (L, H)
        conj_sym (bool):         whether conjugate symmetry is enforced
    Returns:
        ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * jnp.ones(
        (input_sequence.shape[0], Lambda_bar.shape[0])
    )
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if conj_sym:
        return jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)


class S5Layer(eqx.Module):
    Lambda_re: jax.Array
    Lambda_im: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    log_step: jax.Array

    H: int
    P: int
    conj_sym: bool
    clip_eigs: bool = False
    discretisation: str
    step_rescale: float = 1.0

    def __init__(
        self,
        ssm_size,
        blocks,
        H,
        C_init,
        conj_sym,
        clip_eigs,
        discretisation,
        dt_min,
        dt_max,
        step_rescale,
        *,
        key
    ):

        B_key, C_key, D_key, step_key, key = jr.split(key, 5)

        block_size = int(ssm_size / blocks)
        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

        if conj_sym:
            block_size = block_size // 2
            P = ssm_size // 2
        else:
            P = ssm_size

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = (Lambda * jnp.ones((blocks, block_size))).ravel()
        V = block_diag(*([V] * blocks))
        Vinv = block_diag(*([Vc] * blocks))

        self.H = H
        self.P = P
        if conj_sym:
            local_P = 2 * P
        else:
            local_P = P

        self.Lambda_re = Lambda.real
        self.Lambda_im = Lambda.imag

        self.conj_sym = conj_sym

        self.clip_eigs = clip_eigs

        self.B = init_VinvB(lecun_normal(), B_key, (local_P, self.H), Vinv)

        # Initialize state to output (C) matrix
        if C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
        elif C_init in ["lecun_normal"]:
            C_init = lecun_normal()
        elif C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5**0.5)
        else:
            raise NotImplementedError("C_init method {} not implemented".format(C_init))

        if C_init in ["complex_normal"]:
            self.C = C_init(C_key, (self.H, 2 * self.P, 2))
        else:
            self.C = init_CV(C_init, C_key, (self.H, local_P, 2), V)

        self.D = normal(stddev=1.0)(D_key, (self.H,))

        # Initialize learnable discretisation timescale value
        self.log_step = init_log_steps(step_key, (self.P, dt_min, dt_max))

        self.step_rescale = step_rescale
        self.discretisation = discretisation

    def __call__(self, input_sequence):
        if self.clip_eigs:
            Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            Lambda = self.Lambda_re + 1j * self.Lambda_im

        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        step = self.step_rescale * jnp.exp(self.log_step[:, 0])

        # Discretize
        if self.discretisation in ["zoh"]:
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        elif self.discretisation in ["bilinear"]:
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)
        else:
            raise NotImplementedError(
                "Discretization method {} not implemented".format(self.discretisation)
            )

        ys = apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, self.conj_sym)

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


class S5Block(eqx.Module):

    norm: eqx.nn.BatchNorm
    ssm: S5Layer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        ssm_size,
        blocks,
        H,
        C_init,
        conj_sym,
        clip_eigs,
        discretisation,
        dt_min,
        dt_max,
        step_rescale,
        drop_rate=0.05,
        *,
        key
    ):
        ssmkey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=H, axis_name="batch", channelwise_affine=False
        )
        self.ssm = S5Layer(
            ssm_size,
            blocks,
            H,
            C_init,
            conj_sym,
            clip_eigs,
            discretisation,
            dt_min,
            dt_max,
            step_rescale,
            key=ssmkey,
        )
        self.glu = GLU(H, H, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        """Compute S5 block."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.ssm(x)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x
        return x, state


class S5(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: List[S5Block]
    linear_layer: eqx.nn.Linear
    classification: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        num_blocks,
        N,
        ssm_size,
        ssm_blocks,
        H,
        output_dim,
        classification,
        output_step,
        C_init,
        conj_sym,
        clip_eigs,
        discretisation,
        dt_min,
        dt_max,
        step_rescale,
        *,
        key
    ):

        linear_encoder_key, *block_keys, linear_layer_key, weightkey = jr.split(
            key, num_blocks + 3
        )
        self.linear_encoder = eqx.nn.Linear(N, H, key=linear_encoder_key)
        self.blocks = [
            S5Block(
                ssm_size,
                ssm_blocks,
                H,
                C_init,
                conj_sym,
                clip_eigs,
                discretisation,
                dt_min,
                dt_max,
                step_rescale,
                key=key,
            )
            for key in block_keys
        ]
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)
        self.classification = classification
        self.output_step = output_step

    def __call__(self, x, state, key):
        """Compute S5."""
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)
        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)
        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.nn.tanh(jax.vmap(self.linear_layer)(x))
        return x, state
