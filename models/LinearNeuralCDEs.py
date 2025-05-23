"""
This module implements the `LogLinearCDE` class using JAX and Equinox. The model is a
block-diagonal Linear Controlled Differential Equation (CDE), where the output is
approximated during training using the Log-ODE method.

Attributes of the `LogLinearCDE` model:
- `init_layer`: The linear layer used to initialize the hidden state $h_0$ from the input $x_0$.
- `out_layer`: The linear layer used to produce final predictions from the hidden state.
- `vf_A`: Learnable parameters for the linear vector field, shaped as flattened block matrices.
- `hidden_dim`: The dimension of the hidden state $h_t$.
- `block_size`: Size of each square block in the block-diagonal vector field.
- `num_blocks`: Number of blocks, computed as `hidden_dim // block_size`.
- `parallel_steps`: Number of log-flow matrices composed in parallel (using associative scan).
- `logsig_depth`: The depth of the log-signature used in the Log-ODE method.
- `basis_list`: The list of basis elements of the free Lie algebra up to the specified depth.
- `lambd`: Regularization parameter applied to vector field scaling.
- `w_init_std`: Standard deviation for the initial weights of the vector field.
- `classification`: Boolean indicating if the model is used for classification tasks.

The class includes:
- `log_ode`: Method for computing the iterated Lie brackets of the linear vector fields.
- `__call__`: Performs the forward pass, where flows are composed and applied to the hidden state
  either step-by-step or in parallel (using associative scan), followed by output projection.
"""

from __future__ import annotations

from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import roughpy as rp


def to_tuple(el):
    """Convert a basis element which may be an int or a nested [x,y] list into a nested tuple."""
    if isinstance(el, int):
        return (el,)
    else:
        return to_tuple(el[0]), to_tuple(el[1])


def depth(b):
    """Compute the 'depth' of a bracket structure."""
    if isinstance(b, int):
        return 1
    elif isinstance(b, list):
        return max(depth(b[0]), depth(b[1])) + 1
    else:
        raise TypeError("Invalid basis element type.")


class LogLinearCDE(eqx.Module):
    init_layer: eqx.nn.Linear
    out_layer: eqx.nn.Linear
    vf_A: jnp.ndarray
    hidden_dim: int
    block_size: int
    num_blocks: int
    parallel_steps: int
    logsig_depth: int
    basis_list: List[Tuple[int, ...]]
    lambd: float
    w_init_std: float
    classification: bool

    lip2: bool = True
    nondeterministic: bool = False
    stateful: bool = False

    def __init__(
        self,
        *,
        data_dim: int,
        hidden_dim: int,
        label_dim: int,
        block_size: int,
        logsig_depth: int,
        lambd: float = 1.0,
        w_init_std: float = 0.25,
        parallel_steps: int = 128,
        classification: bool = True,
        key,
    ):
        if hidden_dim % block_size != 0:
            raise ValueError("hidden_dim must be divisible by block_size.")
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.num_blocks = hidden_dim // block_size
        self.parallel_steps = parallel_steps
        self.logsig_depth = logsig_depth
        ctx = rp.get_context(width=data_dim, depth=self.logsig_depth, coeffs=rp.DPReal)
        basis = ctx.lie_basis
        basis_list = []
        for i in range(basis.size(self.logsig_depth)):
            basis_list.append(eval(str(basis.index_to_key(i))))
        self.basis_list = basis_list
        self.lambd = lambd
        self.w_init_std = w_init_std

        k_init, k_A, k_B = jr.split(key, 3)
        self.init_layer = eqx.nn.Linear(data_dim, hidden_dim, key=k_init)
        self.out_layer = eqx.nn.Linear(hidden_dim, label_dim, key=k_B)

        self.vf_A = (
            jr.normal(k_A, (data_dim + 1, self.num_blocks * block_size * block_size))
            * self.w_init_std
            / jnp.sqrt(block_size)
        )
        self.classification = classification

    def log_ode(self, vf):

        basis_index = {}
        for i, b in enumerate(self.basis_list):
            basis_index[to_tuple(b)] = i

        depth_to_elements = {}
        for i, b in enumerate(self.basis_list):
            d = depth(b)
            depth_to_elements.setdefault(d, []).append((i, b))

        A_arrays = [None] * len(self.basis_list)

        for i_b, b in depth_to_elements[1]:
            A_arrays[i_b] = vf[b - 1, :, :]

        max_depth = max(depth_to_elements.keys())
        for d in range(2, max_depth + 1):
            curr_elements = depth_to_elements[d]

            left_indices = []
            right_indices = []
            for i_b, b in curr_elements:
                u_tuple = to_tuple(b[0])
                v_tuple = to_tuple(b[1])
                i_u = basis_index[u_tuple]
                i_v = basis_index[v_tuple]
                left_indices.append(i_u)
                right_indices.append(i_v)

            A_left = jnp.stack([A_arrays[i_u] for i_u in left_indices], axis=0)
            A_right = jnp.stack([A_arrays[i_v] for i_v in right_indices], axis=0)

            A_uv = jnp.einsum("ijk,ikl->ijl", A_right, A_left) - jnp.einsum(
                "ijk,ikl->ijl", A_left, A_right
            )

            for idx, (i_b, b) in enumerate(curr_elements):
                A_arrays[i_b] = A_uv[idx]

        return jnp.stack(A_arrays, axis=2)

    def __call__(self, X):
        ts, logsigs, x0 = X

        y0 = self.init_layer(x0)

        vfs = self.vf_A.reshape(-1, self.num_blocks, self.block_size, self.block_size)
        lie_brackets = jax.vmap(self.log_ode, in_axes=(1))(vfs)
        log_flows = jnp.einsum("ijkl,ml->mijk", lie_brackets, logsigs[:, 1:])
        flows = log_flows + jnp.eye(self.block_size)[None, None, :, :]

        def step(y, flow):
            y_block = y.reshape(self.num_blocks, self.block_size, 1)
            y_next = flow @ y_block
            y_next = y_next.reshape(
                self.hidden_dim,
            )
            return y_next, y_next

        def parallel_step(y, flows):
            compose = lambda a, b: jnp.matmul(b, a)
            flow_total = jax.lax.associative_scan(compose, flows)
            y_block = y.reshape(self.num_blocks, self.block_size, 1)
            y_new = jnp.matmul(flow_total, y_block).reshape(-1, self.hidden_dim)
            return y_new[-1], y_new

        if self.parallel_steps == 1:
            scan_fn = step
            remainder = 0
            scan_inp = flows
        else:
            scan_fn = parallel_step
            t = len(flows)
            remainder = (t - 1) % self.parallel_steps
            core = flows[1:] if remainder == 0 else flows[1:-remainder]
            scan_inp = jnp.reshape(
                core,
                (
                    -1,
                    self.parallel_steps,
                    self.num_blocks,
                    self.block_size,
                    self.block_size,
                ),
            )

        _, ys = jax.lax.scan(scan_fn, y0, scan_inp)  # (T‑1, H)
        if len(ys.shape) == 3:
            ys = jnp.reshape(ys, (-1, self.hidden_dim))
        ys = jnp.vstack([y0, ys])
        if remainder != 0:
            inp_rem = flows[-remainder:]
            _, y_rem = jax.lax.scan(step, ys[-1], inp_rem)
            ys = jnp.vstack([ys, y_rem])

        if self.classification:
            ys = jnp.mean(ys, axis=0)
            preds = jax.nn.softmax(self.out_layer(ys))
        else:
            ys = jax.vmap(self.out_layer)(ys)
            preds = jnp.tanh(ys)

        return preds
