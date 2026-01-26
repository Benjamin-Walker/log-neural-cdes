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

from typing import List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import roughpy as rp
from jax.experimental.sparse import random_bcoo
from scipy.linalg import hadamard


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
    hidden_dim: int
    block_size: int
    num_blocks: int
    parallel_steps: int
    logsig_depth: int
    basis_list: List[Tuple[int, ...]]
    lambd: float
    w_init_std: float
    classification: bool
    walsh_hadamard: bool
    diagonal_dense: bool
    sparsity: float
    piecewise_abelian: bool
    rank: int

    vf_A: Optional[jnp.ndarray] = None
    hadamard_matrix: Optional[jnp.ndarray] = None
    vf_A_sparse: Optional[jnp.ndarray] = None
    vf_A_diag: Optional[jnp.ndarray] = None
    vf_A_dense: Optional[jnp.ndarray] = None
    vf_A_u: Optional[jnp.ndarray] = None
    vf_A_v: Optional[jnp.ndarray] = None
    buf_len: Optional[int] = None
    dense_size: int = 0
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
        walsh_hadamard: bool = False,
        w_init_std: float = 0.25,
        parallel_steps: int = 128,
        classification: bool = True,
        diagonal_dense: bool = False,
        rank: int = 0,
        sparsity: float = 1.0,
        piecewise_abelian: bool = True,
        buf_len: Optional[int] = None,
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
        self.piecewise_abelian = piecewise_abelian
        self.buf_len = buf_len

        k_init, k_A, k_B = jr.split(key, 3)
        self.init_layer = eqx.nn.Linear(data_dim, hidden_dim, key=k_init)
        self.out_layer = eqx.nn.Linear(hidden_dim, label_dim, key=k_B)

        if diagonal_dense:
            k_A_diag, k_A_dense = jr.split(k_A, 2)
            self.vf_A_diag = (
                jr.normal(k_A_diag, (data_dim, self.hidden_dim - self.block_size))
                * self.w_init_std
            )
            self.vf_A_dense = (
                jr.normal(k_A_dense, (data_dim, block_size, block_size))
                * self.w_init_std
                / jnp.sqrt(block_size)
            )
            self.dense_size = block_size
        elif sparsity < 1.0:
            total = data_dim * hidden_dim * hidden_dim
            nnz = int(jnp.round(total * sparsity))
            if nnz == 0:
                raise ValueError("sparsity too low, no weights left!")

            self.vf_A_sparse = (
                random_bcoo(
                    key=k_A,
                    shape=(data_dim, self.hidden_dim, self.hidden_dim),
                    nse=nnz,
                    generator=jax.random.normal,
                    dtype=jnp.float32,
                )
                * self.w_init_std
                / jnp.sqrt(self.hidden_dim)
            )
            self.block_size = self.hidden_dim
            self.num_blocks = 1
        else:
            self.vf_A = (
                jr.normal(k_A, (data_dim, self.num_blocks * block_size * block_size))
                * self.w_init_std
                / jnp.sqrt(block_size)
            )
            if rank > 0:
                k_A_u, k_A_v = jr.split(k_A, 2)
                self.vf_A_u = (
                    jr.normal(k_A_u, (data_dim, self.hidden_dim, rank))
                    * self.w_init_std
                    / jnp.sqrt(rank)
                )
                self.vf_A_v = (
                    jr.normal(k_A_v, (data_dim, self.hidden_dim, rank))
                    * self.w_init_std
                    / jnp.sqrt(rank)
                )
                self.block_size = self.hidden_dim
                self.num_blocks = 1

        self.classification = classification

        self.walsh_hadamard = walsh_hadamard
        if self.walsh_hadamard:
            hadamard_matrix = hadamard(self.hidden_dim)
            self.hadamard_matrix = jnp.array(
                hadamard_matrix, dtype=jnp.float32
            ) / jnp.sqrt(self.hidden_dim)
            self.block_size = self.hidden_dim
            self.num_blocks = 1
        else:
            self.hadamard_matrix = None
        self.diagonal_dense = diagonal_dense
        self.sparsity = sparsity
        self.rank = rank

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

    def _scan_with_rewind_joint_flow(
        self,
        *,
        ts,
        logsigs,
        obs_mask,
        y0,
        step_from_logsig,
        lie_brackets=None,
    ):
        n = logsigs.shape[0]
        H = y0.shape[0]
        L = logsigs.shape[1]

        # ts assumed length n+1 boundary times
        dt = ts[1:] - ts[:-1]

        obs_mask = jnp.asarray(obs_mask, dtype=bool).reshape((n,))

        buf_len = self.buf_len + 1
        buf_dt0 = jnp.zeros((buf_len,), dtype=dt.dtype)

        carry0 = (
            y0,  # y
            jnp.array(False),  # in_run
            y0,  # y_saved
            jnp.array(0, jnp.int32),  # buf_count
            buf_dt0,  # buf_dt
            jnp.zeros((L,), logsigs.dtype),  # logsig_sum
            jnp.array(0.0, dt.dtype),  # dt_sum
        )

        def step_outer(carry, inp):
            y, in_run, y_saved, buf_count, buf_dt, logsig_sum, dt_sum = carry
            li, dti, mi = inp

            def do_false():
                start_new = ~in_run
                y_saved_new = jnp.where(start_new, y, y_saved)
                buf_count_new = jnp.where(start_new, jnp.array(0, jnp.int32), buf_count)
                logsig_sum_new = jnp.where(
                    start_new, jnp.zeros_like(logsig_sum), logsig_sum
                )
                dt_sum_new = jnp.where(
                    start_new, jnp.array(0.0, dtype=dti.dtype), dt_sum
                )

                buf_dt_new = buf_dt.at[buf_count_new].set(dti)
                buf_count_new = buf_count_new + 1

                logsig_sum_new = logsig_sum_new + li
                dt_sum_new = dt_sum_new + dti

                y_next = step_from_logsig(y, li)
                # output is committed now
                return (
                    y_next,
                    True,
                    y_saved_new,
                    buf_count_new,
                    buf_dt_new,
                    logsig_sum_new,
                    dt_sum_new,
                ), y_next

            def do_true():
                def no_rewind():
                    y_next = step_from_logsig(y, li)
                    return (
                        y_next,
                        False,
                        y_next,
                        buf_count,
                        buf_dt,
                        logsig_sum,
                        dt_sum,
                    ), y_next

                def do_rewind():
                    total_logsig = logsig_sum + li
                    total_dt = dt_sum + dti
                    total_dt = jnp.where(
                        total_dt > 0, total_dt, jnp.array(1.0, dtype=total_dt.dtype)
                    )

                    buf_dt2 = buf_dt.at[buf_count].set(dti)
                    k = buf_count + 1  # dynamic, but we will use static loop bound

                    # compute generator once if available
                    if lie_brackets is not None:
                        a_total = total_logsig[1:]
                        G = jnp.einsum(
                            "ijkl,l->ijk", lie_brackets, a_total
                        )  # (nb, bs, bs)
                        nb, bs, _ = G.shape

                        def inner(jj, ytmp):
                            def do_step(y_):
                                w = buf_dt2[jj] / total_dt
                                yb = y_.reshape(nb, bs, 1)
                                return (yb + w * (G @ yb)).reshape(
                                    H,
                                )

                            return jax.lax.cond(jj < k, do_step, lambda y_: y_, ytmp)

                    else:

                        def inner(jj, ytmp):
                            def do_step(y_):
                                w = buf_dt2[jj] / total_dt
                                return step_from_logsig(y_, total_logsig * w)

                            return jax.lax.cond(jj < k, do_step, lambda y_: y_, ytmp)

                    # STATIC bound => reverse-mode works
                    y_replayed = jax.lax.fori_loop(0, buf_len, inner, y_saved)

                    # output only computed once for this True interval
                    new_carry = (
                        y_replayed,
                        False,
                        y_replayed,
                        jnp.array(0, jnp.int32),
                        buf_dt,
                        jnp.zeros_like(logsig_sum),
                        jnp.array(0.0, dtype=dt.dtype),
                    )
                    return new_carry, y_replayed

                return jax.lax.cond(in_run, lambda: do_rewind(), lambda: no_rewind())

            return jax.lax.cond(mi, lambda: do_true(), lambda: do_false())

        # IMPORTANT: no ys scatter updates, scan collects outputs
        carryT, ys = jax.lax.scan(step_outer, carry0, (logsigs, dt, obs_mask))
        return ys

    def __call__(self, X):
        ts, logsig_obs, x0 = X
        logsigs, obs = logsig_obs
        y0 = self.init_layer(x0)

        # Each branch prepares `flows` and `step`/`parallel_step` functions
        # for the generic scanner loop at the end.

        if self.walsh_hadamard and self.parallel_steps == 1 and self.logsig_depth == 1:
            flows = logsigs[:, 1:] @ self.vf_A

            def step(y, flow):
                # flowed = fwht(flow * y)
                flowed = self.hadamard_matrix @ (flow * y)
                y_next = y + flowed
                return y_next, y_next

            parallel_step = None

        elif self.rank > 0 and self.logsig_depth == 1 and self.parallel_steps == 1:
            a_seq = logsigs[:, 1:]  # (T-1, C)

            def step(y, a):  # a: (C,)
                diag = (a @ self.vf_A) * y  # (H,)
                vTy = jnp.einsum("chr,h->cr", self.vf_A_v, y)  # (C, R), V_c^T y
                lowrank = jnp.einsum("chr,cr->h", self.vf_A_u, a[:, None] * vTy)  # (H,)
                y_next = y + diag + lowrank
                return y_next, y_next

            parallel_step = None
            flows = a_seq

        elif self.diagonal_dense:
            diag_size = self.hidden_dim - self.dense_size

            # Calculate flows for both parts
            num_log_sig_coeffs = self.vf_A_diag.shape[0]
            update_diag = logsigs[:, 1 : num_log_sig_coeffs + 1] @ self.vf_A_diag
            flows_diag = 1 + update_diag

            lie_brackets_dense = self.log_ode(self.vf_A_dense)
            log_flows_dense = jnp.einsum(
                "jkl,ml->mjk", lie_brackets_dense, logsigs[:, 1:]
            )
            flows_dense = log_flows_dense + jnp.eye(self.dense_size)[None, :, :]
            flows = (flows_diag, flows_dense)

            def step(y, flow_slice):
                flow_diag, flow_dense = flow_slice
                y_diag, y_dense = y[:diag_size], y[diag_size:]
                y_next_diag = y_diag * flow_diag
                y_next_dense = flow_dense @ y_dense
                y_next = jnp.concatenate([y_next_diag, y_next_dense])
                return y_next, y_next

            def parallel_step_comp(flow1, flow2):
                flow1_diag, flow1_dense = flow1
                flow2_diag, flow2_dense = flow2
                return (flow1_diag * flow2_diag, flow2_dense @ flow1_dense)

            def parallel_step(y, flows_slice):
                y_diag, y_dense = y[:diag_size], y[diag_size:]
                total_flows = jax.lax.associative_scan(parallel_step_comp, flows_slice)
                total_flows_diag, total_flows_dense = total_flows
                ys_diag_new = y_diag * total_flows_diag
                ys_dense_new = jnp.einsum("kij,j->ki", total_flows_dense, y_dense)
                ys_new = jnp.concatenate([ys_diag_new, ys_dense_new], axis=-1)
                return ys_new[-1], ys_new

        elif self.block_size == 1 and self.rank == 0 and not self.walsh_hadamard:
            num_log_sig_coeffs = self.vf_A.shape[0]
            updates = logsigs[:, 1 : num_log_sig_coeffs + 1] @ self.vf_A
            flows = 1 + updates

            def step(y, flow):
                return y * flow, y * flow

            def parallel_step_comp(f1, f2):
                return f1 * f2

            def parallel_step(y, flows_slice):
                total_flows = jax.lax.associative_scan(parallel_step_comp, flows_slice)
                ys_new = y * total_flows
                return ys_new[-1], ys_new

            def step_from_logsig(y, logsig):
                flow = 1 + logsig[1 : num_log_sig_coeffs + 1] @ self.vf_A
                return y * flow

        else:  # Generic case for block_size > 1
            if self.sparsity < 1.0:
                vfs = self.vf_A_sparse.todense().reshape(
                    -1, 1, self.hidden_dim, self.hidden_dim
                )
            elif self.rank > 0:
                low_rank = jnp.einsum("cir,cjr->cij", self.vf_A_u, self.vf_A_v)
                diags = jax.vmap(jnp.diag)(self.vf_A)
                vfs = (low_rank + diags)[:, None, :, :]
            elif self.walsh_hadamard:
                vfs = self.vf_A[:, None, :] * self.hadamard_matrix[None, :, :]
                vfs = jnp.reshape(vfs, (-1, 1, self.hidden_dim, self.hidden_dim))
            else:
                vfs = self.vf_A.reshape(
                    -1, self.num_blocks, self.block_size, self.block_size
                )

            lie_brackets = jax.vmap(self.log_ode, in_axes=(1))(vfs)
            log_flows = jnp.einsum("ijkl,ml->mijk", lie_brackets, logsigs[:, 1:])
            flows = log_flows + jnp.eye(self.block_size)[None, None, :, :]

            step_comp = lambda x, y: x @ y
            parallel_step_comp = lambda x, y: jnp.matmul(y, x)

            def step(y, flow):
                y_block = y.reshape(self.num_blocks, self.block_size, 1)
                y_next = step_comp(flow, y_block).reshape(
                    self.hidden_dim,
                )
                return y_next, y_next

            def parallel_step(y, flows):
                flow_total = jax.lax.associative_scan(parallel_step_comp, flows)
                y_block = y.reshape(self.num_blocks, self.block_size, 1)
                y_new = jnp.matmul(flow_total, y_block).reshape(-1, self.hidden_dim)
                return y_new[-1], y_new

            def step_from_logsig(y, logsig):
                a = logsig[1:]
                log_flow = jnp.einsum("ijkl,l->ijk", lie_brackets, a)
                flow = log_flow + jnp.eye(self.block_size)[None, :, :]
                y_block = y.reshape(self.num_blocks, self.block_size, 1)
                y_next = jnp.matmul(flow, y_block).reshape(
                    self.hidden_dim,
                )
                return y_next

        if (not self.piecewise_abelian) and (self.parallel_steps == 1):

            def scan_simple(_):
                _, ys = jax.lax.scan(step, y0, flows)
                return ys

            ys = jax.lax.cond(
                jnp.any(~obs),
                lambda _: self._scan_with_rewind_joint_flow(
                    ts=ts,
                    logsigs=logsigs,
                    obs_mask=obs,
                    y0=y0,
                    step_from_logsig=step_from_logsig,
                    lie_brackets=lie_brackets,
                ),
                scan_simple,
                operand=None,
            )
            ys = jnp.vstack([y0, ys])

            # --- Final output layer (same as yours, but using ys computed above) ---
            if self.classification:
                ys_mean = jnp.mean(ys, axis=0)
                preds = jax.nn.softmax(self.out_layer(ys_mean))
            else:
                ys_out = jax.vmap(self.out_layer)(ys)
                preds = jnp.tanh(ys_out)

            return preds

        # --- Generic scanner logic ---
        if self.parallel_steps == 1 or parallel_step is None:
            scan_fn = step
            scan_inp = flows
            remainder = 0
        else:
            scan_fn = parallel_step
            t = jax.tree_util.tree_leaves(flows)[0].shape[0]
            remainder = (t - 1) % self.parallel_steps

            if remainder == 0:
                core_flows = flows
            else:
                core_flows = jax.tree_util.tree_map(lambda x: x[:-remainder], flows)

            scan_inp = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, self.parallel_steps, *x.shape), core_flows
            )

        _, ys = jax.lax.scan(scan_fn, y0, scan_inp)

        if not (
            self.parallel_steps == 1 or parallel_step is None
        ):  # Unpack parallel results
            ys = ys.reshape(-1, self.hidden_dim)

        ys = jnp.vstack([y0, ys])

        if remainder != 0:
            rem_flows = jax.tree_util.tree_map(lambda x: x[-remainder:], flows)
            _, y_rem = jax.lax.scan(step, ys[-1], rem_flows)
            ys = jnp.vstack([ys, y_rem])

        # --- Final output layer ---
        if self.classification:
            ys = jnp.mean(ys, axis=0)
            preds = jax.nn.softmax(self.out_layer(ys))
        else:
            ys = jax.vmap(self.out_layer)(ys)
            preds = jnp.tanh(ys)

        return preds
