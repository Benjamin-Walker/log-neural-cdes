"""
This module contains functions for generating log-signature of paths over intervals of length stepsize.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from signax.signature import signature
from signax.signature_flattened import flatten
from signax.tensor_ops import log

from data_dir.hall_set import HallSet


def hall_basis_logsig(
    x: jnp.ndarray, depth: int, t2l: Optional[jnp.ndarray]
) -> jnp.ndarray:
    logsig = flatten(log(signature(x, depth)))
    if depth == 1:
        # Keep existing behaviour: prepend 0
        return jnp.concatenate((jnp.array([0], dtype=x.dtype), logsig))
    else:
        return t2l[:, 1:] @ logsig


def calc_paths(
    data: jnp.ndarray,
    stepsize: int,
    depth: int,
    include_time: bool = False,
    time_index: int = 0,
    interval_times: Optional[jnp.ndarray] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate per-interval log-signatures from data.

    Args:
        data: array (B, T, C)
        stepsize: chunk size (only used when interval_times is None)
        depth: signature depth (supported: 1 or 2)
        include_time: whether time is included as one of the channels
        time_index: channel corresponding to time
        interval_times:
            If not None, overrides stepsize splitting. Can be:
              - (K+1,) shared across batch
              - (B, K+1) per-sample

            For each interval (a,b]:
              - include points with time in (a,b], except for the FIRST interval
                where we include [a,b] so observations at t=0 count.
              - ALWAYS stitch continuity by prepending previous interval end at time=a.
              - ALWAYS add a phantom endpoint at time=b (spatial frozen from last point).
              - That phantom endpoint is used as the start (prev_end) of the next interval.

    Returns:
        logsigs: (B, n_intervals, D)
        observation_mask: (B, n_intervals) boolean
            True  -> interval had >= 1 real observation assigned to it
            False -> interval had no real observations (only stitch + phantom end)

        When interval_times is None, observation_mask is all True.

    Assumptions:
        - In interval_times mode, within each sample the time channel is nondecreasing.
    """
    if interval_times is not None and not include_time:
        raise ValueError(
            "interval_times != None requires include_time=True (time must be a channel)."
        )

    if depth > 2:
        raise ValueError(
            "Currently only supports depth <= 2 (HallSet t2l matrix only built for depth=2)."
        )

    # Prepend an initial zero row
    data = jnp.concatenate(
        (jnp.zeros((data.shape[0], 1, data.shape[-1]), dtype=data.dtype), data),
        axis=1,
    )

    if depth == 2:
        hs = HallSet(data.shape[-1], depth)
        t2l = hs.t2l_matrix(depth)
    else:
        t2l = None

    # ---------------------------------------------------------------------
    # interval_times mode (shared or batched)
    # ---------------------------------------------------------------------
    if interval_times is not None:
        interval_times = jnp.asarray(interval_times, dtype=data.dtype)
        B, T_total, C = data.shape

        if interval_times.ndim == 1:
            if interval_times.shape[0] < 2:
                raise ValueError("interval_times must have shape (K+1,) with K >= 1.")
            interval_times = jnp.broadcast_to(
                interval_times[None, :], (B, interval_times.shape[0])
            )

        def process_one_batch(data_b: jnp.ndarray, times_b: jnp.ndarray):
            """
            data_b:  (T_total, C)
            times_b: (K+1,)
            returns:
              logsigs_b: (K, D)
              obs_b:     (K,)
            """
            t_b = data_b[:, time_index]
            K = times_b.shape[0] - 1
            a0 = times_b[0]
            prev_end = data_b[0].at[time_index].set(a0)
            a_arr = times_b[:-1]
            b_arr = times_b[1:]
            i_arr = jnp.arange(K, dtype=jnp.int32)

            def step(prev_end, inp):
                i, a, b = inp

                mask_base = jax.lax.cond(
                    i == 0,
                    lambda _: (t_b >= a) & (t_b <= b),
                    lambda _: (t_b > a) & (t_b <= b),
                    operand=None,
                )
                mask = mask_base

                count = jnp.sum(mask).astype(jnp.int32)
                has_obs = count > 0

                idx = jnp.nonzero(mask, size=T_total, fill_value=0)[0]
                pts = data_b[idx, :]

                start = prev_end.at[time_index].set(a)

                safe_last = jnp.clip(count - 1, 0, T_total - 1)
                last_obs = pts[safe_last, :]

                last_point = jax.lax.select(has_obs, last_obs, start)

                pos = jnp.arange(T_total, dtype=jnp.int32)
                pts_padded = jnp.where((pos < count)[:, None], pts, last_point[None, :])

                path_mid = jnp.concatenate([start[None, :], pts_padded], axis=0)
                end = path_mid[-1].at[time_index].set(b)
                path = jnp.concatenate([path_mid, end[None, :]], axis=0)

                logsig = hall_basis_logsig(path, depth, t2l)
                new_end = end

                return new_end, (logsig, has_obs)

            _, (logsigs_b, obs_b) = jax.lax.scan(step, prev_end, (i_arr, a_arr, b_arr))
            return logsigs_b, obs_b

        logsigs, observation_mask = jax.vmap(process_one_batch)(data, interval_times)
        return logsigs, observation_mask

    if stepsize > data.shape[1]:
        stepsize = data.shape[1]

    B, T_total, C = data.shape

    if stepsize >= (T_total - 1):
        intervals = data[:, None, :, :]  # (B, 1, T_total, C)
        final_interval = None
        n_full = 1
    else:
        n_full = (T_total - 1) // stepsize

        idxs = jnp.arange(stepsize + 1)[None, :] + (
            jnp.arange(n_full)[:, None] * stepsize
        )  # (n_full, stepsize+1)

        intervals = jnp.take(data, idxs, axis=1)  # (B, n_full, stepsize+1, C)

        final_start = n_full * stepsize
        if final_start < (T_total - 1):
            final_interval = data[:, final_start:, :]  # (B, L_final, C)
        else:
            final_interval = None

    vmap_calc_logsig = jax.vmap(hall_basis_logsig, in_axes=(0, None, None))
    logsigs = jax.vmap(vmap_calc_logsig, in_axes=(0, None, None))(intervals, depth, t2l)

    if final_interval is not None:
        final_logsigs = vmap_calc_logsig(final_interval, depth, t2l)[:, None, :]
        logsigs = jnp.concatenate((logsigs, final_logsigs), axis=1)

    observation_mask = jnp.ones((logsigs.shape[0], logsigs.shape[1]), dtype=bool)
    return logsigs, observation_mask


# def rectilinear_interpolate_time_channel(
#     x: jnp.ndarray, time_index: int = 0
# ) -> jnp.ndarray:
#     """
#     Convert a path x(t) with time included as one channel into a rectilinear path.
#
#     If x has points (t_i, y_i), then each increment is replaced by two moves:
#       1) (t_i, y_i) -> (t_{i+1}, y_i)         (time moves, data frozen)
#       2) (t_{i+1}, y_i) -> (t_{i+1}, y_{i+1}) (data moves, time frozen)
#
#     Input:
#       x: shape (L, C)
#
#     Output:
#       shape (2*L - 1, C)
#     """
#     L, C = x.shape
#     if L <= 1:
#         return x
#
#     x0 = x[:-1]  # (L-1, C)
#     x1 = x[1:]  # (L-1, C)
#
#     # Intermediate points: same as x0, but with time replaced by next time
#     mid = x0.at[:, time_index].set(x1[:, time_index])  # (L-1, C)
#
#     out = jnp.zeros((2 * L - 1, C), dtype=x.dtype)
#     out = out.at[0::2].set(x)  # positions 0,2,4,... are original points
#     out = out.at[1::2].set(mid)  # positions 1,3,5,... are "time-only" steps
#     return out
#
#
# IntervalGapMode = Literal["none", "time_only", "linear"]
#
#
# def calc_paths(
#     data: jnp.ndarray,
#     stepsize: int,
#     depth: int,
#     include_time: bool = False,
#     rectilinear_interpolation: bool = False,
#     time_index: int = 0,
#     interval_gap_mode: IntervalGapMode = "none",
#     gap_n_intervals: int = 0,
#     interval_times: Optional[jnp.ndarray] = None,
# ) -> jnp.ndarray:
#     """
#     Generate log-signature objects from data.
#
#     Args:
#         data: array of shape (batch, T, channels)
#         stepsize: number of steps per chunk
#         depth: signature depth
#         include_time: whether time is included as one of the channels
#         rectilinear_interpolation: if True and include_time=True, apply rectilinear
#             interpolation (axis path) before computing log-signatures
#         time_index: which channel corresponds to time (only used if include_time=True)
#
#         interval_gap_mode:
#             "none"      -> no special handling
#             "time_only" -> after each normal interval, make the next gap_n_intervals
#                           intervals time-only (spatial channels frozen)
#             "linear"    -> after each normal interval, replace the next gap_n_intervals
#                           intervals by linear interpolation towards the first true value
#                           of the next normal interval
#
#         gap_n_intervals:
#             Number of consecutive intervals after each normal interval to freeze/linear-fill.
#             Pattern repeats with period (gap_n_intervals + 1).
#
#     Notes:
#         - rectilinear_interpolation=True requires include_time=True
#         - interval_gap_mode != "none" requires include_time=True
#         - "time_only" and "linear" maintain continuity by stitching interval starts
#           to the previous interval end.
#     """
#     if gap_n_intervals < 0:
#         raise ValueError("gap_n_intervals must be >= 0.")
#
#     if interval_gap_mode not in ("none", "time_only", "linear"):
#         raise ValueError(
#             'interval_gap_mode must be one of "none", "time_only", "linear".'
#         )
#
#     if rectilinear_interpolation and not include_time:
#         raise ValueError(
#             "rectilinear_interpolation=True requires include_time=True (time must be a channel)."
#         )
#
#     if interval_gap_mode != "none" and not include_time:
#         raise ValueError("interval_gap_mode != 'none' requires include_time=True.")
#
#     if gap_n_intervals == 0:
#         interval_gap_mode = "none"
#
#     # Prepend an initial zero point (keeps existing behaviour)
#     data = jnp.concatenate(
#         (jnp.zeros((data.shape[0], 1, data.shape[-1]), dtype=data.dtype), data), axis=1
#     )
#
#     if depth == 2:
#         hs = HallSet(data.shape[-1], depth)
#         t2l = hs.t2l_matrix(depth)
#     else:
#         t2l = None
#
#     if stepsize > data.shape[1]:
#         stepsize = data.shape[1]
#
#     # ---------------------------------------------------------------------
#     # Build overlapping intervals directly:
#     # interval i uses data[:, i*stepsize : i*stepsize + stepsize + 1, :]
#     # ---------------------------------------------------------------------
#     B, T_total, C = data.shape
#
#     if stepsize >= (T_total - 1):
#         intervals = data[:, None, :, :]  # (B, 1, T_total, C)
#         final_interval = None
#         n_full = 1
#     else:
#         n_full = (T_total - 1) // stepsize
#
#         idxs = jnp.arange(stepsize + 1)[None, :] + (
#             jnp.arange(n_full)[:, None] * stepsize
#         )  # (n_full, stepsize+1)
#
#         intervals = jnp.take(data, idxs, axis=1)  # (B, n_full, stepsize+1, C)
#
#         final_start = n_full * stepsize
#         if final_start < (T_total - 1):
#             final_interval = data[:, final_start:, :]  # (B, L_final, C)
#         else:
#             final_interval = None
#
#     # ---------------------------------------------------------------------
#     # interval_gap_mode == "time_only"
#     # Pattern: normal, then freeze gap_n_intervals, then normal, ...
#     # ---------------------------------------------------------------------
#     if interval_gap_mode == "time_only":
#         spatial_mask = jnp.ones((C,), dtype=bool).at[time_index].set(False)
#
#         period = gap_n_intervals + 1
#         freeze_flags = jnp.arange(intervals.shape[1]) % period != 0  # (n_full,)
#
#         def process_one_batch(intervals_b):
#             # intervals_b: (n_full, L, C)
#
#             def freeze_interval(interval_):
#                 start = interval_[0:1, :]  # (1, C)
#                 return jnp.where(
#                     spatial_mask[None, :],
#                     jnp.broadcast_to(start, interval_.shape),
#                     interval_,
#                 )
#
#             def step(prev_end, inp):
#                 interval_, freeze_ = inp
#                 interval_ = interval_.at[0].set(prev_end)  # stitch start
#                 interval_ = jax.lax.cond(
#                     freeze_, freeze_interval, lambda x: x, interval_
#                 )
#                 new_end = interval_[-1]
#                 return new_end, interval_
#
#             init_prev_end = intervals_b[0, 0]  # (C,)
#             last_end, out = jax.lax.scan(
#                 step, init_prev_end, (intervals_b, freeze_flags)
#             )
#             return out, last_end
#
#         intervals, last_ends = jax.vmap(process_one_batch)(intervals)
#
#         # Stitch + optionally freeze the final interval too
#         if final_interval is not None:
#             final_interval = final_interval.at[:, 0, :].set(last_ends)
#
#             final_idx = n_full  # original indexing
#             final_freeze = final_idx % period != 0
#
#             if final_freeze:
#                 start_vals_f = final_interval[:, 0:1, :]  # (B, 1, C)
#                 final_interval = jnp.where(
#                     spatial_mask[None, None, :],
#                     jnp.broadcast_to(start_vals_f, final_interval.shape),
#                     final_interval,
#                 )
#
#     # ---------------------------------------------------------------------
#     # interval_gap_mode == "linear"
#     # Pattern: normal, then replace next gap_n_intervals with linear interpolation
#     #          towards the true start of the next normal interval, then normal, ...
#     # ---------------------------------------------------------------------
#     elif interval_gap_mode == "linear":
#         period = gap_n_intervals + 1
#         n_full_int = intervals.shape[1]
#
#         is_normal = jnp.arange(n_full_int) % period == 0  # (n_full,)
#
#         # next normal interval index for each interval i:
#         # next_idx[i] = ((i // period) + 1) * period
#         next_idx = ((jnp.arange(n_full_int) // period) + 1) * period  # (n_full,)
#
#         # True starts of each full interval
#         full_starts = intervals[:, :, 0, :]  # (B, n_full, C)
#
#         # Include final interval true start as a possible anchor
#         if final_interval is not None:
#             final_start = final_interval[:, 0, :]  # (B, C)
#             anchor_starts = jnp.concatenate(
#                 [full_starts, final_start[:, None, :]], axis=1
#             )
#         else:
#             anchor_starts = full_starts
#
#         anchor_len = anchor_starts.shape[1]
#         has_future = next_idx < anchor_len  # (n_full,)
#
#         # If no future anchor exists, treat it as normal (no linear fill)
#         is_normal_eff = jnp.logical_or(is_normal, jnp.logical_not(has_future))
#
#         next_idx_safe = jnp.minimum(next_idx, anchor_len - 1)
#         targets = jnp.take(anchor_starts, next_idx_safe, axis=1)  # (B, n_full, C)
#
#         def make_linear_interval(interval_orig, start_point, target_point):
#             # interval_orig: (L, C)
#             # start_point/target_point: (C,)
#             t = interval_orig[:, time_index]  # (L,)
#             t0 = start_point[time_index]
#             t1 = target_point[time_index]
#             denom = t1 - t0
#             denom = jnp.where(denom == 0, 1.0, denom)
#
#             alpha = (t - t0) / denom
#             alpha = jnp.clip(alpha, 0.0, 1.0)
#
#             start = start_point[None, :]
#             target = target_point[None, :]
#             out = start + alpha[:, None] * (target - start)
#             return out
#
#         def process_one_batch(intervals_b, targets_b):
#             # intervals_b: (n_full, L, C)
#             # targets_b:   (n_full, C)
#
#             def step(prev_end, inp):
#                 interval_orig, normal_flag, target = inp
#
#                 # Always stitch start for continuity
#                 interval = interval_orig.at[0].set(prev_end)
#
#                 def gap_fn(interval_):
#                     return make_linear_interval(interval_, prev_end, target)
#
#                 interval = jax.lax.cond(normal_flag, lambda x: x, gap_fn, interval)
#                 new_end = interval[-1]
#                 return new_end, interval
#
#             init_prev_end = intervals_b[0, 0]  # (C,)
#             last_end, out = jax.lax.scan(
#                 step, init_prev_end, (intervals_b, is_normal_eff, targets_b)
#             )
#             return out, last_end
#
#         intervals, last_ends = jax.vmap(process_one_batch)(intervals, targets)
#
#         # Stitch final interval to the last processed endpoint
#         if final_interval is not None:
#             final_interval = final_interval.at[:, 0, :].set(last_ends)
#
#     # ---------------------------------------------------------------------
#     # Rectilinear interpolation (optional)
#     # ---------------------------------------------------------------------
#     if rectilinear_interpolation:
#         rect_fn = partial(rectilinear_interpolate_time_channel, time_index=time_index)
#         intervals = jax.vmap(jax.vmap(rect_fn))(intervals)
#         if final_interval is not None:
#             final_interval = jax.vmap(rect_fn)(final_interval)
#
#     # ---------------------------------------------------------------------
#     # Compute log-signatures
#     # ---------------------------------------------------------------------
#     vmap_calc_logsig = jax.vmap(hall_basis_logsig, in_axes=(0, None, None))
#     logsigs = jax.vmap(vmap_calc_logsig, in_axes=(0, None, None))(intervals, depth, t2l)
#
#     if final_interval is not None:
#         final_logsigs = vmap_calc_logsig(final_interval, depth, t2l)[:, None, :]
#         logsigs = jnp.concatenate((logsigs, final_logsigs), axis=1)
#
#     return logsigs
#
#
# # -------------------------------------------------------------------------
# # Tests
# # -------------------------------------------------------------------------
# if __name__ == "__main__":
#     import numpy as np
#
#     def _extract_increments(logsigs):
#         """
#         For depth=1, hall_basis_logsig returns:
#             [0, d(time), d(ch1), d(ch2), ...]
#         With time at index 0 and two data channels, this is:
#             [0, dt, dy1, dy2]
#         """
#         dt = np.array(logsigs[0, :, 1])
#         dy1 = np.array(logsigs[0, :, 2])
#         dy2 = np.array(logsigs[0, :, 3])
#         return dt, dy1, dy2
#
#     def _assert_allclose(a, b, msg="", atol=1e-5):
#         a = np.asarray(a)
#         b = np.asarray(b)
#         assert a.shape == b.shape, f"{msg} shape mismatch: {a.shape} vs {b.shape}"
#         assert np.allclose(a, b, atol=atol), f"{msg}\nGot: {a}\nExp: {b}"
#
#     # ---------------------------------------------------------------------
#     # Test setup:
#     # - Include time + two data channels => C = 3
#     # - stepsize = 4 => each interval is 5 points and dt per interval is 4
#     # - choose T so we have n_full = (T_total-1)//4 = 7 full intervals and no final interval
#     #
#     # IMPORTANT: make the channels "wiggly" inside each interval, but keep the
#     # values at interval boundaries (t=0,4,8,12,16,20,24,28) unchanged.
#     #
#     # This tests the fact that depth=1 signatures should only depend on the
#     # endpoint increment, not what happens in the middle.
#     # ---------------------------------------------------------------------
#     B, T = 1, 28
#     times = np.arange(1, T + 1, dtype=np.float32)  # 1..28
#
#     # Boundary-driven baseline values (piecewise constant across boundary blocks)
#     y1_base = np.zeros_like(times)
#     y1_base[times >= 12] = 3.0
#     y1_base[times >= 24] = 6.0
#
#     y2_base = np.zeros_like(times)
#     y2_base[times >= 12] = -2.0
#     y2_base[times >= 24] = -4.0
#
#     # Add "wiggles" that are guaranteed to be zero at t multiple of 4.
#     # That preserves all interval boundary endpoints.
#     bump1 = (
#         0.7 * np.sin(np.pi * times / 4.0)
#         + 0.25 * np.sin(np.pi * times / 2.0)
#         + 0.15 * np.sin(3.0 * np.pi * times / 4.0)
#     ).astype(np.float32)
#
#     bump2 = (
#         0.5 * np.sin(3.0 * np.pi * times / 4.0)
#         - 0.2 * np.sin(np.pi * times / 2.0)
#         + 0.1 * np.sin(5.0 * np.pi * times / 4.0)
#     ).astype(np.float32)
#
#     y1 = (y1_base + bump1).astype(np.float32)
#     y2 = (y2_base + bump2).astype(np.float32)
#
#     data = np.stack([times, y1, y2], axis=-1)[None, :, :]  # (1, 28, 3)
#
#     # ---------------------------------------------------------------------
#     # Test 0: no gaps (depth=1 increments are endpoint differences)
#     #
#     # Interval boundaries (with prepend time 0):
#     # interval0: t=0->4    y1 0->0    y2 0->0
#     # interval1: t=4->8    y1 0->0    y2 0->0
#     # interval2: t=8->12   y1 0->3    y2 0->-2
#     # interval3: t=12->16  y1 3->3    y2 -2->-2
#     # interval4: t=16->20  y1 3->3    y2 -2->-2
#     # interval5: t=20->24  y1 3->6    y2 -2->-4
#     # interval6: t=24->28  y1 6->6    y2 -4->-4
#     #
#     # The bumps are zero at all boundary times, so the expected increments are unchanged.
#     # ---------------------------------------------------------------------
#     logsigs_none = calc_paths(
#         data=jnp.array(data),
#         stepsize=4,
#         depth=1,
#         include_time=True,
#         time_index=0,
#         interval_gap_mode="none",
#         gap_n_intervals=2,
#         rectilinear_interpolation=False,
#     )
#     dt0, dy10, dy20 = _extract_increments(logsigs_none)
#     _assert_allclose(dt0, [4, 4, 4, 4, 4, 4, 4], "none dt")
#     _assert_allclose(dy10, [0, 0, 3, 0, 0, 3, 0], "none dy1")
#     _assert_allclose(dy20, [0, 0, -2, 0, 0, -2, 0], "none dy2")
#
#     # ---------------------------------------------------------------------
#     # Test 1: time_only with gap_n_intervals=2 (period=3)
#     # normal: i=0,3,6; frozen: i=1,2,4,5
#     #
#     # Expected increments:
#     # dt: always 4
#     # dy1: [0, 0, 0, 3, 0, 0, 3]
#     # dy2: [0, 0, 0, -2, 0, 0, -2]
#     # ---------------------------------------------------------------------
#     logsigs_time_only = calc_paths(
#         data=jnp.array(data),
#         stepsize=4,
#         depth=1,
#         include_time=True,
#         time_index=0,
#         interval_gap_mode="time_only",
#         gap_n_intervals=2,
#         rectilinear_interpolation=False,
#     )
#     dt1, dy11, dy21 = _extract_increments(logsigs_time_only)
#     _assert_allclose(dt1, [4, 4, 4, 4, 4, 4, 4], "time_only dt")
#     _assert_allclose(dy11, [0, 0, 0, 3, 0, 0, 3], "time_only dy1")
#     _assert_allclose(dy21, [0, 0, 0, -2, 0, 0, -2], "time_only dy2")
#
#     # ---------------------------------------------------------------------
#     # Test 2: linear with gap_n_intervals=2 (period=3)
#     # normal: i=0,3,6; linear-filled: i=1,2,4,5
#     #
#     # Bridge 1:
#     #   from end of i=0 at t=4  (y1=0, y2=0)
#     #   to start of i=3 at t=12 (y1=3, y2=-2)
#     #   duration 8, so each gap interval (length 4) moves half way:
#     #     i=1: +1.5, -1.0
#     #     i=2: +1.5, -1.0
#     #
#     # Bridge 2:
#     #   from end of i=3 at t=16 (y1=3, y2=-2)
#     #   to start of i=6 at t=24 (y1=6, y2=-4)
#     #   same increments on i=4 and i=5.
#     #
#     # Normal intervals i=3 and i=6 should have ~0 increment in spatial channels.
#     # ---------------------------------------------------------------------
#     logsigs_linear = calc_paths(
#         data=jnp.array(data),
#         stepsize=4,
#         depth=1,
#         include_time=True,
#         time_index=0,
#         interval_gap_mode="linear",
#         gap_n_intervals=2,
#         rectilinear_interpolation=False,
#     )
#     dt2, dy12, dy22 = _extract_increments(logsigs_linear)
#     _assert_allclose(dt2, [4, 4, 4, 4, 4, 4, 4], "linear dt")
#     _assert_allclose(dy12, [0, 1.5, 1.5, 0, 1.5, 1.5, 0], "linear dy1")
#     _assert_allclose(dy22, [0, -1.0, -1.0, 0, -1.0, -1.0, 0], "linear dy2")
#
#     # Sanity: confirm the mid-interval values are genuinely nontrivial
#     # (i.e. the data is not piecewise constant)
#     assert not np.allclose(y1[0:4], y1[3]), "y1 should vary inside the first interval."
#     assert not np.allclose(y2[0:4], y2[3]), "y2 should vary inside the first interval."
#
#     print(
#         "All interval gap mode tests passed (wiggly interior, depth=1 endpoint increments)."
#     )
