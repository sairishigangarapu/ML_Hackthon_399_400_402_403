"""Stage 02B: Monotonic Dynamic Programming alignment of GPS observations to ground-truth .route arcs.

Used ONLY during training label generation and strictly server-side evaluation when .route is supplied.
NEVER invoked during model inference.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_data_processing"))
from io_utils import TripData


def compute_smoothed_headings(
    tx: np.ndarray,
    ty: np.ndarray,
    ts: np.ndarray,
    half_win: int = 3,
    gap_thresh: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute gap-aware smoothed unit heading vectors (hx, hy) and movement speeds."""
    T = len(tx)
    hx = np.zeros(T, dtype=np.float64)
    hy = np.zeros(T, dtype=np.float64)
    spd = np.zeros(T, dtype=np.float64)

    if T <= 1:
        return hx, hy, spd

    dt = np.diff(ts)
    # Segment trajectory into contiguous blocks separated by gaps > gap_thresh
    breaks = np.where(dt > gap_thresh)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [T]])

    for s, e in zip(starts, ends):
        n = e - s
        if n <= 1:
            continue
        bx, by, bts = tx[s:e], ty[s:e], ts[s:e]
        for i in range(n):
            lo = max(0, i - half_win)
            hi = min(n - 1, i + half_win)
            dx = bx[hi] - bx[lo]
            dy = by[hi] - by[lo]
            dist = float(np.hypot(dx, dy))
            dt_win = max(float(bts[hi] - bts[lo]), 1e-6)
            spd[s + i] = dist / dt_win
            if dist > 1.5:
                hx[s + i] = dx / dist
                hy[s + i] = dy / dist

        # Fill stationary stops using nearest non-zero heading in the same contiguous block
        valid = np.where((hx[s:e] != 0.0) | (hy[s:e] != 0.0))[0]
        if len(valid) > 0:
            for i in range(n):
                if hx[s + i] == 0.0 and hy[s + i] == 0.0:
                    nearest = valid[np.argmin(np.abs(valid - i))]
                    hx[s + i] = hx[s + nearest]
                    hy[s + i] = hy[s + nearest]

    return hx, hy, spd


def align_track_to_route(trip: TripData, route_arcs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align each GPS observation t=0..T-1 to an ordered position j_t in route_arcs via Monotonic DP.

    Returns
    -------
    ref_arc_ids : np.ndarray of shape (T,) int32
        Reference ground-truth arc_id aligned to each GPS observation.
    ref_route_idx : np.ndarray of shape (T,) int32
        Index j_t into route_arcs (0..M-1) for each GPS observation.
    """
    T = len(trip.track_x)
    M = len(route_arcs)
    if M == 0:
        return np.full(T, -1, dtype=np.int32), np.full(T, -1, dtype=np.int32)
    if M == 1:
        return np.full(T, int(route_arcs[0]), dtype=np.int32), np.zeros(T, dtype=np.int32)

    rx1 = trip.arc_x1[route_arcs]
    ry1 = trip.arc_y1[route_arcs]
    rx2 = trip.arc_x2[route_arcs]
    ry2 = trip.arc_y2[route_arcs]
    rdx = rx2 - rx1
    rdy = ry2 - ry1
    rl2 = np.maximum(rdx * rdx + rdy * rdy, 1e-12)
    rux = trip.arc_ux[route_arcs]
    ruy = trip.arc_uy[route_arcs]

    hx, hy, spd = compute_smoothed_headings(trip.track_x, trip.track_y, trip.track_ts, half_win=3)
    moving = (spd > 1.0) & ((hx != 0.0) | (hy != 0.0))

    # Build emission cost matrix E[t, j] = dist_m + heading_penalty
    # Vectorized chunked computation
    cost = np.empty((T, M), dtype=np.float32)
    chunk = 512
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        px = trip.track_x[s:e, None]
        py = trip.track_y[s:e, None]
        u = ((px - rx1[None, :]) * rdx[None, :] + (py - ry1[None, :]) * rdy[None, :]) / rl2[None, :]
        u = np.clip(u, 0.0, 1.0)
        dist = np.hypot(px - (rx1[None, :] + u * rdx[None, :]), py - (ry1[None, :] + u * rdy[None, :]))
        cos_a = hx[s:e, None] * rux[None, :] + hy[s:e, None] * ruy[None, :]
        h_pen = np.where(moving[s:e, None], 6.0 * (1.0 - cos_a), 0.0)
        cost[s:e] = (dist + h_pen).astype(np.float32)

    # Monotonic DP: allow staying (j == prev), advancing forward (j > prev) with tiny parsimony slope 0.35*(j-prev),
    # or at most 1-step jitter backtrack with strong penalty 20.0.
    # Running prefix minimum makes forward transition O(M) per time step!
    dp = np.empty(M, dtype=np.float64)
    dp[:] = cost[0] + 0.15 * np.arange(M, dtype=np.float64)  # mild bias toward starting near route start
    backptr = np.zeros((T, M), dtype=np.int32)

    slope = 0.35
    back1_pen = 20.0
    idx_arr = np.arange(M, dtype=np.int32)

    for t in range(1, T):
        # Forward prefix min of (dp[i] - slope * i)
        shifted = dp - slope * idx_arr
        best_val = np.empty(M, dtype=np.float64)
        best_arg = np.empty(M, dtype=np.int32)
        cur_v = shifted[0]
        cur_a = 0
        best_val[0] = cur_v
        best_arg[0] = 0
        for j in range(1, M):
            if shifted[j] < cur_v:
                cur_v = shifted[j]
                cur_a = j
            best_val[j] = cur_v
            best_arg[j] = cur_a

        trans_cost = best_val + slope * idx_arr
        prev_choice = best_arg

        # Optional 1-arc local jitter backtrack (j = i - 1)
        if M > 1:
            back1_cost = dp[1:] + back1_pen
            better_back = back1_cost < trans_cost[:-1]
            trans_cost[:-1] = np.where(better_back, back1_cost, trans_cost[:-1])
            prev_choice[:-1] = np.where(better_back, idx_arr[1:], prev_choice[:-1])

        dp = trans_cost + cost[t]
        backptr[t] = prev_choice

    # Backtrack from minimum end state (with mild bias toward finishing near route end)
    end_score = dp + 0.15 * (M - 1 - idx_arr)
    j_star = int(np.argmin(end_score))
    ref_route_idx = np.empty(T, dtype=np.int32)
    ref_route_idx[T - 1] = j_star
    for t in range(T - 1, 0, -1):
        j_star = int(backptr[t, j_star])
        ref_route_idx[t - 1] = j_star

    ref_arc_ids = route_arcs[ref_route_idx].astype(np.int32)
    return ref_arc_ids, ref_route_idx
