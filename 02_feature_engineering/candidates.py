"""Candidate generation: for every GPS point, the K nearest road arcs (exact point-to-segment distance).

Method
  1. keep only arcs whose bounding box overlaps the trip's GPS bounding box (+ margin)
  2. sample those arcs every SAMPLE_STEP metres (endpoints included) and index the samples in a KD-tree
  3. for each GPS point take the N_NEIGH nearest samples, collapse them to distinct arcs,
     compute the EXACT point-to-segment distance and keep the K closest arcs (sorted by distance)
Because samples are at most SAMPLE_STEP apart, the shortlist can only differ from a brute-force
K-nearest search at the far edge of the shortlist (fewer than K distinct arcs among the N_NEIGH samples).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

K_DEFAULT = 16
SAMPLE_STEP = 20.0
N_NEIGH = 128
BBOX_MARGIN = 600.0


def point_segment(px, py, ax, ay, bx, by):
    """Distance and projection fraction t in [0,1] from points to segments (all arrays broadcastable)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    safe = np.where(l2 > 0, l2, 1.0)
    t = ((px - ax) * dx + (py - ay) * dy) / safe
    t = np.where(l2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t


def sample_segments(ax, ay, bx, by, step=SAMPLE_STEP):
    """Sample every segment at <= step spacing. Returns (segment index, sx, sy)."""
    length = np.hypot(bx - ax, by - ay)
    counts = np.maximum(1, np.ceil(length / step).astype(np.int64) + 1)
    seg = np.repeat(np.arange(len(ax)), counts)
    starts = np.cumsum(counts) - counts
    k = np.arange(counts.sum()) - np.repeat(starts, counts)
    denom = np.repeat(np.maximum(counts - 1, 1), counts)
    frac = k / denom
    sx = ax[seg] + frac * (bx[seg] - ax[seg])
    sy = ay[seg] + frac * (by[seg] - ay[seg])
    return seg, sx, sy


def arcs_near_track(trip, margin=BBOX_MARGIN):
    ax, ay = trip.node_x[trip.arc_from], trip.node_y[trip.arc_from]
    bx, by = trip.node_x[trip.arc_to], trip.node_y[trip.arc_to]
    xmin, xmax = trip.x.min() - margin, trip.x.max() + margin
    ymin, ymax = trip.y.min() - margin, trip.y.max() + margin
    keep = ((np.maximum(ax, bx) >= xmin) & (np.minimum(ax, bx) <= xmax) &
            (np.maximum(ay, by) >= ymin) & (np.minimum(ay, by) <= ymax))
    return np.flatnonzero(keep)


def generate_candidates(trip, K=K_DEFAULT, n_neigh=N_NEIGH, step=SAMPLE_STEP):
    """Returns dict of (n_obs, K) arrays: arc (-1 = padding), dist, frac."""
    n = trip.n_obs
    arc_ids = arcs_near_track(trip)
    out_arc = np.full((n, K), -1, dtype=np.int64)
    out_dist = np.full((n, K), np.inf)
    out_frac = np.zeros((n, K))
    if len(arc_ids) == 0:
        return {"arc": out_arc, "dist": out_dist, "frac": out_frac}

    ax = trip.node_x[trip.arc_from[arc_ids]]
    ay = trip.node_y[trip.arc_from[arc_ids]]
    bx = trip.node_x[trip.arc_to[arc_ids]]
    by = trip.node_y[trip.arc_to[arc_ids]]
    seg, sx, sy = sample_segments(ax, ay, bx, by, step)
    tree = cKDTree(np.column_stack([sx, sy]))
    kk = min(n_neigh, len(sx))
    _, idx = tree.query(np.column_stack([trip.x, trip.y]), k=kk)
    idx = idx.reshape(n, kk)
    loc = seg[idx]                                   # local arc index, (n, kk)

    d, t = point_segment(trip.x[:, None], trip.y[:, None], ax[loc], ay[loc], bx[loc], by[loc])
    order = np.argsort(loc, axis=1, kind="stable")   # collapse duplicate arcs
    loc_s = np.take_along_axis(loc, order, axis=1)
    d_s = np.take_along_axis(d, order, axis=1)
    t_s = np.take_along_axis(t, order, axis=1)
    dup = np.zeros_like(loc_s, dtype=bool)
    dup[:, 1:] = loc_s[:, 1:] == loc_s[:, :-1]
    d_s = np.where(dup, np.inf, d_s)

    take = min(K, kk)
    best = np.argsort(d_s, axis=1, kind="stable")[:, :take]
    b_loc = np.take_along_axis(loc_s, best, axis=1)
    b_d = np.take_along_axis(d_s, best, axis=1)
    b_t = np.take_along_axis(t_s, best, axis=1)
    ok = np.isfinite(b_d)
    out_arc[:, :take] = np.where(ok, arc_ids[b_loc], -1)
    out_dist[:, :take] = b_d
    out_frac[:, :take] = np.where(ok, b_t, 0.0)
    return {"arc": out_arc, "dist": out_dist, "frac": out_frac}
