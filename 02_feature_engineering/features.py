"""Per-(observation, candidate-arc) features. Uses ONLY .track/.nodes/.arcs (never .route),
so exactly the same code runs at inference time on unseen trips.
"""
from __future__ import annotations

import numpy as np

from candidates import generate_candidates, K_DEFAULT

GAP_S = 5.0          # never build a motion window across a GPS gap larger than this
MIN_DISP_M = 3.0     # below this displacement the vehicle heading is treated as undefined
MAX_SPEED = 60.0     # m/s clip for the speed feature

FEATURE_NAMES = [
    "dist_m", "rank", "dist_gap_m", "dist_ratio",
    "log_arc_len", "proj_frac", "log_dist_to_arc_end", "is_zero_len",
    "cos_h3", "sin_h3", "defined_h3",
    "cos_h10", "sin_h10", "defined_h10",
    "has_twin", "cos_h3_x_twin", "cos_h10_x_twin",
    "speed_3s_mps",
]


def _motion(trip, win):
    """Displacement vector and speed over a +/-win second window, truncated at GPS gaps."""
    t = trip.t
    n = len(t)
    brk = np.diff(t) > GAP_S
    seg = np.concatenate([[0], np.cumsum(brk)])
    seg_start = np.searchsorted(seg, seg, side="left")
    seg_end = np.searchsorted(seg, seg, side="right") - 1
    a = np.maximum(np.searchsorted(t, t - win, side="left"), seg_start)
    b = np.minimum(np.searchsorted(t, t + win, side="right") - 1, seg_end)
    dx, dy = trip.x[b] - trip.x[a], trip.y[b] - trip.y[a]
    dt = t[b] - t[a]
    speed = np.where(dt > 0, np.hypot(dx, dy) / np.where(dt > 0, dt, 1.0), 0.0)
    return dx, dy, speed


def _has_twin(trip):
    n_nodes = len(trip.node_x)
    key = trip.arc_from.astype(np.int64) * n_nodes + trip.arc_to
    rev = trip.arc_to.astype(np.int64) * n_nodes + trip.arc_from
    return np.isin(rev, key)


def build_trip_table(trip, K=K_DEFAULT):
    """Returns dict with candidate rows (padding removed) and per-observation arrays."""
    n = trip.n_obs
    cand = generate_candidates(trip, K=K)
    arc, dist, frac = cand["arc"], cand["dist"], cand["frac"]
    valid = arc >= 0
    arc0 = np.maximum(arc, 0)

    ax = trip.node_x[trip.arc_from[arc0]]
    ay = trip.node_y[trip.arc_from[arc0]]
    bx = trip.node_x[trip.arc_to[arc0]]
    by = trip.node_y[trip.arc_to[arc0]]
    ux, uy = bx - ax, by - ay
    alen = np.hypot(ux, uy)
    zero = alen < 1e-6
    twin = _has_twin(trip)[arc0].astype(float)

    dmin = np.where(valid[:, :1], dist[:, :1], 0.0)
    dist_f = np.where(valid, dist, 0.0)
    rank = np.broadcast_to(np.arange(K, dtype=float), (n, K))
    gap = np.where(valid, dist - dmin, 0.0)
    ratio = np.where(valid, dist / (dmin + 1.0), 0.0)
    to_end = (1.0 - frac) * alen

    cols = {
        "dist_m": dist_f, "rank": rank, "dist_gap_m": gap, "dist_ratio": ratio,
        "log_arc_len": np.log1p(alen), "proj_frac": frac,
        "log_dist_to_arc_end": np.log1p(to_end), "is_zero_len": zero.astype(float),
        "has_twin": twin,
    }
    for win, tag in ((3.0, "3"), (10.0, "10")):
        vx, vy, speed = _motion(trip, win)
        vn = np.hypot(vx, vy)
        defined = ((vn >= MIN_DISP_M)[:, None] & ~zero)
        denom = np.where(defined, vn[:, None] * np.where(zero, 1.0, alen), 1.0)
        cos = np.where(defined, (vx[:, None] * ux + vy[:, None] * uy) / denom, 0.0)
        sin = np.where(defined, (vx[:, None] * uy - vy[:, None] * ux) / denom, 0.0)
        cols[f"cos_h{tag}"] = cos
        cols[f"sin_h{tag}"] = sin
        cols[f"defined_h{tag}"] = defined.astype(float)
        cols[f"cos_h{tag}_x_twin"] = cos * twin
        if tag == "3":
            cols["speed_3s_mps"] = np.broadcast_to(np.minimum(speed, MAX_SPEED)[:, None], (n, K))

    X = np.stack([np.broadcast_to(cols[name], (n, K)) for name in FEATURE_NAMES], axis=2)
    obs_i, cand_k = np.nonzero(valid)                 # row-major: grouped by observation
    Xr = X[obs_i, cand_k].astype(np.float32)
    if not np.isfinite(Xr).all():
        raise ValueError(f"trip {trip.trip_id}: non-finite feature values")
    return {
        "X": Xr, "row_obs": obs_i.astype(np.int32), "cand_arc": arc[obs_i, cand_k].astype(np.int32),
        "cand_rank": cand_k.astype(np.int8), "n_cand_per_obs": valid.sum(axis=1),
    }
