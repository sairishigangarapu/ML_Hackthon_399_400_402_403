"""Reference labels: monotonic alignment of GPS points to the reference route (.route).

The .route file only lists the arcs the vehicle used, not which GPS point belongs to which arc.
We therefore align the point sequence to the route arc sequence with dynamic programming:
  choose route position p_i for every point i (p non-decreasing) minimising sum_i d(point_i, route_arc[p_i])
Only route arcs within CAP metres of a point are allowed for that point (sparse DP); points with no route
arc within CAP are flagged label_valid = False and are excluded from supervised training / accuracy.
A (very expensive) backward move in route position is allowed only as a fallback so the DP is always feasible.
The label is a *reference* alignment, not independently verified ground truth.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from candidates import point_segment, sample_segments, SAMPLE_STEP

CAP = 300.0
BACK_PENALTY = 1.0e4


def align_route(trip, cap=CAP, step=SAMPLE_STEP):
    n = trip.n_obs
    route = trip.route
    if route is None:
        raise ValueError("trip has no reference route")
    m = len(route)
    ax = trip.node_x[trip.arc_from[route]]
    ay = trip.node_y[trip.arc_from[route]]
    bx = trip.node_x[trip.arc_to[route]]
    by = trip.node_y[trip.arc_to[route]]
    seg, sx, sy = sample_segments(ax, ay, bx, by, step)
    tree = cKDTree(np.column_stack([sx, sy]))
    lists = tree.query_ball_point(np.column_stack([trip.x, trip.y]), r=cap + step / 2.0)
    lens = np.fromiter((len(l) for l in lists), dtype=np.int64, count=n)
    label_pos = np.full(n, -1, dtype=np.int64)
    label_dist = np.full(n, np.inf)
    if lens.sum() == 0:
        return _pack(trip, label_pos, label_dist, 0)
    flat = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists if len(l)])
    pt = np.repeat(np.arange(n), lens)
    key = np.unique(pt * m + seg[flat])               # unique (point, route position), sorted
    pt_u, pos_u = key // m, key % m
    d, _ = point_segment(trip.x[pt_u], trip.y[pt_u], ax[pos_u], ay[pos_u], bx[pos_u], by[pos_u])
    keep = d <= cap
    pt_u, pos_u, d = pt_u[keep], pos_u[keep], d[keep]
    off = np.searchsorted(pt_u, np.arange(n + 1))

    steps, pos_list, back_list = [], [], []
    prev_pos = prev_f = None
    n_back = 0
    for i in range(n):
        s, e = off[i], off[i + 1]
        if s == e:
            continue
        pos, di = pos_u[s:e], d[s:e]
        if prev_pos is None:
            f = di.copy()
            bp = np.full(len(pos), -1, dtype=np.int64)
        else:
            pm = np.minimum.accumulate(prev_f)
            idxs = np.arange(len(prev_f))
            pam = np.maximum.accumulate(np.where(prev_f == pm, idxs, -1))
            k = np.searchsorted(prev_pos, pos, side="right") - 1
            kc = np.maximum(k, 0)
            fwd = np.where(k >= 0, pm[kc], np.inf)
            fwd_arg = pam[kc]
            g = int(np.argmin(prev_f))
            back = prev_f[g] + BACK_PENALTY
            use_back = back < fwd
            n_back += int(use_back.sum())
            f = di + np.where(use_back, back, fwd)
            bp = np.where(use_back, g, fwd_arg)
        steps.append(i)
        pos_list.append(pos)
        back_list.append(bp)
        prev_pos, prev_f = pos, f

    # backtrack
    cur = int(np.argmin(prev_f))
    for j in range(len(steps) - 1, -1, -1):
        i = steps[j]
        label_pos[i] = pos_list[j][cur]
        cur = int(back_list[j][cur])
    # distance to the chosen route arc
    ok = label_pos >= 0
    dd, _ = point_segment(trip.x[ok], trip.y[ok], ax[label_pos[ok]], ay[label_pos[ok]],
                          bx[label_pos[ok]], by[label_pos[ok]])
    label_dist[ok] = dd
    return _pack(trip, label_pos, label_dist, n_back)


def _pack(trip, label_pos, label_dist, n_back):
    valid = label_pos >= 0
    label_arc = np.where(valid, trip.route[np.maximum(label_pos, 0)], -1)
    nz = label_pos[valid]
    n_nonmono = int((np.diff(nz) < 0).sum()) if len(nz) > 1 else 0
    return {"pos": label_pos, "arc": label_arc, "dist": label_dist,
            "valid": valid, "n_back_moves": n_back, "n_nonmonotone": n_nonmono}
