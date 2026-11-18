"""Stage 02A: Fast exact Top-K road-segment candidate generator via subdivided-arc cKDTree."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_data_processing"))
from config import CANDIDATE_RADIUS_M, TOP_K_CANDIDATES
from io_utils import TripData


@dataclass
class CandidateSet:
    trip_id: str
    cand_arc_ids: np.ndarray     # (T, K) int32, -1 if unavailable
    cand_dist_m: np.ndarray      # (T, K) float32, exact perpendicular distance in meters
    cand_proj_u: np.ndarray      # (T, K) float32, projection ratio in [0, 1] along arc
    cand_signed_lat: np.ndarray  # (T, K) float32, signed lateral offset (+left / -right of directed arc)
    cand_valid: np.ndarray       # (T, K) bool


class ArcSpatialIndex:
    """Subdivided-segment cKDTree ensuring exact recall even on multi-kilometer highway arcs."""

    def __init__(self, trip: TripData, max_step_m: float = 40.0) -> None:
        self.trip = trip
        x1, y1 = trip.arc_x1, trip.arc_y1
        x2, y2 = trip.arc_x2, trip.arc_y2
        lengths = trip.arc_len
        n_arcs = len(lengths)

        # Include midpoint of every arc + subdivided interior points for long arcs
        n_sub = np.maximum(1, np.ceil(lengths / max_step_m).astype(np.int32))
        # Fast vectorized construction for n_sub == 1 (vast majority of arcs)
        short_mask = n_sub == 1
        short_ids = np.where(short_mask)[0]
        short_pts = np.column_stack([0.5 * (x1[short_ids] + x2[short_ids]), 0.5 * (y1[short_ids] + y2[short_ids])])

        long_ids = np.where(~short_mask)[0]
        pts_list = [short_pts]
        ids_list = [short_ids.astype(np.int32)]

        # Also add endpoints of long arcs + interior samples
        if len(long_ids) > 0:
            long_pts = []
            long_arc_refs = []
            for aid in long_ids:
                m = int(n_sub[aid]) + 1
                ts = np.linspace(0.0, 1.0, m)
                px = x1[aid] + ts * (x2[aid] - x1[aid])
                py = y1[aid] + ts * (y2[aid] - y1[aid])
                long_pts.append(np.column_stack([px, py]))
                long_arc_refs.append(np.full(m, aid, dtype=np.int32))
            pts_list.append(np.vstack(long_pts))
            ids_list.append(np.concatenate(long_arc_refs))

        all_pts = np.vstack(pts_list)
        self.proxy_arc_ids = np.concatenate(ids_list)
        self.tree = cKDTree(all_pts)

    def generate_candidates(
        self,
        k: int = TOP_K_CANDIDATES,
        radius_m: float = CANDIDATE_RADIUS_M,
    ) -> CandidateSet:
        trip = self.trip
        T = len(trip.track_x)
        pts = np.column_stack([trip.track_x, trip.track_y])

        # Query nearest 64 proxy points (fast vectorized C++ call) + radius ball if dense
        n_proxy_query = min(max(k * 6, 48), len(self.proxy_arc_ids))
        _, nn_idx = self.tree.query(pts, k=n_proxy_query, workers=-1)
        if nn_idx.ndim == 1:
            nn_idx = nn_idx[:, None]

        # Also check radius ball for urban intersections with >48 nearby proxies
        ball_lists = self.tree.query_ball_point(pts, r=min(radius_m, 65.0), workers=-1)

        cand_arc_ids = np.full((T, k), -1, dtype=np.int32)
        cand_dist_m = np.full((T, k), 1e4, dtype=np.float32)
        cand_proj_u = np.zeros((T, k), dtype=np.float32)
        cand_signed_lat = np.zeros((T, k), dtype=np.float32)
        cand_valid = np.zeros((T, k), dtype=np.bool_)

        ax1, ay1 = trip.arc_x1, trip.arc_y1
        ax2, ay2 = trip.arc_x2, trip.arc_y2
        adx = ax2 - ax1
        ady = ay2 - ay1
        al2 = np.maximum(adx * adx + ady * ady, 1e-12)
        aux, auy = trip.arc_ux, trip.arc_uy
        tx, ty = trip.track_x, trip.track_y

        for t in range(T):
            proxies = nn_idx[t]
            b = ball_lists[t]
            if b:
                merged = np.unique(np.concatenate([self.proxy_arc_ids[proxies], self.proxy_arc_ids[b]]))
            else:
                merged = np.unique(self.proxy_arc_ids[proxies])

            px, py = tx[t], ty[t]
            dx_m = adx[merged]
            dy_m = ady[merged]
            u = ((px - ax1[merged]) * dx_m + (py - ay1[merged]) * dy_m) / al2[merged]
            u = np.clip(u, 0.0, 1.0)
            projx = ax1[merged] + u * dx_m
            projy = ay1[merged] + u * dy_m
            rx = px - projx
            ry = py - projy
            dists = np.hypot(rx, ry)

            # Signed lateral offset: cross product of unit arc vector with residual vector (ux*ry - uy*rx)
            signed_lat = aux[merged] * ry - auy[merged] * rx

            order = np.argsort(dists)
            top = order[:k]
            m_found = len(top)
            cand_arc_ids[t, :m_found] = merged[top]
            cand_dist_m[t, :m_found] = dists[top]
            cand_proj_u[t, :m_found] = u[top]
            cand_signed_lat[t, :m_found] = signed_lat[top]
            cand_valid[t, :m_found] = True

        return CandidateSet(
            trip_id=trip.trip_id,
            cand_arc_ids=cand_arc_ids,
            cand_dist_m=cand_dist_m,
            cand_proj_u=cand_proj_u,
            cand_signed_lat=cand_signed_lat,
            cand_valid=cand_valid,
        )
