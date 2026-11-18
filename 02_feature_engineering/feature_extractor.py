"""Stage 02C: 28-feature spatial, kinematic, listwise-competitive, and topological feature extractor.

Strictly observable from raw .track, .nodes, .arcs ONLY (zero ground-truth leakage).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_data_processing"))
from config import (
    CANDIDATE_RADIUS_M,
    DIR_ARTIFACTS,
    DIR_CACHE,
    DIR_RESULTS,
    GAP_THRESHOLD_S,
    STUDENT_TEST_DIR,
    TOP_K_CANDIDATES,
    TRAIN_DIR,
)
from io_utils import TripData, build_adjacency, list_trips, load_trip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_generator import ArcSpatialIndex, CandidateSet
from route_aligner import align_track_to_route, compute_smoothed_headings


FEATURE_NAMES: List[str] = [
    # Group 1: Geometric / Spatial Proximity (6)
    "dist_m",
    "log_dist_m",
    "proj_ratio",
    "signed_lateral_offset",
    "arc_len_m",
    "log_arc_len_m",
    # Group 2: Kinematic & Multi-Scale Directional Alignment (8)
    "gps_speed_mps",
    "smooth_speed_mps",
    "is_stationary",
    "cos_heading_inst",
    "sin_heading_inst",
    "cos_heading_smooth",
    "cos_heading_wide",
    "heading_progress_dot",
    # Group 3: Listwise Competition within Candidate Set (6)
    "dist_rank_norm",
    "dist_diff_best",
    "dist_ratio_best",
    "has_reverse_pair",
    "heading_rank_near",
    "local_cand_density",
    # Group 4: Local Road-Network Topology & 1-Step Context (5)
    "node_out_degree",
    "node_in_degree",
    "prev_step_same_arc",
    "prev_step_connected",
    "next_step_connected",
    # Group 5: Temporal & GPS-Gap Context (3)
    "dt_prev_log",
    "dt_next_log",
    "is_gap_boundary",
]


def extract_trip_features(
    trip: TripData,
    k: int = TOP_K_CANDIDATES,
    radius_m: float = CANDIDATE_RADIUS_M,
    cands: Optional[CandidateSet] = None,
) -> Tuple[np.ndarray, CandidateSet]:
    """Compute (T, K, F) feature tensor strictly from .track, .nodes, .arcs."""
    if cands is None:
        index = ArcSpatialIndex(trip)
        cands = index.generate_candidates(k=k, radius_m=radius_m)

    T, K = cands.cand_arc_ids.shape
    F = len(FEATURE_NAMES)
    feats = np.zeros((T, K, F), dtype=np.float32)

    # Multi-scale gap-aware headings and speeds
    hx_inst, hy_inst, spd_inst = compute_smoothed_headings(
        trip.track_x, trip.track_y, trip.track_ts, half_win=1, gap_thresh=GAP_THRESHOLD_S
    )
    hx_sm, hy_sm, spd_sm = compute_smoothed_headings(
        trip.track_x, trip.track_y, trip.track_ts, half_win=3, gap_thresh=GAP_THRESHOLD_S
    )
    hx_wd, hy_wd, _ = compute_smoothed_headings(
        trip.track_x, trip.track_y, trip.track_ts, half_win=7, gap_thresh=GAP_THRESHOLD_S
    )

    # Cap speed outliers defensively at 55 m/s (~200 km/h)
    spd_inst = np.clip(spd_inst, 0.0, 55.0)
    spd_sm = np.clip(spd_sm, 0.0, 55.0)
    is_stat = (spd_sm < 0.8).astype(np.float32)

    # Centered displacement vector (gap-aware)
    disp_x = np.zeros(T, dtype=np.float64)
    disp_y = np.zeros(T, dtype=np.float64)
    dt_prev = np.ones(T, dtype=np.float64)
    dt_next = np.ones(T, dtype=np.float64)
    if T > 1:
        dt_raw = np.diff(trip.track_ts)
        dt_prev[1:] = dt_raw
        dt_next[:-1] = dt_raw
        for t in range(T):
            lo = t - 1 if (t > 0 and dt_prev[t] <= GAP_THRESHOLD_S) else t
            hi = t + 1 if (t < T - 1 and dt_next[t] <= GAP_THRESHOLD_S) else t
            disp_x[t] = trip.track_x[hi] - trip.track_x[lo]
            disp_y[t] = trip.track_y[hi] - trip.track_y[lo]

    dt_prev_log = np.log1p(np.clip(dt_prev, 0.0, 1e6)).astype(np.float32)
    dt_next_log = np.log1p(np.clip(dt_next, 0.0, 1e6)).astype(np.float32)
    is_gap = ((dt_prev > GAP_THRESHOLD_S) | (dt_next > GAP_THRESHOLD_S)).astype(np.float32)

    # Graph degrees and reverse-edge map
    adj, out_deg, in_deg, edge_map = build_adjacency(trip)

    # Precompute 1-hop neighbor sets of nodes for fast prev/next connectivity checks
    out_neighbors = [set(v for v, _, _ in adj[u]) for u in range(len(adj))]

    aids = cands.cand_arc_ids
    safe_aids = np.maximum(aids, 0)
    valid = cands.cand_valid

    from_nodes = trip.arcs[safe_aids, 0]
    to_nodes = trip.arcs[safe_aids, 1]
    ux = trip.arc_ux[safe_aids]
    uy = trip.arc_uy[safe_aids]
    alen = trip.arc_len[safe_aids]

    # Group 1: Geometric / Spatial
    dist_m = cands.cand_dist_m
    feats[:, :, 0] = dist_m
    feats[:, :, 1] = np.log1p(np.maximum(dist_m, 0.0))
    feats[:, :, 2] = cands.cand_proj_u
    feats[:, :, 3] = np.clip(cands.cand_signed_lat, -200.0, 200.0)
    feats[:, :, 4] = np.clip(alen, 0.0, 2000.0)
    feats[:, :, 5] = np.log1p(np.maximum(alen, 0.0))

    # Group 2: Kinematic & Multi-Scale Heading
    feats[:, :, 6] = spd_inst[:, None]
    feats[:, :, 7] = spd_sm[:, None]
    feats[:, :, 8] = is_stat[:, None]
    feats[:, :, 9] = hx_inst[:, None] * ux + hy_inst[:, None] * uy
    feats[:, :, 10] = hx_inst[:, None] * uy - hy_inst[:, None] * ux
    cos_sm = hx_sm[:, None] * ux + hy_sm[:, None] * uy
    feats[:, :, 11] = cos_sm
    feats[:, :, 12] = hx_wd[:, None] * ux + hy_wd[:, None] * uy
    feats[:, :, 13] = np.clip(disp_x[:, None] * ux + disp_y[:, None] * uy, -100.0, 100.0)

    # Group 3: Listwise Competition within Candidate Set
    min_dist = dist_m[:, 0:1]  # candidates are sorted by ascending dist_m
    feats[:, :, 14] = (np.arange(K, dtype=np.float32) / max(K - 1, 1))[None, :]
    feats[:, :, 15] = np.clip(dist_m - min_dist, 0.0, 500.0)
    feats[:, :, 16] = np.clip(dist_m / (min_dist + 1.0), 0.0, 50.0)
    feats[:, :, 19] = (dist_m <= (min_dist + 25.0)).sum(axis=1, keepdims=True).astype(np.float32)

    # Reverse pair indicator & near-candidate heading rank
    has_rev = np.zeros((T, K), dtype=np.float32)
    h_rank_near = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        cset = set(aids[t][valid[t]].tolist())
        near_mask = valid[t] & (dist_m[t] <= min_dist[t, 0] + 15.0)
        near_idx = np.where(near_mask)[0]
        if len(near_idx) > 1:
            order_h = np.argsort(-cos_sm[t, near_idx])
            ranks = np.empty(len(near_idx), dtype=np.float32)
            ranks[order_h] = np.linspace(0.0, 1.0, len(near_idx), dtype=np.float32)
            h_rank_near[t, near_idx] = ranks
        h_rank_near[t, ~near_mask] = 1.0

        for j in range(K):
            if not valid[t, j]:
                continue
            rev_id = edge_map.get((int(to_nodes[t, j]), int(from_nodes[t, j])), -1)
            if rev_id in cset:
                has_rev[t, j] = 1.0

    feats[:, :, 17] = has_rev
    feats[:, :, 18] = h_rank_near

    # Group 4: Topology & 1-Step Sequence Context
    feats[:, :, 20] = np.clip(out_deg[to_nodes], 0, 12).astype(np.float32)
    feats[:, :, 21] = np.clip(in_deg[from_nodes], 0, 12).astype(np.float32)

    prev_same = np.zeros((T, K), dtype=np.float32)
    prev_conn = np.zeros((T, K), dtype=np.float32)
    next_conn = np.zeros((T, K), dtype=np.float32)

    for t in range(T):
        if t > 0 and dt_prev[t] <= GAP_THRESHOLD_S:
            p_valid = valid[t - 1]
            p_arcs = set(aids[t - 1][p_valid].tolist())
            p_to = set(to_nodes[t - 1][p_valid].tolist())
            # 2-hop reachable nodes from previous candidates' to_nodes
            p_to_2hop = set(p_to)
            for nd in p_to:
                p_to_2hop.update(out_neighbors[nd])
            for j in range(K):
                if not valid[t, j]:
                    continue
                if int(aids[t, j]) in p_arcs:
                    prev_same[t, j] = 1.0
                    prev_conn[t, j] = 1.0
                elif int(from_nodes[t, j]) in p_to:
                    prev_conn[t, j] = 1.0
                elif int(from_nodes[t, j]) in p_to_2hop:
                    prev_conn[t, j] = 0.5

        if t < T - 1 and dt_next[t] <= GAP_THRESHOLD_S:
            n_valid = valid[t + 1]
            n_arcs = set(aids[t + 1][n_valid].tolist())
            n_from = set(from_nodes[t + 1][n_valid].tolist())
            for j in range(K):
                if not valid[t, j]:
                    continue
                if int(aids[t, j]) in n_arcs or int(to_nodes[t, j]) in n_from:
                    next_conn[t, j] = 1.0
                elif any(nb in n_from for nb in out_neighbors[int(to_nodes[t, j])]):
                    next_conn[t, j] = 0.5

    feats[:, :, 22] = prev_same
    feats[:, :, 23] = prev_conn
    feats[:, :, 24] = next_conn

    # Group 5: Temporal & Gap Context
    feats[:, :, 25] = dt_prev_log[:, None]
    feats[:, :, 26] = dt_next_log[:, None]
    feats[:, :, 27] = is_gap[:, None]

    # Zero out invalid padded slots cleanly
    feats[~valid] = 0.0
    return feats, cands


def build_partition_dataset(partition: str, partition_dir: Path) -> Dict:
    """Extract features, candidates, spatial coords, and aligned labels for all trips in a partition."""
    t0 = time.time()
    trip_ids = list_trips(partition_dir)
    all_X: List[np.ndarray] = []
    all_y_idx: List[np.ndarray] = []       # index in 0..K-1, or -1 if candidate miss
    all_ref_arc: List[np.ndarray] = []     # true arc_id from monotonic alignment
    all_cand_arcs: List[np.ndarray] = []   # (T, K)
    all_cand_dist: List[np.ndarray] = []   # (T, K)
    all_cand_valid: List[np.ndarray] = []  # (T, K)
    all_coords: List[np.ndarray] = []      # (T, 3) -> [x_m, y_m, ts] for NN-GLS spatial covariance
    trip_slices: Dict[str, Tuple[int, int]] = {}
    trip_stats: List[Dict] = []

    offset = 0
    for tid in trip_ids:
        trip = load_trip(partition_dir, tid, partition=partition, load_route=True)
        feats, cands = extract_trip_features(trip)
        ref_arcs, _ = align_track_to_route(trip, trip.route)

        T, K, F = feats.shape
        y_idx = np.full(T, -1, dtype=np.int32)
        matches = (cands.cand_arc_ids == ref_arcs[:, None]) & cands.cand_valid
        hit_t, hit_k = np.where(matches)
        y_idx[hit_t] = hit_k

        miss_rate = float((y_idx == -1).mean())
        top1_dist_acc = float((y_idx == 0).mean())

        all_X.append(feats)
        all_y_idx.append(y_idx)
        all_ref_arc.append(ref_arcs)
        all_cand_arcs.append(cands.cand_arc_ids)
        all_cand_dist.append(cands.cand_dist_m)
        all_cand_valid.append(cands.cand_valid)
        all_coords.append(np.column_stack([trip.track_x, trip.track_y, trip.track_ts]).astype(np.float32))

        trip_slices[tid] = (offset, offset + T)
        offset += T
        trip_stats.append({
            "partition": partition,
            "trip_id": tid,
            "n_obs": T,
            "cand_miss_rate": round(miss_rate, 5),
            "dist_only_top1_acc": round(top1_dist_acc, 5),
        })

    X = np.concatenate(all_X, axis=0)
    y_idx = np.concatenate(all_y_idx, axis=0)
    ref_arc = np.concatenate(all_ref_arc, axis=0)
    cand_arcs = np.concatenate(all_cand_arcs, axis=0)
    cand_dist = np.concatenate(all_cand_dist, axis=0)
    cand_valid = np.concatenate(all_cand_valid, axis=0)
    coords = np.concatenate(all_coords, axis=0)

    out_path = DIR_CACHE / f"features_{partition}.npz"
    np.savez_compressed(
        out_path,
        X=X,
        y_idx=y_idx,
        ref_arc=ref_arc,
        cand_arcs=cand_arcs,
        cand_dist=cand_dist,
        cand_valid=cand_valid,
        coords=coords,
        trip_ids=np.array(trip_ids),
        trip_starts=np.array([trip_slices[t][0] for t in trip_ids], dtype=np.int64),
        trip_ends=np.array([trip_slices[t][1] for t in trip_ids], dtype=np.int64),
    )

    overall_miss = float((y_idx == -1).mean())
    overall_dist_top1 = float((y_idx == 0).mean())
    summary = {
        "partition": partition,
        "n_trips": len(trip_ids),
        "n_observations": int(len(y_idx)),
        "top_k": TOP_K_CANDIDATES,
        "n_features": len(FEATURE_NAMES),
        "overall_candidate_miss_rate": round(overall_miss, 5),
        "naive_nearest_arc_top1_acc": round(overall_dist_top1, 5),
        "elapsed_sec": round(time.time() - t0, 2),
        "cache_file": str(out_path),
        "per_trip": trip_stats,
    }
    return summary


def main() -> None:
    schema = {
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "top_k": TOP_K_CANDIDATES,
        "candidate_radius_m": CANDIDATE_RADIUS_M,
        "gap_threshold_s": GAP_THRESHOLD_S,
    }
    with open(DIR_ARTIFACTS / "feature_schema_model4.json", "w") as f:
        json.dump(schema, f, indent=2)

    train_sum = build_partition_dataset("train", TRAIN_DIR)
    test_sum = build_partition_dataset("student_test", STUDENT_TEST_DIR)

    report = {"train": train_sum, "student_test": test_sum}
    with open(DIR_RESULTS / "feature_engineering_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    print("[Stage 02 Feature Engineering] Done!")
    for k, v in report.items():
        print(
            f"  {k}: {v['n_trips']} trips, {v['n_observations']} obs | "
            f"Candidate Miss Rate = {v['overall_candidate_miss_rate']*100:.2f}% | "
            f"Naive Nearest-Arc Top-1 = {v['naive_nearest_arc_top1_acc']*100:.2f}% | "
            f"{v['elapsed_sec']}s"
        )


if __name__ == "__main__":
    main()
