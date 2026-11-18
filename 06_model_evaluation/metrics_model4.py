"""Stage 06A: Self-contained evaluation metrics for Model 4 (NN-GLS + RLCD).

Computes candidate-level, observation-level, probability-calibration (Log-Loss, Brier, ECE),
route-continuity, and sequence-level arc F1 metrics strictly without HMM/Viterbi dependency.
"""
from __future__ import annotations

import heapq
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_data_processing"))
from config import GAP_THRESHOLD_S
from io_utils import TripData, build_adjacency


class ShortestPathEvaluator:
    """Lazy bounded Dijkstra helper used ONLY to evaluate road-network route continuity."""

    def __init__(self, adj: List[List[Tuple[int, int, float]]]) -> None:
        self.adj = adj
        self._cache: Dict[int, Tuple[float, Dict[int, float]]] = {}

    def shortest_distance(self, src: int, dst: int, max_dist: float) -> float:
        if src == dst:
            return 0.0
        cached = self._cache.get(src)
        if cached is not None:
            explored_r, dists = cached
            if dst in dists:
                d = dists[dst]
                return d if d <= max_dist else math.inf
            if explored_r >= max_dist:
                return math.inf

        dists: Dict[int, float] = {src: 0.0}
        pq: List[Tuple[float, int]] = [(0.0, src)]
        adj = self.adj
        while pq:
            d, u = heapq.heappop(pq)
            if d > max_dist:
                break
            if d > dists.get(u, math.inf):
                continue
            for v, _, w in adj[u]:
                nd = d + w
                if nd <= max_dist and nd < dists.get(v, math.inf):
                    dists[v] = nd
                    heapq.heappush(pq, (nd, v))

        self._cache[src] = (max_dist, dists)
        return dists.get(dst, math.inf)

    def shortest_arc_path(self, src: int, dst: int, max_dist: float = 500.0) -> Optional[List[int]]:
        if src == dst:
            return []
        dists: Dict[int, float] = {src: 0.0}
        parent: Dict[int, Tuple[int, int]] = {}
        pq: List[Tuple[float, int]] = [(0.0, src)]
        adj = self.adj
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                path_arcs = []
                cur = dst
                while cur != src:
                    p_node, aid = parent[cur]
                    path_arcs.append(aid)
                    cur = p_node
                path_arcs.reverse()
                return path_arcs
            if d > max_dist:
                break
            if d > dists.get(u, math.inf):
                continue
            for v, aid, w in adj[u]:
                nd = d + w
                if nd <= max_dist and nd < dists.get(v, math.inf):
                    dists[v] = nd
                    parent[v] = (u, aid)
                    heapq.heappush(pq, (nd, v))
        return None


def normalize_candidate_probs(
    raw_scores: np.ndarray,
    cand_valid: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Normalize per-observation candidate probabilities cleanly over valid slots."""
    p = np.where(cand_valid, np.clip(raw_scores.astype(np.float64), eps, 1.0), 0.0)
    s = p.sum(axis=1, keepdims=True)
    uniform = cand_valid.astype(np.float64)
    uniform = uniform / np.maximum(uniform.sum(axis=1, keepdims=True), 1.0)
    return np.where(s > 0.0, p / np.maximum(s, 1e-12), uniform)


def compute_ece(probs_k: np.ndarray, y_idx: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) on observation-level top-1 confidence (valid-hit obs)."""
    mask = y_idx >= 0
    if not np.any(mask):
        return float("nan")
    p = probs_k[mask]
    y = y_idx[mask]
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        in_bin = (conf >= lo) & (conf <= hi if b == n_bins - 1 else conf < hi)
        cnt = int(in_bin.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(float(correct[in_bin].mean()) - float(conf[in_bin].mean()))
    return float(ece)


def compute_reliability_bins(probs_k: np.ndarray, y_idx: np.ndarray, n_bins: int = 12) -> Dict[str, List[float]]:
    """Compute per-bin confidence, empirical accuracy, and counts for Reliability Diagram plots."""
    mask = y_idx >= 0
    p = probs_k[mask]
    y = y_idx[mask]
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, avg_confs, avg_accs, counts = [], [], [], []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        in_bin = (conf >= lo) & (conf <= hi if b == n_bins - 1 else conf < hi)
        cnt = int(in_bin.sum())
        centers.append(round(float(0.5 * (lo + hi)), 4))
        counts.append(cnt)
        if cnt > 0:
            avg_confs.append(round(float(conf[in_bin].mean()), 5))
            avg_accs.append(round(float(correct[in_bin].mean()), 5))
        else:
            avg_confs.append(None)
            avg_accs.append(None)
    return {"bin_centers": centers, "avg_conf": avg_confs, "avg_acc": avg_accs, "counts": counts}


def compute_multiclass_brier(probs_k: np.ndarray, y_idx: np.ndarray) -> float:
    mask = y_idx >= 0
    if not np.any(mask):
        return float("nan")
    p = probs_k[mask]
    y = y_idx[mask]
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def compute_listwise_log_loss(probs_k: np.ndarray, y_idx: np.ndarray, eps: float = 1e-7) -> float:
    mask = y_idx >= 0
    if not np.any(mask):
        return float("nan")
    p = np.clip(probs_k[mask], eps, 1.0)
    y = y_idx[mask]
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def compute_route_continuity_trip(
    trip: TripData,
    pred_arc_ids: np.ndarray,
    sp_eval: Optional[ShortestPathEvaluator] = None,
) -> Tuple[float, float]:
    """Compute (exact_1hop_continuity, path_connected_continuity) over non-gap consecutive steps."""
    T = len(pred_arc_ids)
    if T <= 1:
        return 1.0, 1.0
    if sp_eval is None:
        adj, _, _, _ = build_adjacency(trip)
        sp_eval = ShortestPathEvaluator(adj)

    dt = np.diff(trip.track_ts)
    gc = np.hypot(np.diff(trip.track_x), np.diff(trip.track_y))

    total = 0
    ok_1hop = 0
    ok_path = 0
    for t in range(1, T):
        if dt[t - 1] > GAP_THRESHOLD_S:
            continue
        total += 1
        a1, a2 = int(pred_arc_ids[t - 1]), int(pred_arc_ids[t])
        if a1 < 0 or a2 < 0:
            continue
        if a1 == a2 or trip.arcs[a1, 1] == trip.arcs[a2, 0]:
            ok_1hop += 1
            ok_path += 1
        else:
            cap = max(2.5 * float(gc[t - 1]) + 100.0, 180.0)
            d = sp_eval.shortest_distance(int(trip.arcs[a1, 1]), int(trip.arcs[a2, 0]), max_dist=cap)
            if math.isfinite(d):
                ok_path += 1

    if total == 0:
        return 1.0, 1.0
    return ok_1hop / total, ok_path / total


def compute_sequence_arc_f1(pred_arc_ids: np.ndarray, ref_route_arcs: np.ndarray) -> Tuple[float, float, float]:
    pred_set = set(int(a) for a in pred_arc_ids if a >= 0)
    ref_set = set(int(a) for a in ref_route_arcs if a >= 0)
    if not pred_set or not ref_set:
        return 0.0, 0.0, 0.0
    inter = len(pred_set & ref_set)
    prec = inter / len(pred_set)
    rec = inter / len(ref_set)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
    return prec, rec, f1


def collapse_predicted_route_arcs(
    trip: TripData,
    pred_arc_ids: np.ndarray,
    sp_eval: Optional[ShortestPathEvaluator] = None,
) -> List[int]:
    """Collapse consecutive duplicate predicted arcs (with light short-hop infill for map polyline continuity)."""
    if len(pred_arc_ids) == 0:
        return []
    collapsed = [int(pred_arc_ids[0])]
    for a in pred_arc_ids[1:]:
        ai = int(a)
        if ai >= 0 and ai != collapsed[-1]:
            collapsed.append(ai)
    if sp_eval is None:
        return collapsed
    out = [collapsed[0]]
    for nxt in collapsed[1:]:
        prev = out[-1]
        v_prev = int(trip.arcs[prev, 1])
        u_nxt = int(trip.arcs[nxt, 0])
        if v_prev != u_nxt:
            mid = sp_eval.shortest_arc_path(v_prev, u_nxt, max_dist=350.0)
            if mid:
                out.extend(mid)
        out.append(nxt)
    return out


def evaluate_predictions(
    cand_probs: np.ndarray,
    cand_arcs: np.ndarray,
    cand_valid: np.ndarray,
    y_idx: np.ndarray,
    ref_arc: np.ndarray,
    trips: Optional[List[TripData]] = None,
    trip_slices: Optional[List[Tuple[int, int]]] = None,
    sp_evals: Optional[Dict[str, ShortestPathEvaluator]] = None,
) -> Dict:
    """Compute complete PathSnap evaluation metrics directly from NN-GLS + RLCD probabilities (zero HMM)."""
    T, K = cand_arcs.shape
    probs_norm = normalize_candidate_probs(cand_probs, cand_valid)

    y_bin = np.zeros((T, K), dtype=np.int32)
    hit_mask = y_idx >= 0
    y_bin[hit_mask, y_idx[hit_mask]] = 1

    pred_k = np.argmax(np.where(cand_valid, probs_norm, -1.0), axis=1)
    pred_arcs = cand_arcs[np.arange(T), pred_k]

    y_pred_bin = np.zeros((T, K), dtype=np.int32)
    y_pred_bin[np.arange(T), pred_k] = 1

    v_flat = cand_valid.ravel()
    yb_flat = y_bin.ravel()[v_flat]
    yp_flat = y_pred_bin.ravel()[v_flat]
    pr_flat = np.clip(probs_norm.ravel()[v_flat], 1e-7, 1.0 - 1e-7)

    cand_acc = float(accuracy_score(yb_flat, yp_flat))
    cand_prec = float(precision_score(yb_flat, yp_flat, zero_division=0))
    cand_rec = float(recall_score(yb_flat, yp_flat, zero_division=0))
    cand_macro_f1 = float(f1_score(yb_flat, yp_flat, average="macro", zero_division=0))
    cand_pos_f1 = float(f1_score(yb_flat, yp_flat, average="binary", zero_division=0))
    cm = confusion_matrix(yb_flat, yp_flat, labels=[0, 1]).tolist()
    bin_ll = float(log_loss(yb_flat, pr_flat, labels=[0, 1]))

    obs_top1_acc = float(np.mean(pred_arcs == ref_arc))
    obs_top1_acc_on_hits = float(np.mean(pred_arcs[hit_mask] == ref_arc[hit_mask])) if np.any(hit_mask) else 0.0
    cand_miss_rate = float(np.mean(~hit_mask))

    listwise_ll = compute_listwise_log_loss(probs_norm, y_idx)
    brier = compute_multiclass_brier(probs_norm, y_idx)
    ece = compute_ece(probs_norm, y_idx)

    result = {
        "n_observations": int(T),
        "candidate_miss_rate": round(cand_miss_rate, 5),
        "obs_top1_accuracy": round(obs_top1_acc, 5),
        "obs_top1_accuracy_given_hit": round(obs_top1_acc_on_hits, 5),
        "macro_f1": round(cand_macro_f1, 5),
        "pos_f1": round(cand_pos_f1, 5),
        "precision": round(cand_prec, 5),
        "recall": round(cand_rec, 5),
        "candidate_accuracy": round(cand_acc, 5),
        "log_loss": round(listwise_ll, 5),
        "binary_log_loss": round(bin_ll, 5),
        "brier_score": round(brier, 5),
        "ece": round(ece, 5),
        "confusion_matrix": cm,
        "reliability_bins": compute_reliability_bins(probs_norm, y_idx),
    }

    if trips is not None and trip_slices is not None:
        cont_1hop_list, cont_path_list = [], []
        seq_prec_list, seq_rec_list, seq_f1_list = [], [], []
        per_trip_rows = []

        for trip, (s, e) in zip(trips, trip_slices):
            pa = pred_arcs[s:e]
            ra = ref_arc[s:e]
            py = y_idx[s:e]
            pp = probs_norm[s:e]
            ev = sp_evals.get(trip.trip_id) if sp_evals else None
            c1, cp = compute_route_continuity_trip(trip, pa, sp_eval=ev)
            cont_1hop_list.append(c1)
            cont_path_list.append(cp)

            t_acc = float(np.mean(pa == ra))
            t_miss = float(np.mean(py < 0))
            t_ll = compute_listwise_log_loss(pp, py)
            t_brier = compute_multiclass_brier(pp, py)
            t_ece = compute_ece(pp, py)
            sp, sr, sf = (0.0, 0.0, 0.0)
            if trip.route is not None:
                sp, sr, sf = compute_sequence_arc_f1(pa, trip.route)
                seq_prec_list.append(sp)
                seq_rec_list.append(sr)
                seq_f1_list.append(sf)

            per_trip_rows.append({
                "trip_id": trip.trip_id,
                "n_obs": int(e - s),
                "candidate_miss_rate": round(t_miss, 5),
                "obs_top1_accuracy": round(t_acc, 5),
                "route_continuity_1hop": round(c1, 5),
                "route_continuity": round(cp, 5),
                "seq_arc_precision": round(sp, 5),
                "seq_arc_recall": round(sr, 5),
                "seq_arc_f1": round(sf, 5),
                "log_loss": round(t_ll, 5),
                "brier_score": round(t_brier, 5),
                "ece": round(t_ece, 5),
            })

        result["route_continuity_1hop"] = round(float(np.mean(cont_1hop_list)), 5)
        result["route_continuity"] = round(float(np.mean(cont_path_list)), 5)
        result["mean_trip_top1_accuracy"] = round(float(np.mean([r["obs_top1_accuracy"] for r in per_trip_rows])), 5)
        if seq_f1_list:
            result["seq_arc_precision"] = round(float(np.mean(seq_prec_list)), 5)
            result["seq_arc_recall"] = round(float(np.mean(seq_rec_list)), 5)
            result["seq_arc_f1"] = round(float(np.mean(seq_f1_list)), 5)
        result["per_trip"] = per_trip_rows

    return result
